import numpy as np
import json
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import argparse
from dataloader import TermLoader
from model import GoModel

def load_library():
    library = []
    with open('../library.json', 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            Q = json.loads(line)
            if Q['chinese_name'] not in library:
                library.append(Q['chinese_name'])
    library.append('None')
    return library

def collect_data():
    trainset_pre = TermLoader(args=args, root="../data/pretrain/train/", mode="train", library=library, pretrain=True)
    testset_pre = TermLoader(args=args, root="../data/pretrain/test/", mode="test", library=library, pretrain=True)
    trainLoader_pre = data.DataLoader(dataset=trainset_pre, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testLoader_pre = data.DataLoader(dataset=testset_pre, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    trainset_fine = TermLoader(args=args, root="../data/finetune/train/", mode="test", library=library, pretrain=False)
    testset_fine = TermLoader(args=args, root="../data/finetune/test/", mode="test", library=library, pretrain=False)
    trainLoader_fine = data.DataLoader(dataset=trainset_fine, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testLoader_fine = data.DataLoader(dataset=testset_fine, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return trainLoader_pre, testLoader_pre, trainLoader_fine, testLoader_fine
    #return trainLoader_fine, testLoader_fine

def total_hit_topk(outputs, label):
    hit = 0
    for row in range(len(outputs)):
        # top k
        _, indices = outputs[row].topk(3, dim = 0, largest = True, sorted = True)
        for j in range(len(indices)):
            if label[row][indices[j].item()].item() == 1.0:
                hit += 1
                break
    return hit

def train(epoch, Loader):
    model.train()
    total_loss = 0.0
    total, hit = 0, 0

    for step, (boards, positions, labels) in enumerate(Loader):
        boards = Variable(boards).to(args.device)
        positions = Variable(positions).to(args.device)
        labels = Variable(labels).to(args.device)
        
        optim.zero_grad()
        outs = model(args, boards, positions)
        loss = loss_fn(outs, labels)
        loss.backward()
        optim.step()
        total_loss += loss.item()

        # evaluate
        total += len(outs)
        hit += total_hit_topk(outs, labels)
        
    print("###############################")
    print("Training")
    print("Epoch:", epoch) 
    print("Loss:", total_loss)
    print("Training accuracy:", hit / total)         
    print(hit, "/", total, "\n")
    
def test(epoch, Loader):
    model.eval()
    total, hit = 0, 0
    for step, (boards, positions, labels) in enumerate(Loader):
        boards = Variable(boards).to(args.device)
        positions = Variable(positions).to(args.device)
        labels = Variable(labels).to(args.device)
        with torch.no_grad():
            outs = model(args, boards, positions)

        # evaluate
        total += len(outs)
        hit += total_hit_topk(outs, labels)
        
    print("Testing")
    print("Epoch:", epoch)
    print("Testing accuracy:", hit / total)         
    print(hit, "/", total, "\n")
    
    return hit / total
    
def run(epochs, trainLoader, testLoader, pretrain):
    max_acc = -1.0
    for epoch in range(epochs):
        # training
        train(epoch+1, trainLoader)
        # testing
        test_acc = test(epoch+1, testLoader)
        if test_acc > max_acc:
            max_acc = test_acc
            torch.save(model.state_dict(), './term_model_weight')
        if pretrain: print("Max Pretrained Testing accuracy:", max_acc, "\n")
        else: print("Max Finetuned Testing accuracy:", max_acc, "\n")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "None")
    parser.add_argument('-d', '--device', default = 'cuda')
    parser.add_argument('--epoch_pre', default = 2, type = int)
    parser.add_argument('--epoch_fine', default = 20, type = int)
    parser.add_argument('--batch_size', default = 128, type = int) 
    parser.add_argument('--lr', default = 1e-4, type = float)
    parser.add_argument('--hidden_size', default = 256, type = int)
    parser.add_argument('--code_size', default = 64, type = int)
    parser.add_argument('--k_hop', default = 2, type = int)
    args = parser.parse_args()
    print('Term Model Training')
    torch.cuda.set_device(1)
    
    library = load_library()
    trainLoader_pre, testLoader_pre, trainLoader_fine, testLoader_fine = collect_data()
    #trainLoader_fine, testLoader_fine = collect_data()
    model = GoModel(label_size=len(library), hidden_size=args.hidden_size, code_size=args.code_size, k_hop=args.k_hop).to(args.device)
    optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-4)
    loss_fn = nn.BCELoss()

    # Pretrain
    run(args.epoch_pre, trainLoader_pre, testLoader_pre, True)
    # Finetune
    run(args.epoch_fine, trainLoader_fine, testLoader_fine, False)
    