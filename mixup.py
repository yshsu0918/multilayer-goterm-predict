import numpy as np
import json
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import argparse
from analysis import draw_auc, total_hit_topk, cal_confusion_matrix, output_predict_result_sgfs
from model import GoModel_frank
from dataloader import DataNegSample
import pandas as pd



def train(epoch, Loader, Loader2=None):
    model.train()
    total_loss = 0.0
    total, hits = 0, [0]*args.label_size

    for step, (boards, labels) in enumerate(Loader): # no position
        if step%5 == 0:
            print('#', end='', flush=True)
        boards = Variable(boards).to(args.device)
        # positions = Variable(positions).to(args.device)
        labels = Variable(labels).to(args.device)
        
        optim.zero_grad()
        outs = model( boards)
        loss = loss_fn(outs, labels)
        loss.backward()
        optim.step()
        total_loss += loss.item()

        # evaluate
        total += len(outs)

        for idx in range(args.label_size):
            hits[idx] += total_hit_topk(outs, labels,k=idx+1)
        
    print("###############################")
    print("Training")
    print("Epoch:", epoch) 
    print("Loss:", total_loss)
    print("Training top1 accuracy:", hits[0] / total) 

    for idx in range(args.label_size):
        print('top {} acc: {}/{}'.format(idx+1,hits[idx],total))

def mixup_train(epoch, Loader1, Loader2):
    model.train()
    total_loss = 0.0
    total, hits = 0, [0]*args.label_size

    step = 0
    for (boards1, labels1), (boards2, labels2) in zip(Loader1,Loader2): # no position
        step += 1
        if step%5 == 0:
            print('#', end='', flush=True)

        lam = np.random.beta(1,1)

        boards = Variable(boards1*lam + boards2*(1-lam)).to(args.device)
        labels = Variable(labels1*lam + labels2*(1-lam)).to(args.device)
        
        optim.zero_grad()
        outs = model( boards)
        loss = loss_fn(outs, labels)
        loss.backward()
        optim.step()
        total_loss += loss.item()

        # evaluate
        total += len(outs)

        for idx in range(args.label_size):
            hits[idx] += total_hit_topk(outs, labels,k=idx+1)
        
    print("###############################")
    print("Training")
    print("Epoch:", epoch) 
    print("Loss:", total_loss)
    print("Training top1 accuracy:", hits[0] / total) 

    for idx in range(args.label_size):
        print('top {} acc: {}/{}'.format(idx+1,hits[idx],total))

    
def test(epoch, Loader, draw=False):
    model.eval()
    total, hits = 0, [0]*args.label_size
    predicts = []
    targets = []    
    for step, (boards, labels) in enumerate(Loader):
        boards = Variable(boards).to(args.device)
        # positions = Variable(positions).to(args.device)
        labels = Variable(labels).to(args.device)
        with torch.no_grad():
            outs = model( boards)

        # evaluate
        total += len(outs)
        for idx in range(args.label_size):
            hits[idx] += total_hit_topk(outs, labels,k=idx+1)
            
        # if draw:
        predicts.append( [ round(x,4) for x in outs.squeeze().tolist() ] )
        targets.append(labels.squeeze().tolist())
        
    print("Testing")
    print("Epoch:", epoch)
    print("Testing top1 accuracy:", hits[0] / total)         
    for idx in range(args.label_size):
        print('top {} acc: {}/{}'.format(idx+1,hits[idx],total))
    
    cal_confusion_matrix(targets, predicts)

    if draw:

        stat = {'target': targets, 'predicts': predicts, 'sgf': [q['sgf_content'].replace('\n','') for q in _DataSimple_test.Q]}

        output_predict_result_sgfs(stat, args.eng_abbrs, tags_ch)

        df_stat = pd.DataFrame(stat)
        df_stat.to_csv("0413_multiclass_noposition_ysctrnivcn.csv",encoding='UTF-8')

    return hits[0] / total
    
def run(epochs, trainLoader, validLoader, pretrain, save_model_path='./weight', mixup=False, train_loader2=None):
    max_acc = -1.0
    for epoch in range(epochs):
        # training
        if mixup:
            mixup_train(epoch+1, trainLoader, train_loader2)
        else:
            train(epoch+1, trainLoader)
        # testing
        test_acc = test(epoch+1, validLoader, draw=0)
        if test_acc > max_acc:
            max_acc = test_acc
            torch.save(model.state_dict(), save_model_path)
        if pretrain: print("Max Pretrained Testing accuracy:", max_acc, "\n")
        else: print("Max Finetuned Testing accuracy:", max_acc, "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "None")
    parser.add_argument('-d', '--device', default = 'cuda')
    parser.add_argument('--epoch', default = 20, type = int)
    parser.add_argument('--batch_size', default = 128, type = int) 
    parser.add_argument('--lr', default = 1e-4, type = float)
    parser.add_argument('--hidden_size', default = 256, type = int)
    # parser.add_argument('--code_size', default = 64, type = int)
    # parser.add_argument('--k_hop', default = 2, type = int)
    parser.add_argument('--train', default = 1, type = int)
    parser.add_argument('--pickle_train', default = '/mnt/nfs/work/yshsu0918/workspace/thesis/Dataset/D_FSHH/D_FSHH_ysctrnivcn_train.pickle', type = str)    
    parser.add_argument('--pickle_valid', default = '/mnt/nfs/work/yshsu0918/workspace/thesis/Dataset/D_FSHH/D_FSHH_ysctrnivcn_valid.pickle', type = str)    
    parser.add_argument('--pickle_test', default = '/mnt/nfs/work/yshsu0918/workspace/thesis/Dataset/D_FSHH/D_FSHH_ysctrnivcn_test.pickle', type = str)    
    parser.add_argument('--eng_abbrs', default = 'ys,ct,rn,iv,cn', type = str)
    parser.add_argument('--model_path', default = './net/mixup_multiclass_noposition_ysctrnivcn.pt', type = str)
    parser.add_argument('--label_size', default = 5, type = int)
    tags_dict = {
        "ys": "官子",
        "ct": "切斷",
        "rn": "逃跑",
        "iv": "打入",
        "cn": "聯絡",
        # "ot": "其他",
    }
    tags_ch = list(tags_dict.items())



    labels = ['']

    args = parser.parse_args()
    args.eng_abbrs = args.eng_abbrs.split(',')
    print('Term Model Training')
    torch.cuda.set_device(1)
    

    _DataSimple_train = DataNegSample(args.pickle_train,need_position=False)
    _DataSimple_train2 = DataNegSample(args.pickle_train,need_position=False)
    # weights = torch.DoubleTensor(_DataSimple_train.make_weights_for_balanced_classes())                                       
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = data.DataLoader(dataset=_DataSimple_train, 
                                    batch_size=32,
                                    shuffle=True,  
                                    num_workers=4, 
                                    pin_memory=True)

    train_loader2 = data.DataLoader(dataset=_DataSimple_train2, 
                                    batch_size=32,
                                    shuffle=True,  
                                    num_workers=4, 
                                    pin_memory=True)


    _DataSimple_valid = DataNegSample(args.pickle_valid,need_position=False)
    valid_loader = data.DataLoader(dataset=_DataSimple_valid,
                                batch_size=1,
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=True)

    _DataSimple_test = DataNegSample(args.pickle_test,need_position=False)
    test_loader = data.DataLoader(dataset=_DataSimple_test,
                                batch_size=1,
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=True)


    model = GoModel_frank(OutputSize = args.label_size, InputChannel = 8, ChannelSize = 256).to(args.device)
    optim = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    # Pretrain
    if args.train:
        # run(args.epoch, train_loader, valid_loader, True,save_model_path=args.model_path)
        run(args.epoch, train_loader, valid_loader, True,save_model_path=args.model_path, mixup =True, train_loader2= train_loader2)
        
        
    else:
        model.load_state_dict(torch.load(args.model_path))
        test(3, test_loader,draw=1)
    
    
    # Finetune
    # run(args.epoch_fine, trainLoader_fine, testLoader_fine, False)
    