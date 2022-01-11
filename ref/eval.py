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
    trainset_fine = TermLoader(args=args, root="../data/finetune/train/", mode="test", library=library, pretrain=False)
    testset_fine = TermLoader(args=args, root="../data/finetune/test/", mode="test", library=library, pretrain=False)
    trainLoader_fine = data.DataLoader(dataset=trainset_fine, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    testLoader_fine = data.DataLoader(dataset=testset_fine, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return trainLoader_fine, testLoader_fine

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

def label_correct_statis(outputs, label, label_dict, correct_dict):
    for row in range(len(label)):
        _, indices = outputs[row].topk(3, dim = 0, largest = True, sorted = True)
        for idx in range(len(label[row])):
            if label[row][idx].item() == 1.0:
                term = library[idx]
                if term not in label_dict: label_dict[term] = 1
                else: label_dict[term] += 1
                # top-k
                if idx in indices:
                #if outputs[row][idx].item() >= 0.5:
                    if term not in correct_dict: correct_dict[term] = 1
                    else: correct_dict[term] += 1
                        
def eval(Loader, label_dict, correct_dict):
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
        label_correct_statis(outs, labels, label_dict, correct_dict)

    print_info(label_dict, correct_dict)    
    print("Testing accuracy:", hit / total)         
    print(hit, "/", total, "\n")
    
    return hit / total

def print_info(label_dict, correct_dict):
    rate_dict = {}
    for key, value in label_dict.items():
        if key not in correct_dict: correct_dict[key] = 0 
        rate = correct_dict[key] / label_dict[key]
        rate_dict[key] = rate
    
    rate_dict = sorted(rate_dict.items(), key = lambda x:x[1], reverse = True)
    for item in rate_dict:
        key, value = item[0], item[1]
        print("{term}: {correct}/{total}, {rate}%".format(term = key, correct = correct_dict[key], total = label_dict[key], rate = round(value*100, 2)))
        #print(" {correct} / {total}".format(correct = correct_dict[key], total = label_dict[key]), file = out)
        #print("{term}".format(term=key), file = out)
        #print("{rate}%".format(rate = round(value*100, 2)), file=out)
    print()

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
    print('Term Model Evaluation')

    library = load_library()
    trainLoader_fine, testLoader_fine = collect_data()
    model = GoModel(label_size=len(library), hidden_size=args.hidden_size, code_size=args.code_size, k_hop=args.k_hop).to(args.device)
    model.load_state_dict(torch.load('term_model_weight'))
    eval(trainLoader_fine, {}, {})
    eval(testLoader_fine, {}, {})