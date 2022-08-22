import numpy as np
import json
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import argparse
from analysis import print_confusion_as_csv, draw_auc, total_hit_topk, cal_confusion_matrix, output_predict_result_sgfs
from model import GoModel_lal
from dataloader import DataNegSample
import pandas as pd
from sklearn.metrics import matthews_corrcoef

def train(epoch, Loader):
    model.train()
    total_loss = 0.0
    total, hits = 0, [0]*args.label_size

    for step, (boards, positions, labels) in enumerate(Loader):
        if step%5 == 0:
            print('#', end='', flush=True)
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
    for step, (boards, positions, labels) in enumerate(Loader):
        boards = Variable(boards).to(args.device)
        positions = Variable(positions).to(args.device)
        labels = Variable(labels).to(args.device)
        with torch.no_grad():
            outs = model(args, boards, positions)

        # evaluate
        total += len(outs)
        for idx in range(args.label_size):
            hits[idx] += total_hit_topk(outs, labels,k=idx+1)
            
        # if draw:
        predicts.append( [ round(x,4) for x in outs.squeeze().tolist() ] )
        targets.append(labels.squeeze().tolist())
        
    # print("Testing")
    # print("Epoch:", epoch)
    # print("Testing top1 accuracy:", hits[0] / total)
    # for idx in range(args.label_size):
    #     print('top {} acc: {}/{}'.format(idx+1,hits[idx],total))

    
    confusion_matrix = cal_confusion_matrix(targets, predicts)
    

    if draw:
        stat = {'target': targets, 'predicts': predicts, 'sgf': [q['sgf_content'].replace('\n','') for q in _DataSimple_test.Q]}
        output_predict_result_sgfs(stat, args.eng_abbrs, tags_ch)
        df_stat = pd.DataFrame(stat)
        df_stat.to_csv("0327_multiclass_ysctrnivcn.csv",encoding='UTF-8')

    # print(targets)
    # print(predicts)



    return targets, predicts, confusion_matrix , hits[0], total, hits[0]/total
    
def run(epochs, trainLoader, validLoader, pretrain, save_model_path='./weight'):
    max_acc = -1.0
    for epoch in range(epochs):
        # training
        train(epoch+1, trainLoader)
        # testing
        _,_,_,_,_,test_acc = test(epoch+1, validLoader, draw=0)
        if test_acc > max_acc:
            max_acc = test_acc
            torch.save(model.state_dict(), save_model_path)
        if pretrain: print("Max Pretrained Testing accuracy:", max_acc, "\n")
        else: print("Max Finetuned Testing accuracy:", max_acc, "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "None")
    parser.add_argument('-d', '--device', default = 'cuda')
    parser.add_argument('--epoch', default = 10, type = int)
    parser.add_argument('--batch_size', default = 128, type = int) 
    parser.add_argument('--lr', default = 1e-4, type = float)
    parser.add_argument('--hidden_size', default = 256, type = int)
    parser.add_argument('--dim_in', default = 4, type = int)    
    
    # parser.add_argument('--code_size', default = 64, type = int)
    # parser.add_argument('--k_hop', default = 2, type = int)
    parser.add_argument('--train', default = 1, type = int)
    parser.add_argument('--pickle_train', default = '/mnt/nfs/work/yshsu0918/workspace/thesis/Dataset_onlybw/onlybw/D_FSHH_ysctrnivlkot_train.pickle', type = str)    
    parser.add_argument('--pickle_valid', default = '/mnt/nfs/work/yshsu0918/workspace/thesis/Dataset_onlybw/onlybw/D_FSHH_ysctrnivlkot_valid.pickle', type = str)    
    parser.add_argument('--pickle_test', default = '/mnt/nfs/work/yshsu0918/workspace/thesis/Dataset_onlybw/onlybw/D_FSHH_ysctrnivlkot_test.pickle', type = str)    
    parser.add_argument('--eng_abbrs', default = 'ys,ct,rn,iv,lk,ot', type = str)
    parser.add_argument('--model_path', default = './net/multiclass_ysctrnivlkot.pt', type = str)
    parser.add_argument('--label_size', default = 6, type = int)
    
    

    # tags_dict = {
    #     "ys": "官子",
    #     "ct": "切斷",
    #     "rn": "逃跑",
    #     "iv": "打入",
    #     "cn": "聯絡",
    #     # "ot": "其他",
    # }


    labels = ['']

    args = parser.parse_args()
    args.eng_abbrs = args.eng_abbrs.split(',')
    print(args)

    tags_dict = {
        "ys": "官子",
        "ct": "切斷",
        "rn": "逃跑",
        "iv": "打入",
        #"cn": "聯絡", 
        "lk": "封鎖",
        #"lv": "做活",
        # "ef": "補 補強 自補 補角",
        #"op": "治孤",
        # "em": "搶佔 急所 搶到",   
        "ot": "其他",
    }
    if 'ot' not in args.eng_abbrs:
        del tags_dict['ot']
    tags_ch = list(tags_dict.items())

    print('Term Model Training')
    torch.cuda.set_device(1)

    _DataSimple_train = DataNegSample(args.pickle_train)

    # weights = torch.DoubleTensor(_DataSimple_train.make_weights_for_balanced_classes())                                       
    # sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = data.DataLoader(dataset=_DataSimple_train, 
                                    batch_size=32,
                                    shuffle=True,  
                                    num_workers=4, 
                                    pin_memory=True)

    _DataSimple_valid = DataNegSample(args.pickle_valid)
    valid_loader = data.DataLoader(dataset=_DataSimple_valid,
                                batch_size=1,
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=True)

    _DataSimple_test = DataNegSample(args.pickle_test)
    test_loader = data.DataLoader(dataset=_DataSimple_test,
                                batch_size=1,
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=True)


    model = GoModel_lal(label_size = args.label_size, hidden_size=256, dim_in=args.dim_in).to(args.device)
    optim = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    # Pretrain
    if args.train:
        run(args.epoch, train_loader, valid_loader, True,save_model_path=args.model_path)
        
    else:
        model.load_state_dict(torch.load(args.model_path))
 
        t1, p1, c1, hits1, total1, _ = test(3, valid_loader,draw=0)       
        t2, p2, c2, hits2, total2, _ = test(3, test_loader,draw=0)

        confusion_matrix = c1 + c2
        hits = hits1 + hits2
        total = total1 + total2 

        tt = [ np.argmax(x) for x in t1+t2]
        pp = [ np.argmax(x) for x in p1+p2]
        print_confusion_as_csv(args.eng_abbrs, confusion_matrix)
        print( 'mcc', matthews_corrcoef(tt, pp))

        print( 'hits:{} total:{} acc:{}'.format( hits,total, hits/total))
    
    # Finetune
    # run(args.epoch_fine, trainLoader_fine, testLoader_fine, False)
    