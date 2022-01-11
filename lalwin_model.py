import json
import pickle
import random
import argparse
import re
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from dataloader import DataA, DataB, DataGoModel
from model import NetA, NetB, GoModel
from global_var import *
from boardprocess import GetBoard

def cal_DataBtarget(args, eng_abbrs,terms_cut_threshold=0.5):
    input_training_pickle = args.input_training_pickle
    training_data = []
    with open(input_training_pickle, 'rb') as fin:
        training_data = pickle.load(fin)
    
    #Count term in each category   
    stat = {}
    for k in eng_abbrs:
        stat[k] = dict()

    for item in training_data:
        for i in range(len(eng_abbrs)):
            if eng_abbrs[i] in item['specific_terms']:
                for term in item['specific_terms'][eng_abbrs[i]]:
                    #print(eng_abbrs[i],term)
                    if term == '接':
                        continue

                    if term not in stat[eng_abbrs[i]]:
                        stat[eng_abbrs[i]][term] = 1
                    else:                            
                        stat[eng_abbrs[i]][term] += 1

    DataBtarget = dict()
    for i in range(len(eng_abbrs)):
        k = eng_abbrs[i]
        buf = sorted(stat[k].items(), key=lambda x:x[1],reverse=True)
        #buf = buf[0 : int(len(buf)**terms_cut_threshold)]
        DataBtarget[k] = [ x[0] for x in buf]
        print(k , DataBtarget[k])
        print(buf)
    return DataBtarget



def trainB_GoModel(args):
    print('Training B...')
    eng_abbrs = args.eng_abbrs
    nets = []
    DataBtarget = cal_DataBtarget(args, eng_abbrs)
    for eng in eng_abbrs :
        print('DataB traindataloader')
        _DataB = DataGoModel(DataBtarget, eng = eng, eng_abbrs = eng_abbrs, ratio = 0.1, is_train = True, input_training_pickle = args.input_training_pickle)
        
        traindataloader = DataLoader(dataset=_DataB,
                                    batch_size=1,
                                    shuffle=True)

        net = GoModel(label_size=len(DataBtarget[eng]), hidden_size=256)
        optimizer = torch.optim.Adam( net.parameters(), lr= 0.02 , weight_decay=3e-4)
        loss_func = torch.nn.BCELoss()  #

        for epoch in range(args.epochB):

            for step, (data, pos, label) in enumerate(traindataloader):
                data = data.to(args.device)
                pos = pos.to(args.device)
                label = label.to(args.device)

                output = net(args, data, pos)          #把data丟進網路中
                loss = loss_func(output, label)
                
                optimizer.zero_grad()      #計算loss,初始梯度
                loss.backward()            #反向傳播
                optimizer.step()       

                if step % 100 == 0:
                    print('Epoch:', epoch, '|step:', step, '|train loss:%.4f'%loss.data)

                #每100steps輸出一次train loss
            print('Epoch:', epoch, '|train loss:%.4f'%loss.data)

        nets.append((eng,net))
    return nets, DataBtarget
                    

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description = "train=0 => load and demo. train=1")
    parser.add_argument('--device', default = 'cuda:2')
    parser.add_argument('--train', default = 1, type = int)
    parser.add_argument('--epochA', default = 5, type = int)
    parser.add_argument('--epochB', default = 1, type = int)
    
    parser.add_argument('--input_training_pickle', default = '/mnt/nfs/work/yshsu0918/lal/other/test/HumanCorrectionTerm_training.pickle', type = str)
    parser.add_argument('--eng_abbrs', default = 'gt', type = str)
    parser.add_argument('--demo_result', default = '/mnt/nfs/work/yshsu0918/lal/other/test/result.pickle', type = str)
    args = parser.parse_args()
    args.eng_abbrs = args.eng_abbrs.split(',')
    print(args)

    trainB_GoModel(args)