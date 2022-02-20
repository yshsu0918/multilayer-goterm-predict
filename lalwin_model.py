import json
import pickle
import random
import argparse
import re
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable

from dataloader import DataA, DataB, DataGoModel, DataNegSampleGoModel
from model import NetA, NetB, GoModel
from global_var import *
from boardprocess import GetBoard
import heapq
from operator import itemgetter

def total_hit_topk(outputs, label, k):
    hit = 0
    for row in range(len(outputs)):
        # top k
        _, indices = outputs[row].topk(k, dim = 0, largest = True, sorted = True)
        for j in range(len(indices)):
            if label[row][indices[j].item()].item() == 1.0:
                hit += 1
                break
    return hit



def cal_DataBtarget(args, eng_abbrs,terms_cut_threshold=0.5):
    input_training_pickle = args.input_training_pickle
    training_data = []
    with open(input_training_pickle, 'rb') as fin:
        training_data = pickle.load(fin)
    
    #Count term in each category
    print(training_data[0])
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
        _DataB = DataGoModel(DataBtarget, eng = eng, eng_abbrs = eng_abbrs, ratio = 0.5, is_train = True, input_training_pickle = args.input_training_pickle)
        
        traindataloader = DataLoader(dataset=_DataB,
                                    batch_size=1,
                                    shuffle=True)

        net = GoModel(label_size=len(DataBtarget[eng]), hidden_size=256).to(args.device)
        optimizer = torch.optim.Adam( net.parameters(), lr= 1e-4 , weight_decay=3e-4)
        loss_func = torch.nn.BCELoss()  #

        for epoch in range(args.epochB):

            for step, (data, pos, label) in enumerate(traindataloader):
                data = Variable(data).to(args.device)
                pos = Variable(pos).to(args.device)
                label = Variable(label).to(args.device)


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


def testB_GoModel(args, nets, DataBtarget, max_topk=10):
    print('Test B...')
    eng_abbrs = args.eng_abbrs
    total = 0
    hits = []
    content = ''
    for i, eng in enumerate(eng_abbrs) :
        print('DataB testdataloader')
        _DataB = DataGoModel(DataBtarget, eng = eng, eng_abbrs = eng_abbrs, ratio = 0.01, is_train = False, input_training_pickle = args.input_training_pickle)
        testdataloader = DataLoader(dataset=_DataB,
                                    batch_size=1,
                                    shuffle=False)

        _, net = nets[i]
        loss_func = torch.nn.BCELoss()  #

        for step, (data, pos, label) in enumerate(testdataloader):
            data = Variable(data).to(args.device)
            pos = Variable(pos).to(args.device)
            label = Variable(label).to(args.device)

            with torch.no_grad():
                output = net(args, data, pos)          #把data丟進網路中
            
        
            hits.append( [ total_hit_topk(output, label,k) for k in range(1,max_topk) ] )

            total += len(output)

            y = label.squeeze().tolist()
            predict_y = output.squeeze().tolist()

            
            idxs = heapq.nlargest(3, enumerate(predict_y), key=itemgetter(1))
            ans_idx = heapq.nlargest(3, enumerate(y), key=itemgetter(1))

            buf = idxs.__str__() + ' / ' + ans_idx.__str__() + ' / '
            #print('Predict: ', end='')
            for idx, value in idxs:
                #print(DataBtarget[eng][idx], end=', ')
                buf += DataBtarget[eng][idx] + ','
            
            buf += ' / '

            
            # print('Ans: ', DataBtarget[eng][ans_idx[0][0]])
            buf += DataBtarget[eng][ans_idx[0][0]] + '\n'

            content += buf
    
    with open('test_GoModel.result', 'w') as fout:
        fout.write(content)
    
    print("Testing")
    
    
    for k in range(1,min(len(hits[0]), max_topk)):
        hitkcount = 0
        for hit in hits:
            hitkcount += hit[k]
        print('top {} accuracy {}'.format(k, float(hitkcount)/float(total)))
                


def test_getboard(args):
    input_training_pickle = args.input_training_pickle
    training_data = []
    with open(input_training_pickle, 'rb') as fin:
        training_data = pickle.load(fin)

    for item in training_data[:10]:
        print(item['sgf_content'])


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description = "train=0 => load and demo. train=1")
    parser.add_argument('--device', default = 'cuda:2')
    parser.add_argument('--train', default = 1, type = int)
    parser.add_argument('--epochA', default = 5, type = int)
    parser.add_argument('--epochB', default = 5, type = int)
    
    parser.add_argument('--input_training_pickle', default = '/mnt/nfs/work/yshsu0918/lal/other/test/VM/lalwinshortsentence_hsusplit_training.pickle', type = str)
    parser.add_argument('--eng_abbrs', default = 'gt', type = str)
    parser.add_argument('--demo_result', default = '/mnt/nfs/work/yshsu0918/lal/other/test/result.pickle', type = str)
    args = parser.parse_args()
    args.eng_abbrs = args.eng_abbrs.split(',')
    print(args)

    if args.train:
        nets, DataBtarget = trainB_GoModel(args)

        with open('B_GoModel.pickle', 'wb') as fout:
            pickle.dump(nets, fout)
        with open('DataBtarget.pickle', 'wb') as fout:
            pickle.dump(DataBtarget, fout)

    else:
        with open('B_GoModel.pickle', 'rb') as fin:
            nets = pickle.load(fin)
        with open('DataBtarget.pickle', 'rb') as fin:
            DataBtarget = pickle.load(fin)

    testB_GoModel(args, nets, DataBtarget)