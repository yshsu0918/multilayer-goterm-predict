import json
import pickle
import random
import argparse
import re
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable

from dataloader import DataMulticlassGoModel, pickle_reader
from model import GoModel

from operator import itemgetter

import numpy as np

from analysis import draw_auc

def train_multiclass_onevsall_A_GoModel(args,data_dict,neg_sampling_k = 5):
    print('Training multiclass A GoModel...')
    eng_abbrs = args.eng_abbrs
    nets = []
    
    for eng in eng_abbrs:
        print('DataA traindataloader eng {} neg_sampling_k {}'.format(eng, neg_sampling_k),flush = True)
        _DataB = DataMulticlassGoModel(args , eng, data_dict, ratio = 0.7,is_train = True, neg_sampling_k = neg_sampling_k)

        traindataloader = DataLoader(dataset=_DataB, 
                                    batch_size=5,
                                    shuffle=True)

        net = GoModel(label_size = 1, hidden_size=256).to(args.device)
        optimizer = torch.optim.Adam( net.parameters(), lr= 1e-4 , weight_decay=3e-4)
        loss_func = torch.nn.BCELoss()  #

        for epoch in range(args.epochA):

            for step, (data, pos, label) in enumerate(traindataloader):

                data = Variable(data).to(args.device)
                pos = Variable(pos).to(args.device)
                label = Variable(label).to(args.device)

                output = net(args, data, pos)          #把data丟進網路中
                loss = loss_func(output, label)
                
                optimizer.zero_grad()      #計算loss,初始梯度
                loss.backward()            #反向傳播
                optimizer.step()       

        nets.append((neg_sampling_k, eng,net))
    return nets


def test_each_model(args,nets, neg_sampling_k = 5,threshold = 0.5):
    # eng_abbrs = args.eng_abbrs

    print('Test each A GoModel and drawauc...')
    data_dict, _ = pickle_reader(args)
    for net_idx in range(len(nets)) :
        neg_sampling_k, eng , net = nets[net_idx]   

        _DataB = DataMulticlassGoModel(args , eng, data_dict, ratio = 0.7,is_train = False, neg_sampling_k = neg_sampling_k)
        testdataloader = DataLoader(dataset=_DataB,
                                    batch_size=1,
                                    shuffle=False)
         
        targets, predicts = [], []

        for step, (data, pos, label) in enumerate(testdataloader):
            data = Variable(data).to(args.device)
            pos = Variable(pos).to(args.device)
            label = Variable(label).to(args.device)
            with torch.no_grad():
                output = net(args, data, pos)          #把data丟進網路中

            predicts.append(output.squeeze().tolist())
            targets.append(label.squeeze().tolist())

        draw_auc(targets, predicts, savefig_filename='./result/multiclass{}_{}vsother_auc.png'.format(neg_sampling_k,eng) )







def test_multiclass_onevsall_A_GoModel(args,nets, neg_sampling_k = 5,threshold = 0.5):
    print('Test multiclass A GoModel...')
    eng_abbrs = args.eng_abbrs
    data_dict, _ = pickle_reader(args)

    for eng in eng_abbrs :

        # print('DataA traindataloader eng {} neg_sampling_k {}'.format(eng, neg_sampling_k),flush = True)
        _DataB = DataMulticlassGoModel(args , eng, data_dict, ratio=0.7, is_train = False, neg_sampling_k = neg_sampling_k)

        testdataloader = DataLoader(dataset=_DataB,
                                    batch_size=1,
                                    shuffle=False)


        correct , wrong = 0.0 , 0.0
        for _eng in ['Target'] + [ _net[1] for _net in nets ]  + ['Predict']:
            print(_eng, end = ' , ')
        print('')        
        
        for step, (data, pos, label) in enumerate(testdataloader):
            other_info = _DataB.other_info[step]
            print(other_info[1][0], end =' , ')

            each_class_predicts =[]
            for j, _eng in enumerate(eng_abbrs):
                neg_sampling_k, _ , net = nets[j]                

                data = Variable(data).to(args.device)
                pos = Variable(pos).to(args.device)
                label = Variable(label).to(args.device)

                with torch.no_grad():
                    output = net(args, data, pos)          #把data丟進網路中

                print(output.squeeze().tolist() , end=' , ')

                each_class_predicts.append(output.squeeze().tolist())

            max_index = each_class_predicts.index(max(each_class_predicts))
            if max(each_class_predicts) > threshold:
                predict_tag = nets[ max_index ][1]
            else:
                predict_tag = 'neg'


            if predict_tag in other_info[1].__str__():
                print('{} , '.format(predict_tag), other_info[0])
                correct+=1
            else:
                print('{} , '.format(predict_tag), other_info[0])
                wrong+=1
    
        print('{} correct {} wrong {} acc {}'.format(eng, correct, wrong, correct/(correct+wrong)))



def test(args,nets,neg_sampling_k=5):
    print('-------TEST START--------')
    test_each_model(args,nets,neg_sampling_k=neg_sampling_k)
    test_multiclass_onevsall_A_GoModel(args,nets,neg_sampling_k=neg_sampling_k)
    print('-------TEST END--------')    

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description = "modelA train=0 => load and demo. train=1")
    parser.add_argument('--device', default = 'cuda:2')
    parser.add_argument('--train', default = 1, type = int)
    parser.add_argument('--epochA', default = 5, type = int)
    
    parser.add_argument('--training_pickle_filepath', 
        default = '/mnt/nfs/work/yshsu0918/lal/other/test/Dataset/D_FSH_ivsncn_training.pickle', type = str)
    parser.add_argument('--neg_training_pickle_filepath', 
        default = '/mnt/nfs/work/yshsu0918/lal/other/test/Dataset/D_FSH_ivsncn_neg_training.pickle', type = str)        
    parser.add_argument('--eng_abbrs', default = 'iv,sn,cn', type = str)
    
    args = parser.parse_args()
    args.eng_abbrs = args.eng_abbrs.split(',')
    print(args)
    abbr_filenameprefix = ''.join(args.eng_abbrs)

    if args.train:
        print('-----train-----')
        data_dict, _ = pickle_reader(args)
        for neg_sampling_k in [5]:
            nets = train_multiclass_onevsall_A_GoModel(args,data_dict,neg_sampling_k = neg_sampling_k)
            with open('./net/multiclass_onevsall_{}_{}_GoModel.pickle'.format(neg_sampling_k,abbr_filenameprefix), 'wb') as fout:
                pickle.dump(nets, fout)

            test(args,nets,neg_sampling_k=neg_sampling_k)

    else:
        for neg_sampling_k in [5]:
            with open('./net/multiclass_onevsall_{}_{}_GoModel.pickle'.format(neg_sampling_k,abbr_filenameprefix),'rb') as fin:
                nets = pickle.load(fin)

            test(args,nets,neg_sampling_k=neg_sampling_k)