import json
import pickle
import random
import argparse
import re
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable

from dataloader import DataMulticlass1v1GoModel, TestMulticlassGoModel, pickle_reader
from model import GoModel

from operator import itemgetter

import numpy as np

from analysis import draw_auc

def train_multiclass_onevsone_A_GoModel(args,data_dict):
    print('Training multiclass_onevsone A GoModel...')
    eng_abbrs = args.eng_abbrs
    nets = []
    
    for enga in eng_abbrs :
        for engb in eng_abbrs:
            if enga == engb:
                continue
            
            print('DataA traindataloader enga {} engb {}'.format(enga,engb),flush = True)
            _DataB = DataMulticlass1v1GoModel(args, enga , engb , data_dict, ratio = 0.9,is_train = True)

            weights = torch.DoubleTensor(_DataB.make_weights_for_balanced_classes())                                       
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

            traindataloader = DataLoader(dataset=_DataB, 
                                        batch_size=32,
                                        sampler=sampler)

            net = GoModel(label_size = 1, hidden_size=256).to(args.device)
            optimizer = torch.optim.Adam( net.parameters(), lr= 1e-4 , weight_decay=3e-2)
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

            nets.append( (enga,engb,net) )
    return nets


def test_each_model(args,nets, data_dict):
    eng_abbrs = args.eng_abbrs

    print('Test each A GoModel and drawauc...')
    
    for enga, engb , net in nets:

        _DataB = DataMulticlass1v1GoModel(args, enga, engb, data_dict,
            ratio = 0.9,is_train = False )

        testdataloader = DataLoader(dataset=_DataB,
                                    batch_size=1,
                                    shuffle=False)
         
        predicts = []
        targets = []

        for step, (data, pos, label) in enumerate(testdataloader):
            data = Variable(data).to(args.device)
            pos = Variable(pos).to(args.device)
            label = Variable(label).to(args.device)
            with torch.no_grad():
                output = net(args, data, pos)          #把data丟進網路中

            predicts.append(output.squeeze().tolist())
            targets.append(label.squeeze().tolist())

        draw_auc(targets, predicts, savefig_filename='./result/multiclass_onevsone_{}vs{}_auc.png'.format(enga,engb) )



def test_multiclass_onevsone_GoModel(args,nets,data_dict):
    print('Test multiclass_onevsall A GoModel...')
    eng_abbrs = args.eng_abbrs

    

    _DataB = TestMulticlassGoModel(args, eng_abbrs , data_dict,
        ratio = 0.9,is_train = True )

    testdataloader = DataLoader(dataset=_DataB,
                                batch_size=1,
                                shuffle=False)


    correct , wrong = 0.0 , 0.0
    for _eng in ['Target'] + eng_abbrs  + ['Predict']:
        print(_eng, end = ' , ')
    print('')
    
    for step, (data, pos, _) in enumerate(testdataloader):
        other_info = _DataB.other_info[step]
        print(other_info[1][0], end =' , ') #target

        each_class_predicts =[]
        
        score = {}
        for _eng in eng_abbrs:
            score[_eng] = 0
        
        for enga, engb , net in nets:

            data = Variable(data).to(args.device)
            pos = Variable(pos).to(args.device)

            with torch.no_grad():
                output = net(args, data, pos)          #把data丟進網路中

            # print(output.squeeze().tolist() , end=' , ')

            if output.squeeze().tolist() > 0.5: # vote for predict
                score[enga] += 1
            else:
                score[engb] += 1
        
        predict_tags = []
        highest_vote = max( [ score[_eng] for _eng in eng_abbrs] )
        for _eng in eng_abbrs:
            print(score[_eng], end=' , ')
            if score[_eng] == highest_vote:
                predict_tags.append(_eng)
        
        print('{} , '.format(predict_tags), other_info[0])

        for predict in predict_tags:
            if predict in other_info[1].__str__():
                correct += 1
                wrong -= 1
                break
        wrong += 1

    print('correct {} wrong {} acc {}'.format(correct, wrong, correct/(correct+wrong)))




if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description = "modelA train=0 => load and demo. train=1")
    parser.add_argument('--device', default = 'cuda:3')
    parser.add_argument('--train', default = 1, type = int)
    parser.add_argument('--epochA', default = 5, type = int)
    
    parser.add_argument('--training_pickle_filepath', 
        default = '/mnt/nfs/work/yshsu0918/lal/other/test/Dataset/D_FSH_ysiftlctrn_training.pickle', type = str)
    parser.add_argument('--neg_training_pickle_filepath', 
        default = '/mnt/nfs/work/yshsu0918/lal/other/test/Dataset/D_FSH_ysiftlctrn_neg_training.pickle', type = str)        
    parser.add_argument('--eng_abbrs', default = 'ys,if,tl,ct,rn', type = str)
    

    args = parser.parse_args()
    args.eng_abbrs = args.eng_abbrs.split(',')
    print(args)
    abbr_filenameprefix = ''.join(args.eng_abbrs)

    data_dict, _ = pickle_reader(args)

    if args.train:
        print('-----train-----')
        
        nets = train_multiclass_onevsone_A_GoModel(args,data_dict)
        with open('./net/multiclass_onevsone_{}_GoModel.pickle'.format(abbr_filenameprefix), 'wb') as fout:
            pickle.dump(nets, fout)

        print('-------TEST START--------')
        test_each_model(args,nets,data_dict)
        test_multiclass_onevsone_GoModel(args,nets,data_dict)
        print('-------TEST END--------')    


    else:
        with open('./net/multiclass_onevsone_{}_GoModel.pickle'.format(abbr_filenameprefix),'rb') as fin:
            nets = pickle.load(fin)

        print('-------TEST START--------')
        test_each_model(args,nets,data_dict)
        test_multiclass_onevsone_GoModel(args,nets,data_dict)
        print('-------TEST END--------')    