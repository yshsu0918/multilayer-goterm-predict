from itertools import accumulate
import json
import pickle
import random
import argparse
import re
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable

from dataloader import DataNegSampleGoModel, pickle_reader
from model import GoModel

from operator import itemgetter

import numpy as np
from analysis import draw_auc

def train_Neg_B_GoModel(args,data_dict, neg_sampling_k = 5):
    print('Training Neg Sampling...')
    eng_abbrs = args.eng_abbrs
    nets = []
    for eng in eng_abbrs :
        print('DataB traindataloader eng {} neg_sampling_k {}'.format(eng, neg_sampling_k),flush = True)
        #(self, args, eng, data_dict, ratio = 0.9, is_train = True, neg_sampling_k = 5, test_shuffle= True)
        _DataB = DataNegSampleGoModel(args,eng,data_dict,
            ratio = 0.9, 
            is_train = True, 
            neg_sampling_k = neg_sampling_k)
    

        traindataloader = DataLoader(dataset=_DataB,
                                    batch_size=32,
                                    shuffle=True)

        net = GoModel(label_size = 1, hidden_size=256).to(args.device)
        optimizer = torch.optim.Adam( net.parameters(), lr= 1e-4 , weight_decay=3e-4)
        loss_func = torch.nn.BCELoss()  #

        for epoch in range(args.epochB):
            accumulate_loss = 0
            for step, (data, pos, label, _) in enumerate(traindataloader):
                data = Variable(data).to(args.device)
                pos = Variable(pos).to(args.device)
                label = Variable(label).to(args.device)


                output = net(args, data, pos)          #把data丟進網路中
                loss = loss_func(output, label)
                

                optimizer.zero_grad()      #計算loss,初始梯度
                loss.backward()            #反向傳播
                optimizer.step()       

                accumulate_loss += loss
            
            
            print('epoch {} last loss {} accumulate_loss {} step {} avgloss {} '.format(epoch,loss,accumulate_loss,step, accumulate_loss/step),flush=True)

        nets.append((neg_sampling_k, eng,net))
    return nets



def test_Neg_B_GoModel(args, data_dict, nets):
    print('Test Neg Sampling...')
    eng_abbrs = args.eng_abbrs
    for neg_sampling_k, eng, net in nets :
        
        print('DataB testdataloader')
        _DataB = DataNegSampleGoModel(args,eng,data_dict,
            ratio = 0.9, 
            is_train = False, 
            neg_sampling_k = neg_sampling_k)

        testdataloader = DataLoader(dataset=_DataB,
                                    batch_size=1,
                                    shuffle=False)


        predicts = []
        targets = []

        print('current eng ', eng)
        print('{} , {} , {}, {}'.format('Target', 'Predict', 'Correct (>0.5)', 'sgf'))
        for step, (data, pos, label, _) in enumerate(testdataloader):
            data = Variable(data).to(args.device)
            pos = Variable(pos).to(args.device)
            label = Variable(label).to(args.device)
            with torch.no_grad():
                output = net(args, data, pos)          #把data丟進網路中


            predicts.append(output.squeeze().tolist())
            targets.append(label.squeeze().tolist())

            
            if output.squeeze().tolist() > 0.5:
                _output = 1
            else:
                _output = 0
            correct = _output == label.squeeze().tolist()

            print('{}, {}, {} , {}'.format(label.squeeze().tolist(), output.squeeze().tolist(), correct,_DataB.other_info[step][0] ))

        draw_auc(targets, predicts, './result/binary_neg_sample_shuffle_{}_{}_auc.png'.format(neg_sampling_k,eng))




def cross_compare(args,nets):
    print('Test cross_compare...')
    eng_abbrs = args.eng_abbrs
    for i, eng in enumerate(eng_abbrs):
        print('DataB testdataloader')
        
        _DataB = DataNegSampleGoModel(args,eng,data_dict,
            ratio = 0.9, 
            is_train = False, 
            neg_sampling_k = neg_sampling_k)
    

        testdataloader = DataLoader(dataset=_DataB,
                                    batch_size=1,
                                    shuffle=False)

        
        result = {}
        for k in eng_abbrs:
            result[k] = []

        for j, _eng in enumerate(eng_abbrs):
            neg_sampling_k, _, net = nets[j]
            labels = []
            for step, (data, pos, label, _) in enumerate(testdataloader):
                data = Variable(data).to(args.device)
                pos = Variable(pos).to(args.device)
                label = Variable(label).to(args.device)
                labels.append(label)

                with torch.no_grad():
                    output = net(args, data, pos)          #把data丟進網路中

                _output = output.squeeze().tolist()

                result[_eng].append(_output)

            result['labels'] = labels
        

        for _eng in eng_abbrs:
            print(_eng, end='\t')
        print('')
        for k in range(len(result[eng])):
            print('#{} Correct LABEL {}  '.format(k, eng if result['labels'][k] == 1 else 'OT' ), end = ',')
            for _eng in eng_abbrs:
                print(', {} '.format(result[_eng][k]), end='')
            
            list1 =  [ result[_eng] for _eng in eng_abbrs ]
            print( ','+ eng_abbrs[list1.index(max(list1))] , end=',')
            print('')



if __name__ == '__main__':
    random.seed( 918 )

    parser = argparse.ArgumentParser(description = "train=0 => load and demo. train=1")
    parser.add_argument('--device', default = 'cuda:2')
    parser.add_argument('--train', default = 1, type = int)
    parser.add_argument('--epochB', default = 10, type = int)
    
    # parser.add_argument('--training_pickle_filepath', 
    #     default = '/mnt/nfs/work/yshsu0918/lal/other/test/Dataset/D_FSH_rn_training.pickle', type = str)
    # parser.add_argument('--neg_training_pickle_filepath', 
    #     default = '/mnt/nfs/work/yshsu0918/lal/other/test/Dataset/D_FSH_ysiftlctrn_neg_training.pickle', type = str)
    # parser.add_argument('--eng_abbrs', default = 'rn', type = str)
    parser.add_argument('--training_pickle_filepath', 
        default = '/mnt/nfs/work/yshsu0918/lal/other/test/Dataset/D_FSH_ysiftlctrn_training.pickle', type = str)
    parser.add_argument('--neg_training_pickle_filepath', 
        default = '/mnt/nfs/work/yshsu0918/lal/other/test/Dataset/D_FSH_ysiftlctrn_neg_training.pickle', type = str)        
    parser.add_argument('--eng_abbrs', default = 'ys,if,tl,ct,rn', type = str)
    
    args = parser.parse_args()
    args.eng_abbrs = args.eng_abbrs.split(',')
    print(args)
    abbr_filenameprefix = ''.join(args.eng_abbrs)

    data_dict, _ = pickle_reader(args, shuffle=False)
    if args.train:
        for neg_sampling_k in [5]:
            nets = train_Neg_B_GoModel(args,data_dict,neg_sampling_k = neg_sampling_k)
            with open('./net/binary_negsample{}_{}_GoModel.pickle'.format(neg_sampling_k,abbr_filenameprefix), 'wb') as fout:
                pickle.dump(nets, fout)
            
            print('-------TEST START--------')
            test_Neg_B_GoModel(args,data_dict,nets)
            print('-------TEST END--------')
    else:
        for neg_sampling_k in [5]:
            with open('./net/binary_negsample{}_{}_GoModel.pickle'.format(neg_sampling_k,abbr_filenameprefix),'rb') as fin:
                nets = pickle.load(fin)
            print('-------TEST START--------')
            test_Neg_B_GoModel(args,data_dict,nets)
            print('-------TEST END--------')
    


            #        count, TP,FP,FN,TN, correct, try1 = 0, [0]*len(thresholds), [0]*len(thresholds), [0]*len(thresholds), [0]*len(thresholds), [0]*len(thresholds), [0]*len(thresholds)
                #print( 'Ans {} / Predict {}'.format(label.squeeze().tolist(), output.squeeze().tolist()) )

            # for j, threshold in enumerate(thresholds):
            #     _output = 1.0 if output.squeeze().tolist() > threshold else 0
            #     try1[j] += _output                
            #     if label.squeeze().tolist() == _output:
            #         correct[j] += 1

                # if label.squeeze().tolist() == 1 and _output == 1:
                #     TP[j] += 1
                # elif label.squeeze().tolist() == 1 and _output == 0:
                #     FN[j] += 1
                # elif label.squeeze().tolist() == 0 and _output == 1:
                #     FP[j] += 1
                # elif label.squeeze().tolist() == 0 and _output == 0:
                #     TN[j] += 1
                # else:
                #     print('WTF?')
                    
                    
            # count += 1
        
        # print('eng {} neg_sampling_k \t TP \t FN \t FP \t TN \t threshold \t try1 \t correct \t total \t accuracy \t' )
        # for j, threshold in enumerate(thresholds):            
        #     print('eng {} neg_sampling_k {} , {} , {} , {} , {} , {} , {} , {} , {} , {}'.format(eng,neg_sampling_k, TP[j],FN[j],FP[j],TN[j], threshold, try1[j] ,correct[j], count, float(correct[j])/float(count)) )

