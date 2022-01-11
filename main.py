import json
import pickle
import random
import argparse
import re
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from dataloader import DataA, DataB
from model import NetA, NetB
from global_var import *


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
        _DataB = DataB(DataBtarget, eng = eng, eng_abbrs = eng_abbrs, ratio = 1, is_train = True, input_training_pickle = args.input_training_pickle)
        traindataloader = DataLoader(dataset=_DataB,
                                    batch_size=1,
                                    shuffle=True)

        net = GoModel(label_size=len(DataBtarget[eng]), hidden_size=args.hidden_size)
        
        optimizer = torch.optim.Adam( net.parameters(), lr= 0.02 , weight_decay=3e-4)
        loss_func = torch.nn.BCELoss()  #

        for epoch in range(args.epochB):

            for step, (data, label) in enumerate(traindataloader):
                data = data.to(args.device)
                label = label.to(args.device)

                output = net(data)          #把data丟進網路中
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
                    


def trainA(args):
    print('Training A...')
    #current_eng_abbr = [ch2abbr[k] for k in ch2abbr.keys()]
    current_eng_abbr = args.eng_abbrs
    print('DataA traindataloader')
    traindataloader = DataLoader(dataset=DataA(eng_abbrs = current_eng_abbr, ratio = 0.9, is_train = True, input_training_pickle = args.input_training_pickle),
                            batch_size=1,
                            shuffle=True)
    print('DataA testdataloader')
    testdataloader = DataLoader(dataset=DataA(eng_abbrs = current_eng_abbr, ratio = 0.9, is_train = False, input_training_pickle = args.input_training_pickle),
                            batch_size=1,
                            shuffle=False)
    net = NetA(args,n_feature=2890, n_hidden=256, n_output=len(current_eng_abbr))     # define the network
    print(net)  # net architecture

    optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
    loss_func = torch.nn.CrossEntropyLoss()  #

    for epoch in range(args.epochA):

        for step, (data, label,_) in enumerate(traindataloader):
            data = data.to(args.device)
            label = label.to(args.device)

            output = net(data)          #把data丟進網路中
            loss = loss_func(output, label)
            
            optimizer.zero_grad()      #計算loss,初始梯度
            loss.backward()            #反向傳播
            optimizer.step()       

            if step % 100 == 0:
                pass
                #print('Epoch:', epoch, '|step:', step, '|train loss:%.4f'%loss.data)

            #每100steps輸出一次train loss
        print('Epoch:', epoch, '|train loss:%.4f'%loss.data)

        correct = 0
        error = 0
        for step, (x, y,_) in enumerate(testdataloader):
            xx = x.to(args.device)
            yy = y.to(args.device)

            output = net(xx).detach()

            for predict_y,y in zip( output.squeeze().tolist()  , yy.squeeze().tolist() ) :
                _predict_y = 1 if predict_y > 0.5 else 0
                _y = int(y)
                #print(predict_y, y)
                if _predict_y == _y:
                    correct += 1
                else:
                    error += 1
        print('TestSet| Correct: {}, Error: {}, Accuracy: {}'.format(correct, error, float(correct)/float(correct+error)))

    return net


def trainB(args):
    print('Training B...')
    eng_abbrs = args.eng_abbrs
    nets = []
    DataBtarget = cal_DataBtarget(args, eng_abbrs)
    for eng in eng_abbrs :
        print('DataB traindataloader ', eng, eng_abbrs)
        _DataB = DataB(DataBtarget, eng = eng, eng_abbrs = eng_abbrs, ratio = 1, is_train = True, input_training_pickle = args.input_training_pickle)
        traindataloader = DataLoader(dataset=_DataB,
                                    batch_size=1,
                                    shuffle=True)


        net = NetB(args, n_feature=2890, n_hidden=256, n_output=len(DataBtarget[eng]))     # define the network
        optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
        loss_func = torch.nn.CrossEntropyLoss()  #

        for epoch in range(args.epochB):

            for step, (data, label) in enumerate(traindataloader):
                data = data.to(args.device)
                label = label.to(args.device)

                output = net(data)          #把data丟進網路中
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

def demo( args, Anet, Bnets, DataBtarget):
    demo_result = args.demo_result
    
    thresholdA = 0.5
    # load data
    current_eng_abbr = list(DataBtarget.keys())
    traindataloader = DataLoader(dataset=DataA(eng_abbrs = current_eng_abbr, ratio = 0.1, is_train = False, input_training_pickle = args.input_training_pickle),
                            batch_size=1,
                            shuffle=False)

    demo_result_content = []

    for step, (data, label, other_info) in enumerate(traindataloader):
        if step > 50:
            break        
        print('-----------------------------------------------------------')
        print('#{} Ground truth'.format(step))
        sgfstr = other_info[0][0]

        last_comment = sgfstr[sgfstr.find('C[')+2:-2] 
        print(sgfstr)
        print(last_comment)
        print(other_info[2])

        need2replace = []


        # print('-----------------------------------')
        print('predict result')
        # get A output
        data = data.to(args.device)
        output = Anet(data).detach()
        Aout = output.view(-1).tolist()
        print('This board may contain: ',end='')
        for i in range(len(current_eng_abbr)):
            if thresholdA < Aout[i]:
                print(current_eng_abbr[i],end=' ')
                need2replace.append(current_eng_abbr[i])
        
        print('')
        # get B output
        buf = {}
        buf['sgf_content'] = sgfstr
        buf['last_comment'] = last_comment
        buf['specific_terms'] = other_info[2]
        buf['A_output'] = need2replace.copy()
        buf['B_output'] = []

        for i, (eng,Bnet) in enumerate(Bnets): 
            #print(eng)
            output = Bnet(data).detach()
            Bout = output.view(-1).tolist()
            max_index = Bout.index(max(Bout))
            if eng in need2replace:
                print(current_eng_abbr[i],DataBtarget[ current_eng_abbr[i] ][max_index])
                buf['B_output'].append((current_eng_abbr[i],DataBtarget[ current_eng_abbr[i] ][max_index]))
        
        
        demo_result_content.append(buf)
    
    print('-----------------------------------')
    for eng in current_eng_abbr:
        for QQ in ch2abbr.keys():
            if ch2abbr[QQ] == eng:
                print(eng, QQ)


    with open( demo_result , 'wb') as fout:
        pickle.dump( demo_result_content , fout)
    # replace comment

if __name__ == '__main__':
    
    Anet = 0
    Bnets = 0
    DataBtarget = 0

    parser = argparse.ArgumentParser(description = "train=0 => load and demo. train=1")
    parser.add_argument('--device', default = 'cuda:2')
    parser.add_argument('--train', default = 1, type = int)
    parser.add_argument('--epochA', default = 5, type = int)
    parser.add_argument('--epochB', default = 1, type = int)
    
    parser.add_argument('--input_training_pickle', default = '/mnt/nfs/work/yshsu0918/lal/other/test/lalwin_shortsentence_dataset_mei_class_training.pickle', type = str)
    parser.add_argument('--eng_abbrs', default = 'gt,ct,in,tn,if,at,df,gd,bd', type = str)
    parser.add_argument('--demo_result', default = '/mnt/nfs/work/yshsu0918/lal/other/test/result.pickle', type = str)
    
    args = parser.parse_args()
    args.eng_abbrs = args.eng_abbrs.split(',')
    print(args)

    if args.train:
        # Anet = trainA(args)
        with open('Anet.pickle', 'rb') as fin:
            Anet = pickle.load(fin)        
        Bnets, DataBtarget = trainB(args)

        with open('Anet.pickle', 'wb') as fout:
            pickle.dump(Anet, fout)
        with open('Bnets.pickle', 'wb') as fout:
            pickle.dump(Bnets, fout)
        with open('DataBtarget.pickle', 'wb') as fout:
            pickle.dump(DataBtarget, fout)
        with open('DataBtarget.txt', 'w') as fout:
            fout.write(DataBtarget.__str__())
    
    else:
        with open('Anet.pickle', 'rb') as fin:
            Anet = pickle.load(fin)
        with open('Bnets.pickle', 'rb') as fin:
            Bnets = pickle.load(fin)
        with open('DataBtarget.pickle', 'rb') as fin:
            DataBtarget = pickle.load(fin)
        with open('DataBtarget.txt', 'r') as fin:
            print(fin.read())

    
    demo(args, Anet, Bnets, DataBtarget)