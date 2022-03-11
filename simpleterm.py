import json
import pickle
import random
import argparse
import re
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable

from dataloader import DataSimpleTermGoModel,pickle_reader
from model import GoModel

from operator import itemgetter
import numpy as np

from analysis import draw_auc, total_hit_topk

def train_simpleterm_A_GoModel(args,net,traindataloader):
    net.train()
    print('Training train_simpleterm_A_GoModel pretask...')

    optimizer = torch.optim.Adam( net.parameters(), lr= 1e-4)
    loss_func = torch.nn.BCELoss()  #

    total_loss = 0
    for step, (data, pos, label) in enumerate(traindataloader):
        if step % 100 == 0:
            print('#', end='',flush=True)

        data = Variable(data).to(args.device)
        pos = Variable(pos).to(args.device)
        label = Variable(label).to(args.device)

        output = net(args, data, pos)   #把data丟進網路中
        loss = loss_func(output, label)
        
        optimizer.zero_grad()           #計算loss,初始梯度
        loss.backward()                 #反向傳播
        optimizer.step()

        total_loss += loss.item()

        
    print('loss {} accumulate_loss {} step {} avgloss {} '.format(loss,total_loss,step, total_loss/step),flush=True)
    



def test_simpleterm_A_GoModel(args, net, testdataloader):
    net.eval()
    print('Test train_simpleterm_A_GoModel pretask...')
    predicts = []
    targets = []
    correct , total,test_loss = 0, 0, 0
    loss_func = torch.nn.BCELoss()
    with torch.no_grad():
        for step, (data, pos, label) in enumerate(testdataloader):
            data = Variable(data).to(args.device)
            pos = Variable(pos).to(args.device)
            label = Variable(label).to(args.device)
            
            output = net(args, data, pos)          #把data丟進網路中
            correct += total_hit_topk( output, label, 3) 
            total += len(output)
            test_loss += loss_func(output, label)

    test_loss /= len(testdataloader.dataset)

    print('correct {} total {} accuracy {} test_avg_loss {}'.format(correct, total, correct/total, test_loss ))

def train(args, data_dict, traindataloader, testdataloader):
    net = GoModel(label_size = args.target_labels, hidden_size=256).to(args.device)

    for epoch in range(1, args.epoch + 1):
        train_simpleterm_A_GoModel(args, net, traindataloader)
        # test_simpleterm_A_GoModel(args, net, testdataloader)

    current_weight = net.state_dict()
    for k,v in current_weight.items():
        print(k,type(v))
    torch.save(net.state_dict(), args.model_path)
    saved_weight = torch.load(args.model_path)


    for a, b in zip( [ (k,v) for k,v in current_weight.items()],[ (k,v) for k,v in saved_weight.items()]) :
        print('aaa', a[0],type(a[1]))
        print('bbb',b[0],type(b[1]))
        try:
            print(torch.equal(b[1],a[1]))
        except Exception as e:
            print(e)
        



def test(args, data_dict, testdataloader):
    print('-------TEST START--------')

    net = GoModel(label_size = args.target_labels, hidden_size=256).to(args.device)
    saved_weight = torch.load(args.model_path)
    net.load_state_dict(saved_weight)

    test_simpleterm_A_GoModel(args, net, testdataloader)
    print('-------TEST END--------')    

if __name__ == '__main__':
    random.seed( 918 )

    parser = argparse.ArgumentParser(description = "modelA train=0 => load and demo. train=1")
    parser.add_argument('--device', default = 'cuda:2')
    parser.add_argument('--train', default = 1, type = int)
    parser.add_argument('--epoch', default = 1, type = int)
    
    # parser.add_argument('--training_pickle_filepath', default = '/mnt/nfs/work/yshsu0918/lal/other/test/Dataset/HumanCorrectionTerm_training.pickle', type = str)
    parser.add_argument('--training_pickle_filepath', default = '/mnt/nfs/work/yshsu0918/lal/other/test/Dataset/test_pretrain20w_sabaki_training.pickle', type = str)
    parser.add_argument('--model_path', default = './simpleterm_GoModel.pt', type = str)
    parser.add_argument('--target_labels', default = 111, type = int)

    

    args = parser.parse_args()
    data_dict, eng_abbr = pickle_reader(args,shuffle=True,need_neg=False)
    _DataSimple_train = DataSimpleTermGoModel(args , data_dict, ratio = 0.9,is_train = True)
    traindataloader = DataLoader(dataset=_DataSimple_train, 
                                    batch_size=32,
                                    shuffle=True)

    _DataSimple_test = DataSimpleTermGoModel(args , data_dict, ratio = 0.9,is_train = False)
    testdataloader = DataLoader(dataset=_DataSimple_test,
                                batch_size=32,
                                shuffle=False)    

    if args.train:
        train(args, data_dict, traindataloader, testdataloader)
        test(args, data_dict, testdataloader)
    else:
        test(args, data_dict, testdataloader)