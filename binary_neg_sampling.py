import numpy as np
import json
import torch as torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import argparse
from analysis import draw_auc
from model import GoModel_lal
from dataloader import DataNegSample
import pandas as pd
import pickle
import re
def total_hit(outputs, label):
    hit = 0
    for row in range(len(outputs)):
        _output = 1 if outputs[row].squeeze().tolist() > 0.5 else 0

        # print(label[row].squeeze().tolist() , _output)

        if label[row].squeeze().tolist() == _output:
            hit+=1
    return hit

def train(epoch, Loader):
    model.train()
    total_loss = 0.0
    total, hit = 0, 0

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
        hit += total_hit(outs, labels)
        
    print("###############################")
    print("Training")
    print("Epoch:", epoch) 
    print("Loss:", total_loss)
    print("Training accuracy:", hit / total)         
    print(hit, "/", total, "\n")
    
def test(epoch, Loader, draw=False):
    model.eval()
    total, hit = 0, 0
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
        hit += total_hit(outs, labels)

        # print(outs.squeeze().tolist(), labels.squeeze().tolist())
        predicts.extend(outs.squeeze().tolist())
        targets.extend(labels.squeeze().tolist())  
        
    print("Testing")
    print("Epoch:", epoch)
    print("Testing accuracy:", hit / total)         
    print(hit, "/", total, "\n")
    
    if draw:

        stat = {'target': targets, 'predicts': predicts, 'sgf': [q['sgf_content'].replace('\n','') for q in _DataSimple_test.Q]}
        df_stat = pd.DataFrame(stat)
        df_stat.to_csv(args.csv_path,encoding='UTF-8')

        # print('target, predict, sgf')
        # for q,t,p in zip( _DataSimple_test.Q , targets, predicts):
        #     print( '{},{},{}'.format(t,p,q['sgf_content'].replace('\n','')))

        draw_auc(targets, predicts, args.result_auc)

    return hit / total
    
def run(epochs, trainLoader, validLoader, pretrain, save_model_path='./weight'):
    max_acc = -1.0
    for epoch in range(epochs):
        # training
        train(epoch+1, trainLoader)
        # testing
        test_acc = test(epoch+1, validLoader)
        if test_acc > max_acc:
            max_acc = test_acc
            torch.save(model.state_dict(), save_model_path)
        if pretrain: print("Max Pretrained Testing accuracy:", max_acc, "\n")
        else: print("Max Finetuned Testing accuracy:", max_acc, "\n")


def gen_extra_dataset(Loader): #找d_fs裡面 被binary model 挑出來的，另行篩選。
    model.eval()
    r =[]
    for step, (boards, positions) in enumerate(Loader):
        # if step>1000:
        #     break
        if step%100 == 0:
            print('#', end='',flush=True)
        boards = Variable(boards).to(args.device)
        positions = Variable(positions).to(args.device)
        with torch.no_grad():
            outs = model(args, boards, positions)

        r.append( (outs.squeeze().tolist(), step) )

    val = 0
    r = sorted(r, key=lambda x: x[0])
    for idx, tuple in enumerate(r):
        if tuple[0] > val:
            print('val {} accumuate index {}'.format(val, idx))
            val += 0.1
    print('val {} accumuate index {}'.format(val, idx))
    
    # print(r[-500:])
    fout_content = []
    for _r in r[-650:]:
        buf = _DataSimple_extra.Q[_r[1]]

        flag = 0

        TagCPattern = '[^AP]C\[([^\]]*)\]'
        comment = re.findall(TagCPattern, buf['sgf_content'])[-1]
        #print(comment)
        lst = comment.split(' ')        
        
        for p in "做活 活棋 安全 救回 脫險".split(' '):
            for q in lst:
                if p == q:
                    flag = 1
                    break
            if flag:
                break
        if flag:
            #print('OUO')
            continue


        buf['order_tags'] = ['lv']
        buf['specific_terms'] = 'BinaryModel feel this board match the order tags'
        fout_content.append(buf)
    print(len(fout_content))

    fout_content.reverse()

    with open(args.extra_dataset_outputpath , 'wb') as fout:
        pickle.dump(fout_content, fout)

            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "None")
    parser.add_argument('-d', '--device', default = 'cuda')
    parser.add_argument('--epoch', default = 3, type = int)
    parser.add_argument('--batch_size', default = 128, type = int) 
    parser.add_argument('--lr', default = 1e-4, type = float)
    parser.add_argument('--hidden_size', default = 256, type = int)
    parser.add_argument('--train', default = 1, type = int)

    #-------當需要使用binary model 過濾資料的時候參考以下參數---------
    # parser.add_argument('--extra_dataset', default = '', type=str)
    parser.add_argument('--extra_dataset_outputpath', default = '../Dataset/D_FSH_extrabybinarymodel_lv.pickle', type=str)
    parser.add_argument('--extra_dataset', default = '../bveyeconnect2trainingpickle/D_FS.pickle', type=str)
    #----------------


    parser.add_argument('--pickle_train', default = '/mnt/nfs/work/yshsu0918/workspace/thesis/Dataset/D_FSHH/D_FSHHneg10_cn_train.pickle', type = str)    
    parser.add_argument('--pickle_valid', default = '/mnt/nfs/work/yshsu0918/workspace/thesis/Dataset/D_FSHH/D_FSHHneg10_cn_test.pickle', type = str)    
    parser.add_argument('--pickle_test', default = '/mnt/nfs/work/yshsu0918/workspace/thesis/Dataset/D_FSHH/D_FSHHneg10_cn_valid.pickle', type = str)    
    parser.add_argument('--result_auc', default = './result/binary_neg10_shuffle_cn_auc.png', type = str)
    parser.add_argument('--model_path', default = './net/0430_binary_lv_neg10.pt', type = str)
    parser.add_argument('--csv_path', default = './0511_cn.csv', type = str)

    '''
python3 binary_neg_sampling.py \
--pickle_train ../Dataset/D_FSHH/D_FSHHneg10_cn_train.pickle \
--pickle_valid ../Dataset/D_FSHH/D_FSHHneg10_cn_valid.pickle \
--pickle_test ../Dataset/D_FSHH/D_FSHHneg10_cn_test.pickle \
--result_auc ./result/binary_neg10_cn_auc.png \
--model_path ./net/binary_cn_neg10.pt \
--csv_path ./log/0430_cn.csv

    '''
    
    args = parser.parse_args()
    print('Term Model Training')
    torch.cuda.set_device(1)
    

    _DataSimple_train = DataNegSample(args.pickle_train)
    train_loader = data.DataLoader(dataset=_DataSimple_train, 
                                    batch_size=32,
                                    shuffle=True, 
                                    num_workers=4, 
                                    pin_memory=True)

    _DataSimple_valid = DataNegSample(args.pickle_valid)
    valid_loader = data.DataLoader(dataset=_DataSimple_valid,
                                batch_size=32,
                                shuffle=True, 
                                num_workers=4, 
                                pin_memory=True)

    _DataSimple_test = DataNegSample(args.pickle_test)
    test_loader = data.DataLoader(dataset=_DataSimple_test,
                                batch_size=32,
                                shuffle=False, 
                                num_workers=4, 
                                pin_memory=True)


    model = GoModel_lal(label_size = 1, hidden_size=256).to(args.device)
    optim = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCELoss()

    # Pretrain
    if args.train:
        run(args.epoch, train_loader, valid_loader, True,save_model_path=args.model_path)
        test(1000, test_loader,draw=1)
    else:
        model.load_state_dict(torch.load(args.model_path))
        
        if args.extra_dataset != '':
            _DataSimple_extra = DataNegSample(args.extra_dataset, need_label=False)
            extra_loader = data.DataLoader(dataset=_DataSimple_extra,
                                        batch_size=1,
                                        shuffle=False, 
                                        num_workers=4, 
                                        pin_memory=True)
            
            gen_extra_dataset(extra_loader)

        else:
            test(1000, test_loader,draw=1)

        
    
    
    # Finetune
    # run(args.epoch_fine, trainLoader_fine, testLoader_fine, False)
    