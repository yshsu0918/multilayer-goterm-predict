import json
import pickle
import random
import argparse
import re
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable

from dataloader import DataMulticlassGoModel, DataNegSampleGoModel, pickle_reader
from model import GoModel




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


    eng = 'iv'
    data_dict, _ = pickle_reader(args)
    _DataB = DataMulticlassGoModel(args , eng, data_dict, ratio = 0.9,is_train = False, neg_sampling_k = 1)
    _DataB.traindatapreview()
    
    _DataC = DataNegSampleGoModel(eng = eng, ratio = 0.9, 
            is_train = False, 
            training_pickle_filepath = args.training_pickle_filepath, 
            neg_training_pickle_filepath = args.neg_training_pickle_filepath, 
            neg_sampling_k = 1)
    _DataC.traindatapreview()
    