import pickle
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from boardprocess import GetBoard
import numpy as np

class DataGoModel(Dataset):
    def __init__(self, DataBtarget, eng,eng_abbrs, ratio = 0.9, is_train = True, input_training_pickle = ''):
        self.data = []
        self.label = []
        self.other_info = []

        training_data = []
        with open(input_training_pickle, 'rb') as fin:
            training_data = pickle.load(fin)

        if is_train:
            begin = 0
            end = int(len(training_data)*ratio)
        else:
            begin = int(len(training_data)*ratio) + 1
            end = -1

        for item in training_data[begin:end]:
            data_buf = []
            for dict_key in ['Policy1', 'Policy2', 'BV1', 'BV2', 'Connect1', 'Connect2', 'Eye1', 'Eye2']:
                data_buf.append( np.array([ float(x) for x in item[dict_key] ]).reshape((19,19)) )

            
            sgf_str = item['sgf_content']
            Black1, White1, Black2, White2, Board, position = GetBoard(sgf_str)
            
            data_buf.append(Black1)
            data_buf.append(White1)
            data_buf.append(Black2)
            data_buf.append(White2)

            
            label_buf = [ ]
            for term in DataBtarget[eng]:
                label_buf.append( 1 if (eng in item['specific_terms']) and (term in item['specific_terms'][eng]) else 0 )

            self.label.append( label_buf )
            self.data.append( (data_buf, position ) )


    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return torch.FloatTensor([ self.data[index][0] ]), torch.LongTensor( [self.data[index][1]]) , torch.FloatTensor([self.label[index]])



class DataA(Dataset):
    def __init__(self, eng_abbrs= [], ratio = 0.9, is_train =True, input_training_pickle = ''):
        training_data = []
        with open(input_training_pickle, 'rb') as fin:
            training_data = pickle.load(fin)
        
        # print(training_data[0].keys())

        #eng_abbrs = [ch2abbr[k] for k in ch2abbr.keys()]

        for i in range(3):
            print(training_data[i]['sgf_content'], training_data[i]['order_tags'], training_data[i]['specific_terms'])
            
        self.data = []
        self.label = []
        self.other_info = []

        if is_train:
            begin = 0
            end = int(len(training_data)*ratio)
        else:
            begin = int(len(training_data)*ratio) + 1
            end = -1

        for item in training_data[begin:end]:
            data_buf = []
            for dict_key in ['Policy1', 'Policy2', 'BV1', 'BV2', 'Connect1', 'Connect2', 'Eye1', 'Eye2']:
                data_buf += [ float(x) for x in item[dict_key] ]
            data_buf += [ float(item['Value1']), float(item['Value2']) ]
            self.data.append(data_buf)

            label_buf = [ ]
            for i in range(len(eng_abbrs)):
                label_buf.append( 1 if eng_abbrs[i] in item['order_tags'] else 0 )
            



            self.label.append(label_buf)
            self.other_info.append((item['sgf_content'], item['order_tags'], item['specific_terms']))

        self.data = torch.FloatTensor(self.data)    
        self.label = torch.FloatTensor(self.label)
        
        print(self.data.shape)
        print(self.label.shape)

        # self.data = torch.tensor([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
        # self.label = torch.LongTensor([1,1,0,0])

    def __getitem__(self,index):
        return self.data[index],self.label[index], self.other_info[index]

    def __len__(self):
        return len(self.data)


class DataB(Dataset):
    def __init__(self,DataBtarget, eng, eng_abbrs= [], ratio = 0.9, is_train = True, input_training_pickle = ''):
        training_data = []
        with open(input_training_pickle, 'rb') as fin:
            training_data = pickle.load(fin)
        self.data = []
        self.label = []


        if is_train:
            begin = 0
            end = int(len(training_data)*ratio)
        else:
            begin = int(len(training_data)*ratio) + 1
            end = -1

        for item in training_data[begin:end]:
            data_buf = []
            for dict_key in ['Policy1', 'Policy2', 'BV1', 'BV2', 'Connect1', 'Connect2', 'Eye1', 'Eye2']:
                data_buf += [ float(x) for x in item[dict_key] ]
            data_buf += [ float(item['Value1']), float(item['Value2']) ]
            self.data.append(data_buf)
            
            label_buf = [ ]
            for term in DataBtarget[eng]:
                label_buf.append( 1 if (eng in item['specific_terms']) and (term in item['specific_terms'][eng]) else 0 )
        
            self.label.append(label_buf)
        
        self.data = torch.FloatTensor(self.data)    
        self.label = torch.FloatTensor(self.label)

    def __getitem__(self,index):
        return self.data[index],self.label[index]

    def __len__(self):
        
        return len(self.data)