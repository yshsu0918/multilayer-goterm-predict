import pickle
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from boardprocess import GetBoard
import numpy as np


def pickle_reader(args):
    print('@pickle reader')
    def pickle_extractor(label, item):
        data_buf = []
        for dict_key in ['Policy1', 'Policy2', 'BV1', 'BV2', 'Connect1', 'Connect2', 'Eye1', 'Eye2']:
            data_buf.append( np.array([ float(x) for x in item[dict_key] ]).reshape((19,19)) )
        
        sgf_str = item['sgf_content']
        Board = item['Board']
        position = item['position']

        data_buf.append(item['Black1'])
        data_buf.append(item['White1'])
        data_buf.append(item['Black2'])
        data_buf.append(item['White2'])

        data_buf = np.stack(data_buf)
        label_buf = np.array([ label ])
        
        return data_buf, position, label_buf

    training_pickle_filepath = args.training_pickle_filepath
    neg_training_pickle_filepath = args.neg_training_pickle_filepath

    with open(training_pickle_filepath, 'rb') as fin:
        training_data = pickle.load(fin)
    with open(neg_training_pickle_filepath, 'rb') as fin:
        neg_training_data = pickle.load(fin)            
    
    _data, _other_info = [],[]
    eng_abbrs = []

    Q = training_data + neg_training_data
    for i, item in enumerate(Q):
        if i % len(Q)//20 == 0:
            print('#' ,end='', flush=True)
        

        data_buf, position, _ = pickle_extractor( 0 , item)

        _data.append( (data_buf, position ) )
        _other_info.append((item['sgf_content'], item['order_tags'], item['specific_terms']))
        eng_abbrs.extend(item['order_tags'])

    
    eng_abbrs = list(set(eng_abbrs))
    print('')
    print(args.eng_abbrs, eng_abbrs)
    data_dict = {}
    for k in eng_abbrs:
        data_dict[k] = {}
        data_dict[k]['data'] = []
        data_dict[k]['other_info'] = []
        
        

    for data, other_info in zip ( _data, _other_info):
        if other_info[1][0] in eng_abbrs:
            k = other_info[1][0]
            data_dict[k]['data'].append( data )
            data_dict[k]['other_info'].append( other_info )
            
        else:
            print('Error found some abbr not in list')

    return data_dict, eng_abbrs



class DataMulticlassGoModel(Dataset):
    def __init__(self, args, eng, data_dict, ratio = 0.9, is_train = True, neg_sampling_k = 5):
        def train_test_split(lst, ratio, is_train):
            if is_train:
                return lst[:int( len(lst) * ratio )]
            else:
                return lst[int( len(lst) * ratio ):]  

        print('@DataMulticlassGoModel')
        print('is_train', is_train, 'neg_sampling_k', neg_sampling_k)
        

        self.label, self.data, self.other_info = [], [], []
        count_pos = 0 
        count_neg = 0
        for k in args.eng_abbrs:
            print('unsplit', k, len(data_dict[k]['data']))

            _data = train_test_split(data_dict[k]['data'], ratio, is_train)
            _other_info = train_test_split( data_dict[k]['other_info'], ratio, is_train)
            
            
            if k == eng:
                _label = [ np.array([ 1 ]) for info in _other_info]
                count_pos += len(_label)
            else:
                _label = [ np.array([ 0 ]) for info in _other_info]
                count_neg += len(_label)

            self.label.extend( _label )
            self.data.extend( _data )
            self.other_info.extend( _other_info )

        print(count_pos, count_neg, neg_sampling_k)
        remain_need = max(count_pos*neg_sampling_k - count_neg, 0)

        print('count_pos',count_pos, 'count_neg',count_neg,'remain_need',remain_need)
        _data = train_test_split( data_dict['neg']['data'], ratio, is_train)[:remain_need]
        _other_info = train_test_split( data_dict['neg']['other_info'], ratio, is_train)[:remain_need]
        _label = [ np.array([ 0 ]) for info in _other_info]

        print('len neg _label', len(_label))

        self.label.extend( _label)
        self.data.extend( _data )
        self.other_info.extend( _other_info )

        print('len self.label', len(self.label))
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index][0]), torch.LongTensor( [self.data[index][1]]) , torch.FloatTensor( self.label[index])

class DataNegSampleGoModel(Dataset):
    def __init__(self, eng, ratio = 0.9, is_train = True, training_pickle_filepath = '', neg_training_pickle_filepath = '', neg_sampling_k = 5):
        def pickle_extractor(eng, item):
            data_buf = []
            for dict_key in ['Policy1', 'Policy2', 'BV1', 'BV2', 'Connect1', 'Connect2', 'Eye1', 'Eye2']:
                data_buf.append( np.array([ float(x) for x in item[dict_key] ]).reshape((19,19)) )
            
            sgf_str = item['sgf_content']
            Board = item['Board']
            position = item['position']

            data_buf.append(item['Black1'])
            data_buf.append(item['White1'])
            data_buf.append(item['Black2'])
            data_buf.append(item['White2'])

            data_buf = np.stack(data_buf)
            if eng == 'neg':
                label_buf = np.array([ 0 ])
            else:
                label_buf = np.array([ 1 ])
            
            return data_buf, position, label_buf
    
        with open(training_pickle_filepath, 'rb') as fin:
            training_data = pickle.load(fin)
        with open(neg_training_pickle_filepath, 'rb') as fin:
            neg_training_data = pickle.load(fin)

        # print(training_data[15])



        _label, _data, _other_info = [],[],[]
        for item in training_data:
            if eng not in item['order_tags']:
                continue                
            data_buf, position, label_buf = pickle_extractor(eng, item)

            _label.append( label_buf )
            _data.append( (data_buf, position ) )
            _other_info.append((item['sgf_content'], item['order_tags'], item['specific_terms']))

        positive_sample_size = int ( len(_label) * ratio )
        print('eng', eng)
        print('positive_sample_size', positive_sample_size )
        print('len(_label)', len(_label))
        print('ratio', ratio)
        print('neg_training_data', len(neg_training_data))
        if is_train:
            self.label = _label[:positive_sample_size]
            self.data = _data[:positive_sample_size]
            self.other_info = _other_info[:positive_sample_size]
            print('is_train (T)', is_train, 'len(self.label)', len(self.label) )
        else:
            self.label = _label[positive_sample_size:]
            self.data = _data[positive_sample_size:]
            self.other_info = _other_info[positive_sample_size:]
            print('is_train (F)', is_train, 'len(self.label)', len(self.label) )
        
        if is_train:
            _neg_training_data = neg_training_data[0: positive_sample_size*neg_sampling_k]
            print('is_train (T) NEG', is_train, 'len(_neg_training_data)', len(_neg_training_data))
        else:
            _neg_training_data = neg_training_data[len(neg_training_data) - len(self.label)*neg_sampling_k: ]
            print('is_train (F) NEG', is_train, 'len(_neg_training_data)', len(_neg_training_data))
        
        for item in _neg_training_data:
            if 'neg' not in item['order_tags']:
                continue                
            data_buf, position, label_buf = pickle_extractor('neg', item)

            self.label.append( label_buf )
            self.data.append( (data_buf, position ) )
            self.other_info.append((item['sgf_content'], item['order_tags'], item['specific_terms']))

        

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index][0]), torch.LongTensor( [self.data[index][1]]) , torch.FloatTensor( self.label[index]), self.other_info

class DataGoModel(Dataset):
    def __init__(self, DataBtarget, eng,eng_abbrs, ratio = 0.9, is_train = True, input_training_pickle = ''):
        self.data = []
        self.label = []
        self.other_info = []

        training_data = []
        with open(input_training_pickle, 'rb') as fin:
            training_data = pickle.load(fin)

        print(training_data[0])
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
            if max(label_buf) == 0:
                continue

            self.label.append( label_buf )
            self.data.append( (data_buf, position ) )


    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index][0]), torch.LongTensor( [self.data[index][1]]) , torch.FloatTensor( self.label[index])


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