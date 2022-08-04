from cProfile import label
from operator import is_
import pickle
import torch 
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
# from boardprocess import GetBoard
import numpy as np
import hashlib
import random
def train_test_split(lst, ratio, is_train):
    print('is_train {} total {} split@ {}'.format(is_train,  len(lst),int( len(lst) * ratio )))
    if is_train:
        return lst[:int( len(lst) * ratio )]
    else:
        return lst[int( len(lst) * ratio ):]  


def getmd5(mystr):
    return hashlib.md5(mystr.encode('utf-8')).hexdigest()


    # if is_train == False and test_shuffle:
    #     random.shuffle(shuffle_lst)
    # _data = [ x[0] for x in shuffle_lst][ : neg_sampling_amount ]
    # _other_info = [ x[1] for x in shuffle_lst] [ : neg_sampling_amount ]

def shuffletwolst(lsta, lstb, outdim=-1):
    if outdim == -1:
        outdim == len(lsta)

    shuffle_lst = [(a,b) for a,b in zip(lsta,lstb)]
    random_indexes = []

    while len(random_indexes) != outdim:
        new_sample = random.randint(0, len(shuffle_lst)-1)
        if new_sample not in random_indexes:
            random_indexes.append(new_sample)
        else:
            pass    

    return [ shuffle_lst[idx][0] for idx in random_indexes], [ shuffle_lst[idx][1] for idx in random_indexes]

def pickle_extractor(label, item):
    # data_buf = []
    # for dict_key in ['BV1', 'BV2', 'Connect1', 'Connect2', 'Eye1', 'Eye2']:
    #     data_buf.append( np.array([ float(x) for x in item[dict_key] ]).reshape((19,19)) )
        
    # sgf_str = item['sgf_content']
    # Board = item['Board']
    # position = item['position']

    # data_buf.append(item['Black1'])
    # data_buf.append(item['White1'])
    # data_buf.append(item['Black2'])
    # data_buf.append(item['White2'])

    # data_buf = np.stack(data_buf)
    label_buf = np.array( label )
        
    return item['features'], item['position'], label_buf

def pickle_reader(args, shuffle=False, need_neg = True):
    print('@pickle reader')
    _data, _other_info = [],[]
    eng_abbrs = []



    with open(args.training_pickle_filepath, 'rb') as fin:
        training_data = pickle.load(fin)

    if need_neg:
        with open( args.neg_training_pickle_filepath, 'rb') as fin:
            neg_training_data = pickle.load(fin)            
        Q = training_data + neg_training_data
    else:
        Q = training_data

    if shuffle:
        print('shuffle all dataset')
        random.shuffle(Q)
        
    for i, item in enumerate(Q):
        if i % len(Q)//20 == 0:
            print('#' ,end='', flush=True)

        data_buf, position, _ = pickle_extractor( [0] , item)

        _data.append( (data_buf, position ) )
        _other_info.append((item['sgf_content'], item['order_tags'], item['specific_terms']))
        eng_abbrs.extend(item['order_tags'])

    
    eng_abbrs = list(set(eng_abbrs))
    print('')
    # print('args', args.eng_abbrs, 'data_dict key', eng_abbrs)
    data_dict = {}
    for k in eng_abbrs:
        data_dict[k] = {}
        data_dict[k]['data'] = []
        data_dict[k]['other_info'] = []
        
    for data, other_info in zip( _data, _other_info):
        if other_info[1][0] in eng_abbrs:
            k = other_info[1][0]
            data_dict[k]['data'].append( data )
            data_dict[k]['other_info'].append( other_info )
            
        else:
            print('Error found some abbr not in list')

    return data_dict, eng_abbrs

class TestMulticlass(Dataset): #for cross compare , between models 

    def __init__(self, args, eng_abbrs, data_dict, ratio = 0.9, is_train = False):
        self.label, self.data, self.other_info = [], [], []
        count_pos = 0 
        count_neg = 0        
        for k in eng_abbrs:
            print('unsplit', k, len(data_dict[k]['data']))

            _data = train_test_split(data_dict[k]['data'], ratio, is_train)
            _other_info = train_test_split( data_dict[k]['other_info'], ratio, is_train)
            _label = [ np.array([ 0 ]) for info in _other_info]
            
            
            self.label.extend( _label )
            self.data.extend( _data )
            self.other_info.extend( _other_info )                

        print('len self.label', len(self.label))
        print(count_pos, count_neg)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index][0]), torch.LongTensor( [self.data[index][1]]) , torch.FloatTensor( self.label[index])


class DataMulticlass1v1(Dataset):
    def make_weights_for_balanced_classes(self, nclasses= 2):                        
        count = [0] * nclasses
        images = self.label                                                 
        for item in images:                                                         
            count[item[0]] += 1                                                     
        weight_per_class = [0.] * nclasses                                      
        N = float(sum(count))                                                   
        for i in range(nclasses):                                                   
            weight_per_class[i] = N/float(count[i])                                 
        weight = [0] * len(images)                                              
        for idx, val in enumerate(images):                                          
            weight[idx] = weight_per_class[val[0]]                                  
        return weight

    def __init__(self, args, enga,engb, data_dict, ratio = 0.9, is_train = True,auto_balance = True):
        self.label, self.data, self.other_info = [], [], []
        count_pos = 0 
        count_neg = 0        
        for k in [enga, engb]:
            print('unsplit', k, len(data_dict[k]['data']))

            _data = train_test_split(data_dict[k]['data'], ratio, is_train)
            _other_info = train_test_split( data_dict[k]['other_info'], ratio, is_train)

            if k == enga:
                _label = [ np.array([ 1 ]) for info in _other_info]
                count_pos += len(_label)
            else:
                _label = [ np.array([ 0 ]) for info in _other_info]
                count_neg += len(_label)
            
            self.label.extend( _label )
            self.data.extend( _data )
            self.other_info.extend( _other_info )                

        print('len self.label', len(self.label))
        print(count_pos, count_neg)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index][0]), torch.LongTensor( [self.data[index][1]]) , torch.FloatTensor( self.label[index])

class DataMulticlassFixedNeg(Dataset):
    def __init__(self, args, eng, data_dict, ratio = 0.9, is_train = True):

        print('@DataMulticlassFixedNeg')
        print('is_train', is_train)
        

        self.label, self.data, self.other_info = [], [], []
        count_pos = 0 
        count_neg = 0

        neg_fix_num = max( [ len(train_test_split(data_dict[k]['data'], ratio, is_train)) for k in args.eng_abbrs ] )
        
        for k in args.eng_abbrs + ['neg']:
            print('unsplit', k, len(data_dict[k]['data']))

            if k == 'neg':
                _data = train_test_split(data_dict[k]['data'], ratio, is_train)[:neg_fix_num]
                _other_info = train_test_split( data_dict[k]['other_info'], ratio, is_train)[:neg_fix_num]
            else:
                _data = train_test_split(data_dict[k]['data'], ratio, is_train)
                _other_info = train_test_split( data_dict[k]['other_info'], ratio, is_train)
            
            
            if k == eng:
                _label = [ np.array([ 1 ]) for info in _other_info]
                count_pos += len(_label)
            else:
                # continue
                _label = [ np.array([ 0 ]) for info in _other_info]
                count_neg += len(_label)

            self.label.extend( _label )
            self.data.extend( _data )
            self.other_info.extend( _other_info )

        print('len self.label', len(self.label))
        print(count_pos, count_neg)
        

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index][0]), torch.LongTensor( [self.data[index][1]]) , torch.FloatTensor( self.label[index])


class DataMulticlass(Dataset):
    def __init__(self, args, eng, data_dict, ratio = 0.9, is_train = True, neg_sampling_k = 5):

        print('@DataMulticlass')
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
                # continue
                _label = [ np.array([ 0 ]) for info in _other_info]
                count_neg += len(_label)

            self.label.extend( _label )
            self.data.extend( _data )
            self.other_info.extend( _other_info )

        print(count_pos, count_neg, neg_sampling_k)
        remain_need = max(count_pos*neg_sampling_k - count_neg, 0)

        print('count_pos',count_pos, 'count_neg',count_neg,'remain_need',remain_need)
        Q = len(train_test_split( data_dict['neg']['data'], ratio, is_train))
        _data = train_test_split( data_dict['neg']['data'], ratio, is_train)[Q-remain_need:]
        _other_info = train_test_split( data_dict['neg']['other_info'], ratio, is_train)[Q-remain_need:]
        _label = [ np.array([ 0 ]) for info in _other_info]

        print('len neg _label', len(_label))

        self.label.extend( _label)
        self.data.extend( _data )
        self.other_info.extend( _other_info )

        print('len self.label', len(self.label))
    
    def traindatapreview(self):
        content = ''
        for item, label in zip(self.data,self.label):
            content += '{} {}\n'.format( label, getmd5(item.__str__()) )
        with open('DataMulticlass_traindata.preview', 'w') as fout:
            fout.write(content)
        

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index][0][0]), torch.LongTensor( [self.data[index][0][1] ] ) , torch.FloatTensor( self.data[index][2] )



class DataNegSample(Dataset): 
    def make_weights_for_balanced_classes(self, nclasses= 6):                        
        count = [0] * nclasses
        images = [ self.Q[i]['label'] for i in range(len(self.Q)) ]

        print('images',len(images))                                              
        for item in images:                             
            count[ item.index(1) ] += 1                                                     
        weight_per_class = [0.] * nclasses                                      
        N = float(sum(count))                                                   
        for i in range(nclasses):                                                   
            weight_per_class[i] = N/float(count[i])                                 
        weight = [0] * len(images)         
        print(weight_per_class)                                     
        for idx, val in enumerate(images):                                          
            weight[idx] = weight_per_class[val[0]]                                  
        return weight       

    def __init__(self, pickle_filepath, shuffle = False,need_position = True, need_label = True):
        # pretrain_labels = ['', '超高目', '斷', '雙', '三目當中', '大壓梁', '衝', '跨', '五五', '挖', '小飛', '雙虎口', '三三', '猴子臉', '小目', '碰', '夾', '反扳', '扳', '超大飛', '刺', '小馬步掛', '天元', '小目二間高締', '自殺', '高目', '裂型', '迷你中國流布局', '拐', '長', '扭斷', '目外', '一間高夾', '門', '星位大馬步締', '象飛', '星位', '台象', '二間跳', '變形迷你中國流布局', '連扳', '小目小馬步締', '擠', '扳斷', '尖頂', '小目大馬步締', '向小目', '退', '穿象眼', '二間低夾', '星位二間高締', '錯小目無憂角布局', '點角', '二間高夾', '補方', '並', '虎口', '大飛', '三連星布局', '帶鉤', '空三角', '覷', '尖衝', '二連星布局', '連接', '雙飛燕', '二四侵分', '二五侵分', '一間高掛', '提劫', '車後推', '擋', '錯小目', '中國流布局', '星位小馬步締', '星位一間高締', '叫吃', '星無憂角布局', '小目一間高締', '小林流布局', '三間高夾', '填', '提', 'Make-Ko', '高中國流布局', '三間低夾', '頂', '尖', '點方', '一間低夾', '一間跳']
        # humantag_labels = ['星無憂角布局', '衝', '連接', '二間跳', '二間低夾', '蓋', '小目一間高締', '高目', '星位一間高締', '二間高掛', '二四侵分', '門', '星位', '象飛', '三連星布局', '一間低掛', '二間高夾', '斷', '點方', '小目', '超大飛', '星位大馬步締', '補方', '開拆', '尖頂', '撲', '尖衝', '目外', '長', '提劫', '逼', '打入', '扳', '小目二間高締', '迷你中國流布局', '扭斷', '跨', '扳斷', '夾', '雙飛燕', '空三角', '立下', '大飛', '超高目', '台象', '拐', '二間低掛', '猴子臉', '退', '阻渡', '穿象眼', '貼', '虎口', '一間高夾', '天元', '黏', '點角', '並', '雙虎口', '壓', '一間高掛', '小馬步掛', '填', '錯小目', '渡過', '托', '一間跳', '點', '一間低夾', '緊氣', '鎮', '星位小馬步締', '提', '團', '尖', '三間低夾', '淺消', '三目當中', '五五', '三間高夾', '反扳', '擠', '向小目', '二五侵分', '叫吃', '小目小馬步締', '三三', '二連星布局', '擋', '連扳', '覷', '碰', '頂', '挖', '雙', '刺', '小飛']
        # self.target_labels = list(set(pretrain_labels + humantag_labels))
        self.target_labels = ['撞', '連接', '高目', '點三三', '挖', '雙', '補方', '斷', '向小目', '扭斷', '一間低掛', '小林流布局', '門', '拐', '三間低夾', '扳斷', '穿象眼', '星位一間高締', '一間低夾', '一間跳', '淺消', '提子', '二間跳', '五五', '點方', '星位大馬步締', '渡過', '衝', '錯小目', '空三角', '小目二間高締', '二五侵分', '星位二間高締', '自殺', '小飛', '鎮', '黏', '扳', '二間高掛', '壓', '貼', '尖', '二間高夾', '星無憂角布局', '星位', '打入', '小目大 馬步締', '雙虎口', '超大飛', '象飛', '緊氣', '超高目', '退', '跨', '反扳', '連扳', '開劫', '中國流布局', '小目小馬步締', '目外', '開拆', '踢', ' 提劫', '三連星布局', '星位小馬步締', '一間高掛', '逼', '三目當中', '二連星布局', '天元', '錯小目無憂角布局', '擠', '小馬步掛', '叫吃', '三三', ' 台象', '阻渡', '迷你中國流布局', '雙飛燕', '蓋', '點', '小目', '小鐵塔', '夾', '尖衝', '小目一間高締', '長', '托', '三間高夾', '覷', '大飛', '並', '高中國流布局', '撲', '填', '擋', '碰', '一間高夾', '立下', '二四侵分', '二間低夾', '團', '二間低掛', '虎口']
        self.label, self.data, self.other_info = [], [], []
        self.need_position = need_position
        self.need_label = need_label

        with open(  pickle_filepath, 'rb') as fin:
            dataset_lst = pickle.load(fin)

        self.Q = dataset_lst
        print('dataset size {} {}'.format(pickle_filepath, len(self.Q)))
        

        try:
            pos, neg = 0,0
            for x in self.Q:
                if x['label'] == [1]:
                    pos += 1
                elif x['label'] == [0]:
                    neg += 1
                else:
                    pass
                    # print(x['label'])
            print( x['order_tags'], x.keys())
            print('{} pos {} neg {} total {}'.format(pickle_filepath , pos,neg, pos+neg))
        except:
            print('Fail to acquire dataset stats...')

            

    def __len__(self):
        return len(self.Q)
        
    def __getitem__(self, index):
        try:
            if self.need_position and self.need_label:
                return torch.FloatTensor(self.Q[index]['features']), torch.LongTensor( [self.Q[index]['position']]) , torch.FloatTensor( self.Q[index]['label']) 
            elif not self.need_position and self.need_label:
                return torch.FloatTensor(self.Q[index]['features']), torch.FloatTensor( self.Q[index]['label']) 
            elif self.need_position and not self.need_label:
                return torch.FloatTensor(self.Q[index]['features']), torch.LongTensor( [self.Q[index]['position']])
            else:
                print('error')
        except:
            print(self.Q[index]['features'])
            print(self.Q[index]['position'])
            print(self.Q[index]['label'])


class DataSimpleTerm(Dataset):
    def __init__(self, args, data_dict, ratio = 0.9, is_train = True, test_shuffle= True):
        pretrain_labels = ['', '超高目', '斷', '雙', '三目當中', '大壓梁', '衝', '跨', '五五', '挖', '小飛', '雙虎口', '三三', '猴子臉', '小目', '碰', '夾', '反扳', '扳', '超大飛', '刺', '小馬步掛', '天元', '小目二間高締', '自殺', '高目', '裂型', '迷你中國流布局', '拐', '長', '扭斷', '目外', '一間高夾', '門', '星位大馬步締', '象飛', '星位', '台象', '二間跳', '變形迷你中國流布局', '連扳', '小目小馬步締', '擠', '扳斷', '尖頂', '小目大馬步締', '向小目', '退', '穿象眼', '二間低夾', '星位二間高締', '錯小目無憂角布局', '點角', '二間高夾', '補方', '並', '虎口', '大飛', '三連星布局', '帶鉤', '空三角', '覷', '尖衝', '二連星布局', '連接', '雙飛燕', '二四侵分', '二五侵分', '一間高掛', '提劫', '車後推', '擋', '錯小目', '中國流布局', '星位小馬步締', '星位一間高締', '叫吃', '星無憂角布局', '小目一間高締', '小林流布局', '三間高夾', '填', '提', 'Make-Ko', '高中國流布局', '三間低夾', '頂', '尖', '點方', '一間低夾', '一間跳']
        humantag_labels = ['星無憂角布局', '衝', '連接', '二間跳', '二間低夾', '蓋', '小目一間高締', '高目', '星位一間高締', '二間高掛', '二四侵分', '門', '星位', '象飛', '三連星布局', '一間低掛', '二間高夾', '斷', '點方', '小目', '超大飛', '星位大馬步締', '補方', '開拆', '尖頂', '撲', '尖衝', '目外', '長', '提劫', '逼', '打入', '扳', '小目二間高締', '迷你中國流布局', '扭斷', '跨', '扳斷', '夾', '雙飛燕', '空三角', '立下', '大飛', '超高目', '台象', '拐', '二間低掛', '猴子臉', '退', '阻渡', '穿象眼', '貼', '虎口', '一間高夾', '天元', '黏', '點角', '並', '雙虎口', '壓', '一間高掛', '小馬步掛', '填', '錯小目', '渡過', '托', '一間跳', '點', '一間低夾', '緊氣', '鎮', '星位小馬步締', '提', '團', '尖', '三間低夾', '淺消', '三目當中', '五五', '三間高夾', '反扳', '擠', '向小目', '二五侵分', '叫吃', '小目小馬步締', '三三', '二連星布局', '擋', '連扳', '覷', '碰', '頂', '挖', '雙', '刺', '小飛']
        
        self.target_labels = list(set(pretrain_labels + humantag_labels))
        

        self.label, self.data, self.other_info = [], [], []
        _label, _data, _other_info = [],[],[]

        print('total {} labels'.format(len(self.target_labels)))

        for k in ['term']:
            _data = train_test_split(data_dict[k]['data'], ratio, is_train)
            _other_info = train_test_split( data_dict[k]['other_info'], ratio, is_train)
            
            for info in _other_info:
                label_buf = [0]*len(self.target_labels)
                for _term in info[2]['term']:
                    label_buf[ self.target_labels.index(_term) ] = 1
                _label.append(label_buf)

        self.label.extend( _label )
        self.data.extend( _data )
        self.other_info.extend( _other_info )


    def traindatapreview(self):
        content = ''
        for item, label in zip(self.data,self.label):
            content += '{} {}\n'.format( label, getmd5(item.__str__()) )
        with open('DataSimpleTerm_traindata.preview', 'w') as fout:
            fout.write(content)

    def __len__(self):
        print('dataset size ', len(self.data))
        return len(self.data)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index][0]), torch.LongTensor( [self.data[index][1]]) , torch.FloatTensor( self.label[index])

class DataSimpleTerm(Dataset):
    def __init__(self, pickle_filepath, ratio = 0.9, is_train = True, shuffle = False):
        # pretrain_labels = ['', '超高目', '斷', '雙', '三目當中', '大壓梁', '衝', '跨', '五五', '挖', '小飛', '雙虎口', '三三', '猴子臉', '小目', '碰', '夾', '反扳', '扳', '超大飛', '刺', '小馬步掛', '天元', '小目二間高締', '自殺', '高目', '裂型', '迷你中國流布局', '拐', '長', '扭斷', '目外', '一間高夾', '門', '星位大馬步締', '象飛', '星位', '台象', '二間跳', '變形迷你中國流布局', '連扳', '小目小馬步締', '擠', '扳斷', '尖頂', '小目大馬步締', '向小目', '退', '穿象眼', '二間低夾', '星位二間高締', '錯小目無憂角布局', '點角', '二間高夾', '補方', '並', '虎口', '大飛', '三連星布局', '帶鉤', '空三角', '覷', '尖衝', '二連星布局', '連接', '雙飛燕', '二四侵分', '二五侵分', '一間高掛', '提劫', '車後推', '擋', '錯小目', '中國流布局', '星位小馬步締', '星位一間高締', '叫吃', '星無憂角布局', '小目一間高締', '小林流布局', '三間高夾', '填', '提', 'Make-Ko', '高中國流布局', '三間低夾', '頂', '尖', '點方', '一間低夾', '一間跳']
        # humantag_labels = ['星無憂角布局', '衝', '連接', '二間跳', '二間低夾', '蓋', '小目一間高締', '高目', '星位一間高締', '二間高掛', '二四侵分', '門', '星位', '象飛', '三連星布局', '一間低掛', '二間高夾', '斷', '點方', '小目', '超大飛', '星位大馬步締', '補方', '開拆', '尖頂', '撲', '尖衝', '目外', '長', '提劫', '逼', '打入', '扳', '小目二間高締', '迷你中國流布局', '扭斷', '跨', '扳斷', '夾', '雙飛燕', '空三角', '立下', '大飛', '超高目', '台象', '拐', '二間低掛', '猴子臉', '退', '阻渡', '穿象眼', '貼', '虎口', '一間高夾', '天元', '黏', '點角', '並', '雙虎口', '壓', '一間高掛', '小馬步掛', '填', '錯小目', '渡過', '托', '一間跳', '點', '一間低夾', '緊氣', '鎮', '星位小馬步締', '提', '團', '尖', '三間低夾', '淺消', '三目當中', '五五', '三間高夾', '反扳', '擠', '向小目', '二五侵分', '叫吃', '小目小馬步締', '三三', '二連星布局', '擋', '連扳', '覷', '碰', '頂', '挖', '雙', '刺', '小飛']
        # self.target_labels = list(set(pretrain_labels + humantag_labels))
        self.target_labels = ['撞', '連接', '高目', '點三三', '挖', '雙', '補方', '斷', '向小目', '扭斷', '一間低掛', '小林流布局', '門', '拐', '三間低夾', '扳斷', '穿象眼', '星位一間高締', '一間低夾', '一間跳', '淺消', '提子', '二間跳', '五五', '點方', '星位大馬步締', '渡過', '衝', '錯小目', '空三角', '小目二間高締', '二五侵分', '星位二間高締', '自殺', '小飛', '鎮', '黏', '扳', '二間高掛', '壓', '貼', '尖', '二間高夾', '星無憂角布局', '星位', '打入', '小目大 馬步締', '雙虎口', '超大飛', '象飛', '緊氣', '超高目', '退', '跨', '反扳', '連扳', '開劫', '中國流布局', '小目小馬步締', '目外', '開拆', '踢', ' 提劫', '三連星布局', '星位小馬步締', '一間高掛', '逼', '三目當中', '二連星布局', '天元', '錯小目無憂角布局', '擠', '小馬步掛', '叫吃', '三三', ' 台象', '阻渡', '迷你中國流布局', '雙飛燕', '蓋', '點', '小目', '小鐵塔', '夾', '尖衝', '小目一間高締', '長', '托', '三間高夾', '覷', '大飛', '並', '高中國流布局', '撲', '填', '擋', '碰', '一間高夾', '立下', '二四侵分', '二間低夾', '團', '二間低掛', '虎口']
        self.label, self.data, self.other_info = [], [], []


        with open(  pickle_filepath, 'rb') as fin:
            dataset_lst = pickle.load(fin)      

        self.Q = train_test_split(dataset_lst, ratio, is_train)
        print('istrain {} dataset size {}'.format(is_train, len(self.Q)))            

    def __len__(self):
        return len(self.Q)
        
    def __getitem__(self, index):
        return torch.FloatTensor(self.Q[index]['features']), torch.LongTensor( [self.Q[index]['position']]) , torch.FloatTensor( self.Q[index]['label']), self.Q[index]['sgf_content']

