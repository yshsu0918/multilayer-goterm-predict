import numpy as np
import re
import os
import torch
import torch.utils.data as data

class AugLoader(data.Dataset):
    def __init__(self, root, library):
        self.root = root
        self.library = library

    def get_label(self, sgf_str):
        label = [0]*len(self.library)
        terms = sgf_str.split('\"')[-2]
        terms = terms.replace(' ', '').replace('\n', '')
        terms = terms.replace('[', '').replace(']', '').replace("\'", '').replace(',', ';')
        for term in terms.split(';'):
            try:
                idx = self.library.index(term)
            except:
                if term == '':
                    continue
                print("terms: ", terms, " term: ", term)
                idx = -1
            label[idx] = 1

        return label

    def find_pos(self, position):
        pos = -1
        for i in range(19*19):
            if position[i] == 1:
                pos = i
        
        if pos == -1:
            print("Wrong!!")
        return [pos]

    def __len__(self):
        return len([name for name in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, name))]) 
        
    def __getitem__(self, idx):
        with open(self.root + "board_" + str(idx+1) + ".sgf", 'r') as fin:
            sgf_str = fin.readline()
            position = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            black1 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            white1 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            policy1 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            bv1 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            connect1 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            black2 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            white2 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            policy2 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            bv2 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            connect2 = fin.readline().split(' ', 1)[1].split(' ')[:-1]

        #position  = np.array([float(i) for i in position]).reshape((19, 19))
        position  = np.array([float(i) for i in position])
        black1 = np.array([float(i) for i in black1]).reshape((19, 19))
        white1 = np.array([float(i) for i in white1]).reshape((19, 19))
        policy1 = np.array([float(i) for i in policy1]).reshape((19, 19))
        bv1 = np.array([float(i) for i in bv1]).reshape((19, 19))
        connect1 = np.array([float(i) for i in connect1]).reshape((19, 19))
        black2 = np.array([float(i) for i in black2]).reshape((19, 19))
        white2 = np.array([float(i) for i in white2]).reshape((19, 19))
        policy2 = np.array([float(i) for i in policy2]).reshape((19, 19))
        bv2 = np.array([float(i) for i in bv2]).reshape((19, 19))
        connect2 = np.array([float(i) for i in connect2]).reshape((19, 19))

        position = self.find_pos(position)
        board = []
        #board.append(position)
        board.append(black1)
        board.append(white1)
        board.append(policy1)
        board.append(bv1)
        board.append(connect1)
        board.append(black2)
        board.append(white2)
        board.append(policy2)
        board.append(bv2)
        board.append(connect2)
        label = self.get_label(sgf_str)
        
        return torch.FloatTensor(board), torch.LongTensor(position), torch.FloatTensor(label)
