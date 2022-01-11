import numpy as np
import re
import os
import torch
import torch.utils.data as data
from boardprocess import GetBoard

class TermLoader(data.Dataset):
    def __init__(self, args, root, mode, library, pretrain):
        self.args = args
        self.root = root
        self.mode = mode
        self.library = library
        self.pretrain = pretrain
        self.TagCPattern = '[^AP]C\[([^\]]*)\]'

    def get_term_label(self, sgf_str):
        label = [0]*len(self.library)
        if self.pretrain == True:
            terms = re.findall(self.TagCPattern, sgf_str)[0]
            term = ""
            for i in range(len(terms)):
                term += terms[i]
                try:
                    idx = self.library.index(term)
                    term = ""
                except:
                    continue
                label[idx] = 1
        else:
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

    def get_surround_label(self, Board, BV, position):
        k_hop = self.args.k_hop
        black_label = [0]*((k_hop*2+1)**2)
        white_label = [0]*((k_hop*2+1)**2)
        empty_label = [0]*((k_hop*2+1)**2)
        outBound_label = [0]*((k_hop*2+1)**2)
        BV_label = []
        for i in range(-k_hop, k_hop+1):
            for j in range(-k_hop, k_hop+1):
                row, col, idx = position//19 + i, position%19+j, (i+k_hop)*(k_hop*2+1)+(j+k_hop)
                if row < 0 or row >=19 or col < 0 or col >= 19:
                    outBound_label[idx] = 1
                    BV_label.append(0)
                else:
                    if Board[row][col] == 1:
                        black_label[idx] = 1
                    elif Board[row][col] == 2:
                        white_label[idx] = 1
                    else:
                        empty_label[idx] = 1
                    BV_label.append(BV[row][col])

        return black_label, white_label, empty_label, outBound_label, BV_label

    def print_surround_label(self, label):
        k_hop = self.args.k_hop
        for i in range(k_hop*2+1):
            for j in range(k_hop*2+1):
                idx = i*(k_hop*2+1)+j
                print(round(label[idx], 2), end='  ')
            print()
        print()

    def __len__(self):
        return len([name for name in os.listdir(self.root) if os.path.isfile(os.path.join(self.root, name))]) 
        
    def __getitem__(self, idx):
        with open(self.root + "board_" + str(idx+1) + ".sgf", 'r') as fin:
            sgf_str = fin.readline()
            fin.readline()
            Policy1 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            Value1 = fin.readline().split(' ')[1].split("\n")[0]
            BV1 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            Eye1 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            Connect1 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            fin.readline()
            Policy2 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            Value2 = fin.readline().split(' ')[1].split("\n")[0]
            BV2 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            Eye2 = fin.readline().split(' ', 1)[1].split(' ')[:-1]
            Connect2 = fin.readline().split(' ', 1)[1].split(' ')[:-1]

        Black1, White1, Black2, White2, Board, position = GetBoard(sgf_str)
        Policy1 = np.array([float(i) for i in Policy1]).reshape((19, 19))
        BV1 = np.array([float(i) for i in BV1]).reshape((19, 19))
        Eye1 = np.array([float(i) for i in Eye1]).reshape((19, 19))
        Connect1 = np.array([float(i) for i in Connect1]).reshape((19, 19))
        Policy2 = np.array([float(i) for i in Policy2]).reshape((19, 19))
        BV2 = np.array([float(i) for i in BV2]).reshape((19, 19))
        Eye2 = np.array([float(i) for i in Eye2]).reshape((19, 19))
        Connect2 = np.array([float(i) for i in Connect2]).reshape((19, 19))
        
        board = []
        board.append(Black1)
        board.append(White1)
        board.append(BV1)
        board.append(Connect1)
        board.append(Black2)
        board.append(White2)
        board.append(BV2)
        board.append(Connect2)
        label = self.get_term_label(sgf_str)

        return torch.FloatTensor(board), torch.LongTensor([position]), torch.FloatTensor(label)
