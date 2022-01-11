import numpy as np
import json
import matplotlib.pyplot as plt
import torch as torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data
from torch.autograd import Variable
import argparse
from dataloader import TermLoader
from model import GoModel

def load_library():
    library = []
    with open('../library.json', 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            Q = json.loads(line)
            if Q['chinese_name'] not in library:
                library.append(Q['chinese_name'])
    library.append('None')
    return library

def collect_data():
    testset = TermLoader(args=args, root="../data/finetune/test/", mode="test", library=library, pretrain=False)
    testLoader = data.DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    return testLoader

def write_term(output, fout):
    output = torch.squeeze(output)
    _, indices = output.topk(3, dim = 0, largest = True, sorted = True)
    for idx in indices:
        fout.write(library[idx] + ' ')
    fout.write("\n")
            
def write_surround(output, type, fout):
    output = torch.squeeze(output)
    for i in range((args.k_hop*2+1)**2):
        if type == 'BV':
            fout.write(str(output[i].item())+' ')
        else:
            if output[i].item() >= 0.5:
                fout.write('1 ')
            else:
                fout.write('0 ')
    fout.write("\n")

def eval():
    model.eval()
    test_idx = 1
    for step, (board, position, term_label, black_label, white_label, empty_label, outBound_label, BV_label) in enumerate(testLoader):
        board = Variable(board).to(args.device)
        position = Variable(position).to(args.device)
        term_label = Variable(term_label).to(args.device)
        black_label = Variable(black_label).to(args.device)
        white_label = Variable(white_label).to(args.device)
        empty_label = Variable(empty_label).to(args.device)
        outBound_label = Variable(outBound_label).to(args.device)
        BV_label = Variable(BV_label).to(args.device)
        with torch.no_grad():
            term_out, black_out, white_out, empty_out, outBound_out, BV_out = model(args, board, position)
        print("handle " + str(test_idx) + " boards...")
        fout = open("./surround_pred/board_" + str(test_idx) + '.sgf', 'w')
        write_term(term_out, fout)
        write_surround(black_out, 'black', fout)
        write_surround(white_out, 'white', fout)
        write_surround(empty_out, 'empty', fout)
        write_surround(outBound_out, 'outBound', fout)
        write_surround(BV_out, 'BV', fout)
        fout.close()
        test_idx += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "None")
    parser.add_argument('-d', '--device', default = 'cuda') 
    parser.add_argument('--hidden_size', default = 128, type = int)
    parser.add_argument('--k_hop', default = 2, type = int)
    args = parser.parse_args()
    torch.cuda.set_device(3)

    library = load_library()
    testLoader = collect_data()
    model = GoModel(label_size=len(library), hidden_size=args.hidden_size, k_hop=args.k_hop).to(args.device)
    model.load_state_dict(torch.load('../term_model_weight'))
    eval()
