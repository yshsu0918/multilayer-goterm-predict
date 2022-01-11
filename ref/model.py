import torch
from torch import nn
from torch.nn import functional as F

def get_block(dim_in, dim_out, kernel, stride, pad):
    block = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel, stride, pad),
            nn.BatchNorm2d(dim_out, eps=1e-5, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(dim_out, dim_out, 3, 1, 1),
            nn.BatchNorm2d(dim_out, eps=1e-5, momentum=0.9),
        )
    return block

def get_add(dim_in, dim_out, kernel, stride, pad):
    block = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel, stride, pad),
            nn.BatchNorm2d(dim_out),
        )
    return block 

class GoModel(nn.Module):
    def __init__(self, label_size, hidden_size, code_size, k_hop, dim_in=8):
        super(GoModel, self).__init__()
        self.hidden_size = hidden_size
        self.init_block = nn.Sequential(
            nn.Conv2d(dim_in, 64, 3, 1, 1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.9),
            nn.ReLU(),
        )
        self.block1 = get_block(64, hidden_size, 3, 1, 1)
        self.block2 = get_block(hidden_size, hidden_size, 3, 1, 1)
        self.block3 = get_block(hidden_size, hidden_size, 3, 1, 1)
        self.add1 = get_add(64, hidden_size, 3, 1, 1)
        self.add2 = get_add(hidden_size, hidden_size, 3, 1, 1)
        self.add3 = get_add(hidden_size, hidden_size, 3, 1, 1)
        self.fc = nn.Linear(hidden_size, label_size)

    def forward(self, args, boards, positions):
        outs = self.init_block(boards)
        outs = F.relu(self.block1(outs) + self.add1(outs))
        outs = F.relu(self.block2(outs) + self.add2(outs))
        outs = F.relu(self.block3(outs) + self.add3(outs))
        outs = outs.permute(0, 2, 3, 1).view(-1, 19*19, self.hidden_size)
        offset = torch.arange(0, outs.size(0)*outs.size(1), outs.size(1)).to(args.device)
        positions = positions.view(-1, positions.size(0)) + offset
        outs = outs.contiguous().view(-1, outs.size(-1))[positions].squeeze(0)
        
        return torch.sigmoid(self.fc(outs))