import torch 
import torch.nn.functional as F
from torch import nn

def get_block(dim_in, dim_out, kernel, stride, pad):
    block = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel, stride, pad),
            nn.BatchNorm2d(dim_out, eps=1e-5, momentum=0.9), #momentum default 0.1
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


class GoModel_lal(nn.Module):
    def __init__(self, label_size, hidden_size, dim_in=8):
        super(GoModel_lal, self).__init__()
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
        self.fc = nn.Linear(hidden_size, label_size) # [1] [0]

    def forward(self, args, boards, positions):

        outs = self.init_block(boards)
        outs = F.relu(self.block1(outs) + self.add1(outs))
        #print(outs.shape)
        outs = F.relu(self.block2(outs) + self.add2(outs))
        #print(outs.shape)
        outs = F.relu(self.block3(outs) + self.add3(outs))
        #print(outs.shape)
        outs = outs.permute(0, 2, 3, 1).view(-1, 19*19, self.hidden_size)  # batch*361*
        #print(outs.shape)

        offset = torch.arange(0, outs.size(0)*outs.size(1), outs.size(1)).to(args.device) # batch si
        
        positions = positions.view(-1, positions.size(0)) + offset
        
        outs = outs.contiguous().view(-1, outs.size(-1))[positions].squeeze(0)
        #print('get position', outs.shape)
        
        return torch.sigmoid(self.fc(outs))

###new model
def MakeBlock(ChannelSize):
    return nn.Sequential(nn.Conv2d(ChannelSize, ChannelSize, 3, 1, 1), nn.BatchNorm2d(ChannelSize))

class GoModel_frank(nn.Module) :
    def __init__(self, OutputSize, InputChannel = 17, BlockNum = 3, ChannelSize = 256) :
        super(GoModel_frank, self).__init__()

        CNN_Start = nn.Sequential(nn.Conv2d(InputChannel, ChannelSize, 3, 1, 1), nn.BatchNorm2d(ChannelSize))
        CNN_End = nn.Sequential(nn.Conv2d(ChannelSize, 1, 1, 1, 0), nn.BatchNorm2d(1))

        self.CNN_List = nn.ModuleList([CNN_Start] + [MakeBlock(ChannelSize) for i in range(BlockNum * 2)] + [CNN_End])
        self.FullyConnect = nn.Linear(361, OutputSize)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, InputData) :
        residual = self.relu(self.CNN_List[0](InputData))
        for CNN_Index in range(1, len(self.CNN_List) - 2, 2) :
        	data = self.relu(self.CNN_List[CNN_Index](residual))
        	residual = self.relu(self.CNN_List[CNN_Index + 1](data) + residual)
        data = self.relu(self.CNN_List[-1](residual)).view(-1, 361)
        return torch.sigmoid(self.FullyConnect(data))


class NetA(torch.nn.Module):
    def __init__(self, args, n_feature, n_hidden, n_output):
        super(NetA, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden).to(args.device)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output).to(args.device)   # output layer


    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x


class NetB(torch.nn.Module):
    def __init__(self, args, n_feature, n_hidden, n_output):
        super(NetB, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden).to(args.device)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output).to(args.device)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = F.softmax(self.out(x))
        return x