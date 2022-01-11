import torch
import torch.nn as nn
from torch.nn import functional as F

class GoModel(nn.Module):
    def __init__(self, label_size, input_channel = 11):
        super(GoModel, self).__init__()

        self.relu = nn.ReLU(inplace = True)

        self.init_block = nn.Sequential(
            nn.Conv2d(input_channel, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

        # 64*19*19

        self.res_block1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.res_add1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
        )
        # 64*19*19

        self.res_block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )

        self.res_add2 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, 0),
            nn.BatchNorm2d(128),
        )
        # 128*10*10

        self.res_block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )

        self.res_add3 = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, 0),
            nn.BatchNorm2d(256),
        )
        # 256*5*5

        self.Linear = nn.Linear(256*5*5, label_size)

    def forward(self, args, input):
        x = self.init_block(input)
        x = F.relu(self.res_block1(x) + self.res_add1(x))
        x = F.relu(self.res_block2(x) + self.res_add2(x))
        x = F.relu(self.res_block3(x) + self.res_add3(x))

        x = x.view(-1, 256*5*5)
        output = self.Linear(x)
        
        return torch.sigmoid(output)