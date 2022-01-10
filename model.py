import torch 
import torch.nn.functional as F
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