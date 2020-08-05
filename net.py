import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 2)
        self.conv2 = nn.Conv2d(10, 10, 2)
        self.p1 = nn.Linear(10 * 8 * 8, 64)
        self.p2 = nn.Linear(64, 64)
        self.p3 = nn.Linear(64, 64)
        self.v1 = nn.Linear(10 * 8 * 8, 64)
        self.v2 = nn.Linear(64, 64)
        self.v3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        o1 = self.p3(self.p2(self.p1(x)))
        o2 = self.v3(self.v2(self.v1(x)))
        return o1, o2

net = Net()
