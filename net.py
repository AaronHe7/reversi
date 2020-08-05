import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

class Wrapper(Net):
    def __init__(self, game):
        self.game = game
        self.net = Net()

    def train(self, trials):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        for i in range(trials):
            while not self.game.game_over():
                board = torch.Tensor(self.game.board)
                inputs = board
                labels = self.game.mcts()
                ouput_policy, ouput_value = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                self.game.make_move()
            
