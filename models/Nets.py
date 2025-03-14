'''
多层感知机 MLP，以及CNN的模型（CNN分别用 cifar以及 mnist 数据集）

'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()

        self.conv1 = nn.Conv2d(1,32,5, padding=2)# mnist是黑白的单通道图片，所以in_channel=1 

        self.conv2 = nn.Conv2d(32,64,5, padding=2)

        self.fc1 = nn.Linear(64*7*7, 1024)# 
        self.fc2 = nn.Linear(1024, 10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)# 1×28×28 -> 32×28×28 -> 32×14×14
        x = F.max_pool2d(F.relu(self.conv2(x)),2)# 32×14×14 -> 64×14×14 -> 64×7×7
        x = x.view(-1, 64*7*7)# 64×7×7 -> 1×(64*7*7)
        x = F.relu(self.fc1(x))# 1×(64*7*7) -> 1×1024

        x = self.fc2(x)# 1×1024 -> 1×10


        return x


class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNCifarPlus(nn.Module):
    """Some Information about Net"""

    def __init__(self, args):
        super(CNNCifarPlus, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1),  # 3*32*32 -> 16*32*32
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 16*32*32 -> 16*16*16
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, padding=1),  # 16*16*16 -> 32*16*16
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 32*16*16 -> 32*8*8
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, padding=1),  # 32*8*8 -> 64*8*8
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)  # 64*8*8 -> 64*4*4
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(64*4*4, 32),
            torch.nn.ReLU(),
            # torch.nn.Dropout()
        )
        self.fc2 = torch.nn.Linear(32, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64*4*4)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
