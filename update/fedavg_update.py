import copy
import torch
import torchvision as tv
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from sklearn import metrics
from utils.AGNEWS import collate_batch

class LocalUpdate(object):
    def __init__(self, args, dataset:Dataset, idxs:list):  # idxs为该本地训练所用数据集索引
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.loader_train = DataLoader(Subset(dataset, idxs), batch_size=self.args.local_bs, shuffle=True, collate_fn=collate_batch if self.args.dataset=='agnews' else None)

    def train(self, net):
        net_this_epoch = copy.deepcopy(net)
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

        epoch_loss = []
        num_batches = 0
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.loader_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.loader_train), 100. * batch_idx / len(self.loader_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            num_batches = len(batch_loss)
        y_delta = net.state_dict()
        old_grad = net_this_epoch.state_dict()
        for k in y_delta.keys():
            if 'weight' in k or 'bias' in k:
                y_delta[k] = torch.div(torch.sub(old_grad[k], y_delta[k]), self.args.lr)


        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), y_delta # 模型参数、损失、本地更新梯度