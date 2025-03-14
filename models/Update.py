#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import torchvision as tv
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np



class LocalUpdate(object):
    def __init__(self, args, dataset:Dataset, idxs:list):  # idxs为该本地训练所用数据集索引
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.loader_train = DataLoader(Subset(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

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
            y_delta[k] = torch.div(torch.sub(old_grad[k], y_delta[k]), self.args.lr)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), y_delta # 模型参数、损失、本地更新梯度

class FEDPROX_LocalUpdate(object):
    def __init__(self, args, dataset:Dataset, idxs:list):  # idxs为该本地训练所用数据集索引
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.loader_train = DataLoader(Subset(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

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
                # prox way
                parameters = []
                global_params = []
                for (_, param), (_, g_param) in zip(net.state_dict(keep_vars=True).items(), net_this_epoch.state_dict(keep_vars=True).items()):
                    if param.requires_grad:
                        parameters.append(param)
                        global_params.append(g_param)
                # gm = torch.cat([p.data.view(-1) for p in global_params], dim=0)
                # pm = torch.cat([p.data.view(-1) for p in parameters], dim=0)
                # loss += 0.5 * self.args.mu * torch.norm(gm-pm, p=2)
                loss.backward()
                for w, w_t in zip(parameters, global_params):
                    w.grad.data += self.args.mu * (w.data - w_t.data)

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
            y_delta[k] = torch.div(torch.sub(old_grad[k], y_delta[k]), self.args.lr)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), y_delta # 模型参数、损失、本地更新梯度

class PROXSKIP_LocalUpdate(object):
    def __init__(self, args, dataset:Dataset, idxs:list):  # idxs为该本地训练所用数据集索引
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.loader_train = DataLoader(Subset(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, w_local, h):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
        hat_x = w_local
        epoch_loss = []
        for iter in range(self.args.local_ep):
            x = net.state_dict()
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
            grad_this_ep = net.state_dict()
            for k in grad_this_ep.keys():
                grad_this_ep[k] = (x[k] - grad_this_ep[k]) / self.args.lr
                x[k] = x[k] - ((grad_this_ep[k] - h[k]) * self.args.lr)
            net.load_state_dict(x)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) # 模型参数、损失、本地更新梯度
    
class SCAFFOLD_LocalUpdate(object):
    def __init__(self, args, dataset:Dataset, idxs:list):  # idxs为该本地训练所用数据集索引
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.start_layer = args.start_layer
        self.loader_train = DataLoader(Subset(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, local_control, global_control):
        x = copy.deepcopy(net).state_dict()
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

        epoch_loss = []
        # num_batches = 0
        for iter in range(self.args.local_ep):
            y_last = net.state_dict()
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.loader_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                for param, sc, cc in zip(net.parameters(), 
                                        (v for k,v in global_control.items() if v.requires_grad), 
                                        (v for k,v in local_control.items() if v.requires_grad)):
                    param.grad += (sc.data - cc.data)
                optimizer.step()
                batch_loss.append(loss.item())
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.loader_train), 100. * batch_idx / len(self.loader_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # num_batches = len(batch_loss)
            grad_this_ep = net.state_dict()
            for k in grad_this_ep.keys():
                grad_this_ep[k] = ((y_last[k] - grad_this_ep[k]) / self.args.lr)
                y_last[k] = y_last[k] - ((grad_this_ep[k] - local_control[k] + global_control[k]) * self.args.lr)
            net.load_state_dict(y_last)
        
        # compute c_plus, which save at client (c+ = ci − c + 1/Kη (x − yi))
        y_delta = net.state_dict()# sum gradient - (x − yi)
        for k in y_delta.keys():
            y_delta[k] = torch.sub(y_delta[k], x[k])
        c_plus = {}
        coef = 1 / (self.args.lr * self.args.local_ep)# 1/Kη
        for i, ((_, c_l), (_, c_g), (k, y_d)) in enumerate(zip(local_control.items(), global_control.items(), y_delta.items())):
            if i < self.start_layer and self.args.method == 'FEDPVR':
                c_plus[k] = torch.zeros_like(c_l.detach(),requires_grad=False).to(torch.float32).to(c_l.device)
            else:
                c_plus[k] = (c_l - c_g - coef * y_d).to(torch.float32).to(c_l.device)

        # compute c_delta which upload to server
        c_delta = {}
        for (k, c_p), (_, c_l) in zip(c_plus.items(), local_control.items()):
            c_delta[k] = (c_p - c_l)

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), c_plus, c_delta, y_delta # 模型参数、损失、本地更新梯度

def layer_wise_scaffold(c_term, args,  device):
    """This function set the client correction and server correction before the conf.start_layer to 0"""
    c_term_update = copy.deepcopy(c_term)
    if args.method == 'FEDPVR':
        for i, key in enumerate(c_term_update.keys()):
            if i < args.start_layer:
                s_value = c_term_update[key]
                c_term_update[key] = torch.zeros_like(s_value).to(s_value.dtype).to(device)
    return c_term_update