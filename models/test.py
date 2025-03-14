#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    confidence = [0 for i in range(args.num_classes)]# 不同类的单一准确率
    confidence_loss = [0 for i in range(args.num_classes)]# 不同类的单一损失
    num_classes = [0 for i in range(args.num_classes)]# 一个batch中各个类的数量
    data_loader = DataLoader(datatest, batch_size=args.bs)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        log_probs = net_g(data) # (batch_size, num_class)，各类的置信度
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1] #(batch_size, 1), 置信度最大的类
        compare = y_pred.eq(target.data.view_as(y_pred)) #(batch_size, 1)， 是否预测对
        correct += compare.long().cpu().sum().item() #预测对的数量
        for i in range(int(target.size(0))):
            label = target[i].item() # 第i个的标签
            confidence[label] += compare[i].item() # 第i个预测对就+1
            num_classes[label] += 1
            if compare[i].item() > 0 :
                confidence_loss[label] += F.cross_entropy(log_probs[i,:], target[i]).item()

    test_loss /= len(data_loader.dataset)
    accuracy = (100.00 * correct / len(data_loader.dataset))
    for i in range(len(confidence)): 
        confidence[i] /=  num_classes[i]
        confidence_loss[i] /=  num_classes[i]
    
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss, confidence, confidence_loss

def test_measures(net_g, datatest, args):
    # net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    trainrunning_grad = 0
    criterion = nn.CrossEntropyLoss()
    confidence = [0 for i in range(args.num_classes)]# 不同类的单一准确率
    confidence_loss = [0 for i in range(args.num_classes)]# 不同类的单一损失
    num_classes = [0 for i in range(args.num_classes)]# 一个batch中各个类的数量
    data_loader = DataLoader(datatest, batch_size=args.bs)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()
        net_g.zero_grad()
        log_probs = net_g(data) # (batch_size, num_class)，各类的置信度
        # sum up batch loss
        loss = criterion(log_probs, target)
        loss.backward()
        test_loss += loss.item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1] #(batch_size, 1), 置信度最大的类
        compare = y_pred.eq(target.data.view_as(y_pred)) #(batch_size, 1)， 是否预测对
        correct += compare.long().cpu().sum().item() #预测对的数量
        for i in range(int(target.size(0))):
            label = target[i].item() # 第i个的标签
            confidence[label] += compare[i].item() # 第i个预测对就+1
            num_classes[label] += 1
            if compare[i].item() > 0 :
                confidence_loss[label] += F.cross_entropy(log_probs[i,:], target[i]).item()
        # 初始化L2梯度范数平方和的变量
        squared_grad_norm = 0
        # 遍历模型参数，对其梯度的L2范数求平方和
        for p in net_g.parameters():
            squared_grad_norm += (torch.norm(p.grad)) ** 2
        trainrunning_grad += squared_grad_norm

    test_loss /= len(data_loader.dataset)
    accuracy = (100.00 * correct / len(data_loader.dataset))
    for i in range(len(confidence)): 
        confidence[i] /=  num_classes[i]
        confidence_loss[i] /=  num_classes[i]
    
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss, confidence, confidence_loss, trainrunning_grad
