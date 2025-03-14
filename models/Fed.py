'''
FedAvg算法
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvgV1(w, total_data_sum, user_idx_this_round, dict_users):
    '''
    以每个client数据的datasize为权重的FedAvg
    :param w:各client的local weight列表
    :param total_data_sum:总样本数
    :param user_idx_this_round:这轮参与聚合的client的索引
    :param dict_users:每个client的数据量
    :return:
    '''
    w_avg = copy.deepcopy(w[0])     # 把第'0'个local_weight拿出来
    print("w_avg: ", type(w_avg))   # <class 'collections.OrderedDict'>
    for k in w_avg.keys():
        w_avg[k] = torch.mul(w_avg[k], len(dict_users[user_idx_this_round[0]]))  # local_weight乘以根据数据量算出来的权重
        #j = 1
        for i in range(1, len(user_idx_this_round)):
            datasize = len(dict_users[user_idx_this_round[i]]) # 第i个client数据量
            w_avg[k] += torch.mul(w[i][k], datasize)
            #w_avg[k] += torch.mul(w[j][k], datasize)    # * len(dict_users[user_idx_this_round[i]]) ::所以为啥不用i而是j
            #j += 1
        w_avg[k] = torch.div(w_avg[k], total_data_sum)
    return w_avg


def FedAvg(w):
    len_of_w = len(w)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len_of_w):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len_of_w)
    return w_avg

def FedwAvg(w,sum_w):
    len_of_w = len(w)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len_of_w):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], sum_w)
    return w_avg

def GradAvg(w, g, lr):
    len_of_w = len(g)
    g_avg = copy.deepcopy(g[0])
    for k in g_avg.keys():
        for i in range(1, len_of_w):
            g_avg[k] += g[i][k]
        g_avg[k] = torch.mul(torch.div(g_avg[k], len_of_w), lr) + w[k]
    return g_avg