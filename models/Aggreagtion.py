'''
FedAvg算法
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn
from utils.straggler_plan import fake_select, find_lcm_of_list


def FedAvgV1(w, total_data_sum, user_idx_this_round, dict_users):
    '''
    以每个client数据的datasize为权重的FedAvg
    '''
    w_avg = copy.deepcopy(w[0])     # 把第'0'个local_weight拿出来,<class 'collections.OrderedDict'>
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
    '''普通联邦平均'''
    len_of_w = len(w)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len_of_w):
            w_avg[k] = torch.add(w_avg[k], w[i][k])
        w_avg[k] = torch.div(w_avg[k], len_of_w)
    return w_avg

def FedAvg_list(w_locals, user_idx):
    '''普通联邦平均'''
    w = []
    for i in user_idx:
        if w_locals[i] != None:
            w.append(w_locals[i])
    len_of_w = len(w)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len_of_w):
            w_avg[k] = torch.add(w_avg[k], w[i][k])
        w_avg[k] = torch.div(w_avg[k], len_of_w)
    return w_avg

def FedAvg_dict(w_locals, user_weight):
    '''普通联邦平均'''
    s = sum(user_weight.values())
    list_key = list(user_weight.keys())
    w_avg = copy.deepcopy(w_locals[list_key[0]])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k] * float(user_weight[list_key[0]])
        for i in list_key[1:]:
            w_avg[k] += w_locals[i][k] * float(user_weight[i])
        w_avg[k] = torch.div(w_avg[k], s)
    return w_avg
# def FedAvg_weight(w_locals, user_idx, weight):
#     lcm = find_lcm_of_list([int(k) for k in weight.keys()])#5
#     user_weight = {str((lcm//int(k))/sum([int(k) for k in weight.keys()])): weight[k]*len(w_locals) for k in weight.keys()}#{'1/6':5,'5/6':5}
#     list_weight = []
#     for i in user_weight.keys():
#         for j in range(user_weight[i]):
#             list_weight.append(float(i))
#     w = []
#     a = []
#     for i in user_idx:
#         w.append(w_locals[i])
#         a.append(list_weight[i])
#     len_of_w = len(w)
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         w_avg[k] = w_avg * a[0]
#         for i in range(1, len_of_w):
#             w_avg[k] += w[i][k] * a[i]
#         w_avg[k] = torch.div(w_avg[k], len_of_w)
#     return w_avg

def FedwAvg(w,num_straggler):
    '''联邦平均，其中straggler的权重减少，非straggler增加'''
    len_of_w = len(w)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len_of_w):
            w_avg[k] = torch.add(w_avg[k], w[i][k])
        w_avg[k] = torch.div(w_avg[k], (len_of_w - 0.1 * num_straggler) + 0.1 * (20-num_straggler))
    return w_avg

def GradAvg(w_glob, g_locals, lr):
    # 梯度归一化
    for grad in g_locals:
        # 将所有梯度展平，拼接到一个向量中
        grad_vector = torch.cat([g.flatten() for g in grad.values()])
        # 计算向量的范数
        grad_norm = torch.norm(grad_vector, p=2)
        # 将所有梯度缩放到单位向量
        for g in grad.values():
            g /= grad_norm

    # 沿梯度方向更新w_glob
    len_of_g = len(g_locals)
    g_avg = copy.deepcopy(g_locals[0])
    for k in g_avg.keys():
        for i in range(1, len_of_g):
            g_avg[k] += g_locals[i][k]
        g_avg[k] = torch.mul(torch.div(g_avg[k], len_of_g), lr) + w_glob[k]
    return g_avg