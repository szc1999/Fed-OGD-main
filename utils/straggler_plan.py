from typing import List
import numpy as np
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
import torch

def select_client(round, num_user, straggler:dict):
    #
    user_idx_this_round = []
    circle_pre_users = {}
    idx_head = 0
    sum_frac = 0.0
    for k,v in straggler.items():
        sum_frac += v
        idx =idx_head + int(num_user * v)
        if round % int(k) == 0:
            for i in range(idx_head,idx):
                user_idx_this_round.append(i)
        for i in range(idx_head,idx):
            circle_pre_users[i] = int(k)
        idx_head = idx
    assert sum_frac == 1.0, 'straggler配比不合理'
        
    return user_idx_this_round, circle_pre_users

import math

def find_lcm_of_list(numbers):
    if len(numbers) < 2:
        raise ValueError("列表中至少需要两个整数来计算最小公倍数。")
    lcm = numbers[0]
    for i in range(1, len(numbers)):
        current_number = numbers[i]
        lcm = lcm * current_number // math.gcd(lcm, current_number)
    return lcm

def fake_select(num_user, straggler:dict):
    user_idx_this_round = []
    lcm = find_lcm_of_list([int(k) for k in straggler.keys()])
    fake_straggler = {str(lcm//int(k)):straggler[k] for k in straggler.keys()}
    idx_head = 0
    sum_frac = 0.0
    for k,v in fake_straggler.items():
        sum_frac += v
        idx =idx_head + int(num_user * v)
        for i in range(int(k)):
            for i in range(idx_head,idx):
                user_idx_this_round.append(i)
        idx_head = idx
    assert sum_frac == 1.0, 'straggler配比不合理'
        
    return user_idx_this_round, lcm

if __name__ == '__main__':
    # 示例用法

    for i in range(20):
        list_user = select_client(i, 10, {'1':0.3, '2':0.3, '3':0.4})
        print('user', list_user)
        
    for i in range(20):
        list_faker, lcm= fake_select(10, {'1':0.3, '2':0.3, '3':0.4})
        print('faker', list_faker,lcm)
        i += lcm