#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import torch
from typing import Dict


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=300, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: K")#
    parser.add_argument('--bs', type=int, default=8, help="local batch size: B")
    parser.add_argument('--local_bs', type=int, default=8, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--g_lr', type=float, default=0.003, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.1, help="SGD momentum (default: 0.5)")
    parser.add_argument('--method', type=str, default="FEDOGD", help="method of baseline")#
    parser.add_argument('--start_layer', type=int, default=84, help="method of baseline")

    # client join
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients join in aggregation: C")#
    parser.add_argument('--straggler', type=str, default= '{"1":0.4,"3":0.3,"5":0.3}',#None,
                        #'{"1":0.6,"3":0.2,"5":0.2}' ,'{"1":0.4,"3":0.3,"5":0.3}'  '{"1":0.2,"3":0.4,"5":0.4}'None,
            help="key: frequry of join aggrecation; value: frac. note:the straggler is fix")
    parser.add_argument('--fake_straggler', type=str, default=None)#'{"1":0.5,"5":0.5}'

    # GradMA
    parser.add_argument('--beta_1', type=float, default=0.9, help="learning rate")
    parser.add_argument('--beta_2', type=float, default=0.0, help="learning rate")
    
    # model arguments
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--dataset', type=str, default="cifar10", help="method of baseline")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")#

    # other arguments
    parser.add_argument('--iid', type=bool, default=False, help='whether i.i.d or not')#
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--device', type=torch.device, default=torch.device("cuda:0"))
    parser.add_argument('--verbose', type=bool, default=False, help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    args = parser.parse_args()

    return args
