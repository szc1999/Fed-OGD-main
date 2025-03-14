'''
Created on March 9 2022 14:52:43
@author(s): HuangLab
'''
import csv
import math
import os
import random



# matplotlib.use('Agg')  # 绘图不显示

# import matplotlib.pyplot as plt
import copy
import numpy as np

import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

import time
import sys
file_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(file_dir)

from utils.AGNEWS import AGNewsDataset
from models.LSTM import RNNnet, TextCNN
from utils.imagenet import TinyImageNet
from utils.sampling import cifar_noniid
from utils.sampling import dataset_iid, dataset_noniid
from options import args_parser
from models.logistic import LogisticRegression
from models.Lenets import LenetMNIST, LenetCifar, LenetMnistPlus, LenetCifarPlus
from models.Resnet import ResNet, ResNet8
from models.Aggreagtion import FedAvgV1, FedAvg
from models.test import test_img
from train.FEDAVG import FedAVG
from train.FEDOGD import FedOGD


def FedLearnSimulate():
    # load args
    args = args_parser()
    for k,v in vars(args).items():
        print(k,v)
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print("cuda is available : ", torch.cuda.is_available())
    
    print("################################################load_dataset########################################################")

    if args.dataset == 'mnist':
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans)
        args.num_classes = 10
    elif args.dataset == 'cifar10':
        trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans)
        args.num_classes = 10    
    elif args.dataset == 'cifar100':
        trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        dataset_train = datasets.CIFAR100('../data/cifar', train=True, download=True, transform=trans)
        dataset_test = datasets.CIFAR100('../data/cifar', train=False, download=True, transform=trans)
        args.num_classes = 100
    elif args.dataset == 'imagenet':
        trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        dataset_train = TinyImageNet('../data/tiny-imagenet-200', train=True,  transform=trans)
        dataset_test = TinyImageNet('../data/tiny-imagenet-200', train=False,  transform=trans)
        args.num_classes = 200
    elif args.dataset == 'agnews':
        dataset_train = AGNewsDataset(root="../data/AG_news",train=True)
        dataset_test = AGNewsDataset(root="../data/AG_news",train=False)
        args.num_classes = 4
    else:
        exit('Error: unrecognized dataset')

    if args.iid:
        print("iid")
        dict_users = dataset_iid(dataset_train, args.num_users)
    else:
        print("non-iid")
        dict_users = dataset_noniid(dataset_train, args.num_users, args.num_classes, main_label_prop=0.80 if args.dataset == 'cifar100' else 0.95)

    print("################################################build model#########################################################")

    if args.model == 'lenet' and args.dataset != 'mnist':
        global_net = LenetCifar(args.num_classes).to(args.device)
    elif args.model == 'lenet' and args.dataset == 'mnist':
        global_net = LenetMNIST(args.num_classes).to(args.device)
    elif args.model == 'logistic':
        global_net = LogisticRegression(num_classes=args.num_classes).to(args.device)
    elif args.model == 'resnet8':
        inchannel = 1 if args.dataset == 'mnist' else 3
        global_net = ResNet8(inchannel, args.num_classes).to(args.device)
    elif args.model == 'resnet18':
        inchannel = 1 if args.dataset == 'mnist' else 3
        global_net = ResNet(inchannel, 18, args.num_classes).to(args.device)
    elif args.model == 'resnet34':
        inchannel = 1 if args.dataset == 'mnist' else 3
        global_net = ResNet(inchannel, 34, args.num_classes).to(args.device)
    elif args.model == 'lstm':
        # global_net = RNNnet(len_vocab=len(dataset_train.vocab),
        #     embedding_size=128,
        #     hidden_size=256,
        #     num_class=4,
        #     num_layers=1,
        #     mode='lstm').to(args.device)
        global_net = TextCNN(len_vocab=len(dataset_train.vocab)).to(args.device)
    else:
        global_net = None
        exit('Error: unrecognized model')
    # print("global_net:\n", global_net)
    global_net.apply(weights_init)
    model = copy.deepcopy(global_net)

    print("################################################training############################################################")
    acc_global_model = []
    loss_avg_client = []
    print('method:',args.method)
    if args.method == "FEDAVG":
        server = FedAVG(args, model)
        acc_global_model, loss_avg_client = server.run_quilk(dict_users, dataset_train, dataset_test)
    elif args.method == "FEDOGD":
        server = FedOGD(args, model)
        acc_global_model, loss_avg_client = server.run_quilk(dict_users, dataset_train, dataset_test)
             
    print('#################################################data_statistics####################################################')
    # acc.to_csv(f'./result/{args.method}_{args.dataset}_{args.iid}_ep{args.local_ep}_client{args.num_users}_acc.csv', encoding='gbk')
    # loss.to_csv(f'./result/{args.method}_{args.dataset}_{args.iid}_ep{args.local_ep}_client{args.num_users}_loss.csv', encoding='gbk')

    with open(f'../loss_{args.method}_{args.dataset}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for idx, row in enumerate(loss_avg_client):
            writer.writerow([idx, row])
    with open(f'../acc_{args.method}_{args.dataset}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for idx, row in enumerate(acc_global_model):
            writer.writerow([idx, row])


import torch.nn.init as init
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 0.01)
        init.constant_(m.bias, 0)

if __name__ == '__main__':
    FedLearnSimulate()


