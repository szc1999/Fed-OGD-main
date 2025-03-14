import copy
import json
import math
import random

import torch
from train.BaseServer import Server
from update.fedavg_update import LocalUpdate
from utils.funcs import angle_and_norm
from utils.ogd import *

from utils.straggler_plan import fake_select, select_client
import numpy as np
from update.fedogd_update import FedogdUpdate
from models.Aggreagtion import FedAvg_list, FedAvg_dict, FedAvgV1, FedAvg
from models.test import test_img

class FedOGD(Server):
    def __init__(self, args, global_net):
        super().__init__(args, global_net)  # include sef.args, self.net

        global_param = self.net.state_dict()
        # 保存每个client训练所得的参数、训练损失、更新
        self.loss_locals = [10.0 for i in range(args.num_users)]
        self.w_locals = [copy.deepcopy(global_param) for i in range(args.num_users)]#每个client保存的本地梯度
        zero_y = copy.deepcopy(global_param)
        for k in zero_y.keys():
            zero_y[k] = zero_y[k].zero_()
        self.y_deltas = [copy.deepcopy(zero_y) for i in range(args.num_users)]
        # 保存到server用于聚合的数据
        self.y_delta_on_server = [copy.deepcopy(zero_y) for i in range(args.num_users)]
        self.name_requires_grad = [n for n,p in self.net.named_parameters()]
        self.old_weight = copy.deepcopy(zero_y)
        # 保存测试的数据
        self.loss_list  = []  # 损失
        self.acc_list = []  # 准确率
        # 模仿延迟的straggler配置 v2
        self.user_idx_this_round = []
        self.straggler_idx_this_round = []
        self.circle_per_users = {}
        if args.fake_straggler != None:
            # 作user_idx_each_round，即有重复元素的长段user_list, step是所有staleness的公因数
            fake_straggler = json.loads(self.args.fake_straggler)
            # 拿了不放回，所以user_idx_each_round会不断清空，需要提前装好
            self.user_idx_each_round, self.step = fake_select(self.args.num_users, fake_straggler)
            self.user_num_each_round = math.ceil(len(self.user_idx_each_round)//self.step)
            print('fake_straggler',self.user_idx_each_round,'num',self.user_num_each_round)
    
    def user_join(self, round):
        # user_weight = {}
        circle_per_users = {i:i for i in range(self.args.num_users)}
        user_idx_this_round = np.random.choice(range(self.args.num_users), int(self.args.num_users * self.args.frac), replace=False).tolist()
        if self.args.fake_straggler != None:
            # v1，从user_idx_each_round中随机选取user_num_each_round个，每轮是独立试验
            user_idx_this_round = np.random.choice(self.user_idx_each_round, self.user_num_each_round, replace=False).tolist()
            # v2，从user_idx_each_round中随机选取user_num_each_round个，但选取之后不放回，是非独立试验
            if len(self.user_idx_each_round) == 0: # user_idx_each_round空了就重新创建
                fake_straggler = json.loads(self.args.fake_straggler)
                self.user_idx_each_round, self.step = fake_select(self.args.num_users, fake_straggler)
                self.user_num_each_round = math.ceil(len(self.user_idx_each_round)//self.step)
            user_idx_this_round = np.random.choice(self.user_idx_each_round, self.user_num_each_round, replace=False).tolist()
            for item in user_idx_this_round: #选中了就删除
                if item in self.user_idx_each_round:
                    self.user_idx_each_round.remove(item)
            print('fake_straggler',self.user_idx_each_round,'num',self.user_num_each_round)
            # # v3，在v1或v2的基础上，取消同一client上的重复训练，改为聚合时增加权重
            # for item in user_idx_this_round:
            #     if item in user_weight:
            #         user_weight[item] += 1
            #     else:
            #         user_weight[item] = 1
        elif self.args.straggler != None:
            straggler = json.loads(self.args.straggler)
            user_idx_this_round, circle_per_users = select_client(round, self.args.num_users, straggler)

        return user_idx_this_round, circle_per_users

    def train(self, dict_users, dataset_train, ogd_straggler=None, ogd_activate=None, epoch=0):
        for idx in self.user_idx_this_round:     # 遍历所有的设备
            # 第idx个client进行本地训练
            # args.local_ep = local_ep*user_weight[idx]
            # self.args.lr = lr * self.circle_per_users[idx]
            local = FedogdUpdate(args=self.args, dataset=dataset_train, idxs=dict_users[idx])
            if idx in self.straggler_idx_this_round:
                ogd = ogd_activate
            else:
                ogd = ogd_straggler
            weight, loss, y_delta = local.train(
                net=copy.deepcopy(self.net).to(self.args.device), 
                ogd=ogd,
                epoch=epoch)
            self.w_locals[idx] = weight
            self.loss_locals[idx] = loss
            self.y_deltas[idx] = y_delta
    
    def test(self, round, dataset_test):
        acc_test, loss_test, confidence, confidence_loss = test_img(self.net, dataset_test, self.args)
        # if np.isnan(loss_test):loss_test = 2.300
        self.acc_list.append(acc_test)
        self.loss_list.append(loss_test)
        print('Round {:3d}, Average loss {:.3f}, Global acc: {:.3f}, Confidence: {}, Confidence loss: {}, valid {:3d}'
            .format(round, loss_test, acc_test, confidence, confidence_loss, len(self.user_idx_this_round)))
        
    def run_quilk(self, dict_users, dataset_train, dataset_test, multilr=False):
        for round in range(0, self.args.epochs):
            self.user_idx_this_round, self.circle_per_users = self.user_join(round)
            self.straggler_idx_this_round = [k for k,v in self.circle_per_users.items() if v!=1]
            if len(self.user_idx_this_round) > 0:
                print("user_idx_this_round: ", self.user_idx_this_round)
            # server接收
            for i in self.user_idx_this_round:
                # for k,v in self.y_delta_on_server[i].items():
                #     self.y_delta_on_server[i][k] = self.y_delta_on_server[i][k] + self.y_deltas[i][k]
                self.y_delta_on_server[i] = copy.deepcopy(self.y_deltas[i])
            y_delta_avg = FedAvg_list(self.y_delta_on_server, [i for i in range(self.args.num_users)])#self.user_idx_this_round
            #y_delta_avg = FedAvg_dict(self.delta_on_server, user_weight)  # v3

            

            # straggler和正常client对比
            ogd_straggler = None
            ogd_activate = None
            ogd_g = None
            list_parameters_straggler = []
            list_parameters_activate = []
            list_parameters_g = []
            delta_straggler = FedAvg_list(self.y_delta_on_server, self.straggler_idx_this_round) # 5
            delta_activate = FedAvg_list(self.y_delta_on_server, [i for i in range(self.args.num_users) if i not in self.straggler_idx_this_round]) # 5
            delta_g = FedAvg_list(self.y_delta_on_server, [i for i in range(self.args.num_users)]) # 5

            list_parameters_straggler.append([delta_straggler[n] for n in self.name_requires_grad])# straggler平均的基向量
            list_parameters_activate.append([delta_activate[n] for n in self.name_requires_grad])# activate平均的基向量
            list_parameters_g.append([delta_g[n] for n in self.name_requires_grad])# g平均的基向量
            
            list_vector_straggler = []
            list_vector_activate = []
            # list_vector_all = []
            for i in range(len(list_parameters_straggler)):
                list_vector_straggler.append(my_parameters_to_vector(list_parameters_straggler[i]))
            G_s = torch.stack(list_vector_straggler).T
            ogd_straggler = orthonormalize(G_s,True)
            for i in range(len(list_parameters_activate)):
                list_vector_activate.append(my_parameters_to_vector(list_parameters_activate[i]))
            G_a = torch.stack(list_vector_activate).T
            ogd_activate = orthonormalize(G_a,True)       

            # for i in range(len(list_parameters_all)):
            #     list_vector_all.append(my_parameters_to_vector(list_parameters_all[i]))
            # G = torch.stack(list_vector_all).T
            # ogd_all = orthonormalize(G,True)
            
            # _, angle_s = project_vec(G.T,G_s)
            # _, angle_a = project_vec(G.T,G_a)
            # print("angle_s: ", angle_s.item())
            # print("angle_a: ", angle_a.item())
            _, dots, angle = project_vec(G_s.T,G_a)
            print("angle: ", dots.item())
            
            # 打印从延迟开始到延迟结束之间各client所训练的delta之差
            # if len(self.straggler_idx_this_round)==0:
            #     big_deltas = []
            #     for w in self.w_locals:
            #         big_delta = []
            #         for k in w.keys():
            #             if k in self.name_requires_grad:
            #                 big_delta.append(w[k] - self.old_weight[k])
            #         big_deltas.append(my_parameters_to_vector(big_delta))
            #     c,n = cosine_norm(torch.stack(big_deltas).T)
            #     print(c, n)
            # 打印保存在server上的y_delta的cos和norm
            # list_all = []    
            # for i in range(self.args.num_users):
            #     list_all.append([self.y_delta_on_server[i][n] for n in self.name_requires_grad])
            # matrix_all = []
            # for i in range(len(list_all)):
            #     matrix_all.append(my_parameters_to_vector(list_all[i]))
            # G_A = torch.stack(matrix_all).T
            # cosine, norm = cosine_norm(G_A)
            # coslist = cosine.tolist()
            # list_cos = []
            # print(norm)
            # for i,e in enumerate(coslist):
            #     l_c =[]
            #     for j,f in enumerate(e):
            #         if i < (self.args.num_users//2) and j > (self.args.num_users//2-1):
            #             l_c.append(f)
            #     if i<(self.args.num_users//2): list_cos.append(l_c)
            # cosine = torch.tensor(list_cos)
            # print(cosine)
            # print(">0",(cosine>0).sum().item())
            # print("<0",(cosine<0).sum().item())


            #delta_avg = FedAvg_dict(self.y_delta_on_server, user_weight)  # v3
            # server 聚合
            global_param = self.net.state_dict()
            for k in y_delta_avg.keys():
                if 'weight' in k or 'bias' in k:
                    global_param[k] = torch.sub(global_param[k], torch.mul((y_delta_avg[k]), self.args.g_lr))
                else:
                    global_param[k] = y_delta_avg[k]
            self.net.load_state_dict(global_param)   # 聚合模型得到全局模型    # 这个方法是原本没有加datasize权重的FedAvg
            if len(self.straggler_idx_this_round)>0: self.old_weight = copy.deepcopy(self.net).state_dict()
            # client训练
            # self.train(dict_users, dataset_train, ogd_straggler=ogd_straggler, ogd_activate=ogd_activate, epoch=round)
            self.train(dict_users, dataset_train, ogd_straggler=ogd_g, ogd_activate=ogd_g, epoch=round)
            # server测试
            self.test(round, dataset_test)
            
        return self.acc_list, self.loss_list
    
    def run_active(self, dict_users, dataset_train, dataset_test, multilr=False):
        for round in range(0, self.args.epochs):
            self.user_idx_this_round, circle_per_users = self.user_join(round)
            if len(self.user_idx_this_round) > 0:
                print("user_idx_this_round: ", self.user_idx_this_round)
            # client训练
            self.train(dict_users, dataset_train)
            # server接收
            for i in self.user_idx_this_round:
                self.y_delta_on_server[i] = copy.deepcopy(self.y_deltas[i])
            # server 聚合
            delta_avg = FedAvg_list(self.y_delta_on_server, self.user_idx_this_round)
            #delta_avg = FedAvg_dict(self.y_delta_on_server, user_weight)  # v3
            global_param = self.net.state_dict()
            for k in delta_avg.keys():
                global_param[k] = torch.sub(global_param[k], torch.mul((delta_avg[k]), self.args.lr))
            self.net.load_state_dict(global_param)   # 聚合模型得到全局模型    # 这个方法是原本没有加datasize权重的FedAvg
            # server测试
            acc_test, loss_test, confidence, confidence_loss = test_img(self.net, dataset_test, self.args)
            self.acc_list.append(acc_test)
            self.loss_list.append(loss_test)
            print('Round {:3d}, Average loss {:.3f}, Global acc: {:.3f}, Confidence: {}, Confidence loss: {}, valid {:3d}'
                .format(round, loss_test, acc_test, confidence, confidence_loss, len(self.user_idx_this_round)))
            
        return self.acc_list, self.loss_list 
            

def FedAVG_(args, global_net, dict_users, dataset_train, dataset_test):
    global_net.train()  # 设置为训练模式
    ###################################################before train###################################################
    # start time
    # time_start = time.time()
    # copy global weights
    y_glob = global_net.state_dict()

    # 这几个最后保存为txt
    loss_list  = []  # 损失
    acc_list = []  # 准确率
    last_loss = 0.0  # 最后的平均损失
    last_acc = 0.0  # 最后全局参数准确率
    loss_locals = [10.0 for i in range(args.num_users)]
    w_locals = [copy.deepcopy(y_glob) for i in range(args.num_users)]#每个client保存的本地梯度
    user_idx_each_round = []
    user_num_each_round = 1
    step = 1
    

    # stragglers固定选为某几个
    # quilk_user_idx = np.random.choice(range(args.num_users), int(args.num_users * (1.0-args.frac_straggler)), replace=False).tolist()
    # print("quilky_user_idx:\n", quilk_user_idx)
    # for x in all_user_idx:
    #     if x not in quilk_user_idx:
    #         straggler_idx.append(x)
    # print("straggler_idx:\n", straggler_idx)

    # if args.random_latency == 1:
    #     #staleness随机的
    #     slice_list = [int(len(straggler_idx)/args.staleness*i) for i in range(1,args.staleness)]  #slice_list表示切的位置 
    #     slice_list.insert(0, 0)
    #     slice_list.append(len(straggler_idx))
    #     staleness_group_idx = [straggler_idx[slice_list[i]: slice_list[i + 1]] for i in range(len(slice_list) - 1)]
    #     print(staleness_group_idx)

    ###################################################training###################################################
    for round in range(0, args.epochs):
        user_weight = {}
        user_idx_this_round = np.random.choice(range(args.num_users), int(args.num_users * args.frac), replace=False).tolist()
        if args.fake_straggler != None:
            # v1，从user_idx_each_round中随机选取user_num_each_round个，每轮是独立试验
            # user_idx_this_round = np.random.choice(user_idx_each_round, user_num_each_round, replace=False).tolist()
            # v2，从user_idx_each_round中随机选取user_num_each_round个，但选取之后不放回，是非独立试验
            if len(user_idx_each_round) == 0: # user_idx_each_round空了就重新创建
                fake_straggler = json.loads(args.fake_straggler)
                user_idx_each_round, step = fake_select(args.num_users, fake_straggler)
                user_num_each_round = math.ceil(len(user_idx_each_round)//step)
            user_idx_this_round = np.random.choice(user_idx_each_round, user_num_each_round, replace=False).tolist()
            for item in user_idx_this_round: #选中了就删除
                if item in user_idx_each_round:
                    user_idx_each_round.remove(item)
            print(user_idx_each_round,'num',user_num_each_round)
            # v3，在v1或v2的基础上，取消同一client上的重复训练，改为聚合时增加权重
            # for item in user_idx_this_round:
            #     if item in user_weight:
            #         user_weight[item] += 1
            #     else:
            #         user_weight[item] = 1
        elif args.straggler != None:
            straggler = json.loads(args.straggler)
            user_idx_this_round, circle_per_users = select_client(round, args.num_users, straggler)

        if len(user_idx_this_round) > 0:
            print("user_idx_this_round: ", user_idx_this_round)
            # copy weight to net_glob
            w_glob = FedAvg_list(w_locals, user_idx_this_round)  # v1\v2
            # w_glob = FedAvg_dict(w_locals, user_weight)  # v3
            global_net.load_state_dict(w_glob)   # 聚合模型得到全局模型    # 这个方法是原本没有加datasize权重的FedAvg
            # Local Training start
            local_ep = args.local_ep
            for idx in user_idx_this_round:     # 遍历所有的设备
                # 第idx个client进行本地训练
                # args.local_ep = local_ep*user_weight[idx]
                local = FedogdUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                weight, loss, y_delta = local.train(net=copy.deepcopy(global_net).to(args.device))
                w_locals[idx] = weight
                loss_locals[idx] = loss
            
            # print trian loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            # test
            acc_test, loss_test, confidence, confidence_loss = test_img(global_net, dataset_test, args)
            acc_list.append(acc_test)
            loss_list.append(loss_test)
            last_loss = loss_avg
            last_acc = acc_test
            print('Round {:3d}, Average loss {:.3f}, Global acc: {:.3f}, Confidence: {}, Confidence loss: {}, valid {:3d}'
                .format(round, loss_avg, acc_test, confidence, len(user_idx_this_round)))
        else:
            print('Round {:3d}, Average loss {:.3f}, Global acc: {:.3f}, Confidence: {} 0 !'
                .format(round, last_loss, last_acc))
            loss_list.append(last_loss)
            acc_list.append(last_acc)
    
    return acc_list, loss_list 