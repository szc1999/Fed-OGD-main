import copy
import math
import torch
class EMA_si():
    def __init__(self, model, num_usr, alpha):
        self.model = model # 初始化时是第一个值，之后保存待平滑的值

        self.alpha = alpha #记忆衰减因子,越大忘得越快
        
        grad_dict = copy.deepcopy(model)
        for key in model.keys():
            grad_dict[key] = model[key] - model[key]
        self.shadow = [grad_dict
         for i in range(num_usr)]  #用于存放最新的参数
        
        self.staleness = [0 for i in range(num_usr)]  # 各client的陈旧度

        self.backup = []  #用于倒退

        self.warm_up = math.log(0.01,math.e)/math.log(1-self.alpha,math.e) + 1

    def update(self,model,idx):
        weight = copy.deepcopy(model)
        for k in weight.keys():  #根据EMA公式更新self.shadow的参数
            assert k in self.shadow[idx]
            k_shadow = copy.deepcopy(self.shadow[idx][k])
            for i in range(self.staleness[idx]):
                k_shadow = (1.0 - self.alpha) * k_shadow  + self.alpha * weight[k] # 一次指数EMA迭代公式
            weight[k] = k_shadow

        return weight 


class EMA_bi():
    def __init__(self, model, num_usr, alpha, beta):
        self.model = model  #初始化时是第一个值，之后保存待平滑的值

        self.alpha = alpha  #记忆衰减因子,越大忘得越快
        self.beta = beta  

        self.shadow = [model for i in range(num_usr)]  #用于存放经过指数平滑的值s
        temp_model = copy.deepcopy(model)
        for k in temp_model.keys():
            temp_model[k] = torch.mul(temp_model[k], 0.0)
        self.tend = [temp_model for i in range(num_usr)]  #用于存放各client趋势t
        self.g_tend = temp_model  #用于存放全局趋势g_t

        self.warm_up = math.log(0.01,math.e)/math.log(self.alpha,math.e) + 1 #0.01可以换成更小的,启动指数移动平均的时刻

        self.staleness = [0 for i in range(num_usr)]  #对应的h
        self.pred = []  # 利用二次EMA,根据staleness进行值预测
    
    def update(self,model,idx):
        weight = copy.deepcopy(model)
        for k in weight.keys():  #根据EMA公式更新self.shadow的参数
            assert k in self.shadow[idx]
            k_shadow = copy.deepcopy(self.shadow[idx][k])
            for i in range(self.staleness[idx]):
                k_shadow = (1.0 - self.alpha) * (k_shadow + self.tend[idx][k]) + self.alpha * weight[k] # 一次指数EMA迭代公式
            weight[k] = k_shadow

        return weight 

    def update(self, model, idx):
        self.model = model # 待更新的参数
        for k in self.model.keys():  #根据EMA公式更新self.shadow的参数
                assert k in self.shadow[idx]
                new_shadow = (1.0 - self.alpha) * (self.shadow[idx][k] + self.tend[idx][k]) + self.alpha * self.model[k]  # 二次指数EMA迭代公式
                new_tend = (1.0 - self.beta) * self.tend[idx][k] + self.beta * (new_shadow - self.shadow[idx][k])
                self.tend[idx][k] = new_tend.clone() 
    
    def update_g(self, model, idx):
        self.model = model # 待更新的参数
        for k in self.model.keys():  #根据EMA公式更新self.shadow的参数
                assert k in self.shadow[idx]
                new_shadow = (1.0 - self.alpha) * (self.shadow[idx][k] + self.g_tend[k]) + self.alpha * self.model[k]  # 二次指数EMA迭代公式
                self.shadow[idx][k] = new_shadow.clone()

    def predict(self,idx):
        for k in self.model.keys(): 
            new_predict = self.shadow[idx][k] + self.staleness * self.tend[idx][k]
            self.pred[idx].update({k : new_predict})


class EMA_th():
    def __init__(self, model, num_usr, alpha, beta, gamma, circle):
        self.model = model

        self.alpha = alpha  #记忆衰减因子,越大忘得越快
        self.beta = beta  
        self.gamma = gamma

        self.shadow = [model for i in range(num_usr)]  #用于存放经过指数平滑的值s
        temp_model = [torch.mul(copy.deepcopy(model[k]), 0.0) for k in model.keys()]
        self.tend = [temp_model for i in range(num_usr)]  #用于存放趋势t
        self.peak = {} #用于存储季节分量p

        self.warm_up = math.log(0.01,math.e)/math.log(self.alpha,math.e) + 1 #0.01可以换成更小的,启动指数移动平均的时刻

        self.staleness = [0 for i in range(num_usr)]
        self.h=0#对应的h
        self.circle = circle  #周期/季节长度k
        self.pred = {}  # 利用二次EMA,根据staleness进行值预测

    def update(self,model):
        self.model = model # 待平滑的参数
        for k, param in self.model.kd_parameters():  #根据EMA公式更新self.shadow的参数
            if param.requires_grad:
                assert k in self.shadow
                assert k in self.tend
                assert k in self.peak
                new_shadow = (1.0 - self.alpha) * (self.shadow[k] + self.tend[k]) + self.alpha * (param.data - self.peak[k]) # 二次指数EMA迭代公式
                new_tend = (1.0 - self.beta) * self.tend[k] + self.beta * (new_shadow - self.shadow[k])
                new_peak = (1.0 - self.gamma) * self.peak[k] + self.gamma * (param.data - self.shadow[k])
                self.shadow[k] = new_shadow.clone()
                self.tend[k] = new_tend.clone()
                self.peak[k] = new_peak.clone()

    def predict(self,peak_ikh):
        for k, param in peak_ikh.kd_parameters(): 
            new_predict = self.shadow[k] + self.h * self.tend[k] + param.data
            self.pred.update({k : new_predict})
