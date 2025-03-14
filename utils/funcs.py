import os
import pickle
from collections import OrderedDict

import torch
from sklearn.utils import shuffle as shuffle_func

def func_on_params(func, *params_args):
    """在模型参数上计算公式

    Args:
        func: 公式函数，参数顺序要和params_args中的保持一致
        params_args: 输入公式的模型参数字典，OrderedDict或Dict类型

    Returns:
        OrderedDict类型的公式计算结果
    """
    res = OrderedDict()
    for name in params_args[0].keys():
        weight = func(*[params[name] for params in params_args])
        res[name] = weight.detach()
    return res

def get_parameters(params_model, deepcopy=True):
    ans = OrderedDict()
    for name, params in params_model.items():
        if deepcopy:
            if 'weight' in name or 'bias' in name:
                params = params.clone().detach()
                ans[name] = params
    return ans

def get_buffers(params_model, deepcopy=True):
    ans = OrderedDict()
    for name, buffers in params_model.items():
        if deepcopy:
            if 'weight' in name or 'bias' in name:
                continue
            buffers = buffers.clone().detach()
            ans[name] = buffers
    return ans

def get_grad(model, weight_decay=0.0):
    grad = OrderedDict()
    for name, weight in model.named_parameters():
        _g = weight.grad.detach()
        if 'bias' not in name: 
            _g += (weight * weight_decay).detach()
        grad[name] = _g
    return grad

def params_zero_like(params):
    ans = OrderedDict()
    for name, weight in params.items():
        if 'weight' in name or 'bias' in name:
            ans[name] = torch.zeros_like(weight).detach()
    return ans

def model_size(model):
    """
    获取模型大小，可以传入模型或模型的state_dict
    """
    if isinstance(model, torch.nn.Module):
        params_iter = model.named_parameters()
    elif isinstance(model, dict):
        params_iter = model.items()
    else:
        raise Exception(f"unknow type: {type(model)}, expected is torch.nn.Module or dict")
    res = 0.0
    for _, weight in params_iter:
        res += (weight.element_size() * weight.nelement())
    return res

# 计算余弦相似度
def cos_similarity(gradients, other_gradients):
    similarities = {}
    if isinstance(gradients, torch.Tensor):
        similarity = torch.nn.functional.cosine_similarity(gradients.unsqueeze(0), other_gradients.unsqueeze(0), dim=0)
        return similarity.item()
    elif isinstance(gradients, dict):
        for (layer_name, grad), (other_layer_name, other_grad) in zip(gradients.items(), other_gradients.items()):
            if layer_name != other_layer_name:
                similarity = torch.nn.functional.cosine_similarity(grad.unsqueeze(0), other_grad.unsqueeze(0), dim=0)
                # 存储余弦相似度
                similarities[layer_name] = similarity.item()
    elif isinstance(gradients, list):
        for idx, (grad, other_grad) in enumerate(zip(gradients, other_gradients)):
                similarity = torch.nn.functional.cosine_similarity(grad.unsqueeze(0), other_grad.unsqueeze(0), dim=0)
                # 存储余弦相似度
                similarities[idx] = similarity
    return similarities


def angle_and_norm(list_deltas, name_requires_grad):
    grad_numels = []
    for k, v in list_deltas[0].items():
        if k in name_requires_grad:
            grad_numels.append(v.data.numel())
    G = torch.zeros((sum(grad_numels), len(list_deltas)))
    for i, delta in enumerate(list_deltas):
        stpt = 0
        endpt = 0
        for j, (name, param)  in enumerate(delta.items()):
            if name in name_requires_grad:
                endpt += grad_numels[j]
                G[stpt:endpt, i].data.copy_(param.data.view(-1))
                stpt = endpt
    norms = []
    for i in range(len(list_deltas)):
        x = G[:,i]
        x_norm = torch.norm(x, p=2) # type: ignore L-2范数
        norms.append(x_norm)
    sum_norms = sum(norms)
    norms = [i/sum_norms for i in norms]

    dotprod = torch.zeros((len(list_deltas), len(list_deltas)))
    for u in range(len(list_deltas)):
        dotprod[u] = (torch.mm(G[:, u].unsqueeze(0), G[:, :]))# u与其他梯度的内积
    ones = torch.full_like(dotprod, 1)
    zeros = torch.full_like(dotprod, 0)
    neg_ones = torch.full_like(dotprod, -1)
    return torch.where(dotprod == 0 , zeros, torch.where(dotprod > 0, ones, neg_ones)), norms  # 列表第i个元素记录第i个梯度和其他梯度的内积


def mf(file_path):  # mkdir for file_path
    _dir = os.path.dirname(file_path)
    if(_dir): os.makedirs(_dir, exist_ok=True)
    return file_path

def save_pkl(obj, file_path):
    with open(mf(file_path), "wb") as _f:
        pickle.dump(obj, _f)


def load_pkl(file_path):
    with open(file_path, "rb") as _f:
        return pickle.load(_f)
    
if __name__ == '__main__':
    t_list = [
        {f'{i}': torch.zeros(i,i).fill_(i) for i in range(1,4)},
        {f'{i}': torch.zeros(i,i).fill_(i*(-1)) for i in range(1,4)},
        {f'{i}': torch.zeros(i,i).fill_(i*(2)) for i in range(1,4)},
        ]
    for t in t_list: 
        for i in t.values():
            print(i)
    a, b = angle_and_norm(t_list, ['1','2'])
    print(a, b)