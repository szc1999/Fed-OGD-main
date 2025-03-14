import torch
from torch.nn.utils.convert_parameters import _check_param_device, parameters_to_vector, vector_to_parameters
import torch.nn as nn


def orthonormalize(vectors, normalize=True, start_idx=0, end_idx=0 ): # 施密特正交化vectors(dim, num)
    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension' # 因为正交基向量的个数不会超过维度
    # TODO : Check if start_idx is correct :)
    # orthonormalized_vectors = torch.zeros_like(vectors)
    if normalize and 0<torch.norm(vectors[:, 0], p=2):  # 第一个向量归一化到单位向量
        vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)
    else:
        vectors[:, 0] = vectors[:, 0]
    
    if start_idx == 0 :
        start_idx = 1
    for i in range(start_idx, vectors.size(1)):# 将后面的向量投影到第一个向量上
        vector = vectors[:, i] # vector(dim, 1)
        V = vectors[:, :i] # 第i个vector之前已经正交化了的正交基们
        PV_vector = torch.mv(V, torch.mv(V.t(), vector)) #计算了vector 在前i个正交基构成的子空间上的投影；vector先是与正交基逐个点乘得长度，再乘以单位化的正交集得投影
        # vector减去在正交基上的投影，就能得到垂直于正交基的一个新的基向量
        if normalize and 0<torch.norm(vector - PV_vector, p=2):
            vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)
        else:
            vectors[:, i] = (vector - PV_vector)

    return vectors


def project_vec(vec, proj_basis):# 将一个向量 vec (x1,y) 投影到给定基 proj_basis (y,x2)，这个proj_basis是单位向量
    if proj_basis.shape[1] > 0:  # 如果proj_basis是个列向量，就可以进行矩阵乘法
        dots = torch.matmul(vec, proj_basis)  # (x1,y)·(y,x2)=(x1,x2)，表示vec投影在proj_basis上的长度,proj_basis是单位向量所以不用除以|proj_basis|
        out = torch.matmul(proj_basis, dots)

        angle = torch.div(torch.div(dots, torch.norm(vec, p=2).data),torch.norm(proj_basis, p=2).data)

        # ones = torch.full_like(dots, 1)
        # zeros = torch.full_like(dots, 0)
        # neg_ones = torch.full_like(dots, -1)
        # angle = torch.where(dots == 0 , zeros, torch.where(dots > 0, ones, neg_ones))  # 列表第i个元素记录第i个梯度和其他梯度的内积
        # TODO : Check !!!!
        # out = torch.matmul(proj_basis, dots.T) # (y,x2)·(x2,x1)=(y,x), 长度乘以单位向量，就是投影的向量，要dots.T就是为了防止出现x1≠x2的情况
        return out, dots, angle
    else:
        return torch.zeros_like(vec), torch.zeros_like(vec), torch.zeros_like(vec)


def cosine_norm(vectors): # 施密特正交化vectors(dim, num)
    assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension' # 因为正交基向量的个数不会超过维度
    list_norm = []
    for i in range(vectors.size(1)):
        list_norm.append(torch.norm(vectors[:, i], p=2).data)
    
    list_cosine = []
    for i in range(vectors.size(1)):# 将后面的向量投影到第一个向量上
        cosine_i = []
        for j in range(vectors.size(1)):# 将后面的向量投影到第一个向量上
            vector_i = vectors[:, i] # vector(dim, 1)
            vector_j = vectors[:, j] # vector(dim, 1)
            PV = torch.div(torch.mv(vector_i.unsqueeze(1).t(), vector_j),torch.mul(list_norm[i],list_norm[j]))
            cosine_i.append(PV.data)
        list_cosine.append(cosine_i)

    return torch.tensor(list_cosine), torch.tensor(list_norm)

def parameters_to_grad_vector(parameters):  # 将参数的梯度展平成（1，n）的向量，并进行拼接成（1，n × num_layer）
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        # param_device = _check_param_device(param, param_device)
        vec.append(param.grad.view(-1))
    return torch.cat(vec)


def grad_vector_to_parameters(vec, parameters):  # 将梯度（1，n × num_layer）放回参数的梯度中去
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        # param.data = vec[pointer:pointer + num_param].view_as(param).data
        param.grad = vec[pointer:pointer + num_param].view_as(param).clone()

        # Increment the pointer
        pointer += num_param

def my_parameters_to_vector(parameters):  # 将参数的梯度展平成（1，n）的向量，并进行拼接成（1，n × num_layer）
    # Flag for the device where the parameter is located
    param_device = None

    vec = []
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)
        vec.append(param.view(-1))
    return torch.cat(vec)

def my_vector_to_parameters(vec, parameters):  # 将梯度（1，n × num_layer）放回参数的梯度中去
    # Ensure vec of type Tensor
    if not isinstance(vec, torch.Tensor):
        raise TypeError('expected torch.Tensor, but got: {}'
                        .format(torch.typename(vec)))
    # Flag for the device where the parameter is located
    param_device = None

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # Ensure the parameters are located in the same device
        param_device = _check_param_device(param, param_device)

        # The length of the parameter
        num_param = param.numel()
        # Slice the vector, reshape it, and replace the old data of the parameter
        # param.data = vec[pointer:pointer + num_param].view_as(param).data
        param = vec[pointer:pointer + num_param].view_as(param).clone()

        # Increment the pointer
        pointer += num_param

    return parameters

if __name__ == '__main__':
    t_list = [
        [i*(1.) for i in range(1,4)],
        [i*(-1.) for i in range(1,4)],
        [i*(2.) for i in range(1,4)],
        ]
    t = torch.tensor(t_list).t()
    print(t)
    a, b = cosine_norm(t)
    print(a, b)