import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parameters_to_vector
import skopt
from skopt import Optimizer
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
import copy
import math
import random
import more_itertools
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import scale
from torch.autograd import Variable
import time
from itertools import combinations
import argparse
import os
from torch.utils.data import Dataset
from matplotlib import rcParams
import multiprocessing as mp
import json
from collections import OrderedDict
from scipy.optimize import linprog
import matplotlib.ticker as ticker

def generate_matrix(m, n, mean=0, std_dev=1):
    # 生成一个 m x n 的矩阵，元素服从均值为 mean、标准差为 std_dev 的正态分布
    # measurement_matrix = np.random.normal(loc=mean, scale=std_dev, size=(n, n))
    # u, s, vh = np.linalg.svd(measurement_matrix)
    measurement_matrix = np.load('saved_array.npy')
    Phi = measurement_matrix[:int(m),]
    return (torch.tensor(Phi)).float()

def dict_to_vector(ordered_dict):
    """
    将有序字典的所有value转换为一个一维向量，并返回该向量及索引信息。
    
    Args:
        ordered_dict (OrderedDict): 包含键值对的有序字典，字典的值是1维或多维的numpy数组。
    
    Returns:
        tuple: 包含一维向量和索引信息的元组。索引信息为一个字典，键是有序字典的键，值是该键对应数据在向量中的位置范围。
    """
    vector = []
    index_map = {}
    start_idx = 0

    for key, value in ordered_dict.items():
        flattened_value = value.flatten()
        end_idx = start_idx + len(flattened_value)
        vector.extend(flattened_value)
        index_map[key] = (start_idx, end_idx)
        start_idx = end_idx

    return torch.tensor(vector), index_map
    
def vector_to_dict(vector, index_map, original_dict):
    """
    根据索引信息将一维向量还原为原始有序字典的形状。
    
    Args:
        vector (np.array): 从原始有序字典得到的一维向量。
        index_map (dict): 键是有序字典的键，值为该键对应数据在向量中的位置范围。
        original_dict (OrderedDict): 原始的有序字典，用于获取各键值的形状信息。
    
    Returns:
        OrderedDict: 还原后的有序字典，保持了原始字典的键顺序和各键值的形状。
    """
    restored_dict = OrderedDict()
    
    for key, (start, end) in index_map.items():
        shape = original_dict[key].shape
        restored_dict[key] = vector[start:end].reshape(shape)
    
    return restored_dict

def decompress_(args, matrix_S, compressed_signal, index_map, original_dict):
    mat_dct_1d=torch.zeros((matrix_S.shape[1],matrix_S.shape[1])).to(matrix_S.device)
    v=torch.tensor(range(matrix_S.shape[1]))
    for k in range(0,matrix_S.shape[1]):  
        dct_1d=torch.cos(v*k*math.pi/matrix_S.shape[1])
        if k>0:
            dct_1d=dct_1d-torch.mean(dct_1d)
        mat_dct_1d[:,k]=dct_1d/torch.norm(dct_1d)
    
    #IHT算法函数
    def cs_IHT(y,D,k, tol):    
        # K=math.floor(y.shape[0])  #稀疏度  
        K=100  
        result_temp=torch.zeros((D.shape[1])).to(D.device)  #初始化重建信号   
        u=0.3 #影响因子
        result=result_temp
        for j in range(K):  #迭代次数
            x_increase=torch.matmul(D.T,(y-torch.sign(torch.matmul(D,result_temp))))    #x=D*(y-D*y0)
            # print('IHT第',j,'轮差值的各个元素和',sum(x_increase))
            # print('整个increase的向量',x_increase)
            result=result_temp+x_increase*u #   x(t+1)=x(t)+D*(y-D*y0)
            result_thresh = torch.zeros((D.shape[1])).to(D.device)
            # print('IHT第',j,'轮result',result)
            abs_x = torch.abs(result)
            indices = torch.argsort(abs_x, descending=True)[:k]  # 选择最大的k个绝对值元素
            result_thresh[indices] = result[indices]  # 创建稀疏向量
            if torch.norm(torch.abs(result_thresh - result_temp)) < tol:
                print(j,'已经收敛')
                break
            result_temp=result_thresh
                   
        return  result
    
    #重建
    # sparse_rec_1d=torch.zeros((matrix_S.shape[1],1)).to(mat_dct_1d.device)   # 初始化稀疏系数矩阵    
    # Theta_1d=torch.matmul(matrix_S,mat_dct_1d)   #测量矩阵乘上基矩阵
    # Theta_1d = matrix_S

    column_rec=cs_IHT(compressed_signal,matrix_S,10000,1e-5)  #利用IHT算法计算稀疏系数
    # sparse_rec_1d[:,0]=column_rec;        
    
    # img_rec=torch.matmul(mat_dct_1d,sparse_rec_1d)          #稀疏系数乘上基矩阵
    # img_rec = sparse_rec_1d
    
    # img_rec = img_rec.float().view(-1)

    if args.if_quantize:
        column_rec = torch.sign(column_rec)

    # 从一维向量还原为有序字典
    restored_dict = vector_to_dict(column_rec, index_map, original_dict)

    return restored_dict

def decompress(args, matrix_S, compressed_signal, index_map, original_dict):
    
    def iht(y, Phi, k, max_iter, tol):
        x = torch.zeros((Phi.shape[1])).to(Phi.device)  # 初始化估计向量
        for i in range(max_iter):
            x_full = Phi.T @ y  # 计算完整图像估计
            x_thresh = torch.zeros((Phi.shape[1])).to(Phi.device)  # 初始化稀疏向量
            abs_x = torch.abs(x_full)
            indices = torch.argsort(abs_x, descending=True)[:k]  # 选择最大的k个绝对值元素
            x_thresh[indices] = x_full[indices]  # 创建稀疏向量
            
            # 检查收敛条件（基于估计向量的变化量）
            if torch.norm(x - x_thresh) < tol:
                # print(i,'已经收敛')
                break
            x = x_thresh.clone()

                # 重塑为原始图像的形状
        return x
    
    column_rec=iht(compressed_signal, matrix_S, k = int(args.V*(1-args.prune_rate_max)/2), max_iter = 100, tol = 1e-5)
    if args.if_quantize:
        column_rec = torch.sign(column_rec)
    restored_dict = vector_to_dict(column_rec, index_map, original_dict)

    return restored_dict

def compress(args, ordered_dict, matrix_S):
    signal_flatten, index_map = dict_to_vector(ordered_dict)
    if args.if_topk:
        signal_flatten = topk_sparse(signal_flatten, int(args.V*args.topk))
    compressed_signal = torch.matmul(matrix_S.to(signal_flatten.device), signal_flatten)
    
    return compressed_signal, index_map

def topk_sparse(arr, k):
    # 获取Top-k元素的索引
    values, indices = torch.topk(arr, k)
    
    # 创建一个与原数组相同大小的全零Tensor
    result = torch.zeros_like(arr)
    
    # 将Top-k元素赋值到对应位置
    result[indices] = values
    
    return result

def calculate_T(args,power,S,prune_rates, alpha):
    data_rates = np.array([data_rate(p_u=p_u, B_u=args.B_u, h_u=args.h_us[index], N0 = args.N0, I_u=args.I_us[index]) for index,p_u in enumerate(power)])
    if args.pattern == 'PROPOSED' or args.pattern == 'baseline4' or args.pattern == 'baseline5':
        bit_totals = S*2 
    elif args.pattern == 'FEDAVG' or args.pattern == 'FEDSGD':
        bit_totals = 32*args.V
    elif args.pattern == 'SIGNSGD':
        bit_totals = 2*args.V
    
    if args.if_prune:
        T_1 = np.array(args.N_us)*args.local_ep/args.scale*args.c0*(1-np.array(prune_rates))/np.array(args.computing_resources)
    else:
        T_1 = np.array(args.N_us)*args.local_ep/args.scale*args.c0/np.array(args.computing_resources)
    
    T_2 = bit_totals/data_rates
    return ((T_1+T_2+args.s)*np.array(alpha)).max()

def calculate_E(args,power,S,prune_rates):
    data_rates = np.array([data_rate(p_u=p_u, B_u=args.B_u, h_u=args.h_us[index], N0 = args.N0, I_u=args.I_us[index]) for index,p_u in enumerate(power)])
    if args.pattern == 'PROPOSED' or args.pattern == 'baseline4' or args.pattern == 'baseline5':
        bit_totals = S*2 
    elif args.pattern == 'FEDAVG' or args.pattern == 'FEDSGD':
        bit_totals = 32*args.V
    elif args.pattern == 'SIGNSGD':
        bit_totals = 2*args.V

    if args.if_prune:
        T_1 = np.array(args.N_us)*args.local_ep/args.scale*args.c0*(1-np.array(prune_rates))/np.array(args.computing_resources)
    else:
        T_1 = np.array(args.N_us)*args.local_ep/args.scale*args.c0/np.array(args.computing_resources)
    
    T_2 = bit_totals/data_rates
    E_1 = args.k*np.array(args.computing_resources)**args.sigma*T_1
    E_2 = np.array(power)*T_2
    return sum(E_1+E_2), max(T_1), min(T_1), max(T_2), min(T_2), max(E_1), min(E_1), max(E_2), min(E_2)

def calculate_dataamount(args,S,alpha):
    if args.pattern == 'PROPOSED' or args.pattern == 'baseline4' or args.pattern == 'baseline5':
        bit_totals = S*2 
    elif args.pattern == 'FEDAVG' or args.pattern == 'FEDSGD':
        bit_totals = 32*args.V
    elif args.pattern == 'SIGNSGD':
        bit_totals = args.V
    dataamount = sum(bit_totals*np.array(alpha))
    return dataamount

def getOmegaE(args, index_set, t, transmit_power, prune_rate, S, data_rates, error_rates):
    # data_rates = np.array([data_rate(p_u=p_u, B_u=args.B_u, h_u=args.h_us[index], N0 = args.N0, I_u=args.I_us[index]) for index,p_u in enumerate(transmit_power)])
    # error_rates = [error_rate(p_u=p_u, B_u=args.B_u, h_u=args.h_us[index], N0 = args.N0, I_u=args.I_us[index], thre=args.waterfall_thre) for index, p_u in enumerate(transmit_power)]
    K = 0
    for u in range(args.num_clients):
        K += args.N_us[u]
    
    
    total_sum = 0
    # Iterate over all possible sizes of U_1

    for U_1_indices in combinations(index_set, t+1):
        U_1 = set(U_1_indices)
        U_2 = index_set - U_1

        # Compute the products
        prod_q_U1 = np.prod([1 - error_rates[u] for u in U_1])
        prod_q_U2 = np.prod([error_rates[u] for u in U_2])

        sum_K_U1 = np.sum([args.N_us[u] for u in U_1])

        term = (1/K-prod_q_U1 * prod_q_U2 * (1 / sum_K_U1))**2

        total_sum += term
    

    return total_sum

def calculate_rates(args, transmit_power):
    error_rates = []
    data_rates = []
    for index, p_u in enumerate(transmit_power):
        h_u = args.h_us[index]
        I_u = args.I_us[index]
        error_rate_val = error_rate(p_u=p_u, B_u=args.B_u, h_u=h_u, N0=args.N0, I_u=I_u, thre=args.waterfall_thre)
        data_rate_val = data_rate(p_u=p_u, B_u=args.B_u, h_u=h_u, N0=args.N0, I_u=I_u)
        error_rates.append(error_rate_val)
        data_rates.append(data_rate_val)
    return error_rates, np.array(data_rates)

def H_BO_rho(args, prune_rate, transmit_power, S):
    for i in range(args.num_clients):
        globals()['r'+str(i)] = prune_rate[i]
    error_rates, data_rates = calculate_rates(args, transmit_power)
    index_set = set(range(args.num_clients))
    results = []
    # with mp.Pool() as pool:
    #     for t in range(args.num_clients):
    #         results.append(pool.apply_async(getOmegaE, (args, index_set, t, transmit_power, prune_rate, S, data_rates, error_rates)))
    #     gammas = [result.get() for result in results]
    for t in range(args.num_clients):
        results.append(getOmegaE(args, index_set, t, transmit_power, prune_rate, S, data_rates, error_rates))
    gammas = [result for result in results]
    
    total_sum = sum(gammas)
    
    sum_q = 0
    for u in range(args.num_clients):
        sum_q += error_rates[u]
    K = sum(args.N_us)

    part1 = (2/args.learning_rate) * (args.F_0 - args.F_1)
    part2 = args.epsilon - 2*args.L**2 * args.D**2 * sum(prune_rate) - 2*args.G**2 / K-(12 * (args.delta + 1) * args.C**2 * args.G**2) / (S * K)-12 * args.C**2 * sum_q-args.L*args.learning_rate* args.G**2/K-12*args.C**2-(12 * (args.delta + 1) * args.C**2 * args.G**2)*sum_q / (S * K)
    part3 = 2*K*args.G**2*total_sum
    Omega = part1 / (part2 - part3)

    E_t = 0
    for u in range(args.num_clients):
        E_u = args.k * (args.computing_resources[u]) ** (args.sigma-1) * args.N_us[u] * args.c0 * (1 -prune_rate[u]) + transmit_power[u] * 2 * S / data_rates[u]
        E_t += E_u
    
    OmegaE = Omega*E_t
    return OmegaE

def H_BO_p(args, transmit_power, prune_rate, S):
    for i in range(args.num_clients):
        globals()['p'+str(i)] = transmit_power[i]
    error_rates = [error_rate(p_u=p_u, B_u=args.B_u, h_u=args.h_us[index], N0 = args.N0, I_u=args.I_us[index], thre=args.waterfall_thre) for index, p_u in enumerate(transmit_power)]
    data_rates = np.array([data_rate(p_u=p_u, B_u=args.B_u, h_u=args.h_us[index], N0 = args.N0, I_u=args.I_us[index]) for index,p_u in enumerate(transmit_power)])
    index_set = {i for i in range(args.num_clients)}
    results = []
    # with mp.Pool() as pool:
    #     for t in range(args.num_clients):
    #         results.append(pool.apply_async(getOmegaE, (args,index_set, t, transmit_power, prune_rate, S, data_rates, error_rates)))
    #     gammas = [result.get() for result in results]
    for t in range(args.num_clients):
        results.append(getOmegaE(args, index_set, t, transmit_power, prune_rate, S, data_rates, error_rates))
    gammas = [result for result in results]
    
    total_sum = sum(gammas)

    sum_q = 0
    for u in range(args.num_clients):
        sum_q += error_rates[u]
    K = sum(args.N_us)

    part1 = (2/args.learning_rate) * (args.F_0 - args.F_1)
    part2 = args.epsilon - 2*args.L**2 * args.D**2 * sum(prune_rate) - 2*args.G**2 / K-(12 * (args.delta + 1) * args.C**2 * args.G**2) / (S * K)-12 * args.C**2 * sum_q-args.L*args.learning_rate* args.G**2/K-12*args.C**2-(12 * (args.delta + 1) * args.C**2 * args.G**2)*sum_q / (S * K)
    part3 = 2*K*args.G**2*total_sum
    Omega = part1 / (part2 - part3)

    E_t = 0
    for u in range(args.num_clients):
        E_u = args.k * (args.computing_resources[u]) ** (args.sigma-1) * args.N_us[u] * args.c0 * (1 -prune_rate[u]) + transmit_power[u] * 2 * S / data_rates[u]
        E_t += E_u
    
    OmegaE = Omega*E_t
    return OmegaE
    
def get_S(args, prune_rate, transmit_power, S):
    error_rates, data_rates = calculate_rates(args, transmit_power)
    index_set = set(range(args.num_clients))
    results = []
    # with mp.Pool() as pool:
    #     for t in range(args.num_clients):
    #         results.append(pool.apply_async(getOmegaE, (args, index_set, t, transmit_power, prune_rate, S, data_rates, error_rates)))
    #     gammas = [result.get() for result in results]
    for t in range(args.num_clients):
        results.append(getOmegaE(args, index_set, t, transmit_power, prune_rate, S, data_rates, error_rates))
    gammas = [result for result in results]
    
    total_sum = sum(gammas)

    sum_q = 0
    for u in range(args.num_clients):
        sum_q += error_rates[u]
    K = sum(args.N_us)

    part1 = (2/args.learning_rate) * (args.F_0 - args.F_1)
    part3 = 2*K*args.G**2*total_sum
    
    E_t = 0
    for u in range(args.num_clients):
        E_u = args.k * (args.computing_resources[u]) ** (args.sigma-1) * args.N_us[u] * args.c0 * (1 -prune_rate[u]) + transmit_power[u] * 2 * S / data_rates[u]
        E_t += E_u
    
    # 下面是S的表达式部分，用到了上面的part1,part3
    ps1 = 0
    for u in range(args.num_clients):
        ps1 = args.k * (args.computing_resources[u]) ** (args.sigma-1) * args.N_us[u] * args.c0 * (1 -prune_rate[u])
        ps1 += ps1
        a1 = transmit_power[u] * 2 / data_rates[u]
        a1 += a1
    ps2 = args.epsilon - 2*args.L**2 * args.D**2 * sum(prune_rate) - 2*args.G**2 / K-12 * args.C**2 * sum_q-args.L*args.learning_rate* args.G**2/K-12*args.C**2
    a2 = -(12 * (args.delta + 1) * args.C**2 * args.G**2) / K-(12 * (args.delta + 1) * args.C**2 * args.G**2)*sum_q / K
    # S_max = D_min #D_min是梯度的维数的最小值
    # S_min = k_max*args.num_clients # k_max 剪枝之后所有模型的参数都是一个包含0与非0元素的向量，k_max是非零的参数个数最多的向量的 非零参数的个数
    d = (part1*ps1*(part3-ps2)*a2)/(part1*a1)
    if d>=0:
        S1=(1/(ps2-part3))(math.sqrt((part1*ps1*(part3-ps2)*a2)/part1*a1)-a2)
    # 请注意 H(S)_min = min {H(S_max), H(S_min), H(S1)}
    else:
        S1 = args.S_max/2

    return S1
    
def H_tolal(args, transmit_power, prune_rate, S):
    index_set = {i for i in range(args.num_clients)}
    # total_gamma = 0
    error_rates = [error_rate(p_u=p_u, B_u=args.B_u, h_u=args.h_us[index], N0 = args.N0, I_u=args.I_us[index], thre=args.waterfall_thre) for index, p_u in enumerate(transmit_power)]
    data_rates = np.array([data_rate(p_u=p_u, B_u=args.B_u, h_u=args.h_us[index], N0 = args.N0, I_u=args.I_us[index]) for index,p_u in enumerate(transmit_power)])
    #-----------------------------------开始计算---------------------------------
    results = []
    # with mp.Pool() as pool:
    #     for t in range(args.num_clients):
    #         results.append(pool.apply_async(getOmegaE, (args,index_set, t, transmit_power, prune_rate, S, data_rates, error_rates)))
    #     gammas = [result.get() for result in results]
    for t in range(args.num_clients):
        results.append(getOmegaE(args, index_set, t, transmit_power, prune_rate, S, data_rates, error_rates))
    gammas = [result for result in results]
    
    total_H = sum(gammas)
    return total_H

def data_rate(p_u, B_u, h_u, N0, I_u): 
    # B_u is the allocated bandwidth for u; h_u is the channel gain; N0 is the power spectral density of noise; I_u is the interference
    temp = 1 + p_u*h_u/(I_u + B_u*N0)
    E = math.log(temp,2)
    R = B_u*E
    return R
 
def error_rate(p_u, B_u, h_u, N0, I_u, thre):
    # q^n_u; thre is the waterfall threshold
    temp = -thre*(I_u + B_u*N0)/(p_u*h_u)
    rate = 1 - math.exp(temp)
    return rate

def error_rate_forBO(p_u, B_u, h_u, N0, I_u, thre):
    # q^n_u; thre is the waterfall threshold
    temp = -thre*(I_u + B_u*N0)/(p_u*h_u)
    rate = 1 - np.exp(temp)
    return rate

def generate_alpha(alpha, transmit_power, num_clients, I_us, h_us, B_u=1, N0=1, waterfall_thre=1):
    error_rates = [error_rate(p_u=p_u, B_u=B_u, h_u=h_us[index], N0 = N0, I_u=I_us[index], thre=waterfall_thre) for index, p_u in enumerate(transmit_power)]
    # np.random.seed(0)
    p = list(zip(np.array(error_rates), 1-np.array(error_rates)))
    for i in range(num_clients): 
        alpha[i] = np.random.choice([0,1], p=p[i])


G = []

def Save_to_Csv(data, file_name, Save_format = 'csv', Save_type = 'col', file_path = './'):
    # data
    # 输入为一个字典，格式： { '列名称': 数据,....} 
    # 列名即为CSV中数据对应的列名， 数据为一个列表
    
    # file_name 存储文件的名字
    # Save_format 为存储类型， 默认csv格式， 可改为 excel
    # Save_type 存储类型 默认按列存储， 否则按行存储
    
    # 默认存储在当前路径下
    
    import pandas as pd
    import numpy as np
    
    Name = []
    times = 0
 
    if Save_type == 'col':
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List).reshape(-1,1)
            else:
                Data = np.hstack((Data, np.array(List).reshape(-1,1)))
                
            times += 1
            
        Pd_data = pd.DataFrame(columns=Name, data=Data) 
        
    else:
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List)
            else:
                Data = np.vstack((Data, np.array(List)))
        
            times += 1
    
        Pd_data = pd.DataFrame(index=Name, data=Data)  

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    if Save_format == 'csv':
        Pd_data.to_csv(file_path + file_name +'.csv',encoding='utf-8')
    else:
        Pd_data.to_excel(file_path + file_name +'.xls',encoding='utf-8')

def cal_ref(args, wer,bitwidth_max,resource_max,resource_min,dis_max,dis_min,power_max,power_min,I_max,I_min,N_us,B_u,N0,V,k,c0,s):
    h_min = wer/(dis_max**2)
    h_max = wer/(dis_min**2)
    Tmax_ref = s + max(N_us)/args.scale*c0/resource_min + (bitwidth_max*V+64+V)/data_rate(power_min,B_u,h_min,N0,I_max)
    Emax_ref = k*(resource_max)**2*max(N_us)/args.scale*c0+(bitwidth_max*V+64+V)*power_max/data_rate(power_min,B_u,h_min,N0,I_max)
    Tmin_ref = s + min(N_us)/args.scale*c0/resource_max + (1*V+64+V)/data_rate(power_max,B_u,h_max,N0,I_min)
    Emin_ref = k*(resource_min)**2*min(N_us)/args.scale*c0+(1*V+64+V)*power_min/data_rate(power_max,B_u,h_max,N0,I_min)
    print(f'当前T范围:({Tmin_ref},{Tmax_ref}), 当前E范围:({Emin_ref},{Emax_ref})')

    with open(f'referce.txt', 'w') as f:
        # 写入数组字符串
        f.write(f'当前T范围:({Tmin_ref},{Tmax_ref}), 当前E范围:({Emin_ref},{Emax_ref})')

def adjust(args):
    ''' 
    bitwidths: [delta^n_1,...,delta^n_U],量化比特
    prune_rates: [rho^n_1,...,rho^n_U],剪枝率
    power: [p^n_1,...p^n_U],传输功率
    threshold: 判断块梯度下降是否收敛的阈值
    I_us: [I^n_1,...I^n_U],interference,用于计算误码率
    h_us: [h^n_1,...h^n_U],channel gain,用于计算误码率
    g_maxs: [[gm_1_1,...gm_1_V],...,[gm_U_1,...gm_U_V]],各个client的各个分量的上界列表
    g_mins: 各个client的各个分量的下界列表
    f: [f^n_1,...,f^n_u],computing_resources
    xis: [xi_1,...,xi_U]表达g_max和g_min和符号的bit数
    max_iter: 贝叶斯优化的最大优化次数
    '''
    # N_us = [3601,3601,3601,3601,3602]
    start_ini = time.perf_counter()

    
    # 深拷贝，用于判断是否收敛
    # b = copy.deepcopy(bitwidths)
    # r = copy.deepcopy(prune_rates)
    # p = copy.deepcopy(power)
    # check = Gamma(r,b,p, g_maxs, g_mins, h_us, I_us)
    # S0 = random.randint(args.S_min, args.S_max)
    S0 = args.S_max/2
    prune_rate0 = [args.prune_rate_max for i in range(args.num_clients)]
    transmit_power0 = [random.uniform(args.power_min, args.power_max) for i in range(args.num_clients)]
    # power = [0.1 for i in range(num_clients)]

    best_power = copy.deepcopy(transmit_power0)
    best_S = copy.deepcopy(S0)
    best_prune_rate = copy.deepcopy(prune_rate0)
    best_Gamma = 1e9
    # while abs(Gamma(r,b,p, g_max, g_min, h_us, I_us)-Gamma(prune_rates,bitwidths,power, g_max, g_min, h_us, I_us)) > threshold:
    for time_ in range(args.bcd_epoch):
        # b = copy.deepcopy(bitwidths)
        # r = copy.deepcopy(prune_rates)
        # p = copy.deepcopy(power)
        print('当前块坐标下降轮次：',time_+1)
        start = time.perf_counter()
        
        # update prune_rates by BO
                     
        namespace = ['r'+str(i) for i in range(args.num_clients)]
        space = [Real(args.prune_rate_min,args.prune_rate_max,name=namespace[i]) for i in range(args.num_clients)]
        
        opt = Optimizer(space, base_estimator="GP", random_state=0, acq_func=args.acq_func)
        n_calls = args.BO_epoch
        BO_results = []
        count = 0
        for i in range(n_calls):
            start_BO = time.perf_counter()
            suggested = opt.ask()
            y = H_BO_rho(args, suggested, best_power, best_S)
            count += 1
            BO_results.append(y)
            opt.tell(suggested,y)
            end_BO = time.perf_counter()
            # print(f'第{i}轮BO for rho搜索用时: {end_BO-start_BO}')
        
        
        result = opt.get_result()
        rate = result.x
            
        # update S

        # S_temp = get_H(args, set(range(args.num_clients)), args.num_clients, best_power, best_prune_rate, best_S, data_ratesS, error_ratesS)
        S_temp = get_S(args, best_prune_rate, best_power, best_S)
        
        # update power by BO
        
        namespace = ['p'+str(i) for i in range(args.num_clients)]
        space = [Real(args.power_min,args.power_max,name=namespace[i]) for i in range(args.num_clients)]
        
        opt = Optimizer(space, base_estimator="GP", random_state=0, acq_func=args.acq_func)
        n_calls = args.BO_epoch
        BO_results = []
        count = 0
        for i in range(n_calls):
            start_BO = time.perf_counter()
            suggested = opt.ask()
            y = H_BO_p(args, suggested, best_prune_rate, best_S)
            count += 1
            BO_results.append(y)
            opt.tell(suggested,y)
            end_BO = time.perf_counter()
            # print(f'第{i}轮BO for p搜索用时: {end_BO-start_BO}')
        # print('kexingjie for p:',count)    
        
        result = opt.get_result()
        power = result.x

        # plt.figure(figsize=(12,5))
        # plot_convergence(result)
        # plt.show()
        # plt.figure(figsize=(12,5))
        # plot_objective(result,size=3,dimensions=['p0','p1'])

        # if np.all(np.array(prune_rates)==0) or np.std(np.array(prune_rates))==0:
        #     power = [random.uniform(power_max/2, power_max) for i in range(num_clients)]
        #     print('random')
        #     temp = 0
        #     while np.any(constraint_E_forBO(power)>0) or np.any(constraint_T_forBO(power)>0):
        #         # print('not available')
        #         print(temp)
        #         temp +=1
        #         power = [random.uniform(power_min, power_max) for i in range(num_clients)]
            

        # print(Gamma(r,b,p, g_maxs, g_mins, h_us, I_us))
        temp_g = H_tolal(args=args,prune_rate=rate,S=S_temp,transmit_power=power)
        # output_record.append([power,bitwidths,prune_rates])
        if temp_g <= best_Gamma:
            best_Gamma = temp_g
            best_power = copy.deepcopy(power)
            best_S = copy.deepcopy(S_temp)
            best_prune_rate = copy.deepcopy(rate)

        print(temp_g)
        G.append(temp_g)
        print('本轮优化的power',power,'S',S_temp,'prune_rate',rate)
        print('best_power',best_power,'best_S',best_S,'best_prune_rate',best_prune_rate)
        end = time.perf_counter()
        print(f'本轮优化用时：{end-start}')
        print(f'目前优化总用时：{end-start_ini}')

        # if abs(Gamma(r,b,p, g_maxs, g_mins, h_us, I_us)-Gamma(prune_rates,bitwidths,power, g_maxs, g_mins, h_us, I_us)) < threshold:
        #     break
    print('final: best_power',best_power,'best_S',best_S,'best_prune_rate',best_prune_rate)
    return best_power,best_S,best_prune_rate

def read_converg(file_name):
    df = pd.read_csv(file_name, header=None)

    # 将第一列设置为变量名称
    variable_names = df.iloc[:, 0]
    df = df.iloc[:, 1:]

    # 转置DataFrame
    df_transposed = df.T

    # 给DataFrame设置列名
    df_transposed.columns = variable_names

    # 提取数据
    index_values = df_transposed.index  # 序号
    losses_train = df_transposed['losses_train']  # 对应的losses_train列
    accuracies_train = df_transposed['accuracies_train']  # 对应的acc_train列
    accuracies_test = df_transposed['accuracies_test']  # 对应的acc_train列
    time_consumption = df_transposed['time_consumption']
    energy_consumption = df_transposed['energy_consumption']
    return index_values, losses_train, accuracies_train, accuracies_test, time_consumption, energy_consumption

def read_TE(file_name):
    df = pd.read_csv(file_name, header=None)

        # 将第一列设置为变量名称
    variable_names = df.iloc[:, 0]
    df = df.iloc[:, 1:]

        # 转置DataFrame
    df_transposed = df.T

        # 给DataFrame设置列名
    df_transposed.columns = variable_names

        # 提取数据
    # index_values = df_transposed.index  # 序号
    data_T = df_transposed['T_step']  # 对应的losses_train列
    data_E = df_transposed['E_step']  # 对应的acc_train列

    return data_T, data_E

def read_condition(file_name):
    df = pd.read_csv(file_name, header=None)

    # 将第一列设置为变量名称
    variable_names = df.iloc[:, 0]
    df = df.iloc[:, 1:]

    # 转置DataFrame
    df_transposed = df.T

    # 给DataFrame设置列名
    df_transposed.columns = variable_names

    # 提取数据
    I_us = df_transposed['I_us']  # 对应的losses_train列
    computing_resources = df_transposed['computing_resources']  # 对应的computing_resources列
    distances = df_transposed['distances']  # 对应的distances列

    return list(I_us), list(computing_resources), list(distances)

def plot_single_converg(args, save_path, file_name, if_loss=True):
    # 读取CSV文件
    index_values, losses_train, accuracies_train, accuracies_test, time_consumption, _ = read_converg(file_name)
    font_size = 30
    # rcParams.update({'font.size': font_size, 'font.family': 'Times New Roman'})
    
    # 示例输出
    if if_loss:
        plt.plot(index_values, losses_train, label='Losses Train', marker='o',markevery=args.markevery)
    plt.plot(index_values, accuracies_train, label='Train Accuracy', marker='o',markevery=args.markevery)
    plt.plot(index_values, accuracies_test, label='Test Accuracy', marker='o',markevery=args.markevery)

    plt.xlabel('Index Values')
    plt.ylabel('Values')
    plt.title('Training Metrics Over Time')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path +'convergence.png')

    # plt.show()

def plot_temp(args,save_path, path_1, path_2, path_3):
    index_values_1, losses_train_1, accuracies_train_1, accuracies_test_1 = read_converg(path_1)
    index_values_2, losses_train_2, accuracies_train_2, accuracies_test_2 = read_converg(path_2)
    index_values_3, losses_train_3, accuracies_train_3, accuracies_test_3 = read_converg(path_3)

    plt.figure()
    interval = args.markevery
    index_values_1 = index_values_1[::interval]
    index_values_2 = index_values_2[::interval]
    index_values_3 = index_values_3[::interval]
    accuracies_test_1 = accuracies_test_1[::interval]
    accuracies_test_2 = accuracies_test_2[::interval]
    accuracies_test_3 = accuracies_test_3[::interval]
    # plt.plot(index_values_fedsgd, accuracies_test_fedsgd, label='Accuracy_fedsgd', marker='o',markevery=args.markevery)
    plt.plot(index_values_1, accuracies_test_1, label='Accuracy_1', marker='s',markevery=1)
    plt.plot(index_values_2, accuracies_test_2, label='Accuracy_2', marker='D',markevery=1)
    plt.plot(index_values_3, accuracies_test_3, label='Accuracy_3', marker='*',markevery=1)
    
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training Metrics Over Time')
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path +'convergence_TEMP.png')

def plot_multi_converg(args, save_path, save_name, **kwargs):
    marker_list = ['o','s','D','*','^','p','H','v']
    plt.rcParams['font.family'] = 'times new roman'  # 设置全局字体，例如 'serif', 'sans-serif'
    plt.rcParams['font.size'] = 12         # 设置全局字体大小
    # 绘制精度曲线图
    i = 0
    plt.figure()
    cut = 100
    for key, value in kwargs.items():
        index_values, losses_train, accuracies_train, accuracies_test, _, _ = read_converg(value)
        interval = args.markevery
        index_values = index_values[:cut:interval]
        accuracies_test = accuracies_test[:cut:interval]
        if key == 'baseline1':       
            plt.plot(index_values, accuracies_test, label='FedSGD', marker=marker_list[i],markevery=1)
        elif key == 'baseline2':
            plt.plot(index_values, accuracies_test, label='FedAVG', marker=marker_list[i],markevery=1)
        elif key == 'baseline3':
            plt.plot(index_values, accuracies_test, label='SignSGD', marker=marker_list[i],markevery=1)
        elif key == 'baseline4':
            plt.plot(index_values, accuracies_test, label='FL-1-Bit-Spar-CS', marker=marker_list[i],markevery=1)
        elif key == 'baseline5':
            plt.plot(index_values, accuracies_test, label='FedUEE-Non-Compensation', marker=marker_list[i],markevery=1)
        elif key == 'baseline6':
            plt.plot(index_values, accuracies_test, label='FedUEE', marker=marker_list[i],markevery=1)
        i += 1
    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    # plt.title('Training Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path +'convergence'+save_name+'.pdf', dpi=300, format='pdf')

    # 绘制loss曲线图
    i = 0
    plt.figure()
    for key, value in kwargs.items():
        index_values, losses_train, accuracies_train, accuracies_test, _, _ = read_converg(value)
        interval = args.markevery
        index_values = index_values[:cut:interval]
        losses_train = losses_train[:cut:interval]
        if key == 'baseline1':       
            plt.plot(index_values, losses_train, label='FedSGD', marker=marker_list[i],markevery=1)
        elif key == 'baseline2':
            plt.plot(index_values, losses_train, label='FedAVG', marker=marker_list[i],markevery=1)
        elif key == 'baseline3':
            plt.plot(index_values, losses_train, label='SignSGD', marker=marker_list[i],markevery=1)
        elif key == 'baseline4':
            plt.plot(index_values, losses_train, label='FL-1-Bit-Spar-CS', marker=marker_list[i],markevery=1)
        elif key == 'baseline5':
            plt.plot(index_values, losses_train, label='FedUEE-Non-Compensation', marker=marker_list[i],markevery=1)
        elif key == 'baseline6':
            plt.plot(index_values, losses_train, label='FedUEE', marker=marker_list[i],markevery=1)
        i += 1

    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    # plt.title('Training Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path+'loss'+save_name+'.pdf',dpi=300, format='pdf')
    
    # 绘制时延曲线图
    i = 0
    plt.figure()
    for key, value in kwargs.items():
        index_values, _, _, _, time_consumption, _ = read_converg(value)
        interval = args.markevery
        index_values = index_values[:cut:interval]
        time_consumption = time_consumption[:cut:interval]
        if key == 'baseline1':       
            plt.plot(index_values, time_consumption, label='FedSGD', marker=marker_list[i],markevery=1)
        elif key == 'baseline2':
            plt.plot(index_values, time_consumption, label='FedAVG', marker=marker_list[i],markevery=1)
        elif key == 'baseline3':
            plt.plot(index_values, time_consumption, label='SignSGD', marker=marker_list[i],markevery=1)
        elif key == 'baseline4':
            plt.plot(index_values, time_consumption, label='FL-1-Bit-Spar-CS', marker=marker_list[i],markevery=1)
        elif key == 'baseline5':
            plt.plot(index_values, time_consumption, label='FedUEE-Non-Compensation', marker=marker_list[i],markevery=1)
        elif key == 'baseline6':
            plt.plot(index_values, time_consumption, label='FedUEE', marker=marker_list[i],markevery=1)
        
        i += 1
    
    ax = plt.gca()  # 获取当前的坐标轴对象
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())  # 设置纵轴为标准格式
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # 使用科学计数法

    plt.xlabel('Training Epochs')
    plt.ylabel('Delay')
    # plt.title('Training Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path +'Delay'+save_name+'.pdf', dpi=300, format='pdf')

    # 绘制能耗曲线图
    i = 0
    plt.figure()
    for key, value in kwargs.items():
        index_values, _, _, _, _, energy_consumption = read_converg(value)
        interval = args.markevery
        index_values = index_values[:cut:interval]
        energy_consumption = energy_consumption[:cut:interval]
        if key == 'baseline1':       
            plt.plot(index_values, energy_consumption, label='FedSGD', marker=marker_list[i],markevery=1)
        elif key == 'baseline2':
            plt.plot(index_values, energy_consumption, label='FedAVG', marker=marker_list[i],markevery=1)
        elif key == 'baseline3':
            plt.plot(index_values, energy_consumption, label='SignSGD', marker=marker_list[i],markevery=1)
        elif key == 'baseline4':
            plt.plot(index_values, energy_consumption, label='FL-1-Bit-Spar-CS', marker=marker_list[i],markevery=1)
        elif key == 'baseline5':
            plt.plot(index_values, energy_consumption, label='FedUEE-Non-Compensation', marker=marker_list[i],markevery=1)
        elif key == 'baseline6':
            plt.plot(index_values, energy_consumption, label='FedUEE', marker=marker_list[i],markevery=1)
        
        i += 1
    
    ax = plt.gca()  # 获取当前的坐标轴对象
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())  # 设置纵轴为标准格式
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # 使用科学计数法

    plt.xlabel('Training Epochs')
    plt.ylabel('Energy consumption')
    # plt.title('Training Metrics Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path +'Energy consumption'+save_name+'.pdf', dpi=300, format='pdf')  
# def plot_bar(args, save_path, save_name,  **kwargs):
#     for key, value in kwargs.items():
#         data_T_fedsgd, data_E_fedsgd = read_TE(value)
#         data = [data_T_fedsgd[1],data_T_fedsgd[2],data_T_fedsgd[3]]
#         plt.plot(index_values, accuracies_test, label=f'{key}', marker=marker_list[i],markevery=1)
#         i += 1

#     data_T_fedsgd, data_E_fedsgd = read_TE(file_fedsgd)
#     data_T_signsgd, data_E_signsgd = read_TE(file_signsgd)
#     data_T_fedavg, data_E_fedavg = read_TE(file_fedavg)
#     data_T_proposed, data_E_proposed = read_TE(file_proposed)

#     proposed = [data_T_proposed[1], data_T_proposed[2], data_T_proposed[3]]                  # Proposed 2.0 1.5 0.006
#     fedavg = [data_T_fedavg[1],data_T_fedavg[2],data_T_fedavg[3]]               # SGD 2. 1.5
#     signsgd = [data_T_signsgd[1],data_T_signsgd[2],data_T_signsgd[3]]                # signSGD
#     fedsgd = [data_T_fedsgd[1],data_T_fedsgd[2],data_T_fedsgd[3]]             # avg
#     font_size = 20
#     labels = ['0.6', '0.7', '0.8']
#     bar_width = 0.15
#     # rcParams.update({'font.size': font_size, 'font.family': 'Times New Roman'})

#     maxlim = max([max(proposed),max(signsgd),max(fedsgd),max(fedavg)])

#     # 绘图
#     plt.figure(figsize=(15, 12))
#     plt.bar(np.arange(3), proposed, label='LTFL (Proposed)', color='royalblue', alpha=1, width=bar_width, edgecolor="k", hatch='/')
#     # plt.bar(np.arange(3) + 3 *bar_width, fedsgd, label=u'FedSGD', color='orange', alpha=1, edgecolor="k",
#     #         width=bar_width, hatch="o")
#     plt.bar(np.arange(3) + 1 * bar_width, signsgd, label=u'SignSGD', color='limegreen', alpha=1, edgecolor="k",
#             width=bar_width, hatch="x")
#     plt.bar(np.arange(3) + 2 * bar_width, fedavg, label=u'FedAVG', color='saddlebrown', alpha=1, edgecolor="k",
#             width=bar_width, hatch="/")
#     # 添加刻度标签
#     plt.xticks(np.arange(3) + 2*bar_width, labels)
#     # plt.tick_params(labelsize=20)
#     # 设置Y轴的刻度范围
#     plt.ylim([0, math.ceil(maxlim/10)*10])
#     plt.grid(True)
#     # 为每个条形图添加数值标签
#     # for x2016, proposed in enumerate(proposed):
#     #     plt.text(x2016, proposed + 2, '%s' % proposed, ha='center', fontsize=font)

#     # for x2017, fedsgd in enumerate(fedsgd):
#     #     plt.text(x2017 + bar_width, fedsgd + 2, '%s' % fedsgd, ha='center', fontsize=font)

#     # for x2018, signsgd in enumerate(signsgd):
#     #     plt.text(x2018 + 2 * bar_width, signsgd + 2, '%s' % signsgd, ha='center', fontsize=font)

#     # for x2019, y2019 in enumerate(Y2019):
#     #     plt.text(x2019 + 3*bar_width, y2019 + 2, '%s' % y2019, ha='center', fontsize=font)

#     # for x2020, y2020 in enumerate(Y2020):
#     #     plt.text(x2020 + 4 * bar_width, y2020 + 2, '%s' % y2020, ha='center', fontsize=font)

#     # 显示图例
#     # plt.legend(bbox_to_anchor=(0.5, 1), loc=5, borderaxespad=0, fontsize=font)
#     plt.legend(loc='upper left')

#     # plt.title("test", fontsize=font)
#     plt.xlabel("Accuracy")
#     plt.ylabel("Delay(s)")
#     plt.savefig(save_path +'Delay'+file_name+'.png', dpi=300, format = 'png')

#     #显示图形
#     # plt.show()

#     proposed = [data_E_proposed[1], data_E_proposed[2], data_E_proposed[3]]                  # Proposed 2.0 1.5 0.006
#     fedavg = [data_E_fedavg[1],data_E_fedavg[2],data_E_fedavg[3]]               # SGD 2. 1.5
#     signsgd = [data_E_signsgd[1],data_E_signsgd[2],data_E_signsgd[3]]                # signSGD
#     fedsgd = [data_E_fedsgd[1],data_E_fedsgd[2],data_E_fedsgd[3]]             # avg
#     font_size = 20
#     labels = ['0.6', '0.7', '0.8']
#     bar_width = 0.15
#     # rcParams.update({'font.size': font_size, 'font.family': 'Times New Roman'})
    
#     maxlim = max([max(proposed),max(signsgd),max(fedsgd),max(fedavg)])

#     # 绘图
#     plt.figure(figsize=(15, 12))
#     plt.bar(np.arange(3), proposed, label='LTFL (Proposed)', color='royalblue', alpha=1, width=bar_width, edgecolor="k", hatch='/')
#     # plt.bar(np.arange(3) + 3 *bar_width, fedsgd, label=u'FedSGD', color='orange', alpha=1, edgecolor="k",
#     #         width=bar_width, hatch="o")
#     plt.bar(np.arange(3) + 1 * bar_width, signsgd, label=u'SignSGD', color='limegreen', alpha=1, edgecolor="k",
#             width=bar_width, hatch="x")
#     plt.bar(np.arange(3) + 2 * bar_width, fedavg, label=u'FedAVG', color='saddlebrown', alpha=1, edgecolor="k",
#             width=bar_width, hatch="/")
#     # 添加刻度标签
#     plt.xticks(np.arange(3) + 2*bar_width, labels)
#     # plt.tick_params(labelsize=20)
#     # 设置Y轴的刻度范围
#     plt.ylim([0, math.ceil(maxlim/10)*10])
#     plt.grid(True)
#     # 为每个条形图添加数值标签
#     # for x2016, proposed in enumerate(proposed):
#     #     plt.text(x2016, proposed + 2, '%s' % proposed, ha='center', fontsize=font)

#     # for x2017, fedsgd in enumerate(fedsgd):
#     #     plt.text(x2017 + bar_width, fedsgd + 2, '%s' % fedsgd, ha='center', fontsize=font)

#     # for x2018, signsgd in enumerate(signsgd):
#     #     plt.text(x2018 + 2 * bar_width, signsgd + 2, '%s' % signsgd, ha='center', fontsize=font)

#     # for x2019, y2019 in enumerate(Y2019):
#     #     plt.text(x2019 + 3*bar_width, y2019 + 2, '%s' % y2019, ha='center', fontsize=font)

#     # for x2020, y2020 in enumerate(Y2020):
#     #     plt.text(x2020 + 4 * bar_width, y2020 + 2, '%s' % y2020, ha='center', fontsize=font)

#     # 显示图例
#     # plt.legend(bbox_to_anchor=(0.5, 1), loc=5, borderaxespad=0, fontsize=font)
#     plt.legend(loc='upper left')

#     # plt.title("test", fontsize=font)
#     plt.xlabel("Accuracy")
#     plt.ylabel("Energy Consumption(J)")
#     plt.savefig(save_path +'Energy Consumption'+file_name+'.png', dpi=300, format = 'png')

#     #显示图形
#     # plt.show()

def record_condition(save_path,vers,I_us,computing_resources,distance,h_us):
    with open(save_path+f'array_proposed_step_v{vers}.txt', 'w') as f:
        # 写入数组字符串
        f.write('I_us:'+str(I_us)+'\n')
        f.write('computing_resources:'+str(computing_resources)+'\n')
        f.write('distances:'+str(distance)+'\n')
        f.write('h_us:'+str(h_us)+'\n')

def cifar_iid(dataset, args):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # num_items = int(len(dataset)/args.num_clients)
    num_items = args.num_items
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(args.num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items+50*(-1)**(i+1),
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def mnist_iid(dataset, args):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    # num_items = int(len(dataset)/args.num_clients)
    num_items = args.num_items
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(args.num_clients):
        dict_users[i] = set(np.random.choice(all_idxs, num_items+5*(-1)**(i+1),
                                             replace=False))
        # all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# 根据（节点本地）字典划分全局数据集，得到各个节点的数据集
class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # return torch.tensor(image), torch.tensor(label)
        return image.clone().detach(), torch.tensor(label)
    
def cal_power_ref(args, computing_resources, N_us):
    f_us = np.array(computing_resources)
    powers_ref = args.B_u*(args.Emax-args.k*args.c0*np.array(N_us)*f_us**(args.sigma-1))/(32*args.V)
    print('参考功率最大值',powers_ref)
    for a in range(len(powers_ref)):
        if powers_ref[a] >= args.power_max:
            powers_ref[a] = args.power_max
        elif powers_ref[a] <= args.power_min:
            powers_ref[a] = args.power_min
    print('初始功率值：',powers_ref)
    return powers_ref

def save_args(args, path, filename='args.json'):
    with open(path+filename, 'w') as f:
        json.dump(vars(args), f, indent=4)

def calculate_gamma(t, num_clients, index_set, error_rates, N_us, client_gmaxs, client_gmins, V, bitwidths, prune_rates, L, D):
        gamma = 0.0
        
        for u1 in combinations(index_set, t + 1):
            u1 = set(u1)
            u2 = index_set - u1

            p = 1
            for u in u1:
                p *= (1 - error_rates[u])
            for u in u2:
                p *= error_rates[u]

            n2 = sum(N_us[u]**2 for u in u1)
            n3 = sum(N_us[u] for u in u1)**2

            g_list = [(t * V) for t in (np.array(client_gmaxs) - np.array(client_gmins))**2]
            b_list = [1 / (4 * (2**(i) - 1)**2) for i in bitwidths]
            temp = np.multiply(g_list, b_list)
            
            e = sum(temp[u] + (L**2) * (D**2) * prune_rates[u] for u in u1)
            # e = 0
            # for u in u1:
            #     e += temp[u] + (L**2) * (D**2) * prune_rates[u]
            
            gamma += p * n2 * e / n3
        
        return gamma

def Gamma(prune_rates, bitwidths, transmit_power, client_gmins, client_gmaxs, h_us, I_us, num_clients=10, N_us=[100 for i in range(10)], B_u=1000000*10, N0=3.98e-21, V=62984, waterfall_thre=1, L=100, D=0.3):
    index_set = {i for i in range(num_clients)}
    # total_gamma = 0
    error_rates = [error_rate(p_u=p_u, B_u=B_u, h_u=h_us[index], N0 = N0, I_u=I_us[index], thre=waterfall_thre) for index, p_u in enumerate(transmit_power)]
    
    #-----------------------------------开始计算---------------------------------
    results = []
    
    with mp.Pool() as pool:
        for t in range(num_clients):
            results.append(pool.apply_async(calculate_gamma, (t, num_clients, index_set, error_rates, N_us, client_gmaxs, client_gmins, V, bitwidths, prune_rates, L, D)))
        
        gammas = [result.get() for result in results]
    
    total_gamma = sum(gammas)
    
    # for t in range(num_clients):
        
    #     # 对通信节点为t个的每一种情况遍历
    #     for u1 in combinations(index_set, t+1):
    #         u1 = set(u1)
    #         u2 = index_set - u1

    #         # 计算概率
    #         p = 1
    #         for u in u1:
    #             p = p*(1-error_rates[u])
    #         for u in u2:
    #             p = p*error_rates[u]
            
    #         # 计算N_u的平方和
    #         n2 = 0
    #         for u in u1:
    #             n2 += N_us[u]**2 

    #         # 计算N_u的和平方
    #         n3 = 0
    #         for u in u1:
    #             n3 += N_us[u]
    #         n3 = n3**2

    #         # 计算小期望和
    #         g_list = [t*V for t in (np.array(client_gmaxs)-np.array(client_gmins))**2]
    #         b_list = [1/(4 * (2**(i)-1)**2) for i in bitwidths] 
    #         temp = np.multiply(g_list, b_list)
    #         e = 0
    #         for u in u1:
    #             e += temp[u] + (L**2) * (D**2) * prune_rates[u]
            
    #         # 计算该情况的最终数值并加上去
    #         gamma += p*n2*e/n3

    return total_gamma

def Gamma_for_BO_v2(transmit_power, bitwidths, prune_rates, client_gmins, client_gmaxs, h_us, I_us, num_clients=10, N_us=[100 for i in range(10)], B_u=1, N0=1, V=1, waterfall_thre=1, L=100, D=0.3):
   

    for i in range(num_clients):
            globals()['p'+str(i)] = transmit_power[i]
    error_rates = [error_rate_forBO(p_u=globals()['p'+str(i)], B_u=B_u, h_u=h_us[i], N0 = N0, I_u=I_us[i], thre=waterfall_thre) for i in range(num_clients)]
    
    index_set = {i for i in range(num_clients)}
    # total_gamma = 0


    # def calculate_gamma(t, num_clients, index_set, error_rates, N_us, client_gmaxs, client_gmins, V, bitwidths, prune_rates, L, D):


    results = []
    
    with mp.Pool() as pool:
        for t in range(num_clients):
            results.append(pool.apply_async(calculate_gamma, (t, num_clients, index_set, error_rates, N_us, client_gmaxs, client_gmins, V, bitwidths, prune_rates, L, D)))
        
        gammas = [result.get() for result in results]
    
    total_gamma = sum(gammas)
    #-----------------------------------开始计算---------------------------------
    # for t in range(num_clients):
        
    #     # 对通信节点为t个的每一种情况遍历
    #     for u1 in combinations(index_set, t+1):
    #         u1 = set(u1)
    #         u2 = index_set - u1

    #         # 计算概率
    #         p = 1
    #         for u in u1:
    #             p = p*(1-error_rates[u])
    #         for u in u2:
    #             p = p*error_rates[u]
            
    #         # 计算N_u的平方和
    #         n2 = 0
    #         for u in u1:
    #             n2 += N_us[u]**2 

    #         # 计算N_u的和平方
    #         n3 = 0
    #         for u in u1:
    #             n3 += N_us[u]
    #         n3 = n3**2

    #         # 计算小期望和
    #         g_list = [t*V for t in (np.array(client_gmaxs)-np.array(client_gmins))**2]
    #         b_list = [1/(4 * (2**(i)-1)**2) for i in bitwidths] 
    #         temp = np.multiply(g_list, b_list)
    #         e = 0
    #         for u in u1:
    #             e += temp[u] + (L**2) * (D**2) * prune_rates[u]
            
    #         # 计算该情况的最终数值并加上去
    #         gamma += p*n2*e/n3

    return total_gamma


def count_different_elements(tensor1, tensor2):
    """
    计算两个同维度张量中不同元素的个数。

    Args:
        tensor1 (torch.Tensor): 第一个张量。
        tensor2 (torch.Tensor): 第二个张量，必须与第一个张量形状相同。

    Returns:
        int: 不同元素的个数。
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("两个张量必须具有相同的维度")

    # 计算不同的元素
    difference = tensor1 != tensor2
    # 统计不同元素的个数
    different_count = difference.sum().item()

    return different_count

if __name__ == '__main__':
    # file_name='./FEDSGD/LA_SGD_T2.5_E0.015_w0.0065_c1.csv'
    
    # plot_single_converg('./', file_name, if_loss=True)
    parser = argparse.ArgumentParser(description='Example script with global variable.')

    parser.add_argument('--dataset', type=str, help='dataset, cifar或者mnist', default='cifar')
    parser.add_argument('--model', type=str, help='model, cnn(cifar的cnn 343946个参数)或者mlp(50890个参数)', default='res')
    parser.add_argument('--if_batch', type=int, help='是否使用minibatchgd', default=1)
    parser.add_argument('--if_prune', type=int, help='是否prune', default=1)
    parser.add_argument('--if_quantize', type=int, help='是否quantize', default=1)
    parser.add_argument('--if_one_hot', type=int, help='是否独热编码，目前这个参数没用', default=0)
    parser.add_argument('--if_SCG', type=int, help='是否使用SCG, 注意不能与MINIbatchSGD一起用,还没写', default=0)
    parser.add_argument('--if_aug', type=int, help='是否数据增强', default=1)
    parser.add_argument('--pattern', type=str, help='exp1', default='PROPOSED')
    ''' 
    pattarn:
        FEDSGD:单独进行FEDSGD算法
        FEDAVG:单独进行FEDAVG算法
        SIGNSGD:单独进行SIGNSGD算法
        PROPOSED:单独进行PROPOSED算法
        exp1:四种算法的对比实验, 在相同特定信道条件、节点数量下, 不同的方案在不同通信轮次上的收敛性、能耗、时延
        exp2:四种算法的对比实验, 在相同特定信道条件下, 不同的方案在不同节点上的能耗、时延
    '''
    parser.add_argument('--B_u', type=int, help='B_u', default=1e6)
    parser.add_argument('--num_items', type=int, help='num_items是每个节点的平均数据量, 在fl_utils.cifar_iid里面认为设置了波动值', default=500) 
    parser.add_argument('--scale', type=int, help='scale是每个节点选取batch的大小比例, 在各个Client类的train函数中用到', default=1)    
    parser.add_argument('--num_clients', type=int, help='num_clients是参与训练的节点数量', default=20)

    parser.add_argument('--local_bs', type=int, help='local_bs, 本地的batch_size大小', default=32)
    parser.add_argument('--local_ep', type=int, help='local_ep', default=1)
    parser.add_argument('--num_epoch', type=int, help='num_epoch是全局迭代的最大轮次', default=1000) 
    parser.add_argument('--init_param', type=float, help='初始功率的系数', default=0.5)   
    # parser.add_argument('--mean_datanum', type=int, help='节点平均的数据量', default=10000)

    parser.add_argument('--wer', type=float, help='wer是信道条件Rayleigh fading factor', default=0.01)                                                     
    parser.add_argument('--Tmax', type=float, help='Tmax是每轮全局迭代的最大时延(约束条件)', default=500)     
    parser.add_argument('--Emax', type=float, help='Emax是每轮全局迭代的最大能耗(约束条件)', default=400)      
    

    parser.add_argument('--V', type=int, help='V; cnn(cifar的cnn 343946个参数)或者mlp(50890个参数)', default=11171146)
    parser.add_argument('--c0', type=int, help='c0是通过反向传播算法训练一个样本数据所需的CPU周期数', default=2.7e8) 

    parser.add_argument('--count_py', type=int, help='count_py是文件名序号,用于扫参数的时候区分随机性', default=3)       
    parser.add_argument('--learning_rate', type=int, help='learning_rate是学习率', default=0.01)
    parser.add_argument('--s', type=float, help='s是梯度聚合、模型更新并广播的时延。一个常数。', default=0.1)    
    parser.add_argument('--waterfall_thre', type=int, help='waterfall_thre是阈值', default=1)
    parser.add_argument('--D', type=float, help='D', default=0.3)
    parser.add_argument('--sigma', type=int, help='sigma', default=3)
    parser.add_argument('--loss_func', type=str, help='loss_func, 可以为crossentropy或nll', default='crossentropy')
    parser.add_argument('--optimizer', type=str, help='optimizer, 可以为sgd或adam', default='sgd')
    
    parser.add_argument('--N0', type=float, help='N0', default=3.98e-21)
    parser.add_argument('--k', type=float, help='k', default=1.25e-26)
    parser.add_argument('--I_min', type=str, help='I_min', default=1e-8)
    parser.add_argument('--I_max', type=str, help='I_max', default=2e-8)
    parser.add_argument('--dis_min', type=str, help='dis_min', default=100)
    parser.add_argument('--dis_max', type=str, help='dis_max', default=300)

    parser.add_argument('--bcd_epoch', type=int, help='bcd_epoch是块坐标下降法的迭代次数', default=0)                
    parser.add_argument('--BO_epoch', type=int, help='BO_epoch是贝叶斯优化的迭代次数', default=10)                
    parser.add_argument('--power_min', type=float, help='power_min', default=0.01)
    parser.add_argument('--power_max', type=float, help='power_max', default=0.1)
    parser.add_argument('--S_min', type=int, help='S_min', default=1)
    parser.add_argument('--S_max', type=int, help='S_max', default=20)
    parser.add_argument('--prune_rate_min', type=float, help='prune_rate_min', default=0.0)
    parser.add_argument('--prune_rate_max', type=float, help='prune_rate_max', default=0.3)
    parser.add_argument('--resource_min', type=float, help='resource_min', default=2e8)
    parser.add_argument('--resource_max', type=float, help='resource_max', default=5e8)

    parser.add_argument('--acq_func', type=str, help='acq_func', default='PI')

    parser.add_argument('--markevery', type=int, help='画折线图时点的间隔', default=20)

    parser.add_argument('--L', type=float, help='optimizer, 可以为sgd或adam', default=100)

    parser.add_argument('--F_0', type=float, help='F_0', default=1)
    parser.add_argument('--F_1', type=float, help='F_1', default=0.8)
    parser.add_argument('--epsilon', type=float, help='epsilon', default=5e-2)
    parser.add_argument('--G', type=float, help='G', default=0.1)
    parser.add_argument('--C', type=float, help='C', default=0.1)
    parser.add_argument('--delta', type=float, help='delta', default=2)

    args = parser.parse_args()

    I_us,computing_resources,distance = read_condition(file_name='condition.csv')
    I_us = I_us[:args.num_clients]
    computing_resources = computing_resources[:args.num_clients]
    distance = distance[:args.num_clients]
    h_us = [args.wer/(i**(2)) for i in distance] 
    N_us = [500 for i in range(args.num_clients)]

    args.I_us = I_us
    args.computing_resources = computing_resources
    args.distance = distance
    args.h_us = h_us
    args.N_us = N_us

    # adjust(args)

    S0 = random.randint(args.S_min, args.S_max)
    prune_rate0 = [random.uniform(args.prune_rate_min, args.prune_rate_max) for i in range(args.num_clients)]
    transmit_power0 = [random.uniform(args.power_min, args.power_max) for i in range(args.num_clients)]
    
    # while(1):
    #     start = time.perf_counter()
    #     error_ratesS = [error_rate(p_u=p_u, B_u=args.B_u, h_u=args.h_us[index], N0 = args.N0, I_u=args.I_us[index], thre=args.waterfall_thre) for index, p_u in enumerate(transmit_power0)]
    #     data_ratesS = np.array([data_rate(p_u=p_u, B_u=args.B_u, h_u=args.h_us[index], N0 = args.N0, I_u=args.I_us[index]) for index,p_u in enumerate(transmit_power0)])
    #     get_H(args, set(range(args.num_clients)), args.num_clients, transmit_power0, prune_rate0, S0, data_ratesS, error_ratesS)
    #     end = time.perf_counter()
    #     print(f'本轮计算用时：{end-start}')
    
    # while(1):
    #     alpha = torch.tensor([1 for i_ in range(args.num_clients)])
    #     generate_alpha(alpha, transmit_power0, args.num_clients, args.I_us, args.h_us, args.B_u, args.N0, args.waterfall_thre)
    #     print(alpha)

    # 现在的H函数，算一次就要12s

    # original_dict = OrderedDict({
    # 'a': torch.tensor([[10, 0.], [5, -124]]),
    # 'b': torch.tensor([56, 0., 0.]),
    # 'c': torch.tensor([[0]]),
    # 'a1': torch.tensor([[12, -45], [89, -12]]),
    # 'b1': torch.tensor([15, 0, -69]),
    # 'c1': torch.tensor([[-56]]),
    # 'a2': torch.tensor([[106, 0.], [7, -56]]),
    # 'b2': torch.tensor([0, 86, 27]),
    # 'c3': torch.tensor([[-108]]),
    # 'a4': torch.tensor([[12, -45], [89, -2]]),
    # 'b4': torch.tensor([0, 0, -6]),
    # 'c4': torch.tensor([[-56]]),
    # 'a5': torch.tensor([[10, 0.], [57, -5]]),
    # 'b5': torch.tensor([0, 8, 27]),
    # 'c5': torch.tensor([[-8]]),
    # 'a6': torch.tensor([[0, -45], [8, -12]]),
    # 'b6': torch.tensor([15, 0, -9]),
    # 'c6': torch.tensor([[0]]),
    # 'a7': torch.tensor([[0, 0.], [5, -56]]),
    # 'b7': torch.tensor([0, 0, 2]),
    # 'c7d': torch.tensor([[0]])
    # })

    # original_dict_sign = OrderedDict()
    # for key in original_dict.keys():
    #     original_dict_sign[key] = torch.sign(original_dict[key])
    # 转换为一维向量
    
    # print("一维向量:", vector)
    # print("索引信息:", index_map)

    # matrix_S = generate_matrix(m=12, n=vector.numel(), mean=0, std_dev=1)

    # temp_compress = torch.matmul(matrix_S, vector)

    
    


    

    
    