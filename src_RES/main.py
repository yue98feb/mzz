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
import argparse

import fl_utils
import model

import MINISGD
import FEDAVG
import SIGNSGD
import PROPOSED
import multiprocessing as mp
import os
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

parser = argparse.ArgumentParser(description='Example script with global variable.')

#------------- 以下是训练模式参数 ----------------
parser.add_argument('--dataset', type=str, help='dataset, cifar或者mnist', default='mnist')
parser.add_argument('--model', type=str, help='model, cnn(cifar的cnn 343946个参数)或者mlp(50890个参数)', default='mlp')
parser.add_argument('--if_batch', type=int, help='是否使用minibatchgd', default=1)
parser.add_argument('--if_prune', type=int, help='是否剪枝', default=0)
parser.add_argument('--if_quantize', type=int, help='是否1-bit量化', default=1)
parser.add_argument('--if_compress', type=int, help='是否加入压缩', default=0)
parser.add_argument('--if_topk', type=int, help='是否topk稀疏', default=0)
parser.add_argument('--feedback_coe', type=float, help='feedback机制的衰减因子, 取0.3/0.5/1', default=0.1) 
parser.add_argument('--V', type=int, help='训练模型的参数个数, 由网络本身确定; 目前MNIST采用的mlp网络是50890个参数;CIFAR10采用的resnet网络参数是11171146个', default=50890)
parser.add_argument('--num_clients', type=int, help='参与训练的节点数量', default=10)
parser.add_argument('--pattern', type=str, help='训练模式', default='baseline4')
parser.add_argument('--optimizer', type=str, help='优化器, 可以为sgd; adam; sign', default='sign')
parser.add_argument('--loss_func', type=str, help='损失函数类型, 目前使用的是crossentropy', default='crossentropy')
''' 
pattarn:
    FEDSGD:单独进行FEDSGD算法
    FEDAVG:单独进行FEDAVG算法
    SIGNSGD:单独进行SIGNSGD算法
    baseline4: baseline4
    baseline5:baseline5
    PROPOSED:单独进行PROPOSED算法


optimizer:
    sgd: SGD 优化器
    adam: adam 优化器
    sign: 1-bit量化下的优化器
    SIGNSGD、PROPOSED都用sign优化器; 其他的用sgd/adam二选一,之前一般用的sgd
'''

#------------- 以下是训练本身的超参数 ----------------                
parser.add_argument('--num_items', type=int, help='每个节点的平均数据量, 在fl_utils.cifar_iid里面人为设置了波动值;【根据num_clients数量调节】', default=5000) 
parser.add_argument('--scale', type=int, help='每个节点总数据/每轮参与训练的数据的值, 在各个Client类的train函数中用到; 【一般设置1】', default=200)    
parser.add_argument('--learning_rate', type=int, help='学习率', default=0.005)
parser.add_argument('--local_bs', type=int, help='local_bs, 本地的batch_size大小', default=16)
parser.add_argument('--local_ep', type=int, help='local_ep, 每个节点本地重复训练的次数, 直接设置为1', default=1)
parser.add_argument('--num_epoch', type=int, help='num_epoch是全局迭代的最大轮次', default=200) 
parser.add_argument('--init_param', type=float, help='初始功率的系数', default=0.5)   
parser.add_argument('--topk', type=float, help='topk稀疏保留的比例, 取0.5/0.7', default=0.7)

#------------- 以下是优化问题的约束条件 ----------------
parser.add_argument('--power_min', type=float, help='发射功率最小值', default=0.01)
parser.add_argument('--power_max', type=float, help='发射功率最大值', default=0.1)
parser.add_argument('--S_min', type=int, help='压缩感知的维度最小值', default=20000)
parser.add_argument('--S_max', type=int, help='压缩感知的维度最大值=模型参数个数', default=50890)
parser.add_argument('--prune_rate_min', type=float, help='剪枝率最小值', default=0.0)
parser.add_argument('--prune_rate_max', type=float, help='剪枝率最大值', default=0.5)

#------------- 以下是优化问题的参数 ----------------
parser.add_argument('--bcd_epoch', type=int, help='整体块坐标下降法的迭代次数上限; 由于时间问题, 这里就设置一个比较小的值5, 看看效果如何', default=10)                
parser.add_argument('--BO_epoch', type=int, help='贝叶斯优化的迭代次数', default=10)
parser.add_argument('--acq_func', type=str, help='贝叶斯优化采集函数, 直接用PI即可', default='PI')
parser.add_argument('--L', type=float, help='李普希兹常数', default=10)
parser.add_argument('--F_0', type=float, help='损失函数的初值; cifar10+resnet情况下是1.8(avg), 1-bit量化下是6.4', default=6.4)
parser.add_argument('--F_1', type=float, help='损失函数的收敛值; cifar10+resnet情况下取0.1', default=0.1)
parser.add_argument('--epsilon', type=float, help='epsilon', default=0.9)
parser.add_argument('--G', type=float, help='G是梯度上界', default=0.1)
parser.add_argument('--C', type=float, help='C根据delta计算', default=8)
parser.add_argument('--delta', type=float, help='delta', default=0.4)

#------------- 以下是环境模拟参数 ----------------
parser.add_argument('--wer', type=float, help='wer是信道条件Rayleigh fading factor, 数值越小，信道越差', default=0.05)                                                     
parser.add_argument('--Tmax', type=float, help='Tmax是每轮全局迭代的最大时延(约束条件)', default=500)     
parser.add_argument('--Emax', type=float, help='Emax是每轮全局迭代的最大能耗(约束条件)', default=400)      
parser.add_argument('--B_u', type=int, help='信道带宽/Hz', default=1e6)
parser.add_argument('--c0', type=int, help='c0是通过反向传播算法训练一个样本数据所需的CPU周期数', default=2.7e8) 
parser.add_argument('--resource_min', type=float, help='计算资源的最小值', default=2e8)
parser.add_argument('--resource_max', type=float, help='计算资源的最大值', default=5e8)
parser.add_argument('--N0', type=float, help='N0', default=3.98e-21)
parser.add_argument('--k', type=float, help='k', default=1.25e-26)
parser.add_argument('--I_min', type=str, help='I_min', default=1e-8)
parser.add_argument('--I_max', type=str, help='I_max', default=2e-8)
parser.add_argument('--dis_min', type=str, help='dis_min', default=100)
parser.add_argument('--dis_max', type=str, help='dis_max', default=300)
parser.add_argument('--s', type=float, help='s是梯度聚合、模型更新并广播的时延。一个常数。', default=0.1)    
parser.add_argument('--waterfall_thre', type=int, help='waterfall_thre是阈值', default=1)
parser.add_argument('--D', type=float, help='D', default=0.3)
parser.add_argument('--sigma', type=int, help='sigma', default=3)

#------------- 其他参数 ----------------
parser.add_argument('--count_py', type=int, help='count_py单纯是文件名序号,用于扫参数的时候同一组参数跑多次区分随机性', default=0)       
parser.add_argument('--markevery', type=int, help='画折线图时标注点的间隔', default=1)
parser.add_argument('--if_SCG', type=int, help='是否使用SCG, 注意不能与MINIbatchSGD一起用,还没写', default=0)
parser.add_argument('--if_one_hot', type=int, help='是否独热编码，目前这个参数没用', default=0)
parser.add_argument('--if_aug', type=int, help='是否数据增强,默认1即可,用处不大', default=1)


args = parser.parse_args()

def main():
    mp.set_start_method('spawn')

    # 设置一些环境参数，直接从已经随机生成好的condition.csv文件中读取
    I_us,computing_resources,distance = fl_utils.read_condition(file_name='./condition.csv')
    I_us = I_us[:args.num_clients]
    computing_resources = computing_resources[:args.num_clients]
    distance = distance[:args.num_clients]
    h_us = [args.wer/(i**(2)) for i in distance] 
    train_dataset, test_dataset, user_groups = model.get_dataset(args=args)
    N_us = [len(user_groups[i]) for i in range(args.num_clients)]
    
    args.I_us = I_us
    args.computing_resources = computing_resources
    args.distance = distance
    args.h_us = h_us
    args.N_us = N_us

    # 初始化3个优化变量：S是压缩感知的维度，prune_rates是各个client的剪枝率，transmit_power是各个client的功率
    ini_S = args.V/2
    ini_prune_rates = [0 for i in range(args.num_clients)]
    ini_transmit_power = [args.power_max*args.init_param for i in range(args.num_clients)]

    if args.pattern=='FEDSGD':
        print('现在进行实验:FEDSGD')
        args.if_batch = 1
        args.if_prune = 0
        args.if_quantize = 0
        args.if_compress = 0
        args.feedback_coe = 0.0
        args.if_topk = 0
        args.optimizer = 'sgd'
        MINISGD.FEDSGD(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./FEDSGD/', transmit_power=ini_transmit_power, bitwidths=[32 for i in range(args.num_clients)], prune_rates=[0 for i in range(args.num_clients)],computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        # file_path='./FEDSGD/', train_dataset=None, test_dataset=None, user_groups=None, num_epoch=100, num_clients=10, learning_rate=0.01, bitwidths=[8 for i in range(10)], prune_rates=[0.5 for i in range(10)], transmit_power=[0.1 for i in range(10)], Tmax=0.1, Emax=0.1, wer=0.1, count_py=0.1, N_us=[100 for i in range(10)], I_us=[1.5e-08 for i in range(10)], h_us=[0.1 for i in range(10)], computing_resources=[6e7 for i in range(10)], c0=200000, s=0.1, waterfall_thre=1, D=0.3, sigma=3, V=62984, B_u=1000000*10, N0=3.98e-21, k=1.25e-26
        fl_utils.plot_single_converg(args=args, save_path='./FEDSGD/', file_name=f'./FEDSGD/LA_SGD_w{args.wer}_节点数{args.num_clients}_e{args.epsilon}_rho{args.prune_rate_max}_p{args.power_max}.csv')

    elif args.pattern=='FEDAVG':
        print('现在进行实验:FEDAVG')
        args.if_batch = 1
        args.if_prune = 0
        args.if_quantize = 0
        args.if_compress = 0
        args.feedback_coe = 0.0
        args.if_topk = 0
        args.optimizer = 'sgd'
        FEDAVG.FEDAVG(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./FEDAVG/', transmit_power=ini_transmit_power, bitwidths=[32 for i in range(args.num_clients)], prune_rates=[0 for i in range(args.num_clients)],computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        # file_path='./FEDAVG/', train_dataset=None, test_dataset=None, user_groups=None, num_epoch=100, num_clients=10, learning_rate=0.01, bitwidths=[8 for i in range(10)], prune_rates=[0.5 for i in range(10)], transmit_power=[0.1 for i in range(10)], Tmax=0.1, Emax=0.1, wer=0.1, count_py=0.1, N_us=[100 for i in range(10)], I_us=[1.5e-08 for i in range(10)], h_us=[0.1 for i in range(10)], computing_resources=[6e7 for i in range(10)], c0=200000, s=0.1, waterfall_thre=1, D=0.3, sigma=3, V=62984, B_u=1000000*10, N0=3.98e-21, k=1.25e-26
        fl_utils.plot_single_converg(args=args, save_path='./FEDAVG/', file_name=f'./FEDAVG/LA_AVG_w{args.wer}_节点数{args.num_clients}_e{args.epsilon}_rho{args.prune_rate_max}_p{args.power_max}.csv')
    
    elif args.pattern=='SIGNSGD':
        print('现在进行实验:SIGNSGD')
        args.if_batch = 1
        args.if_prune = 0
        args.if_quantize = 0
        args.if_compress = 0
        args.feedback_coe = 0.0
        args.if_topk = 0
        args.optimizer = 'sign'
        SIGNSGD.SIGNSGD(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./SIGNSGD/', transmit_power=ini_transmit_power, bitwidths=[2 for i in range(args.num_clients)], prune_rates=[0 for i in range(args.num_clients)],computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        # file_path='./SIGNSGD/', train_dataset=None, test_dataset=None, user_groups=None, num_epoch=100, num_clients=10, learning_rate=0.01, bitwidths=[8 for i in range(10)], prune_rates=[0.5 for i in range(10)], transmit_power=[0.1 for i in range(10)], Tmax=0.1, Emax=0.1, wer=0.1, count_py=0.1, N_us=[100 for i in range(10)], I_us=[1.5e-08 for i in range(10)], h_us=[0.1 for i in range(10)], computing_resources=[6e7 for i in range(10)], c0=200000, s=0.1, waterfall_thre=1, D=0.3, sigma=3, V=62984, B_u=1000000*10, N0=3.98e-21, k=1.25e-26
        fl_utils.plot_single_converg(args=args, save_path='./SIGNSGD/', file_name=f'./SIGNSGD/LA_SIGNSGD_w{args.wer}_节点数{args.num_clients}_e{args.epsilon}_rho{args.prune_rate_max}_p{args.power_max}.csv')
    
    elif args.pattern=='baseline4':
        print('现在进行实验:baseline4')
        args.bcd_epoch = 0
        args.if_batch = 1
        args.if_prune = 0
        args.if_quantize = 1
        args.if_compress = 1
        args.feedback_coe = 0.0
        args.if_topk = 1
        args.optimizer = 'sign'
        PROPOSED.PROPOSED(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./baseline4/', transmit_power=ini_transmit_power, S=ini_S, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        # file_path='./PROPOSED/', train_dataset=None, test_dataset=None, user_groups=None, num_epoch=100, num_clients=10, learning_rate=0.01, bitwidths=[8 for i in range(10)], prune_rates=[0.5 for i in range(10)], transmit_power=[0.1 for i in range(10)], Tmax=0.1, Emax=0.1, wer=0.1, count_py=0.1, N_us=[100 for i in range(10)], I_us=[1.5e-08 for i in range(10)], h_us=[0.1 for i in range(10)], computing_resources=[6e7 for i in range(10)], c0=200000, s=0.1, waterfall_thre=1, D=0.3, sigma=3, V=62984, B_u=1000000*10, N0=3.98e-21, k=1.25e-26
        fl_utils.plot_single_converg(args=args, save_path='./baseline4/', file_name=f'./baseline4/LA_baseline4_w{args.wer}_节点数{args.num_clients}_e{args.epsilon}_rho{args.prune_rate_max}_p{args.power_max}_topk{args.topk}.csv')
    
    elif args.pattern=='baseline5':
        print('现在进行实验:baseline5')
        args.if_batch = 1
        args.if_prune = 1
        args.if_quantize = 1
        args.if_compress = 1
        args.feedback_coe = 0.0
        args.if_topk = 0
        args.optimizer = 'sign'
        PROPOSED.PROPOSED(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./baseline5/', transmit_power=ini_transmit_power, S=ini_S, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        # file_path='./PROPOSED/', train_dataset=None, test_dataset=None, user_groups=None, num_epoch=100, num_clients=10, learning_rate=0.01, bitwidths=[8 for i in range(10)], prune_rates=[0.5 for i in range(10)], transmit_power=[0.1 for i in range(10)], Tmax=0.1, Emax=0.1, wer=0.1, count_py=0.1, N_us=[100 for i in range(10)], I_us=[1.5e-08 for i in range(10)], h_us=[0.1 for i in range(10)], computing_resources=[6e7 for i in range(10)], c0=200000, s=0.1, waterfall_thre=1, D=0.3, sigma=3, V=62984, B_u=1000000*10, N0=3.98e-21, k=1.25e-26
        fl_utils.plot_single_converg(args=args, save_path='./baseline5/', file_name=f'./baseline5/LA_baseline5_w{args.wer}_节点数{args.num_clients}_e{args.epsilon}_rho{args.prune_rate_max}_p{args.power_max}.csv')

    elif args.pattern=='PROPOSED':
        print('现在进行实验:PROPOSED')
        args.if_batch = 1
        args.if_prune = 1
        args.if_quantize = 1
        args.if_compress = 1
        # args.feedback_coe = 0.1
        args.if_topk = 0
        args.optimizer = 'sign'
        PROPOSED.PROPOSED(args=args, train_dataset=train_dataset, test_dataset=test_dataset, user_groups=user_groups, N_us=N_us, file_path='./PROPOSED/', transmit_power=ini_transmit_power, S=ini_S, prune_rates=ini_prune_rates,computing_resources=computing_resources, I_us=I_us, h_us=h_us)
        # file_path='./PROPOSED/', train_dataset=None, test_dataset=None, user_groups=None, num_epoch=100, num_clients=10, learning_rate=0.01, bitwidths=[8 for i in range(10)], prune_rates=[0.5 for i in range(10)], transmit_power=[0.1 for i in range(10)], Tmax=0.1, Emax=0.1, wer=0.1, count_py=0.1, N_us=[100 for i in range(10)], I_us=[1.5e-08 for i in range(10)], h_us=[0.1 for i in range(10)], computing_resources=[6e7 for i in range(10)], c0=200000, s=0.1, waterfall_thre=1, D=0.3, sigma=3, V=62984, B_u=1000000*10, N0=3.98e-21, k=1.25e-26
        fl_utils.plot_single_converg(args=args, save_path='./PROPOSED/', file_name=f'./PROPOSED/LA_PROPOSED_w{args.wer}_节点数{args.num_clients}_e{args.epsilon}_rho{args.prune_rate_max}_p{args.power_max}_feedback{args.feedback_coe}.csv')

    else:
        pass

if __name__ == "__main__":
    start_time = time.time()
    # array_S = np.random.normal(loc=0, scale=1, size=(args.V, args.V))
    # np.save('saved_array.npy', array_S)
    main()
    # fl_utils.plot_single_converg(args=args, save_path='./FEDAVG/', file_name=f'./FEDAVG/LA_AVG_w{args.wer}_c{args.count_py}.csv')
    
    # fl_utils.plot_multi_converg(args=args, save_path='./', save_name='_test', baseline6 = './scale=100/PROPOSED/LA_PROPOSED_w0.05_节点数10_e0.9_rho0.5_p0.1_feedback0.1.csv', baseline4 = './scale=100/baseline4/LA_baseline4_w0.05_节点数10_e0.9_rho0.5_p0.1.csv', baseline5 = './scale=100/baseline5/LA_baseline5_w0.05_节点数10_e0.9_rho0.5_p0.1.csv')

    end_time = time.time()
    execution_time = end_time - start_time
    print("程序运行时间：", execution_time/60, "分钟")
    torch.cuda.empty_cache()
    os._exit(1)