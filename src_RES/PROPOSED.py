import torch
import torchvision
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
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader, Dataset
# from tensorboardX import SummaryWriter
import fl_utils
import model
from collections import OrderedDict
from resnet import ResNet18
from res18 import PreActResNet18
import multiprocessing as mp
from scipy.optimize import linprog

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# logger = SummaryWriter('../logs/SGD')
device_1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_2 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Server:
    def __init__(self, model, matrix_S, test_dataset, optimizer,device, learning_rate,loss_func, N_us):
        self.model = model
        self.test_dataset = test_dataset
        self.optimizer = optimizer
        self.device = device
        self.learning_rate = learning_rate
        if loss_func =='crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_func == 'nll':
            self.criterion = nn.NLLLoss()
        self.N_us = N_us
        self.matrix_S = matrix_S

    # 梯度聚合
    def aggregate(self, gradients, alpha, args):
        
        # new_list = []
        # for j in range(len(gradients[0])):
        #     new_sub_list = [sub_l[j] for sub_l in gradients]
        #     new_list.append(new_sub_list)
        
        # stacked_grads = [torch.stack(sublist) for sublist in new_list]
        
        # if torch.all(alpha==0):
        #     agg_grads = [torch.sum(grad*alpha.reshape(num_clients,1).view((num_clients,)+(1,)*(len(grad.shape)-1)),dim=0) for grad in stacked_grads]
        # else:
        #     agg_grads = [torch.sum(grad*alpha.reshape(num_clients,1).view((num_clients,)+(1,)*(len(grad.shape)-1))*torch.tensor(self.N_us).reshape(num_clients,1).view((num_clients,)+(1,)*(len(grad.shape)-1)).to(self.device),dim=0)/torch.sum(alpha*torch.tensor(self.N_us).to(self.device)).item() for grad in stacked_grads]
        
        # 重建
        alpha_weight = [self.N_us[i]*alpha[i] for i in range(len(gradients))]
        alpha_weight = [i/sum(alpha_weight) for i in alpha_weight]

        gradient_size = OrderedDict()
        for name, param in self.model.named_parameters():
            gradient_size[name] = torch.zeros_like(param.data)

        _, index_map = fl_utils.dict_to_vector(gradient_size)
        
        if args.if_compress:
            avg_gradients_flatten = torch.zeros_like(gradients[0]).to(self.device)
            for i in range(args.num_clients):
                avg_gradients_flatten += gradients[i].to(self.device)*alpha_weight[i]
            
            avg_gradients = fl_utils.decompress(args, self.matrix_S.to(self.device), avg_gradients_flatten, index_map, gradient_size)
        
        else:
            avg_gradients = copy.deepcopy(gradients[0])
            for key in avg_gradients.keys():
                avg_gradients[key] = torch.zeros_like(avg_gradients[key])

            alpha_weight = [self.N_us[i]*alpha[i] for i in range(len(gradients))]
            alpha_weight = [i/sum(alpha_weight) for i in alpha_weight]
            if torch.all(alpha==0):
                print('alpha all zero')
                for key in avg_gradients.keys():
                    avg_gradients[key] = torch.zeros_like(avg_gradients[key])
            else:
                for key in avg_gradients.keys():
                    for i in range(len(gradients)):
                        avg_gradients[key] += gradients[i][key].to(device)*alpha_weight[i]
                    # avg_gradients[key] = torch.div(avg_gradients[key], sum(alpha))
            
            if args.if_quantize:
                for key in avg_gradients.keys():
                    avg_gradients[key] = torch.sign(avg_gradients[key])

        return avg_gradients

    def aggregate_SCG(self, gradients, alpha):
        # new_list = []
        # num_clients = len(deltas)
        # for j in range(len(deltas[0])):
        #     new_sub_list = [sub_l[j] for sub_l in deltas]
        #     new_list.append(new_sub_list)

        # stacked_grads = [torch.stack(sublist) for sublist in new_list]

        # if torch.all(alpha==0):
        #     agg_delta = [torch.sum(grad*alpha.reshape(num_clients,1).view((num_clients,)+(1,)*(len(grad.shape)-1)),dim=0) for grad in stacked_grads]
        # else:
        #     agg_delta = [torch.sum(grad*alpha.reshape(num_clients,1).view((num_clients,)+(1,)*(len(grad.shape)-1))*torch.tensor(self.N_us).reshape(num_clients,1).view((num_clients,)+(1,)*(len(grad.shape)-1)).to(self.device),dim=0)/torch.sum(alpha*torch.tensor(self.N_us).to(self.device)).item() for grad in stacked_grads]
        
        # return agg_delta
        avg_gradients = copy.deepcopy(gradients[0])
        for key in avg_gradients.keys():
            avg_gradients[key] = torch.zeros_like(avg_gradients[key])

        alpha_weight = [self.N_us[i]*alpha[i] for i in range(len(gradients))]
        alpha_weight = [i/sum(alpha_weight) for i in alpha_weight]
        if torch.all(alpha==0):
            print('alpha all zero')
            for key in avg_gradients.keys():
                avg_gradients[key] = torch.zeros_like(avg_gradients[key])
        else:
            for key in avg_gradients.keys():
                for i in range(len(gradients)):
                    avg_gradients[key] += gradients[i][key].to(device)*alpha_weight[i]
        return avg_gradients

    # 梯度下降更新模型参数
    def update(self, avg_gradients, if_converge):
        # with torch.no_grad():
        #     # checkg = [p.grad.data for p in self.model.parameters()]
        #     self.model.train()
        #     self.model.zero_grad()
        #     gradients = list(avg_gradients.values())
        #     for p, g in zip(self.model.parameters(), gradients):
        #         # p.grad.data = g.to(device)
        #         p.grad.data = g
        #     # checkg2 = [p.grad.data for p in self.model.parameters()]
        #     self.optimizer.step()
        if if_converge==0:
            self.model.to(device)
            self.model.train()
            global_weights = self.model.state_dict()
            for key in avg_gradients.keys():
                global_weights[key] -= self.learning_rate * avg_gradients[key].to(device)
            self.model.load_state_dict(global_weights)


    def update_SCG(self, avg_deltas): 
        self.model.train()   
        for (name, param), (delta_name, dt) in zip(self.model.named_parameters(), avg_deltas.items()):
            assert name == delta_name, "Parameter names do not match!"  # 检查参数名匹配
            param.data.add_(dt)  # 使用 avg_deltas 中对应的梯度增量更新参数

    # 测试准确率
    def test(self):
        # self.model.eval()
        self.model.to(device)
        loss, total, correct = 0.0, 0.0, 0.0
        testloader = DataLoader(self.test_dataset, batch_size=128,
                                shuffle=False)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = self.model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

        accuracy = correct/total
        return accuracy
    
    # def test(self):
    #     test_data, test_target = self.x_test.to(self.device,dtype=torch.float32),self.y_test.to(self.device,dtype=torch.long)
    #     num_correct = 0
    #     num_samples = 0
    #     self.model.eval()   # set model to evaluation mode
    #     with torch.no_grad():
    #         scores = self.model(test_data)
    #         _,preds = scores.max(1)
    #         num_correct = (preds==test_target).sum()
    #         num_samples = preds.size(0)
    #         acc = float(num_correct) / num_samples
    #         print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 *acc ))
    #         return acc

class Client: 
    def __init__(self, model, matrix_S, logger, prune_rate, train_dataset, test_dataset, idxs, device, loss_func, local_bs, optimizer, learning_rate, local_ep, N_u):
        self.model = model
        self.logger = logger
        self.prune_rate = prune_rate
        self.local_bs = local_bs
        self.trainloader = DataLoader(fl_utils.DatasetSplit(train_dataset, idxs),
                                 batch_size=self.local_bs, shuffle=True)
        self.testloader = DataLoader(test_dataset, batch_size=self.local_bs, shuffle=False)
        self.device = device
        self.optimizer = optimizer
        self.learning_rate = learning_rate 
        # if optimizer == 'sgd':
        #     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.5)
        # elif optimizer == 'adam':
        #     self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.local_ep = local_ep
        if loss_func =='crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_func == 'nll':
            self.criterion = nn.NLLLoss()
        self.N_u = N_u
        self.matrix_S = matrix_S
        self.feedback_grad = None
        self.if_feedback = False
        self.if_converge = 0
    # 梯度量化
    def quantize(self, grad, g_max, g_min): 
        g_max = abs(g_max)
        g_min = abs(g_min)
        sign = torch.sign(grad)
        grad = abs(grad)

        # 计算最值，确定边界
        # max = grad.max()
        # min = grad.min()

        # 计算量化间隔大小
        scale = (g_max - g_min)/(2.**self.bitwidth - 1.) 

        if scale==0:
            print("scale=0")
            print(g_min,g_max)
            print(grad)
            raise ValueError("量化区间为0")
            return grad
        else:
            # 计算可能的量化值
            level = (grad - g_min)/scale
            level_down = level.floor_()
            level_up = (level+0.0001).ceil_()
            q_down = level_down*scale + g_min
            q_up = level_up*scale + g_min
            
            # 计算概率tensor
            pro = (grad-q_down)/(q_up-q_down)

            # 生成随机tensor
            rand_tensor = torch.randn(grad.shape).to(self.device)

            # 生成量化梯度
            q_grad = torch.where(rand_tensor>pro, q_up, q_down)

            if torch.isnan(q_grad).any():
                print(scale)
                print(g_min,g_max)
                print(grad)
                raise ValueError("梯度有nan")
            if torch.all(q_grad==0):
                print(scale)
                print(g_min,g_max)
                print(grad)
                raise ValueError("量化梯度全0")
            q_grad = q_grad*sign
            return q_grad
    
    # 剪枝
    def prune(self, model): 
        parameters_to_prune = ()
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d):
                # parameters_to_prune += ((module, 'weight'),)
                prune.l1_unstructured(module, 'weight', amount=self.prune_rate)
                if module.bias is not None:
                    prune.l1_unstructured(module, 'bias', amount=self.prune_rate)
        
    def re_prune(self, model):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.BatchNorm2d):
                # parameters_to_prune += ((module, 'weight'),)
                prune.remove(module, 'weight')
                if module.bias is not None:
                    prune.remove(module, 'bias')
        
    
    # 训练
    def train(self,global_epoch, args, if_fail):  

        # self.model.to(device)

        # -----------------------------------------------------
        self.model.train()
        epoch_loss = []
        epoch_gradients = OrderedDict()
        delta_gradients = OrderedDict()
        # Initialize epoch_gradients with zero tensors
        for name, param in self.model.named_parameters():
            epoch_gradients[name] = torch.zeros_like(param.data)
            delta_gradients[name] = torch.zeros_like(param.data)
        # Set optimizer for the local updates
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.5)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        elif self.optimizer == 'sign':
            optimizer = signSGD_opt(self.model.parameters(), lr=args.learning_rate, rand_zero=1)
        
        if args.if_batch:
            for iter in range(self.local_ep):
                total_batch, correct_batch = 0.0, 0.0
                batch_loss = []
                local_iter = 0
                for batch_idx, (images, labels) in enumerate(self.trainloader):
                    local_iter += 1
                    if args.if_prune:
                        self.prune(self.model)                        # 剪枝
                    images, labels = images.to(self.device), labels.to(self.device)

                    self.model.zero_grad()
                    log_probs = self.model(images)
                    loss = self.criterion(log_probs, labels)
                    # print('loss:',loss)
                    loss.backward()

                    # batch acc-------------------------------------------
                    _, pred_labels_batch = torch.max(log_probs, 1)
                    pred_labels_batch = pred_labels_batch.view(-1)
                    correct_batch += torch.sum(torch.eq(pred_labels_batch, labels)).item()
                    total_batch += len(labels)
                    
                    # Accumulate gradients for each parameter
                    if args.if_prune:
                        for name, param in self.model.named_parameters():
                            # epoch_gradients[name[:-len('_orig')]] += param.grad.data
                            epoch_gradients[name[:-len('_orig')]] += param.grad.data
                    else:
                        for name, param in self.model.named_parameters():
                            # epoch_gradients[name] += param.grad.data
                            epoch_gradients[name] += param.grad.data
                    
                    # if self.if_converge == 0:
                    optimizer.step()

                    # if (batch_idx % 10 == 0):
                    #     print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #         global_epoch, iter, batch_idx * len(images),
                    #         len(self.trainloader.dataset),
                    #                             100. * batch_idx / len(self.trainloader), loss.item()))
                    # self.logger.add_scalar('loss', loss.item())
                    batch_loss.append(loss.item())
                    if args.if_prune:
                        self.re_prune(self.model)                        # 去剪枝
                    if local_iter >= self.N_u/(args.local_bs*args.scale):    break
                
                # feedback操作
                if self.if_feedback:
                    for key in epoch_gradients.keys():
                        epoch_gradients[key] = (1-args.feedback_coe)*epoch_gradients[key]+args.feedback_coe*self.feedback_grad[key]

                self.feedback_grad = copy.deepcopy(epoch_gradients)


                if args.if_compress:
                    epoch_gradients_compress, _ = fl_utils.compress(args=args, ordered_dict=epoch_gradients, matrix_S=self.matrix_S)
                    if args.if_quantize:
                        epoch_gradients_compress = torch.sign(epoch_gradients_compress)
                    
                

                if args.if_quantize:
                    for name in epoch_gradients.keys():
                        # if sum(batch_loss) / len(batch_loss) > args.epsilon:   # 检查loss是否达到设定标准（epsilon），如果已经达到，则停止训练（梯度直接置为0，不再更新模型）
                        epoch_gradients[name] = torch.sign(epoch_gradients[name])
                        

                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                train_accuracy = correct_batch/total_batch
                    # print('train accuracy: ', train_accuracy)
        else:
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                if args.if_prune:
                    self.prune(self.model)                        # 剪枝
                images, labels = images.to(self.device), labels.to(self.device)

                self.model.zero_grad()
                log_probs = self.model(images)

                loss = self.criterion(log_probs, labels)
                loss.backward()

                # SCG 
                if batch_idx == 0:
                    gold = [copy.deepcopy(parm.grad) for parm in self.model.parameters()]
                    delta = [torch.zeros_like(parm.grad) for parm in self.model.parameters()]

                # Compute conjugate gradient direction
                with torch.no_grad():
                    gnew = [parm.grad for parm in self.model.parameters()]
                    beta = [((gn**2).sum()/(go**2).sum()).item() for gn, go in zip(gnew, gold)]
                    delta = [-gn + bet * d for gn, bet, d in zip(gnew, beta, delta)]
                    
                # Accumulate gradients for each parameter
                if args.if_prune:
                    i = 0
                    for name, param in self.model.named_parameters():
                        epoch_gradients[name[:-len('_orig')]] += param.grad.data
                        delta_gradients[name] += delta[i]
                        i += 1
                else:
                    i = 0
                    for name, param in self.model.named_parameters():
                        epoch_gradients[name] += param.grad.data
                        delta_gradients[name] += delta[i]
                        i += 1

                if args.if_prune:
                    self.re_prune(self.model)                        # 去剪枝
                epoch_loss.append(loss.item()) 

                break    # 只训练一次，一个batch就结束

        # if sum(epoch_loss) / len(epoch_loss) < args.epsilon or self.if_converge == 1:
        #     self.if_converge = 1
        #     for name in epoch_gradients.keys():
        #         epoch_gradients[name] = torch.zeros_like(epoch_gradients[name])
        # ----------------------------------------------------
        g_maxs = [epoch_gradients[name].max().cpu() for name in epoch_gradients.keys()]
        g_mins = [epoch_gradients[name].min().cpu() for name in epoch_gradients.keys()]
        g_max = max(g_maxs)
        g_min = min(g_mins)

        if not args.if_batch:
            d_maxs = [delta[i].max().cpu() for i in range(len(delta))]   # 此处的d_max为当前client各分量的上界列表
            d_mins = [delta[i].min().cpu() for i in range(len(delta))]
            d_max = max(d_maxs)
            d_min = min(d_mins)
        
        
        # gradients = [self.quantize(p.grad.data, g_max, g_min) for p in self.model.parameters()]  # p为遍历g_u的每一个分量
        
        #----------------------------------------------------------------------------------------------------------
        
        max_weight = float('-inf')
        for name,param in self.model.named_parameters():
            if 'weight' in name:
                max_weight = max(max_weight, param.max().item())
        
        q_max = g_max
        
        xi = 64

        # 计算准确率
        # self.model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        with torch.no_grad():
            # self.model.eval()
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)

                # Inference
                outputs = self.model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
            # self.model.train()

        test_accuracy = correct/total
        
        # print('test accuracy: ', test_accuracy)
        
        # self.re_prune(self.model)                        # 剪枝
        # return gradients
        # print(g_max, max(q_max))
        
        
        
        
            # epoch_gradients = self.feedback_grad

        

        if ~if_fail and global_epoch >= 1:        
            self.if_feedback = True
        else:
            self.if_feedback = False
            
        if args.if_SCG:
            return delta_gradients, sum(epoch_loss) / len(epoch_loss), g_max, g_min, q_max, xi, test_accuracy, test_accuracy
        else:
            if args.if_batch:
                if args.if_compress:
                    return epoch_gradients_compress, sum(epoch_loss) / len(epoch_loss), g_max, g_min, q_max, xi, train_accuracy, test_accuracy
                else:
                    return epoch_gradients, sum(epoch_loss) / len(epoch_loss), g_max, g_min, q_max, xi, train_accuracy, test_accuracy
            else:
                return epoch_gradients, sum(epoch_loss) / len(epoch_loss), g_max, g_min, q_max, xi, test_accuracy, test_accuracy

    # 下面是照抄的SIGNSGD的train
   
class signSGD_opt(optim.Optimizer):

    def __init__(self, params, lr=0.01, rand_zero=True):
        defaults = dict(lr=lr)
        self.rand_zero = rand_zero
        super(signSGD_opt, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # take sign of gradient
                grad = torch.sign(p.grad)

                # randomise zero gradients to ±1
                if self.rand_zero:
                    grad[grad==0] = torch.randint_like(grad[grad==0], low=0, high=2)*2 - 1
                    assert not (grad==0).any()
                
                # make update
                p.data -= group['lr'] * grad

        return loss
       
def client_train(args, if_fail, matrix_S, queue, queue_model, event, idx, global_model, prune_rates, train_dataset, test_dataset, idxs, device, N_u):
    # global_model = global_model.to(device)
    # device_1 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device_2 = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    global_model.to(device)
    # for epo_id in range(args.num_epoch):
    #     if not queue_model.empty():
    #         model_state_dict = queue_model.get()
    #         global_model.load_state_dict(model_state_dict)
    #     if idx < args.num_clients/2:
    #         global_model.to(device_1)
    #         client = Client(model=copy.deepcopy(global_model), logger=None, prune_rate=prune_rates[idx], train_dataset=train_dataset, test_dataset=test_dataset, idxs=idxs, device=device_1, loss_func=args.loss_func, local_bs=args.local_bs, optimizer=args.optimizer, learning_rate=args.learning_rate, local_ep=args.local_ep, N_u = N_u)
    #     else:
    #         global_model.to(device_2)
    #         client = Client(model=copy.deepcopy(global_model), logger=None, prune_rate=prune_rates[idx], train_dataset=train_dataset, test_dataset=test_dataset, idxs=idxs, device=device_2, loss_func=args.loss_func, local_bs=args.local_bs, optimizer=args.optimizer, learning_rate=args.learning_rate, local_ep=args.local_ep, N_u = N_u)
    #     weight, loss, g_max, g_min, q_max, xi, train_accuracy, test_accuracy = client.train(global_epoch=epo_id, args=args)
    #     print(f"子进程 {idx}  第{epo_id}轮训练完成")
    #     queue.put((idx, weight, loss, g_max, g_min, q_max, xi, train_accuracy, test_accuracy))
    #     torch.cuda.empty_cache()
    #     event.wait()
    #     event.clear()
    if idx < args.num_clients/2:
        global_model.to(device_1)
        client = Client(model=copy.deepcopy(global_model), matrix_S=matrix_S, logger=None, prune_rate=prune_rates[idx], train_dataset=train_dataset, test_dataset=test_dataset, idxs=idxs, device=device_1, loss_func=args.loss_func, local_bs=args.local_bs, optimizer=args.optimizer, learning_rate=args.learning_rate, local_ep=args.local_ep, N_u = N_u)
    else:
        global_model.to(device_2)
        client = Client(model=copy.deepcopy(global_model), matrix_S=matrix_S, logger=None, prune_rate=prune_rates[idx], train_dataset=train_dataset, test_dataset=test_dataset, idxs=idxs, device=device_2, loss_func=args.loss_func, local_bs=args.local_bs, optimizer=args.optimizer, learning_rate=args.learning_rate, local_ep=args.local_ep, N_u = N_u)
    
    for epo_id in range(args.num_epoch):
        client.trainloader = DataLoader(fl_utils.DatasetSplit(train_dataset, idxs),
                                 batch_size=args.local_bs, shuffle=True)
        if not queue_model.empty():
            model_state_dict = queue_model.get()
            global_model.load_state_dict(model_state_dict)
        if idx < args.num_clients/2:
            global_model.to(device_1)
            client.model = copy.deepcopy(global_model)
        else:
            global_model.to(device_2)
            client.model = copy.deepcopy(global_model)
        weight, loss, g_max, g_min, q_max, xi, train_accuracy, test_accuracy = client.train(global_epoch=epo_id, args=args, if_fail=if_fail)
        # print(f"子进程 {idx}  第{epo_id}轮训练完成")
        queue.put((idx, weight, loss, g_max, g_min, q_max, xi, train_accuracy, test_accuracy))
        torch.cuda.empty_cache()
        event.wait()
        event.clear()

def PROPOSED(args, test_dataset, train_dataset, user_groups, N_us, file_path, transmit_power, S, prune_rates, computing_resources, I_us, h_us):
    start_time = time.time()
    L = 1/args.learning_rate

    

    # global_model = model.Net(n_feature=520, n_hidden1=64, n_hidden2=64, n_output=8)
    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = model.CNNMnist()
        
        elif args.dataset == 'cifar':
            # global_model = model.CNNCifar()
            global_model = model.Net_CNN()

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = model.MLP(dim_in=len_in, dim_hidden=64,
                            dim_out=10)
    elif args.model == 'res':
        global_model = PreActResNet18()

    else:
        exit('Error: unrecognized model')
    
   
    transmit_power, S ,prune_rates = fl_utils.adjust(args)

    # 生成观测矩阵
    matrix_S = fl_utils.generate_matrix(S, args.V)
    
    

    # global_model = mobilenet_v2(num_classes=10)
    global_model = global_model.to(device)

    # optimizer = torch.optim.Adam(global_model.parameters(),lr=learning_rate)
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(global_model.parameters(), lr=args.learning_rate, momentum=0.5, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(global_model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    elif args.optimizer == 'sign':
        optimizer = signSGD_opt(global_model.parameters(), lr=args.learning_rate, rand_zero=1)
        
    global_model.train()
    
    
    server = Server(model=global_model, matrix_S=matrix_S, test_dataset=test_dataset, optimizer=optimizer, device=device, learning_rate=args.learning_rate, loss_func ='crossentropy', N_us=N_us)
    # 创建客户端
    
    alpha = torch.tensor([1 for i_ in range(args.num_clients)])
    
    
    
    # 试训练一次，获取梯度的最大值和最小值；不用更新
    # print('试训练一次，获取梯度的最大值和最小值')
    client_gmaxs = []
    client_gmins = []
   
    for idx in range(args.num_clients):
        epo = 0
        # print('client:',idx)
        if idx < args.num_clients/2:
            client = Client(model=copy.deepcopy(global_model), matrix_S=matrix_S, logger=None, prune_rate=prune_rates[idx], train_dataset=train_dataset, test_dataset=test_dataset, idxs=user_groups[idx], device=device_1, loss_func=args.loss_func, local_bs=args.local_bs, optimizer=args.optimizer, learning_rate=args.learning_rate, local_ep=args.local_ep, N_u = N_us[idx])
        else:
            global_model.to(device_2)
            client = Client(model=copy.deepcopy(global_model), matrix_S=matrix_S, logger=None, prune_rate=prune_rates[idx], train_dataset=train_dataset, test_dataset=test_dataset, idxs=user_groups[idx], device=device_2, loss_func=args.loss_func, local_bs=args.local_bs, optimizer=args.optimizer, learning_rate=args.learning_rate, local_ep=args.local_ep, N_u = N_us[idx])
        gradient, loss, g_max, g_min, q_max, xi, train_accuracy, test_accuracy = client.train(global_epoch=epo, args=args, if_fail=0)
        client_gmaxs.append(g_max)
        client_gmins.append(g_min)
    
    # 用提出方法调整超参数
    

    # bitwidths = [32 for i in range(args.num_clients)]

    T_PROPOSED = fl_utils.calculate_T(args,transmit_power,S,prune_rates, alpha)
    E_PROPOSED, T_1max, T_1min, T_2max, T_2min, E_1max, E_1min, E_2max, E_2min  = fl_utils.calculate_E(args,transmit_power,S,prune_rates)
    dataamount_PROPOSED = fl_utils.calculate_dataamount(args,S,alpha)
    
    fl_utils.generate_alpha(alpha, transmit_power, args.num_clients, I_us, h_us, args.B_u, args.N0, args.waterfall_thre)
    a = sum(alpha)
    # print('alpha:',alpha)
    if torch.all(alpha==0):
        return
    # gama = fl_utils.Gamma(prune_rates,bitwidths,transmit_power, client_gmaxs, client_gmaxs, h_us, I_us, args.num_clients, N_us, args.B_u, args.N0, args.V, args.waterfall_thre, args.L, args.D)
    gama = 0
    datatosave2 = {'T':[T_PROPOSED],'E':[E_PROPOSED],'dataamount':[dataamount_PROPOSED],'a':[a], 'gamma':[gama],'T_calculate_max': [T_1max], 'T_calculate_min':[T_1min], 'T_transmit_max':[T_2max], 'T_transmit_min':[T_2min], 'E_calculate_max':[E_1max], 'E_calculate_min':[E_1min], 'E_transmit_max':[E_2max], 'E_transmit_min':[E_2min]}
    if args.pattern == 'baseline4':
        file_name2 = f'TE_baseline4_w{args.wer}_节点数{args.num_clients}_e{args.epsilon}_rho{args.prune_rate_max}_p{args.power_max}_topk{args.topk}'
        fl_utils.Save_to_Csv(data = datatosave2, file_name = file_name2, Save_format = 'csv',Save_type = 'row', file_path=file_path)
    elif args.pattern == 'baseline5':
        file_name2 = f'TE_baseline5_w{args.wer}_节点数{args.num_clients}_e{args.epsilon}_rho{args.prune_rate_max}_p{args.power_max}'
        fl_utils.Save_to_Csv(data = datatosave2, file_name = file_name2, Save_format = 'csv',Save_type = 'row', file_path=file_path)
    elif args.pattern == 'PROPOSED':
        file_name2 = f'TE_PROPOSED_w{args.wer}_节点数{args.num_clients}_e{args.epsilon}_rho{args.prune_rate_max}_p{args.power_max}_feedback{args.feedback_coe}'
        fl_utils.Save_to_Csv(data = datatosave2, file_name = file_name2, Save_format = 'csv',Save_type = 'row', file_path=file_path)
    
    
    array_power = str(transmit_power)
    # array_bitwidths = str(bitwidths)
    array_prune_rates = str(prune_rates)
    array_S = str(S)
    # 打开一个文件用于写入
    with open(f'array_proposed_step_w{args.wer}.txt', 'w') as f:
        # 写入数组字符串
        f.write('power:'+array_power+'\n')
        f.write('S:'+array_S+'\n')
        f.write('prune_rates:'+array_prune_rates+'\n')
        
    
    #--------------------------------------------------开始训练---------------------------------
    losses_train = []
    losses_test = []
    accuracies_train = []
    accuracies_test = []
    energy_consumption = []
    time_consumption = []
    if_converge = 0
    # epo用于记录训练轮数，但其实每个iteration都会更新一次，所以其实没啥用
    epo = 0

    # step1,step2,step3用于记录训练到0.6,0.7,0.8（可调节）的轮数
    step1 = 0
    step2 = 0
    step3 = 0
    step_count = 0

    queue = mp.Queue()
    queue_model = mp.Queue()
    event = mp.Event()
    processes = []

    # 开启多进程训练
    for idx in range(args.num_clients):
        process = mp.Process(target=client_train, args=(args, alpha[idx], matrix_S, queue, queue_model, event, idx, global_model, prune_rates, train_dataset, test_dataset, user_groups[idx], device, N_us[idx] ))
        processes.append(process)
        process.start()

    # 每个全局轮次的操作
    for epo in range(args.num_epoch):
        local_weights, local_losses = [], []
        local_gradient = {}
        list_acc = []
        list_test_acc = []
        
        print(f'\n | Global Training Round : {epo+1} |\n')
        client_gmaxs = []
        client_gmins = []
        for id in range(args.num_clients):
            worker_id, gradient, loss, g_max, g_min, q_max, xi, train_accuracy, test_accuracy = queue.get()
            # print('get',id)
            local_losses.append(copy.deepcopy(loss))
            local_gradient[worker_id]=copy.deepcopy(gradient)  
            list_acc.append(train_accuracy) 
            list_test_acc.append(test_accuracy)
            
            client_gmaxs.append(g_max)
            client_gmins.append(g_min)
    
        
        
        acc_train = max(list_acc)
        
        loss_avg = sum(local_losses) / len(local_losses)
        

        # 打印每个iteration的部分训练情况，可注释
        # print(f'epoch:{epo}, [{i + 1}] train_loss of client 0: {client_loss[0]}, accuracy of client 0: {client_acces[0]}, g_max[0]: {client_gmaxs[0]}, q_max[0]: {client_qmaxs[0][0]}, alpha: {alpha}')                                 # 打印训练时的loss
        if torch.all(alpha==0):
            print('alpha all zero')

        # 更新alpha，即各个client是否成功传输    
        fl_utils.generate_alpha(alpha, transmit_power, args.num_clients, I_us, h_us, args.B_u, args.N0, args.waterfall_thre)
            
        # 服务器操作（梯度聚合+更新）
        if args.if_SCG:
            agg_deltas = server.aggregate_SCG(gradients=local_gradient, alpha=alpha.to(device))
            server.update_SCG(avg_deltas=agg_deltas)
        else:
            agg_grads = server.aggregate(gradients=local_gradient, alpha=alpha.to(device), args=args)   # 梯度聚合   
            server.update(avg_gradients=agg_grads,if_converge=if_converge)                                                           # 梯度更新
        
        #------------------测试-------------------------------
        acc_test = server.test()
       
        

        print(f'epoch:{epo+1}, [{epo + 1}] train_loss: {loss_avg},acccuracy_train:{acc_train}, accuracy_v from server: {acc_test}, g_max[0]: {client_gmaxs[0]}, alpha: {alpha}') 
        losses_train.append(loss_avg)
        accuracies_train.append(acc_train)
        accuracies_test.append(acc_test)
        if loss_avg > args.epsilon and if_converge == 0:  # 检查loss是否达到设定标准（epsilon），如果已经达到，则停止训练（即不再有能耗时延开销）
            time_consumption.append(T_PROPOSED*epo)
            energy_consumption.append(E_PROPOSED*epo)
        else:
            time_consumption.append(time_consumption[-1])
            energy_consumption.append(energy_consumption[-1])
            if_converge = 1

        end_time = time.time()
        train_time = end_time - start_time
        
        print(f'已训练时间：{train_time/60}分钟')

        # global_model.cpu()
        for _ in range(args.num_clients):
            queue_model.put(global_model.state_dict())
        global_model.to(device)
        event.set()  

    for process in processes:
        process.join()
    torch.cuda.empty_cache()

    end_time = time.time()
    train_time = end_time - start_time
        
    print(f'训练时间：{train_time/60}分钟')

    losses_train_PROPOSED = torch.tensor(losses_train)
    accuracies_train_PROPOSED = torch.tensor(accuracies_train)
    accuracies_test_PROPOSED = torch.tensor(accuracies_test)

    datatosave = {'losses_train':losses_train_PROPOSED, 'accuracies_train':accuracies_train_PROPOSED, 'accuracies_test':accuracies_test_PROPOSED, 'time_consumption':time_consumption, 'energy_consumption':energy_consumption}
    if args.pattern == 'baseline4':
        file_name1 = f'LA_baseline4_w{args.wer}_节点数{args.num_clients}_e{args.epsilon}_rho{args.prune_rate_max}_p{args.power_max}_topk{args.topk}'
        fl_utils.Save_to_Csv(data = datatosave, file_name = file_name1, Save_format = 'csv',Save_type = 'row', file_path=file_path)
    elif args.pattern == 'baseline5':
        file_name1 = f'LA_baseline5_w{args.wer}_节点数{args.num_clients}_e{args.epsilon}_rho{args.prune_rate_max}_p{args.power_max}'
        fl_utils.Save_to_Csv(data = datatosave, file_name = file_name1, Save_format = 'csv',Save_type = 'row', file_path=file_path)
    elif args.pattern == 'PROPOSED':
        file_name1 = f'LA_PROPOSED_w{args.wer}_节点数{args.num_clients}_e{args.epsilon}_rho{args.prune_rate_max}_p{args.power_max}_feedback{args.feedback_coe}'
        fl_utils.Save_to_Csv(data = datatosave, file_name = file_name1, Save_format = 'csv',Save_type = 'row', file_path=file_path)

    return
