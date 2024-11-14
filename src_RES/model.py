import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parameters_to_vector
from skopt import Optimizer
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.utils.prune as prune
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import scale
from torch.autograd import Variable
import argparse
import fl_utils

# 得到全局train_dataset和test_dataset，以及随机得到各个节点的数据集序号字典
def get_dataset(args):
    if args.dataset == 'cifar':
        data_dir = './data/cifar/'
        if args.if_aug:
            apply_transform = transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                        ])
        else:
            apply_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        normal_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))])
        
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                    transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                    transform=normal_transform)
        user_groups = fl_utils.cifar_iid(train_dataset, args)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = './data/mnist/'
        else:
            data_dir = './data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)
        user_groups = fl_utils.mnist_iid(train_dataset, args)

    return train_dataset, test_dataset, user_groups

class Net_finger(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2,n_output):
        super(Net_finger, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        # self.drop1 = torch.nn.Dropout(0.7)
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        # self.drop2 = torch.nn.Dropout(0.7)
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden2)
        # self.drop3 = torch.nn.Dropout(0.1)
        self.hidden4 = torch.nn.Linear(n_hidden2, n_hidden2)

        # # self.drop4 = torch.nn.Dropout(0.1)
        self.hidden5 = torch.nn.Linear(n_hidden2, n_hidden2)
        self.drop5 = torch.nn.Dropout(0.1)
        self.hidden6 = torch.nn.Linear(n_hidden2, n_hidden2)
        # # self.drop6 = torch.nn.Dropout(0.1)
        self.hidden7 = torch.nn.Linear(n_hidden2, n_hidden2)
        # # self.drop7 = torch.nn.Dropout(0.1)
        self.hidden8 = torch.nn.Linear(n_hidden2, n_hidden2)
        self.drop8 = torch.nn.Dropout(0.1)
        # self.bn1 = torch.nn.BatchNorm1d(n_hidden)
        # self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden2)
        # self.predict = torch.nn.Linear(n_hidden2, n_output)   # output layer

        self.num = torch.nn.Linear(n_hidden2, 3)
        self.layer = torch.nn.Linear(n_hidden2, 5)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        #x = self.drop1(x)
        x = F.relu(self.hidden2(x))
        #x = self.drop2(x)
        xx = F.relu(self.hidden3(x))
        #xx = self.drop3(xx)
        xx = F.relu(self.hidden4(xx))
        #xx = self.drop4(xx)
        xx = F.relu(self.hidden5(xx))
        xx = self.drop5(xx)

        x=xx+x

        xxx = F.relu(self.hidden6(x))
        #xxx = self.drop6(xxx)
        xxx = F.relu(self.hidden7(xxx))
        #xxx = self.drop7(xxx)
        xxx = F.relu(self.hidden8(xxx))
        xxx = self.drop8(xxx)

        x=xxx+x

        # x = self.bn1(x)
        #x = F.relu(self.hidden3(x))
        

        num = self.num(x)
        # num = torch.softmax(num,dim=1)

        layer = self.layer(x)
        # layer = torch.softmax(layer,dim=1)

        return num, layer

class CNNCifar(nn.Module):
    def __init__(self):
        super(CNNCifar, self).__init__()

        # 增加卷积层和批归一化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)


        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, 512 * 2 * 2)  # 展平

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNMnist(nn.Module):
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.sigmoid(x)
        x = self.layer_hidden(x)
        return x

class Net_CNN(nn.Module):

    def __init__(self):
        super(Net_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv7 = nn.Conv2d(128, 128, 1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.conv8 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv9 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 1, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.conv11 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv12 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv13 = nn.Conv2d(512, 512, 1, padding=1)
        self.pool5 = nn.MaxPool2d(2, 2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()

        self.fc14 = nn.Linear(512 * 4 * 4, 1024)
        self.drop1 = nn.Dropout2d()
        self.fc15 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout2d()
        self.fc16 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.pool5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        # print(" x shape ",x.size())
        x = x.view(-1, 512 * 4 * 4)
        x = F.relu(self.fc14(x))
        x = self.drop1(x)
        x = F.relu(self.fc15(x))
        x = self.drop2(x)
        x = self.fc16(x)

        return x 

if __name__ == '__main__':
    num=10
    train_dataset, test_dataset, user_groups = get_dataset(num)
    print(len(train_dataset))
    print(len(test_dataset))
    print(len(user_groups))
