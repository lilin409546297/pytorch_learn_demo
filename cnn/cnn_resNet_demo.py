# !/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

from cnn.lenet5 import LeNet5
from cnn.resnet import ResNet18


def cnn_train():
    # 1、准备数据集
    cifar_dataset = datasets.CIFAR10("./cifar", train=True, download=True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    cifar_train = DataLoader(cifar_dataset, batch_size=32, shuffle=True)

    # 2、加载模型、loss、优化器
    device = torch.device("cpu")
    net = ResNet18().to(device)
    cel = nn.CrossEntropyLoss().to(device)
    optimer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # 3、训练
    if os.path.exists("./model/resNet18.pkl"):
        net.load_state_dict(torch.load("./model/resNet18.pkl"))

    for num in range(100):
        for idx, (cifar_train_data, cifar_train_target) in enumerate(cifar_train):
            cifar_train_data, cifar_train_target = cifar_train_data.to(device), cifar_train_target.to(device)
            logits = net(cifar_train_data)
            loss = cel(logits, cifar_train_target)
            optimer.zero_grad()
            loss.backward()
            optimer.step()
            if idx % 100 == 0:
                print("num : {}, batch num : {}, loss : {}".format((num + 1), idx, loss.item()))
        if not os.path.exists("./model"):
            os.mkdir("./model")
        torch.save(net.state_dict(), "./model/resNet18.pkl")


def cnn_test():
    # 1、加载模型、设备
    device = torch.device("cpu")
    net = LeNet5().to(device)
    net.load_state_dict(torch.load("./model/resNet18.pkl"))

    # 2、准备测试数据
    cifar_dataset = datasets.CIFAR10("./cifar", train=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]))
    cifar_test = DataLoader(cifar_dataset, batch_size=32, shuffle=True)

    # 3、测试
    predict_true_num = 0
    dataset_num = 0
    for cifar_test_data, cifar_test_target in cifar_test:
        cifar_test_data, cifar_test_target = cifar_test_data.to(device), cifar_test_target.to(device)
        predict = net(cifar_test_data).argmax(dim=1)
        predict_true =torch.eq(predict, cifar_test_target).float().sum().item()
        predict_true_num += predict_true
        dataset_num += len(cifar_test_data)

    print("accuracy: {}%".format(predict_true_num * 100. / dataset_num))



if __name__ == '__main__':
    cnn_train()
    # cnn_test()