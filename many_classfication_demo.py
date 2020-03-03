# !/usr/bin/python
# -*- coding: UTF-8 -*-
"""
    基本的数据训练过程
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

w1 = torch.randn(200, 784, requires_grad=True)
b1 = torch.zeros(200, requires_grad=True)
w2 = torch.randn(200, 200, requires_grad=True)
b2 = torch.zeros(200, requires_grad=True)
w3 = torch.randn(10, 200, requires_grad=True)
b3 = torch.zeros(10, requires_grad=True)

w1 = torch.nn.init.kaiming_normal_(w1)
w2 = torch.nn.init.kaiming_normal_(w2)
w3 = torch.nn.init.kaiming_normal_(w3)

def forward(x):
    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x@w3.t() + b3
    x = F.relu(x)
    return x

optimizer = torch.optim.Adam([w1, w2, w3, b1, b2, b3], lr=1e-3)
cel = torch.nn.CrossEntropyLoss()

train_data_batch = DataLoader(datasets.MNIST("./data", train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ])), batch_size=200, shuffle=True)
test_data_batch = DataLoader(datasets.MNIST("./data", train=False,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ])), batch_size=200, shuffle=True)

for n in range(10):
    for batch_idx, (train_data, train_target) in enumerate(train_data_batch):
        train_data = train_data.view(-1, 784)
        logits = forward(train_data)
        loss = cel(logits, train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            print("num : {}, train batch : {}, loss : {}".format(n + 1, batch_idx + 1, loss.item()))

    correct = 0
    for test_data, test_target in test_data_batch:
        test_data = test_data.view(-1, 784)
        test_logits = forward(test_data)
        predict = test_logits.data.max(dim=1)[1]
        correct += int(predict.eq(test_target.data).sum().item())

    print("accuracy : {}".format((correct/len(test_target.data))))