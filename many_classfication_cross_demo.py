# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
    交叉验证,dropout(处理过拟合)
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 200),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(200, 200),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(200, 10),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

# 分割数据为训练集和验证集
train_dataset, validation_dataset = random_split(datasets.MNIST("./data", train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ])), [50000, 10000])

train_data_batch = DataLoader(train_dataset, batch_size=200, shuffle=True)
validation_data_batch = DataLoader(validation_dataset, batch_size=200, shuffle=True)

test_data_batch = DataLoader(datasets.MNIST("./data", train=False,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.1307,), (0.3081,))
                                       ])), batch_size=200, shuffle=True)


device = torch.device("cpu:0")
mlp = MLP().to(device)
# weight_decay:权重衰减 momentum:动量（惯性）
# optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-3, weight_decay=1e-2, momentum=1e-1)
optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-2)
cel = torch.nn.CrossEntropyLoss().to(device)

for n in range(20):
    for batch_idx, (train_data, train_target) in enumerate(train_data_batch):
        train_data = train_data.view(-1, 784)
        train_data, train_target = train_data.to(device), train_target.to(device)
        logits = mlp(train_data)
        loss = cel(logits, train_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            print("num : {}, train batch : {}, loss : {}".format(n + 1, batch_idx + 1, loss.item()))

    correct = 0
    for validation_data, validation_target in validation_data_batch:
        validation_data = validation_data.view(-1, 784)
        validation_data, validation_target = validation_data.to(device), validation_target.to(device)
        test_logits = mlp(validation_data)
        predict = test_logits.data.max(dim=1)[1]
        # predict = test_logits.data.argmax(dim=1)
        # correct += int(predict.eq(test_target.data).sum().item())
        correct += torch.eq(validation_target.data, predict).sum()

    print("validation accuracy : {}%".format((correct * 100./len(test_data_batch.dataset))))

correct = 0
for validation_data, validation_target in validation_data_batch:
    validation_data = validation_data.view(-1, 784)
    validation_data, validation_target = validation_data.to(device), validation_target.to(device)
    test_logits = mlp(validation_data)
    predict = test_logits.data.max(dim=1)[1]
    # predict = test_logits.data.argmax(dim=1)
    # correct += int(predict.eq(test_target.data).sum().item())
    correct += torch.eq(validation_target.data, predict).sum()

print("test accuracy : {}%".format((correct * 100./len(test_data_batch.dataset))))