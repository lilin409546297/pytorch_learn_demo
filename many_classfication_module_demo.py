# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
    visdom,基础模型,正则化,动量,lr衰减
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from visdom import Visdom

# pip install visdom
# python -m visdom.server / visdom

class MLP(torch.nn.Module):
    def __init__(self) -> None:
        super(MLP, self).__init__()

        # inplace=True的作用覆盖前面的值，较少内存消耗
        self.model = torch.nn.Sequential(
            torch.nn.Linear(784, 200),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(200, 10),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)

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
device = torch.device("cpu:0")
mlp = MLP().to(device)
# weight_decay:权重衰减 momentum:动量（惯性）
optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-3, weight_decay=1e-2, momentum=1e-1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
cel = torch.nn.CrossEntropyLoss().to(device)
# viz = Visdom()
# viz.line([0.], [0,], win="train_loss", opts=dict(title="train_loss"))
# viz.line([0., 0.], [0,], win="test_accuracy", opts=dict(title="test_accuracy"))

for n in range(10):
    for batch_idx, (train_data, train_target) in enumerate(train_data_batch):
        train_data = train_data.view(-1, 784)
        train_data, train_target = train_data.to(device), train_target.to(device)
        logits = mlp(train_data)
        loss = cel(logits, train_target)
        optimizer.zero_grad()
        loss.backward()
        # viz.line([loss.item()], [0], win="train_loss", update="append")
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            scheduler.step(loss.item())
            print("num : {}, train batch : {}, loss : {}".format(n + 1, batch_idx + 1, loss.item()))


    correct = 0
    for test_data, test_target in test_data_batch:
        test_data = test_data.view(-1, 784)
        test_data, test_target = test_data.to(device), test_target.to(device)
        test_logits = mlp(test_data)
        predict = test_logits.data.max(dim=1)[1]
        # predict = test_logits.data.argmax(dim=1)
        # correct += int(predict.eq(test_target.data).sum().item())
        correct += torch.eq(test_target.data, predict).sum()

    # viz.image([test_data.view(-1, 28, 28)], win="x")
    # viz.text(str(predict.detach.cpu().numpy()), win="predict", opts=dict(title="predict"))
    # viz.line([(correct * 100./len(test_data_batch.dataset))], [0], win="test_accuracy", update="append")
    print("accuracy : {}%".format((correct * 100./len(test_data_batch.dataset))))