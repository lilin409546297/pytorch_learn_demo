# !/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
from torch import nn

class LeNet5(nn.Module):
    def __init__(self) -> None:
        super(LeNet5, self).__init__()

        # 卷积
        self.cnn_unit = nn.Sequential(
            # 1、第一层
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),

            # 2、第二层
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        )

        # 全连接
        self.fc_unit = nn.Sequential(
            nn.Linear(32 * 26 * 26, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.cnn_unit(x)
        x = x.view(x.size(0), 32 * 26 * 26)
        logits = self.fc_unit(x)
        return logits