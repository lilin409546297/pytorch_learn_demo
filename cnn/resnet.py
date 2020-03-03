# !/usr/bin/python
# -*- coding: UTF-8 -*-
from torch import nn
from torch.nn import functional as F

class ResBlk(nn.Module):

    def __init__(self, ch_in, ch_out, stride=1):
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)

        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out  = self.extra(x) + out
        return out

class ResNet18(nn.Module):

    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64)
        )

        self.blk1 = ResBlk(64, 128, stride=2)
        self.blk2 = ResBlk(128, 256, stride=2)
        self.blk3 = ResBlk(256, 512, stride=2)
        self.blk4 = ResBlk(512, 512, stride=2)

        self.outLayer = nn.Linear(512 * 1 * 1, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.blk1(out)
        out = self.blk2(out)
        out = self.blk3(out)
        out = self.blk4(out)

        out = F.adaptive_avg_pool2d(out, [1,1])

        out = out.view(out.size(0), -1)

        out = self.outLayer(out)
        return out
