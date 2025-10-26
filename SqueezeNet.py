# -*- coding: utf-8 -*-
"""


@author: rouxm
"""


import torch
import torch.nn as nn


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.expand_activation(torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1))

class SqueezeNet(nn.Module):
    def __init__(self, num_classes=10): #nombre de classes attendues par SqueezeNet
        super(SqueezeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=7, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(96, 16, 64, 64),
            #Fire(128, 16, 64, 64),
            #Fire(128, 32, 128, 128),
            #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            #Fire(256, 32, 128, 128),
            #Fire(256, 48, 192, 192),
            #Fire(384, 48, 192, 192),
            #Fire(384, 64, 256, 256),
            #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            #Fire(128, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), -1)
