#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


class AdveraryGeneratorImageNet(torch.nn.Module):
    def __init__(self, nz, xi=6./255):

        super(AdveraryGeneratorImageNet, self).__init__()
        self.xi = xi
        self.nz = nz
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_channels=nz, out_channels=1024, kernel_size=4, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(256, 128, 4, 2, 2, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(128, 64, 4, 2, 2, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(64, 3, 4, 4, 4, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.xi * self.main(x) # Scaling of ε


class AdveraryGeneratorCIFAR(torch.nn.Module):
    def __init__(self, nz, xi=6./255):

        super(AdveraryGeneratorCIFAR, self).__init__()
        self.xi = xi
        self.nz = nz
        self.main = torch.nn.Sequential(

            torch.nn.ConvTranspose2d(in_channels=nz, out_channels=256, kernel_size=4, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.xi * self.main(x) # Scaling of ε