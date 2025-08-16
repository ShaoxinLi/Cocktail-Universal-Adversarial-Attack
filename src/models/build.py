#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torchvision
import numpy as np
from .cifar_resnet import resnet20, resnet56
from .cifar_vgg import vgg11_bn, vgg16_bn, vgg19_bn
from .convnet import convnet


def get_network(net_arch, n_classes=1000, n_channels=3, pretrained=False, finetune=False):
    """Get a network"""

    if net_arch == "convnet":
        net = convnet(num_classes=n_classes, num_channels=n_channels)
    elif net_arch == "resnet20":
        net = resnet20(num_classes=n_classes)
    elif net_arch == "resnet56":
        net = resnet56(num_classes=n_classes)
    elif net_arch == "vgg11_cifar":
        net = vgg11_bn(num_classes=n_classes)
    elif net_arch == "vgg16_cifar":
        net = vgg16_bn(num_classes=n_classes)
    elif net_arch == "vgg19_cifar":
        net = vgg19_bn(num_classes=n_classes)
    elif net_arch == "alexnet":
        net = torchvision.models.alexnet(pretrained=pretrained, progress=True)
    elif net_arch == "resnet18":
        net = torchvision.models.resnet18(pretrained=pretrained, progress=True)
        if pretrained and finetune:
            set_parameter_requires_grad(net, requires_grad=False)
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, n_classes)
        if n_classes != 1000:
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, n_classes)
    elif net_arch == "resnet50":
        net = torchvision.models.resnet50(pretrained=pretrained, progress=True)
        if pretrained and finetune:
            set_parameter_requires_grad(net, requires_grad=False)
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, n_classes)
        if n_classes != 1000:
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, n_classes)
    elif net_arch == "resnet152":
        net = torchvision.models.resnet152(pretrained=pretrained, progress=True)
        if pretrained and finetune:
            set_parameter_requires_grad(net, requires_grad=False)
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, n_classes)
        if n_classes != 1000:
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, n_classes)
    elif net_arch == "vgg16":
        net = torchvision.models.vgg16_bn(pretrained=pretrained, progress=True)
        if pretrained and finetune:
            set_parameter_requires_grad(net, requires_grad=False)
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = torch.nn.Linear(num_ftrs, n_classes)
        if n_classes != 1000:
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = torch.nn.Linear(num_ftrs, n_classes)
    elif net_arch == "vgg19":
        net = torchvision.models.vgg19_bn(pretrained=pretrained, progress=True)
        if pretrained and finetune:
            set_parameter_requires_grad(net, requires_grad=False)
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = torch.nn.Linear(num_ftrs, n_classes)
        if n_classes != 1000:
            num_ftrs = net.classifier[6].in_features
            net.classifier[6] = torch.nn.Linear(num_ftrs, n_classes)
    elif net_arch == "inception_v3":
        net = torchvision.models.inception_v3(pretrained=pretrained, progress=True)
        if pretrained and finetune:
            set_parameter_requires_grad(net, requires_grad=False)
            # Handle the auxilary net
            num_ftrs = net.AuxLogits.fc.in_features
            net.AuxLogits.fc = torch.nn.Linear(num_ftrs, n_classes)
            # Handle the primary net
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, n_classes)
        if n_classes != 1000:
            num_ftrs = net.AuxLogits.fc.in_features
            net.AuxLogits.fc = torch.nn.Linear(num_ftrs, n_classes)
            # Handle the primary net
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, n_classes)
    elif net_arch == "squeezenet":
        net = torchvision.models.squeezenet1_1(pretrained=pretrained, progress=True)
        if pretrained and finetune:
            set_parameter_requires_grad(net, requires_grad=False)
            net.classifier[1] = torch.nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
        if n_classes != 1000:
            net.classifier[1] = torch.nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    elif net_arch == "shufflenet":
        net = torchvision.models.shufflenet_v2_x0_5(pretrained=pretrained, progress=True)
        if pretrained and finetune:
            set_parameter_requires_grad(net, requires_grad=False)
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, n_classes)
        if n_classes != 1000:
            num_ftrs = net.fc.in_features
            net.fc = torch.nn.Linear(num_ftrs, n_classes)
    else:
        assert False, f"Net arch {net_arch} is not supported yet."
    return net


def set_parameter_requires_grad(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad


def get_n_parameters(net):
    return sum(p.numel() for p in net.parameters())


def get_n_trainable_parameters(net):
    net_parameters = filter(lambda p: p.requires_grad is True, net.parameters())
    return sum([np.prod(p.size()) for p in net_parameters])


def get_n_non_trainable_parameters(net):
    net_parameters = filter(lambda p: p.requires_grad is False, net.parameters())
    return sum([np.prod(p.size()) for p in net_parameters])


class Normalize(torch.nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, inputs):
        size = inputs.size()
        x = inputs.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x