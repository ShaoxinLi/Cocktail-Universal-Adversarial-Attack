#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torchvision
from PIL import Image


class CustomizedFashionMNIST(torchvision.datasets.FashionMNIST):

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image.numpy(), mode="L")
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class CustomizedImageNet(torchvision.datasets.ImageFolder):

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        target = self.targets[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def mnist_dataset(dataset_dir, is_train, transform=None, download=True):

    if transform is None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
    dataset = torchvision.datasets.MNIST(dataset_dir, train=is_train, transform=transform, download=download)
    return dataset


def fmnist_dataset(dataset_dir, is_train, transform=None, download=True):

    if transform is None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
    dataset = CustomizedFashionMNIST(dataset_dir, train=is_train, transform=transform, download=download)
    return dataset


def cifar10_dataset(dataset_dir, is_train, transform=None, download=True):

    if transform is None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
    dataset = torchvision.datasets.CIFAR10(dataset_dir, train=is_train, transform=transform, download=download)
    return dataset


def cifar100_dataset(dataset_dir, is_train, transform=None, download=True):

    if transform is None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
    dataset = torchvision.datasets.CIFAR100(dataset_dir, train=is_train, transform=transform, download=download)
    return dataset


def imagenet_dataset(dataset_dir, is_train, transform=None):

    if is_train:
        dataset_dir = os.path.join(dataset_dir, "train")
    else:
        dataset_dir = os.path.join(dataset_dir, "val")
    if transform is None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ])
    dataset = CustomizedImageNet(dataset_dir, transform)
    return dataset

