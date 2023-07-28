#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
import torchvision


class NpzDataset(torch.utils.data.Dataset):
    """Create a torch image dataset from npz files"""

    def __init__(self, data_file, target_file, data_shape, target_shape):

        self.data_memmap = np.memmap(data_file, dtype="float32", mode="r", shape=data_shape)
        self.target_memmap = np.memmap(target_file, dtype="int32", mode="r", shape=target_shape)

    def __len__(self):
        return self.data_memmap.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(np.copy(self.data_memmap[index])), torch.from_numpy(np.copy(self.target_memmap[index]))


class ArrayImageDataset(torch.utils.data.Dataset):
    """Create a torch image dataset from numpy arrays"""

    def __init__(self, image_array, target_array, transform=None):

        self.images = image_array
        self.targets = target_array
        if transform is None:
            transform = torchvision.transforms.transforms.Compose([
                torchvision.transforms.transforms.ToPILImage(),
                torchvision.transforms.transforms.ToTensor()
            ])
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        if self.transform is not None:
            image = self.transform(image)
        target = self.targets[index]
        return image, target