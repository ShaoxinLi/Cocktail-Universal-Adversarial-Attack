#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from .benchmark import *


def get_image_dataset_info(dataset_name):
    """Get the specific information about an image dataset"""

    if dataset_name == "mnist":
        n_classes = 10
        img_size = 28
        n_channels = 1
        mean = [0.130]
        std = [0.308]
    elif dataset_name == "fmnist":
        n_classes = 10
        img_size = 28
        n_channels = 1
        mean = [0.286]
        std = [0.353]
    elif dataset_name == "cifar10":
        n_classes = 10
        img_size = 32
        n_channels = 3
        mean = [0.491, 0.482, 0.446]
        std = [0.247, 0.243, 0.261]
    elif dataset_name == "cifar100":
        n_classes = 100
        img_size = 32
        n_channels = 3
        mean = [0.507, 0.486, 0.441]
        std = [0.267, 0.256, 0.276]
    elif dataset_name == "imagenet10":
        n_classes = 1000
        img_size = 224
        n_channels = 3
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset_name == "sub_imagenet":
        n_classes = 5
        img_size = 224
        n_channels = 3
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        assert False, f"Dataset {dataset_name} is not supported yet."
    return n_channels, img_size, n_classes, (mean, std)


def get_image_dataset(data_root_dir, dataset_name, is_train=True, use_classes=None, train_images_per_class=-1,
                      validation_split=-1, transform=None, dataset_dir=None):
    """Get the image dataset(s)"""

    # load the original dataset
    dataset_dir = os.path.join(data_root_dir, dataset_name) if dataset_dir is None else dataset_dir
    if dataset_name == "mnist":
        dataset = mnist_dataset(dataset_dir=dataset_dir, is_train=is_train, transform=transform, download=True)
    elif dataset_name == "fmnist":
        dataset = fmnist_dataset(dataset_dir=dataset_dir, is_train=is_train, transform=transform, download=True)
    elif dataset_name == "cifar10":
        dataset = cifar10_dataset(dataset_dir=dataset_dir, is_train=is_train, transform=transform, download=True)
    elif dataset_name == "cifar100":
        dataset = cifar100_dataset(dataset_dir=dataset_dir, is_train=is_train, transform=transform, download=True)
    elif dataset_name == "imagenet10" or dataset_name == "sub_imagenet":
        if is_train:
            dataset = imagenet_dataset(dataset_dir=dataset_dir, is_train=True, transform=transform)
        else:
            dataset = imagenet_dataset(dataset_dir="/home/share/Datasets/imagenet10", is_train=False, transform=transform)
    else:
        assert False, f"Dataset {dataset_name} is not supported yet."

    # convert classes of string types to integers (i.e., airplane --> 0)
    dataset.class_names = dataset.classes
    dataset.classes = list(range(len(dataset.classes)))

    # convert targets of list type to tensors
    if isinstance(dataset.targets, list):
        dataset.targets = torch.tensor(dataset.targets, dtype=torch.int64)

    # process the training dataset if it is required
    if is_train:

        # downsampling the images of each class
        if train_images_per_class > 0:

            # keep the original classes, class names and targets
            orig_classes = dataset.classes
            orig_class_names = dataset.class_names
            orig_targets = dataset.targets

            # get the idxs of sampled images
            sampled_image_idxs = []
            for cls in dataset.classes:
                idxs_wrt_cls = np.argwhere(dataset.targets.numpy() == cls)[:, 0]
                sampled_image_idxs.append(idxs_wrt_cls[:train_images_per_class])
            sampled_image_idxs = np.concatenate(sampled_image_idxs)

            # get the downsampled training set
            dataset = torch.utils.data.Subset(dataset, sampled_image_idxs)
            dataset.classes = orig_classes
            dataset.class_names = orig_class_names
            dataset.targets = orig_targets[sampled_image_idxs]

        # use images of specific classes
        if use_classes is not None:
            assert set(use_classes).issubset(set(dataset.classes))

            # keep the original class names and targets
            orig_class_names = dataset.class_names
            orig_targets = dataset.targets

            # get the idxs of samples of specific classes
            sampled_image_idxs = []
            for cls in use_classes:
                idxs_wrt_cls = np.argwhere(dataset.targets.numpy() == cls)[:, 0]
                sampled_image_idxs.append(idxs_wrt_cls)
            sampled_image_idxs = np.concatenate(sampled_image_idxs)

            # get the training set of specific classes
            dataset = torch.utils.data.Subset(dataset, sampled_image_idxs)
            dataset.classes = use_classes
            dataset.class_names = [orig_class_names[i] for i in use_classes]
            dataset.targets = orig_targets[sampled_image_idxs]

        # split the dataset into a training set and a validating set
        if validation_split > 0:
            assert 0.0 < validation_split < 1.0

            # keep the original classes, class names and targets
            orig_classes = dataset.classes
            orig_class_names = dataset.class_names
            orig_targets = dataset.targets

            # get the idxs of training and validating
            idxs_of_train, idxs_of_val = [], []
            for cls in dataset.classes:
                idxs_wrt_cls = np.argwhere(dataset.targets.numpy() == cls)[:, 0]
                idxs_of_val.append(idxs_wrt_cls[:int(len(idxs_wrt_cls) * validation_split)])
                idxs_of_train.append(idxs_wrt_cls[int(len(idxs_wrt_cls) * validation_split):])
            idxs_of_train = np.concatenate(idxs_of_train)
            idxs_of_val = np.concatenate(idxs_of_val)

            # get the training set
            train_dataset = torch.utils.data.Subset(dataset, idxs_of_train)
            train_dataset.classes = orig_classes
            train_dataset.class_names = orig_class_names
            train_dataset.targets = orig_targets[idxs_of_train]

            # get the validating set
            val_dataset = torch.utils.data.Subset(dataset, idxs_of_val)
            val_dataset.classes = orig_classes
            val_dataset.class_names = orig_class_names
            val_dataset.targets = orig_targets[idxs_of_val]
            return train_dataset, val_dataset
        else:
            return dataset, None

    # process the testing dataset if it is required
    else:
        # use images of specific classes
        if use_classes is not None:
            assert set(use_classes).issubset(set(dataset.classes))

            # keep the original class names and targets
            orig_class_names = dataset.class_names
            orig_targets = dataset.targets

            # get the idxs of samples of specific classes
            sampled_image_idxs = []
            for cls in use_classes:
                idxs_wrt_cls = np.argwhere(dataset.targets.numpy() == cls)[:, 0]
                sampled_image_idxs.append(idxs_wrt_cls)
            sampled_image_idxs = np.concatenate(sampled_image_idxs)

            # get the testing set of specific classes
            dataset = torch.utils.data.Subset(dataset, sampled_image_idxs)
            dataset.classes = use_classes
            dataset.class_names = [orig_class_names[i] for i in use_classes]
            dataset.targets = orig_targets[sampled_image_idxs]
        return dataset
