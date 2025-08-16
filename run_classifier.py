#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import torch

from src.utils import CheckpointIO
from src.utils.config import setup_cfg, print_cfg
from src.data import get_image_dataset_info, get_image_dataset
from src.models import get_network, get_n_parameters, get_n_trainable_parameters, Normalize
from src.solvers import Classifier


def parse_arguments():

    def str_or_int_or_none(value):
        if value == "":
            return None
        elif value.isdigit():
            return int(value)
        return value

    def list_or_none(value):
        if value == "":
            return None
        else:
            return [int(i) for i in value.split(",")]

    def true_or_false(value):
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            assert False

    parser = argparse.ArgumentParser()

    # Experiment args
    parser.add_argument("--exp_root_dir", type=str, default="./archive", help="The root dir for storing results")
    parser.add_argument("--dir_suffix", type=str, default="", help="The suffix of the result directory name")
    parser.add_argument("--device", type=str_or_int_or_none, default=None, help="Device for computing (default: None)")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of used GPUs (default: 1)")
    parser.add_argument("--seed", type=str_or_int_or_none, default=3407, help="Random seed (default: 3407)")

    # Network args
    parser.add_argument("--net_arch", type=str, default="convnet_mnist", choices=["convnet_mnist", "resnet20", "vgg16_cifar", "resnet56", "vgg19_cifar", "vgg16", "vgg19", "resnet18", "resnet50", "resnet152", "squeezenet", "shufflenet"], help="Architecture for the classifier (default: convnet_mnist)")
    parser.add_argument("--net_ckpt_path", type=str, default="", help="The checkpoint file for the classifier (default: '')")
    parser.add_argument("--finetune", type=true_or_false, default=False, help="Whether to fintune the pretrained classifier (default: False)")

    # Dataset args
    parser.add_argument("--data_root_dir", type=str, default="/home/share/Datasets", help="The root dir for storing datasets")
    parser.add_argument("--dataset", type=str, default="fmnist", choices=["fmnist", "cifar10", "cifar100", "imagenet10", "sub_imagenet"], help="Used dataset (default: fmnist)")
    parser.add_argument("--dataset_dir", type=str_or_int_or_none, default=None)
    parser.add_argument("--n_samples", type=int, default=-1, help="Number of samples used for training the classifier, if -1 then all samples are used (default: -1)")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Percentage used to split the validation set (default: 0.2)")
    parser.add_argument("--train_classes", type=list_or_none, default=None, help="The classes of the selected training samples (default: None)")
    parser.add_argument("--test_classes", type=list_or_none, default=None, help="The classes of the selected testing samples (default: None)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--n_workers", type=int, default=int(os.cpu_count() / 2), help="Number of data loading workers (default: int(os.cpu_count() / 2))")

    # Trainer args
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs to train (dfault: 100)")
    parser.add_argument("--patience", type=int, default=-1, help="The patience in the earlystopping strategy (default: -1)")
    parser.add_argument("--print_freq", type=int, default=200, help="Frequency of printing training logs (default: 200)")

    # Optimizing args
    parser.add_argument("--opt_alg", type=str, default="adam", choices=["sgd", "adam"], help="Used optimizer (default: adam)")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate (default: 1e-1)")

    args = parser.parse_args()
    return args


def prepare_dataloaders(args):

    args.logger.info(f"===========================> Loading dataset {args.dataset}:")
    args.n_channels, args.img_size, args.n_classes, (args.mean, args.std) = get_image_dataset_info(args.dataset)

    # get datasets
    train_dataset, val_dataset = get_image_dataset(
        data_root_dir=args.data_root_dir, dataset_name=args.dataset, is_train=True, use_classes=args.train_classes,
        train_images_per_class=int(args.n_samples / args.n_classes), validation_split=args.validation_split,
        dataset_dir=args.dataset_dir
    )
    test_dataset = get_image_dataset(
        data_root_dir=args.data_root_dir, dataset_name=args.dataset, is_train=False, use_classes=args.test_classes,
        train_images_per_class=-1, validation_split=-1, dataset_dir=args.dataset_dir
    )

    # prepare training dataloader
    args.train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True
    )
    if len(train_dataset.classes) != args.n_classes:
        args.n_classes = len(train_dataset.classes)
    args.train_loader.data_info = [args.n_channels, args.img_size, args.img_size, args.n_classes, (args.mean, args.std)]
    args.logger.info(f"# Training images: {len(train_dataset)}")

    # prepare validating dataloader
    if val_dataset is not None:
        args.val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=True
        )
        args.logger.info(f"# Validating images: {len(val_dataset)}")
    else:
        args.val_loader = None

    # prepare testing dataloader
    args.test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=False
    )
    args.test_loader.data_info = [args.n_channels, args.img_size, args.img_size, args.n_classes, (args.mean, args.std)]
    args.logger.info(f"# Testing images: {len(test_dataset)}")


def prepare_model(args):

    args.logger.info(f"===========================> Loading network: {args.net_arch}")
    args.net = get_network(
        net_arch=args.net_arch, n_classes=args.n_classes, n_channels=args.n_channels,
        pretrained=args.net_ckpt_path == "pretrained", finetune=args.finetune,
    )
    args.net = torch.nn.Sequential(Normalize(args.mean, args.std), args.net)

    # initialize from checkpoint
    if args.net_ckpt_path and args.net_ckpt_path != "pretrained":
        ckptio = CheckpointIO(
            ckpt_dir=args.exp_dir, multi_gpu=torch.cuda.device_count() > 1 and args.n_gpus > 1, state_dict=args.net
        )
        ckptio.load()
    args.logger.info(f"===========================> Network :\n {args.net}")
    args.logger.info(f"Total # parameters: {get_n_parameters(args.net)}")
    args.logger.info(f"# Trainable parameters: {get_n_trainable_parameters(args.net)}")


if __name__ == "__main__":

    args = parse_arguments()
    args.exp_type = "classification"
    setup_cfg(args)
    print_cfg(args)

    prepare_dataloaders(args)
    prepare_model(args)

    # instantiate a classifier instance
    classifier = Classifier(
        exp_dir=args.exp_dir, opt_alg=args.opt_alg, lr=args.lr, n_epochs=args.n_epochs, logger=args.logger,
        patience=args.patience, device=args.device, n_gpus=args.n_gpus, seed=args.seed, print_freq=args.print_freq
    )

    # train
    net = classifier.fit(net=args.net, train_loader=args.train_loader, val_loader=args.val_loader)

    # test
    classifier.test(net=args.net, test_loader=args.test_loader, auto_restore=False)
