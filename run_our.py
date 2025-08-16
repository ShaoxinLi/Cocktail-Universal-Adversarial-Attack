#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import torch

from src.utils import CheckpointIO
from src.utils.config import setup_cfg, print_cfg
from src.data import get_image_dataset_info, get_image_dataset
from src.models import get_network, get_n_parameters, get_n_trainable_parameters, Normalize, set_parameter_requires_grad
from src.solvers import Classifier, Our


def parse_arguments():

    def empty_to_none(value):
        if value == "":
            return None
        elif value.isdigit():
            return int(value)
        return value

    def list_or_none(value):
        if value == "":
            return None
        else:
            ele = value.split(",")[0]
            if ele.isdigit():
                return [int(i) for i in value.split(",")]
            else:
                return [i for i in value.split(",")]

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
    parser.add_argument("--device", type=empty_to_none, default=None, help="Device for computing (default: None)")
    parser.add_argument("--n_gpus", type=int, default=1, help="Number of used GPUs (default: 1)")
    parser.add_argument("--seed", type=empty_to_none, default=3407, help="Random seed (default: 3407)")

    # Network args
    parser.add_argument("--net_arch", type=str, default="convnet_mnist", choices=["convnet_mnist", "resnet20", "vgg16_cifar", "resnet56", "vgg19_cifar", "vgg16", "vgg19", "resnet18", "resnet50", "resnet152", "squeezenet", "shufflenet"], help="Architecture for the classifier (default: convnet_mnist)")
    parser.add_argument("--assign_net_arch", type=str, default="convnet_mnist", choices=["convnet_mnist", "resnet20", "vgg16_cifar", "resnet56", "vgg19_cifar", "vgg16", "vgg19", "resnet18", "resnet50", "resnet152", "squeezenet", "shufflenet"], help="Architecture for the classifier (default: convnet_mnist)")
    parser.add_argument("--target_net_ckpt_path", type=str, default="", help="The checkpoint file for the classifier (default: '')")
    parser.add_argument("--assign_net_ckpt_path", type=str, default="", help="The checkpoint file for the classifier (default: '')")

    # Dataset args
    parser.add_argument("--data_root_dir", type=str, default="/home/share/Datasets", help="The root dir for storing datasets")
    parser.add_argument("--dataset", type=str, default="fmnist", choices=["fmnist", "cifar10", "cifar100", "imagenet10", "sub_imagenet"], help="Used dataset (default: fmnist)")
    parser.add_argument("--dataset_dir", type=empty_to_none, default=None)
    parser.add_argument("--n_samples", type=int, default=-1, help="Number of samples used for training the classifier, if -1 then all samples are used (default: -1)")
    parser.add_argument("--train_classes", type=list_or_none, default=None, help="The classes of the selected training samples (default: None)")
    parser.add_argument("--test_classes", type=list_or_none, default=None, help="The classes of the selected testing samples (default: None)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--n_workers", type=int, default=int(os.cpu_count() / 2), help="Number of data loading workers (default: int(os.cpu_count() / 2))")

    # Trainer args
    parser.add_argument("--xi", type=int, default=10, help="Controls the magnitude of the perturbation (default: 10)")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--pretrain_uaps_paths", type=list_or_none, default=None)
    parser.add_argument("--loss_fn", type=str, default="neg_bounded_ce", choices=["neg_logits_ce", "neg_bounded_ce", "neg_logit", "neg_bounded_logit", "neg_cosine_sim", "neg_cosine_sim_max"], help="Used loss function (default: neg_bounded_ce)")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs to train perturbations (dfault: 10)")
    parser.add_argument("--confidence", type=float, default=12.0, help="The loss threshold used in loss function (default: 12.0)")
    parser.add_argument("--init_method", type=str, default="zero", help="The method to initialize the uap (default: zero)")
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
    train_dataset, _ = get_image_dataset(
        data_root_dir=args.data_root_dir, dataset_name=args.dataset, is_train=True, use_classes=args.train_classes,
        train_images_per_class=int(args.n_samples / args.n_classes), validation_split=-1.0,
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
    args.train_loader.data_info = [args.n_channels, args.img_size, args.img_size, args.n_classes, (args.mean, args.std)]
    args.logger.info(f"# Training images: {len(train_dataset)}")

    # prepare testing dataloader
    args.test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers, pin_memory=False
    )
    args.test_loader.data_info = [args.n_channels, args.img_size, args.img_size, args.n_classes, (args.mean, args.std)]
    args.logger.info(f"# Testing images: {len(test_dataset)}")


def prepare_model(args):

    args.logger.info(f"===========================> Loading network: {args.net_arch}")
    pretrained = True if not args.target_net_ckpt_path else False
    args.target_net = get_network(
        net_arch=args.net_arch, n_classes=args.n_classes, n_channels=args.n_channels,
        pretrained=pretrained, finetune=False,
    )
    args.target_net = torch.nn.Sequential(Normalize(args.mean, args.std), args.target_net)

    # initialize from checkpoint
    if args.target_net_ckpt_path:
        ckptio = CheckpointIO(
            ckpt_dir=os.path.dirname(args.target_net_ckpt_path),
            multi_gpu=torch.cuda.device_count() > 1 and args.n_gpus > 1,
            net_state=args.target_net
        )
        ckptio.load()
    set_parameter_requires_grad(args.target_net, requires_grad=False)
    args.logger.info(f"===========================> Network :\n {args.target_net}")
    args.logger.info(f"Total # parameters: {get_n_parameters(args.target_net)}")
    args.logger.info(f"# Trainable parameters: {get_n_trainable_parameters(args.target_net)}")

    args.logger.info(f"===========================> Loading network: {args.assign_net_arch}")
    pretrained = True if not args.assign_net_ckpt_path else False
    args.assign_net = get_network(
        net_arch=args.assign_net_arch, n_classes=args.k, n_channels=args.n_channels,
        pretrained=pretrained, finetune=False,
    )
    args.assign_net = torch.nn.Sequential(Normalize(args.mean, args.std), args.assign_net)

    # initialize from checkpoint
    if args.assign_net_ckpt_path:
        ckptio = CheckpointIO(
            ckpt_dir=os.path.dirname(args.assign_net_ckpt_path),
            multi_gpu=torch.cuda.device_count() > 1 and args.n_gpus > 1,
            net_state=args.assign_net
        )
        ckptio.load()
    set_parameter_requires_grad(args.assign_net, requires_grad=True)
    args.logger.info(f"===========================> Network :\n {args.assign_net}")
    args.logger.info(f"Total # parameters: {get_n_parameters(args.assign_net)}")
    args.logger.info(f"# Trainable parameters: {get_n_trainable_parameters(args.assign_net)}")


if __name__ == "__main__":

    args = parse_arguments()
    args.exp_type = "our"
    setup_cfg(args)
    print_cfg(args)

    prepare_dataloaders(args)
    prepare_model(args)

    # # get the test accuracy of the target net
    # classifier = Classifier(
    #     exp_dir=args.exp_dir, opt_alg=None, lr=None, n_epochs=None, logger=args.logger,
    #     patience=-1, device=args.device, n_gpus=args.n_gpus, seed=args.seed, print_freq=-1
    # )
    # classifier.test(net=args.net, test_loader=args.test_loader, auto_restore=False)

    # instantiate an instance
    attacker = Our(
        exp_dir=args.exp_dir, xi=args.xi, k=args.k, loss_fn=args.loss_fn, opt_alg=args.opt_alg,
        lr=args.lr, n_epochs=args.n_epochs, logger=args.logger, confidence=args.confidence,
        init_method=args.init_method, pretrain_uaps_paths=args.pretrain_uaps_paths,
        device=args.device, n_gpus=args.n_gpus, seed=args.seed, print_freq=args.print_freq
    )

    # train
    uap = attacker.fit(
        target_net=args.target_net, assign_net=args.assign_net, train_loader=args.train_loader
    )

    # test
    attacker.test(
        target_net=args.target_net, assign_net=args.assign_net, test_loader=args.test_loader,
        uaps=None, auto_restore=True
    )




