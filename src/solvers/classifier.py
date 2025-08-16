#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import torch

from ..utils import EarlyStopper, seed_everything, save_record, CheckpointIO
from ..utils.time import convert_secs2time
from ..utils.plot import plot_metrics
from ..metrics import AverageMeter, accuracy


class Classifier():
    def __init__(self, exp_dir, opt_alg, lr, n_epochs, logger, patience=-1, device=None, n_gpus=1,
                 seed=None, print_freq=100):

        self.exp_dir = exp_dir
        self.opt_alg = opt_alg
        self.lr = lr
        self.n_epochs = n_epochs
        self.logger = logger
        self.patience = patience
        self.n_gpus = n_gpus
        self.seed = seed
        self.print_freq = print_freq
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def fit(self, net, train_loader, val_loader=None):

        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader

        # prepare the routine for fitting
        self.on_fit_start()

        # start training
        for epoch in range(self.n_epochs):

            # train one epoch
            self.on_train_epoch_start(epoch)
            train_loss, train_acc = self.train_epoch(epoch)
            self.on_train_epoch_end(epoch)

            # validate one epoch
            if self.val_loader is not None:
                self.on_val_epoch_start()
                val_loss, val_acc = self.val_epoch()
                self.on_val_epoch_end(epoch)
            else:
                val_loss, val_acc = None, None

            # callbacks
            self.callback_save_record(
                file=self.his_file, train_loss=train_loss, train_acc=train_acc,
                val_loss=val_loss, val_acc=val_acc,
                epoch=epoch + 1, lr=self.opt.param_groups[0]["lr"]
            )
            self.callback_save_ckpt(val_loss, epoch)

            # measure elapsed time
            self.epoch_time_meter.update(time.time() - self.epoch_start_time)
            self.epoch_start_time = time.time()

            # check the status of the earlystopper
            if self.earlystopper is not None and self.earlystopper.early_stop:
                break

        # on fit end
        self.on_fit_end()
        return self.net

    def on_fit_start(self):

        self.img_size = self.train_loader.data_info[:3]
        self.n_classes = self.train_loader.data_info[3]
        self.n_imgs = self.train_loader.data_info[4]

        # set the random seed
        seed_everything(self.seed)

        # parallel the network if possible
        self.net = self._place_and_parallel_net(self.net, self.device, self.n_gpus)

        # configure the optimizer and the scheduler
        self.opt, self.sch = self.configure_optimizers(paras=self.net.parameters(), opt_alg=self.opt_alg, lr=self.lr)

        # set the criterion
        self.criterion = torch.nn.BCEWithLogitsLoss() if self.n_classes == 2 else torch.nn.CrossEntropyLoss()
        self.criterion = self.criterion.to(self.device)

        # set the training history file path
        self.his_file = os.path.join(self.exp_dir, "train_history.csv")

        # set a timer
        self.epoch_start_time = time.time()
        self.epoch_time_meter = AverageMeter()

        # set the earlystopper
        self.earlystopper = EarlyStopper(self.patience) if self.patience > 0 else None

        # set the monitors
        self.best_val_loss = 999999.
        self.best_val_loss_epoch = 0

        # set the ckptio
        self.ckptio = CheckpointIO(
            ckpt_dir=self.exp_dir,
            multi_gpu=torch.cuda.device_count() > 1 and self.n_gpus > 1,
            net_state=self.net, opt_state=self.opt
        )

    def on_train_epoch_start(self, epoch):

        need_hour, need_mins, need_secs = convert_secs2time(self.epoch_time_meter.avg * (self.n_epochs - epoch))
        need_time = f"[Need: {need_hour:02d}:{need_mins:02d}:{need_secs:02d}]"
        self.logger.info(f"===========================> [Epoch={epoch + 1:03d} / {self.n_epochs:03d}] {need_time:s}")

        # init average meters
        self.train_loss_meter = AverageMeter()
        self.train_acc_meter = AverageMeter()

        # init timers
        self.batch_start_time = time.time()
        self.data_time_meter = AverageMeter()
        self.batch_time_meter = AverageMeter()

        # switch net mode
        self.net.train()

    def train_epoch(self, epoch):

        for i, (batch_inputs, batch_gt_targets) in enumerate(self.train_loader):

            batch_inputs, batch_gt_targets = batch_inputs.to(self.device), batch_gt_targets.to(self.device)

            # measure data loading time
            self.data_time_meter.update(time.time() - self.batch_start_time)

            # train one step
            batch_outputs, loss = self.train_step(batch_inputs, batch_gt_targets)

            # on train step end
            self.on_train_step_end(batch_outputs, batch_gt_targets, loss, epoch, i)

        return self.train_loss_meter.avg, self.train_acc_meter.avg

    def train_step(self, batch_inputs, batch_gt_targets):

        # forward
        batch_outputs = self.forward(batch_inputs)

        # compute the loss
        loss = self.criterion(batch_outputs, batch_gt_targets)

        # backward
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return batch_outputs, loss.item()

    @torch.no_grad()
    def on_train_step_end(self, batch_outputs, batch_gt_targets, loss, epoch, iteration):

        acc = accuracy(batch_outputs, batch_gt_targets)[0].item()

        # update metric meters
        self.train_loss_meter.update(loss, batch_outputs.size(0))
        self.train_acc_meter.update(acc, batch_outputs.size(0))

        # measure elapsed time
        self.batch_time_meter.update(time.time() - self.batch_start_time)
        self.batch_start_time = time.time()

        # batch log
        if (iteration + 1) % self.print_freq == 0:
            self.logger.info(
                f"Epoch: [{epoch + 1:03d}][{iteration + 1:03d} / {len(self.train_loader):03d}]  "
                f"Batch Time {self.batch_time_meter.val:.3f} ({self.batch_time_meter.avg:.3f})  "
                f"Data {self.data_time_meter.val:.3f} ({self.data_time_meter.avg:.3f})  "
                f"Accuracy {self.train_acc_meter.val * 100.:.3f} ({self.train_acc_meter.avg * 100.:.3f})  "
                f"Loss {self.train_loss_meter.val:.3f} ({self.train_loss_meter.avg:.3f})  "
            )

    def on_train_epoch_end(self, epoch):

        self.logger.info(
            f"*********Train*********\tEpoch [{epoch + 1:03d}]  "
            f"Loss {self.train_loss_meter.avg:.3f}  "
            f"Accuracy {self.train_acc_meter.avg * 100.:.3f}"
        )

        # use learning rate scheduler if it exists
        if self.sch is not None:
            self.sch.step()

    def on_val_epoch_start(self):

        # init metric meters
        self.val_loss_meter = AverageMeter()
        self.val_acc_meter = AverageMeter()

        # switch net mode
        self.net.eval()

    @torch.no_grad()
    def val_epoch(self):

        for i, (batch_inputs, batch_gt_targets) in enumerate(self.val_loader):

            batch_inputs, batch_gt_targets = batch_inputs.to(self.device), batch_gt_targets.to(self.device)

            # val one step
            batch_outputs, loss = self.val_step(batch_inputs, batch_gt_targets)

            # on val step end
            self.on_val_step_end(batch_outputs, batch_gt_targets, loss)

        return self.val_loss_meter.avg, self.val_acc_meter.avg

    @torch.no_grad()
    def val_step(self, batch_inputs, batch_gt_targets):

        # forward
        batch_outputs = self.forward(batch_inputs)

        # compute the loss
        loss = self.criterion(batch_outputs, batch_gt_targets)

        return batch_outputs, loss.item()

    @torch.no_grad()
    def on_val_step_end(self, batch_outputs, batch_gt_targets, loss):

        acc = accuracy(batch_outputs, batch_gt_targets)[0].item()

        # update metric meters
        self.val_loss_meter.update(loss, batch_outputs.size(0))
        self.val_acc_meter.update(acc, batch_outputs.size(0))

    def on_val_epoch_end(self, epoch):

        self.logger.info(
            f"*********Val*********\tEpoch [{epoch + 1:03d}]  "
            f"Loss {self.val_loss_meter.avg:.3f}  "
            f"Accuracy {self.val_acc_meter.avg * 100.:.3f}"
        )

        if self.earlystopper is not None:
            self.earlystopper(self.val_loss_meter.avg)

    def on_fit_end(self):

        if self.val_loader is not None:
            self.logger.info(f"Best val loss at Epoch {self.best_val_loss_epoch}: {self.best_val_loss:.3f}")

        # restore the best parameters
        self.ckptio.load()

        # plot figures
        if self.val_loader is not None:
            name_keys_map = {"train_loss": ["train_loss", "val_loss"], "train_acc": ["train_acc", "val_acc"]}
        else:
            name_keys_map = {"train_loss": ["train_loss"], "train_acc": ["train_acc"]}

        for img_name, keys in name_keys_map.items():
            plot_metrics(
                csv_file=self.his_file, keys=keys, img_name=img_name,
                rolling_window_size=1, average_window_size=1
            )

    def forward(self, batch_inputs):
        return self.net(batch_inputs)

    @torch.no_grad()
    def test(self, net, test_loader, auto_restore=False):

        self.net = net
        self.test_loader = test_loader

        # on test start
        self.on_test_start(auto_restore)

        for i, (batch_inputs, batch_gt_targets) in enumerate(test_loader):
            batch_inputs, batch_gt_targets = batch_inputs.to(self.device), batch_gt_targets.to(self.device)

            # test one step
            batch_outputs = self.test_step(batch_inputs)

            # on test step end
            self.on_test_step_end(batch_outputs, batch_gt_targets)

        # on test end
        self.on_test_end()

    @torch.no_grad()
    def test_step(self, batch_inputs):

        # forward
        batch_outputs = self.forward(batch_inputs)
        return batch_outputs

    def on_test_start(self, auto_restore):

        # auto restore the trained net
        if auto_restore:
            self.ckptio = CheckpointIO(
                ckpt_dir=self.exp_dir,
                multi_gpu=torch.cuda.device_count() > 1 and self.n_gpus > 1,
                net_state=self.net
            )
            self.ckptio.load()

        # parallel the network if possible
        self.net = self._place_and_parallel_net(self.net, self.device, self.n_gpus)

        # init metric meters
        self.test_acc_meter = AverageMeter()

        # switch net mode
        self.net.eval()

    @torch.no_grad()
    def on_test_step_end(self, batch_outputs, batch_gt_targets):

        acc = accuracy(batch_outputs, batch_gt_targets)[0].item()

        # update metric meters
        self.test_acc_meter.update(acc, batch_outputs.size(0))

    def on_test_end(self):

        # testing log
        self.logger.info(f"*********Test*********  Accuracy {self.test_acc_meter.avg * 100.:.3f}")

    @staticmethod
    def configure_optimizers(paras, opt_alg, lr, momentum=0.9, betas=(0.9, 0.999), weight_decay=5e-4,
                             milestones=(60, 90), gamma=0.1):
        if opt_alg == "sgd":
            opt = torch.optim.SGD(paras, lr=lr, momentum=momentum, weight_decay=weight_decay)
            sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
        elif opt_alg == "adam":
            opt = torch.optim.AdamW(paras, lr=lr, betas=betas, weight_decay=weight_decay)
            sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
        else:
            assert False, f"Unknown optimizer: {opt_alg}"
        return opt, sch

    @staticmethod
    def callback_save_record(file, **kwargs):

        record = {k: v for k, v in kwargs.items() if v is not None}
        save_record(file, **record)

    def callback_save_ckpt(self, val_loss, epoch):

        if val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_loss_epoch = epoch
                self.ckptio.save()
        else:
            self.ckptio.save()

    @staticmethod
    def _place_and_parallel_net(net, device, n_gpus):

        net.to(device)
        if device == torch.device("cuda") and torch.cuda.device_count() > 1 and n_gpus > 1:
            net = torch.nn.DataParallel(net, device_ids=list(range(n_gpus)))
        return net