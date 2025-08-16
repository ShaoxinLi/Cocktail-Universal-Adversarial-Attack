#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import copy
import torch
import torch.nn.functional as F

from ..utils import seed_everything, save_record, CheckpointIO, EarlyStopper
from ..utils.time import convert_secs2time
from ..utils.plot import plot_metrics, plot_images, plot_image_grid
from ..metrics import AverageMeter, accuracy, LossConstructor


class UAP():
    def __init__(self, exp_dir, xi, uap_size, loss_fn, opt_alg, lr, n_epochs, logger, p="inf",
                 confidence=0.0, init_method="zero", patience=-1, device=None, n_gpus=1, seed=None, print_freq=100):

        self.exp_dir = exp_dir
        self.xi = xi / 255.
        self.uap_size = uap_size
        self.loss_fn = loss_fn
        self.opt_alg = opt_alg
        self.lr = lr
        self.n_epochs = n_epochs
        self.logger = logger
        self.p = int(p) if p.isdigit() else p
        self.confidence = confidence
        self.init_method = init_method
        self.patience = patience
        self.n_gpus = n_gpus
        self.seed = seed
        self.print_freq = print_freq
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def fit(self, net, train_loader, val_loader=None, target=None, if_copy=False, loc=None):

        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.target = target
        if self.target is not None:
            self.target = torch.tensor([target], device=self.device)

        self.if_copy = if_copy
        self.loc = loc
        if self.uap_size == self.train_loader.data_info[1]:
            assert not self.if_copy and self.loc is None, f"The size of UAP is the same as the size of images"
        else:
            assert (self.if_copy and self.loc is None) or (not self.if_copy and self.loc is not None), f"Copy the adversarial patch or provide a location for it"

        # prepare the routine for fitting
        self.on_fit_start()

        # start training
        for epoch in range(self.n_epochs):

            # train one epoch
            self.on_train_epoch_start(epoch)
            train_loss, train_acc, train_fr = self.train_epoch(epoch)
            self.on_train_epoch_end(epoch)

            # validate one epoch
            if self.val_loader is not None:
                self.on_val_epoch_start()
                val_loss, val_acc, val_fr = self.val_epoch()
                self.on_val_epoch_end(epoch)
            else:
                val_loss, val_acc, val_fr = None, None, None

            # callbacks
            self.callback_save_record(
                train_loss=train_loss, train_acc=train_acc, train_fr=train_fr,
                val_loss=val_loss, val_acc=val_acc, val_fr=val_fr, epoch=epoch + 1)
            self.callback_save_ckpt(val_loss, epoch)

            # measure elapsed time
            self.epoch_time_meter.update(time.time() - self.epoch_start_time)
            self.epoch_start_time = time.time()

            # check the status of the earlystopper
            if self.earlystopper is not None and self.earlystopper.early_stop:
                break

        # on fit end
        self.on_fit_end()
        return self.uap

    def on_fit_start(self):

        # set the random seed
        seed_everything(self.seed)

        # parallel the network if possible
        self.net.to(self.device)
        if self.device == torch.device("cuda") and torch.cuda.device_count() > 1 and self.n_gpus > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=list(range(self.n_gpus)))

        # init the uap
        self.uap = self.init_uap()

        # rescale the perturbation magnitude
        if self.p != "inf" and self.uap_size != self.train_loader.data_info[1]:
            self.xi = self.xi * (self.uap_size / self.train_loader.data_info[1]) ** 2

        # configure the optimizer and the scheduler
        self.opt, self.sch = self.configure_optimizers(paras=[self.uap], opt_alg=self.opt_alg, lr=self.lr)

        # set the criterion
        if self.target is None:
            self.criterion = LossConstructor(
                src_loss_fn_name=self.loss_fn, other_loss_fn_name="", src_classes=None,
                n_classes=self.train_loader.data_info[3], confidence=self.confidence, alpha=0, device=self.device
            ).to(self.device)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="none").to(self.device)

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
            opt_state=self.opt, uap=self.uap
        )

    def on_train_epoch_start(self, epoch):

        need_hour, need_mins, need_secs = convert_secs2time(self.epoch_time_meter.avg * (self.n_epochs - epoch))
        need_time = f"[Need: {need_hour:02d}:{need_mins:02d}:{need_secs:02d}]"
        self.logger.info(f"===========================> [Epoch={epoch + 1:03d} / {self.n_epochs:03d}] {need_time:s}")

        # init average meters
        self.train_loss_meter = AverageMeter()
        self.train_acc_meter = AverageMeter()
        self.train_fr_meter = AverageMeter()

        # init timers
        self.batch_start_time = time.time()
        self.data_time_meter = AverageMeter()
        self.batch_time_meter = AverageMeter()

        # switch net mode
        self.net.eval()

    def train_epoch(self, epoch):

        for i, (batch_inputs, batch_gt_targets) in enumerate(self.train_loader):

            batch_inputs, batch_gt_targets = batch_inputs.to(self.device), batch_gt_targets.to(self.device)

            # measure data loading time
            self.data_time_meter.update(time.time() - self.batch_start_time)

            # train one step
            batch_pert_logits, batch_clean_logits, loss = self.train_step(batch_inputs, batch_gt_targets)

            # on train step end
            self.on_train_step_end(batch_pert_logits, batch_clean_logits, batch_gt_targets, loss, epoch, i)

        return self.train_loss_meter.avg, self.train_acc_meter.avg, self.train_fr_meter.avg

    def train_step(self, batch_inputs, batch_gt_targets):

        # get the clean logits
        with torch.no_grad():
            batch_clean_logits = self.forward(batch_inputs)

        # repeat the adversarial patch
        if self.uap_size != self.train_loader.data_info[1] and self.if_copy:
            n_repeats = int(self.train_loader.data_info[1] / self.uap_size)
            self.uap_ = self.uap.repeat(1, 1, n_repeats, n_repeats)
            assert self.uap_.size(2) == self.train_loader.data_info[1]
        else:
            self.uap_ = self.uap

        # get the perturbed inputs
        if self.loc is not None:
            batch_pert_inputs = []
            for input_ in batch_inputs:
                input_clone = input_.clone()
                x, y = self.loc[0], self.loc[1]
                input_clone[:, x: x + self.uap_size, y: y + self.uap_size] += self.uap_[0]
                batch_pert_inputs.append(input_clone)
            batch_pert_inputs = torch.stack(batch_pert_inputs, dim=0)
        else:
            batch_pert_inputs = batch_inputs + self.uap_
        batch_pert_inputs = torch.clamp(batch_pert_inputs, 0.0, 1.0)

        # forward
        batch_pert_logits = self.forward(batch_pert_inputs)

        # zero grads
        if self.opt is not None:
            self.opt.zero_grad()
        elif self.uap.grad is not None: # pgd
            self.uap.grad.data.zero_()

        # compute the loss
        if self.target is None:
            if self.loss_fn == "neg_logit_loss_nag":
                batch_pert_logits = F.softmax(batch_pert_logits)
                batch_clean_logits = F.softmax(batch_clean_logits)
            loss = torch.mean(self.criterion(batch_pert_logits, batch_clean_logits, batch_gt_targets))
        else:
            loss = torch.mean(self.criterion(batch_pert_logits, self.target.repeat(batch_inputs.size(0))))

        # backward
        loss.backward()
        if self.opt is not None:
            self.opt.step()
        else:   # pgd
            grad_sign = self.uap.grad.data.sign()
            self.uap.data = self.uap.data - grad_sign * (self.xi * 0.8)

        # project to l-p ball
        if self.p == "inf":
            self.uap.data = torch.clamp(self.uap.data, -self.xi, self.xi)
        else:
            self.uap.data = self.uap.data * min(1, self.xi / (torch.norm(self.uap.data, self.p)))

        return batch_pert_logits, batch_clean_logits, loss.item()

    def on_train_step_end(self, batch_pert_logits, batch_clean_logits, batch_gt_targets, loss, epoch, iteration):

        with torch.no_grad():
            acc = accuracy(batch_pert_logits, batch_gt_targets)[0].item()
            if self.target is None:
                fr = 1. - accuracy(batch_pert_logits, batch_clean_logits.argmax(dim=1))[0].item()
            else:
                fr = accuracy(batch_pert_logits, self.target.repeat(batch_pert_logits.size(0)))[0].item()

        # update metric meters
        self.train_loss_meter.update(loss, batch_pert_logits.size(0))
        self.train_acc_meter.update(acc, batch_pert_logits.size(0))
        self.train_fr_meter.update(fr, batch_pert_logits.size(0))

        # measure elapsed time
        self.batch_time_meter.update(time.time() - self.batch_start_time)
        self.batch_start_time = time.time()

        # batch log
        if (iteration + 1) % self.print_freq == 0:
            self.logger.info(
                f"Epoch: [{epoch + 1:03d}][{iteration + 1:03d} / {len(self.train_loader):03d}]  "
                f"Batch Time {self.batch_time_meter.val:.3f} ({self.batch_time_meter.avg:.3f})  "
                f"Data {self.data_time_meter.val:.3f} ({self.data_time_meter.avg:.3f})  "
                f"Loss {self.train_loss_meter.val:.3f} ({self.train_loss_meter.avg:.3f})  "
                f"Accuracy {self.train_acc_meter.val * 100.:.3f} ({self.train_acc_meter.avg * 100.:.3f})  "
                f"Fooling ratio {self.train_fr_meter.val * 100.:.3f} ({self.train_fr_meter.avg * 100.:.3f})"
            )

    def on_train_epoch_end(self, epoch):

        self.logger.info(
            f"*********Train*********\tEpoch [{epoch + 1:03d}]  "
            f"Loss {self.train_loss_meter.avg:.3f}  "
            f"Accuracy {self.train_acc_meter.avg * 100.:.3f}  "
            f"Fooling ratio {self.train_fr_meter.avg * 100.:.3f}"
        )

        # use learning rate scheduler if it exists
        if self.sch is not None:
            self.sch.step()

        if self.p == "inf":
            assert torch.max(self.uap.data) <= self.xi
            assert torch.min(self.uap.data) >= -self.xi
        else:
            # assert torch.norm(self.uap.data, self.p) <= self.xi
            pass

    def on_val_epoch_start(self):

        # init metric meters
        self.val_loss_meter = AverageMeter()
        self.val_acc_meter = AverageMeter()
        self.val_fr_meter = AverageMeter()

        # switch net mode
        self.net.eval()

    def val_epoch(self):

        uap = copy.deepcopy(self.uap.detach())  # avoid any unaware modification
        with torch.no_grad():
            for i, (batch_inputs, batch_gt_targets) in enumerate(self.val_loader):

                batch_inputs, batch_gt_targets = batch_inputs.to(self.device), batch_gt_targets.to(self.device)

                # val one step
                batch_pert_logits, batch_clean_logits, loss = self.val_step(uap, batch_inputs, batch_gt_targets)

                # on val step end
                self.on_val_step_end(batch_pert_logits, batch_clean_logits, batch_gt_targets, loss)

        return self.val_loss_meter.avg, self.val_acc_meter.avg, self.val_fr_meter.avg

    def val_step(self, uap, batch_inputs, batch_gt_targets):

        # get the clean logits
        batch_clean_logits = self.forward(batch_inputs)

        # repeat the adversarial patch
        if self.uap_size != self.train_loader.data_info[1] and self.if_copy:
            n_repeats = int(self.train_loader.data_info[1] / self.uap_size)
            uap_ = uap.repeat(1, 1, n_repeats, n_repeats)
            assert uap_.size(2) == self.train_loader.data_info[1]
        else:
            uap_ = uap

        # get the perturbed inputs
        if self.loc is not None:
            batch_pert_inputs = []
            for input_ in batch_inputs:
                x, y = self.loc[0], self.loc[1]
                input_[:, x: x + self.uap_size, y: y + self.uap_size] += uap_[0]
                batch_pert_inputs.append(input_)
            batch_pert_inputs = torch.stack(batch_pert_inputs, dim=0)
        else:
            batch_pert_inputs = batch_inputs + uap_
        batch_pert_inputs = torch.clamp(batch_pert_inputs, 0.0, 1.0)

        # forward
        batch_pert_logits = self.forward(batch_pert_inputs)

        # compute the loss
        if self.target is None:
            loss = torch.mean(self.criterion(batch_pert_logits, batch_clean_logits, batch_gt_targets))
        else:
            loss = torch.mean(self.criterion(batch_pert_logits, self.target.repeat(batch_inputs.size(0))))

        return batch_pert_logits, batch_clean_logits, loss.item()

    def on_val_step_end(self, batch_pert_logits, batch_clean_logits, batch_gt_targets, loss):

        with torch.no_grad():
            acc = accuracy(batch_pert_logits, batch_gt_targets)[0].item()
            if self.target is None:
                fr = 1. - accuracy(batch_pert_logits, batch_clean_logits.argmax(dim=1))[0].item()
            else:
                fr = accuracy(batch_pert_logits, self.target.repeat(batch_pert_logits.size(0)))[0].item()

        # update metric meters
        self.train_loss_meter.update(loss, batch_pert_logits.size(0))
        self.train_acc_meter.update(acc, batch_pert_logits.size(0))
        self.train_fr_meter.update(fr, batch_pert_logits.size(0))

    def on_val_epoch_end(self, epoch):

        self.logger.info(
            f"*********Val*********\tEpoch [{epoch + 1:03d}]  "
            f"Loss {self.val_loss_meter.avg:.3f}  "
            f"Accuracy {self.val_acc_meter.avg * 100.:.3f}  "
            f"Fooling ratio {self.val_fr_meter.avg * 100.:.3f}"
        )

        if self.earlystopper is not None:
            self.earlystopper(self.val_loss_meter.avg)

        if self.p == "inf":
            assert torch.max(self.uap.data) <= self.xi
            assert torch.min(self.uap.data) >= -self.xi
        else:
            # assert torch.norm(self.uap.data, self.p) <= self.xi
            pass

    def on_fit_end(self):

        if self.val_loader is not None:
            self.logger.info(f"\nBest val loss at Epoch {self.best_val_loss_epoch}: {self.best_val_loss:.3f}")

        # restore the best uap
        self.uap = self.ckptio.load()["uap"]

        #
        # set the testing result file path
        self.train_file = os.path.join(self.exp_dir, "train_result.csv")
        self.train_loader.shuffle = False
        with torch.no_grad():
            for i, (batch_inputs, batch_gt_targets) in enumerate(self.train_loader):
                batch_inputs, batch_gt_targets = batch_inputs.to(self.device), batch_gt_targets.to(self.device)
                batch_clean_logits = self.forward(batch_inputs)
                batch_pert_inputs = batch_inputs + self.uap
                batch_pert_inputs = torch.clamp(batch_pert_inputs, 0.0, 1.0)
                batch_pert_logits = self.forward(batch_pert_inputs)
                for pert_logits, clean_logits, gt_target in zip(batch_pert_logits, batch_clean_logits, batch_gt_targets):
                    record = {
                        "gt_target": int(gt_target),
                        "clean_target": int(clean_logits.argmax()),
                        "clean_target_prob": float(clean_logits.max()),
                        "pert_target": int(pert_logits.argmax()),
                        "pert_target_prob": float(pert_logits.max())
                    }
                    save_record(self.train_file, **record)

        # plot figures
        if self.val_loader is not None:
            name_keys_map = {
                "train_loss": ["train_loss", "val_loss"],
                "train_acc": ["train_acc", "val_acc"],
                "train_fr": ["train_fr", "val_fr"],
            }
        else:
            name_keys_map = {
                "train_loss": ["train_loss"],
                "train_acc": ["train_acc"],
                "train_fr": ["train_fr"]
            }
        for img_name, keys in name_keys_map.items():
            plot_metrics(
                csv_file=self.his_file, keys=keys, img_name=img_name,
                rolling_window_size=1, average_window_size=1
            )

        uap = torch.permute(copy.deepcopy(self.uap.detach().cpu()), (0, 2, 3, 1)).numpy()
        if self.p == "inf":
            uap = (uap / self.xi + 1.) / 2.
        else:
            uap = (uap - uap.min()) / (uap.max() - uap.min())
        plot_images(uap, os.path.join(self.exp_dir, "uap"))

    def forward(self, batch_inputs):
        return self.net(batch_inputs)

    def test(self, net, uap, test_loader, target=None, if_copy=False, loc=None, auto_restore=False):

        if uap is None and not auto_restore:
            assert False, f"Please provide a UAP for testing."

        self.net = net
        self.uap = uap
        self.test_loader = test_loader

        self.target = target
        if self.target is not None:
            self.target = torch.tensor([target], device=self.device)

        self.if_copy = if_copy
        self.loc = loc
        if self.uap_size == self.test_loader.data_info[1]:
            assert not self.if_copy and self.loc is None
        else:
            assert not self.if_copy or self.loc is None

        # on test start
        self.on_test_start(auto_restore)

        with torch.no_grad():
            for i, (batch_inputs, batch_gt_targets) in enumerate(test_loader):
                batch_inputs, batch_gt_targets = batch_inputs.to(self.device), batch_gt_targets.to(self.device)

                # test one step
                batch_pert_logits, batch_clean_logits = self.test_step(batch_inputs)

                # on test step end
                self.on_test_step_end(batch_pert_logits, batch_clean_logits, batch_gt_targets)

        # on test end
        self.on_test_end()

    def on_test_start(self, auto_restore):

        # auto restore the trained uap
        if auto_restore:
            self.ckptio = CheckpointIO(
                ckpt_dir=self.exp_dir,
                multi_gpu=torch.cuda.device_count() > 1 and self.n_gpus > 1,
                uap=self.uap
            )
            self.uap = self.ckptio.load()["uap"]

        # rescale the perturbation magnitude
        if self.p != "inf" and self.uap_size != self.test_loader.data_info[1]:
            self.xi = self.xi * (self.uap_size / self.test_loader.data_info[1]) ** 2

        self.uap = self.uap.to(self.device)
        if self.p == "inf":
            assert torch.max(self.uap.data) <= self.xi
            assert torch.min(self.uap.data) >= -self.xi
        else:
            # assert torch.norm(self.uap.data, self.p) <= self.xi
            pass

        # parallel the network if possible
        self.net.to(self.device)
        if self.device == torch.device("cuda") and torch.cuda.device_count() > 1 and self.n_gpus > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=list(range(self.n_gpus)))

        # init metric meters
        self.test_acc_meter = AverageMeter()
        self.test_fr_meter = AverageMeter()

        # set the testing result file path
        self.test_file = os.path.join(self.exp_dir, "test_result.csv")

        # init distribution of perturbed targets
        self.pert_targets_dis = [0 for _ in range(self.test_loader.data_info[3])]

        # switch net mode
        self.net.eval()

    def test_step(self, batch_inputs):

        # get the clean logits
        batch_clean_logits = self.forward(batch_inputs)

        # repeat the adversarial patch
        if self.uap_size != self.test_loader.data_info[1] and self.if_copy:
            n_repeats = int(self.test_loader.data_info[1] / self.uap_size)
            self.uap_ = self.uap.repeat(1, 1, n_repeats, n_repeats)
            assert self.uap_.size(2) == self.test_loader.data_info[1]
        else:
            self.uap_ = self.uap

        # get the perturbed inputs
        if self.loc is not None:
            batch_pert_inputs = []
            for input_ in batch_inputs:
                x, y = self.loc[0], self.loc[1]
                input_[:, x: x + self.uap_size, y: y + self.uap_size] += self.uap_[0]
                batch_pert_inputs.append(input_)
            batch_pert_inputs = torch.stack(batch_pert_inputs, dim=0)
        else:
            batch_pert_inputs = batch_inputs + self.uap_
        batch_pert_inputs = torch.clamp(batch_pert_inputs, 0.0, 1.0)

        # forward
        batch_pert_logits = self.forward(batch_pert_inputs)

        return batch_pert_logits, batch_clean_logits

    def on_test_step_end(self, batch_pert_logits, batch_clean_logits, batch_gt_targets):

        with torch.no_grad():
            acc = accuracy(batch_pert_logits, batch_gt_targets)[0].item()
            if self.target is None:
                fr = 1. - accuracy(batch_pert_logits, batch_clean_logits.argmax(dim=1))[0].item()
            else:
                fr = accuracy(batch_pert_logits, self.target.repeat(batch_pert_logits.size(0)))[0].item()

        # update metric meters
        self.test_acc_meter.update(acc, batch_pert_logits.size(0))
        self.test_fr_meter.update(fr, batch_pert_logits.size(0))

        for pert_logits, clean_logits, gt_target in zip(batch_pert_logits, batch_clean_logits, batch_gt_targets):
            record = {
                "gt_target": int(gt_target),
                "clean_target": int(clean_logits.argmax()),
                "clean_target_prob": float(clean_logits.max()),
                "pert_target": int(pert_logits.argmax()),
                "pert_target_prob": float(pert_logits.max())
            }
            save_record(self.test_file, **record)
            self.pert_targets_dis[int(pert_logits.argmax())] += 1

    def on_test_end(self):

        arrays_list = []

        # store the original inputs
        inputs = next(iter(self.test_loader))[0][:5]
        inputs = inputs.to(self.device)
        arrays_list.append(torch.permute(copy.deepcopy(inputs.detach().cpu()), (0, 2, 3, 1)).numpy())

        # store the perturbed inputs
        with torch.no_grad():
            if self.uap_size != self.test_loader.data_info[1] and self.if_copy:
                n_repeats = int(self.test_loader.data_info[1] / self.uap_size)
                self.uap_ = self.uap.repeat(1, 1, n_repeats, n_repeats)
                assert self.uap_.size(2) == self.test_loader.data_info[1]
            else:
                self.uap_ = self.uap

            if self.loc is not None:
                pert_inputs = []
                for input_ in inputs:
                    x, y = self.loc[0], self.loc[1]
                    input_[:, x: x + self.uap_size, y: y + self.uap_size] += self.uap_[0]
                    pert_inputs.append(input_)
                pert_inputs = torch.stack(pert_inputs, dim=0)
            else:
                pert_inputs = inputs + self.uap_
            pert_inputs = torch.clamp(pert_inputs, 0.0, 1.0)
            arrays_list.append(torch.permute(pert_inputs.detach().cpu(), (0, 2, 3, 1)).numpy())

        # plot the images grid
        plot_image_grid(arrays_list, os.path.join(self.exp_dir, "examples"))

        # testing log
        self.logger.info(
            f"*********Test*********  "
            f"Accuracy {self.test_acc_meter.avg * 100.:.3f}  "
            f"Fooling ratio {self.test_fr_meter.avg * 100.:.3f}"
        )
        self.logger.info(
            f"The distribution of perturbed targets: \n{self.pert_targets_dis}"
        )

    def init_uap(self):

        if self.p == "inf":
            if self.init_method == "zero":
                uap = torch.zeros(1, self.train_loader.data_info[0], self.uap_size, self.uap_size, device=self.device)
            else:
                uap = torch.FloatTensor(1, self.train_loader.data_info[0], self.uap_size, self.uap_size).uniform_(-self.xi, self.xi).to(self.device)
        else:
            uap = torch.zeros(1, self.train_loader.data_info[0], self.uap_size, self.uap_size, device=self.device)
        uap.requires_grad_(True)
        return uap

    @staticmethod
    def configure_optimizers(paras, opt_alg, lr, momentum=0.9, betas=(0.9, 0.999),
                             milestones=(10000, 15000), gamma=0.1):

        if opt_alg == "sgd":
            opt = torch.optim.SGD(paras, lr=lr, momentum=momentum, weight_decay=0.0)
            sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
        elif opt_alg == "adam":
            opt = torch.optim.AdamW(paras, lr=lr, betas=betas, weight_decay=0.0)
            sch = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=gamma)
        elif opt_alg == "pgd":
            opt, sch = None, None
        else:
            assert False, f"Unknown optimizer: {opt_alg}"
        return opt, sch

    def callback_save_record(self, **kwargs):

        record = {k: v for k, v in kwargs.items() if v is not None}
        save_record(self.his_file, **record)

    def callback_save_ckpt(self, val_loss, epoch):

        if val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_loss_epoch = epoch
                self.ckptio.save()
        else:
            self.ckptio.save()

    def test_multi(self, net, uaps, locs, test_loader, target=None):

        self.net = net
        self.uaps = uaps
        self.locs = locs
        self.test_loader = test_loader
        self.target = target

        # parallel the network if possible
        self.net.to(self.device)
        if self.device == torch.device("cuda") and torch.cuda.device_count() > 1 and self.n_gpus > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=list(range(self.n_gpus)))
        self.uaps = self.uaps.to(self.device)

        # switch net mode
        self.net.eval()

        # init the indexes of the images attacked by each uap
        idxs_wrt_uap = [set() for _ in range(self.uaps.size(0))]

        # init the idx counter
        counter = 0

        with torch.no_grad():
            for batch_inputs, batch_gt_targets in test_loader:
                batch_inputs, batch_gt_targets = batch_inputs.to(self.device), batch_gt_targets.to(self.device)

                # get the idxs of the images in this batch
                idxs_of_batch = list(range(counter, counter + batch_inputs.size(0)))
                counter += batch_inputs.size(0)

                # get the clean logits
                batch_clean_logits = self.forward(batch_inputs)

                for i in range(self.uaps.size(0)):
                    uap, loc = self.uaps[i], self.locs[i]

                    # get the perturbed inputs
                    batch_pert_inputs = []
                    for input_ in batch_inputs:
                        x, y = loc[0], loc[1]
                        input_[:, x: x + self.uap_size, y: y + self.uap_size] += uap
                        batch_pert_inputs.append(input_)
                    batch_pert_inputs = torch.stack(batch_pert_inputs, dim=0)
                    batch_pert_inputs = torch.clamp(batch_pert_inputs, 0.0, 1.0)

                    # forward
                    batch_pert_logits = self.forward(batch_pert_inputs)

                    # get the idxs of the successfully attacked images
                    for pert_logits, clean_logits, idx in zip(batch_pert_logits, batch_clean_logits, idxs_of_batch):
                        if self.target is None:
                            if int(pert_logits.argmax()) != int(clean_logits.argmax()):
                                idxs_wrt_uap[i].add(idx)
                        else:
                            if int(pert_logits.argmax()) != self.target:
                                idxs_wrt_uap[i].add(idx)

        n_images = counter - batch_inputs.size(0)
        for i in range(self.uaps.size(0)):
            self.logger.info(
                f"The idxs of the images attacked by UAP {i} "
                f"(fooling ratio = {len(idxs_wrt_uap[i]) / n_images * 100.:.3f}) is: {idxs_wrt_uap[i]}"
            )

