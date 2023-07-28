#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import copy
import torch
import torch.nn.functional as F

from ..utils import seed_everything, save_record, CheckpointIO
from ..utils.time import convert_secs2time
from ..utils.plot import plot_metrics, plot_images, plot_image_grid
from ..metrics import AverageMeter, accuracy, LossConstructor


class Our():
    def __init__(self, exp_dir, xi, k, loss_fn, opt_alg, lr, n_epochs, logger, confidence=0.0, init_method="zero",
                 pretrain_uaps_paths=None, device=None, n_gpus=1, seed=None, print_freq=100):

        self.exp_dir = exp_dir
        self.xi = xi / 255.
        self.k = k
        self.loss_fn = loss_fn
        self.opt_alg = opt_alg
        self.lr = lr
        self.n_epochs = n_epochs
        self.logger = logger
        self.confidence = confidence
        self.init_method = init_method
        self.pretrain_uaps_paths = pretrain_uaps_paths
        self.n_gpus = n_gpus
        self.seed = seed
        self.print_freq = print_freq
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def fit(self, target_net, assign_net, train_loader):

        self.target_net = target_net
        self.assign_net = assign_net
        self.train_loader = train_loader

        # prepare the routine for fitting
        self.on_fit_start()

        # start training
        for epoch in range(self.n_epochs):

            # train one epoch
            self.on_train_epoch_start(epoch)
            train_loss, train_acc, train_fr = self.train_epoch(epoch)
            self.on_train_epoch_end(epoch)

            # callbacks
            self.callback_save_record(
                train_loss=train_loss, train_acc=train_acc, train_fr=train_fr, epoch=epoch + 1)
            self.callback_save_ckpt()

            # measure elapsed time
            self.epoch_time_meter.update(time.time() - self.epoch_start_time)
            self.epoch_start_time = time.time()

        # on fit end
        self.on_fit_end()
        return self.uaps

    def on_fit_start(self):

        # set the random seed
        seed_everything(self.seed)

        # parallel the network if possible
        self.target_net.to(self.device, non_blocking=True)
        self.assign_net.to(self.device, non_blocking=True)
        if self.device == torch.device("cuda") and torch.cuda.device_count() > 1 and self.n_gpus > 1:
            self.target_net = torch.nn.DataParallel(self.target_net, device_ids=list(range(self.n_gpus)))
            self.assign_net = torch.nn.DataParallel(self.assign_net, device_ids=list(range(self.n_gpus)))

        # init the uap
        self.uaps = self.init_uaps()
        uaps = []
        if self.pretrain_uaps_paths is not None:
            for path in self.pretrain_uaps_paths:
                ckpt = torch.load(path)
                uap = ckpt["uap"]
                uaps.append(uap)
            self.uaps = torch.concat(uaps, dim=0)
            self.uaps = self.uaps.to(self.device, non_blocking=True)

        # configure the optimizer and the scheduler
        params_to_train = list(self.assign_net.parameters()) + [self.uaps]
        self.opt, self.sch = self.configure_optimizers(paras=params_to_train, opt_alg=self.opt_alg, lr=self.lr)

        # set the criterion
        self.criterion = LossConstructor(
            src_loss_fn_name=self.loss_fn, other_loss_fn_name="", src_classes=None,
            n_classes=self.train_loader.data_info[3], confidence=self.confidence, alpha=0, device=self.device
        ).to(self.device, non_blocking=True)

        # set the training history file path
        self.his_file = os.path.join(self.exp_dir, "train_history.csv")

        # set a timer
        self.epoch_start_time = time.time()
        self.epoch_time_meter = AverageMeter()

        # set the ckptio
        self.ckptio = CheckpointIO(
            ckpt_dir=self.exp_dir,
            multi_gpu=torch.cuda.device_count() > 1 and self.n_gpus > 1,
            opt_state=self.opt,  assign_net=self.assign_net, uaps=self.uaps.detach()
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
        self.target_net.eval()
        self.assign_net.train()

    def train_epoch(self, epoch):

        for i, (batch_inputs, batch_gt_targets) in enumerate(self.train_loader):

            batch_inputs, batch_gt_targets = batch_inputs.to(self.device, non_blocking=True), batch_gt_targets.to(self.device, non_blocking=True)

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
            batch_clean_logits = self.target_net(batch_inputs)

        # compute the assign scores
        batch_scores = self.assign_net(batch_inputs)
        batch_scores = F.softmax(batch_scores, dim=1)

        # perturb images and forward through the target net
        batch_inputs = torch.unsqueeze(batch_inputs, 1)
        unsqueezed_uaps = self.uaps.unsqueeze(0)
        batch_pert_inputs = batch_inputs + unsqueezed_uaps
        batch_pert_inputs = torch.clamp(batch_pert_inputs, 0.0, 1.0)
        batch_pert_inputs = batch_pert_inputs.view(-1, *batch_pert_inputs.size()[2:])
        batch_pert_logits = self.target_net(batch_pert_inputs)

        # zero grads
        if self.opt is not None:
            self.opt.zero_grad(set_to_none=True)
        elif self.uaps.grad is not None: # pgd
            self.uaps.grad.data.zero_()

        # compute the loss
        loss = torch.mean(self.criterion(batch_pert_logits, batch_clean_logits, batch_gt_targets) * batch_scores.view(-1) * batch_scores.size(-1))

        # backward
        loss.backward()
        if self.opt is not None:
            self.opt.step()
        else:   # pgd
            grad_sign = self.uaps.grad.data.sign()
            self.uaps.data = self.uaps.data - grad_sign * (self.xi * 0.8)

        # project to l-p ball
        self.uaps.data = torch.clamp(self.uaps.data, -self.xi, self.xi)

        return batch_pert_logits, batch_clean_logits, loss.item()

    def on_train_step_end(self, batch_pert_logits, batch_clean_logits, batch_gt_targets, loss, epoch, iteration):

        with torch.no_grad():
            k = int(batch_pert_logits.size(0) / batch_clean_logits.size(0))
            batch_gt_targets = batch_gt_targets.repeat_interleave(k, dim=0)
            batch_clean_logits = batch_clean_logits.repeat_interleave(k, dim=0)

            acc = accuracy(batch_pert_logits, batch_gt_targets)[0].item()
            fr = 1. - accuracy(batch_pert_logits, batch_clean_logits.argmax(dim=1))[0].item()

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

        assert torch.max(self.uaps.data) <= self.xi
        assert torch.min(self.uaps.data) >= -self.xi

    def on_fit_end(self):

        # restore the best uap
        self.uaps = self.ckptio.load()["uaps"]

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

        uaps = torch.permute(copy.deepcopy(self.uaps.detach().cpu()), (0, 2, 3, 1)).numpy()
        uaps = (uaps / self.xi + 1.) / 2.
        plot_images(uaps, os.path.join(self.exp_dir, "uaps"))

    def test(self, target_net, assign_net, test_loader, uaps, auto_restore=False):

        if uaps is None and not auto_restore:
            assert False, f"Please provide a UAP for testing."

        self.target_net = target_net
        self.assign_net = assign_net
        self.uaps = uaps
        self.test_loader = test_loader

        # on test start
        self.on_test_start(auto_restore)

        with torch.no_grad():
            for i, (batch_inputs, batch_gt_targets) in enumerate(test_loader):
                batch_inputs, batch_gt_targets = batch_inputs.to(self.device, non_blocking=True), batch_gt_targets.to(self.device, non_blocking=True)

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
                assign_net=self.assign_net, uaps=self.uaps,
            )
            self.uaps = self.ckptio.load()["uaps"]

        self.uaps = self.uaps.to(self.device, non_blocking=True)
        assert torch.max(self.uaps.data) <= self.xi
        assert torch.min(self.uaps.data) >= -self.xi

        # parallel the network if possible
        self.target_net.to(self.device, non_blocking=True)
        self.assign_net.to(self.device, non_blocking=True)
        if self.device == torch.device("cuda") and torch.cuda.device_count() > 1 and self.n_gpus > 1:
            self.target_net = torch.nn.DataParallel(self.target_net, device_ids=list(range(self.n_gpus)))
            self.assign_net = torch.nn.DataParallel(self.assign_net, device_ids=list(range(self.n_gpus)))

        # init metric meters
        self.test_acc_meter = AverageMeter()
        self.test_fr_meter = AverageMeter()

        # init statistics variables
        self.top_counts_all_uaps = [0 for _ in range(len(self.uaps))]

        # set the testing result file path
        self.test_file = os.path.join(self.exp_dir, "test_result.csv")

        # init distribution of perturbed targets
        self.pert_targets_dis = [0 for _ in range(self.test_loader.data_info[3])]

        # switch net mode
        self.target_net.eval()
        self.assign_net.eval()

    def test_step(self, batch_inputs):

        # get the clean logits
        batch_clean_logits = self.target_net(batch_inputs)

        # compute the scores obtained from the score net
        batch_scores = self.assign_net(batch_inputs)
        batch_scores = F.softmax(batch_scores, dim=1)

        # perturb images and forward through the target net
        uap_idxs = [int(i) for i in torch.argmax(batch_scores, dim=1)]
        batch_pert_inputs = [x + self.uaps[idx] for x, idx in zip(batch_inputs, uap_idxs)]
        batch_pert_inputs = torch.stack(batch_pert_inputs)
        batch_pert_inputs = torch.clamp(batch_pert_inputs, 0.0, 1.0)
        batch_pert_logits = self.target_net(batch_pert_inputs)

        for idx in uap_idxs:
            self.top_counts_all_uaps[idx] += 1
        return batch_pert_logits, batch_clean_logits

    def on_test_step_end(self, batch_pert_logits, batch_clean_logits, batch_gt_targets):

        with torch.no_grad():
            acc = accuracy(batch_pert_logits, batch_gt_targets)[0].item()
            fr = 1. - accuracy(batch_pert_logits, batch_clean_logits.argmax(dim=1))[0].item()

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
        inputs = inputs.to(self.device, non_blocking=True)
        arrays_list.append(torch.permute(copy.deepcopy(inputs.detach().cpu()), (0, 2, 3, 1)).numpy())

        # store the perturbed inputs
        with torch.no_grad():
            # compute the scores obtained from the score net
            scores = self.assign_net(inputs)
            scores = F.softmax(scores, dim=1)

            # perturb images and forward through the target net
            uap_idxs = [int(i) for i in torch.argmax(scores, dim=1)]
            pert_inputs = [x + self.uaps[idx] for x, idx in zip(inputs, uap_idxs)]
            pert_inputs = torch.stack(pert_inputs)
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
        self.logger.info(
            f"The counts of all uaps: \n{self.top_counts_all_uaps}"
        )

    def init_uaps(self):

        if self.init_method == "zero":
            uaps = torch.zeros(self.k, self.train_loader.data_info[0], self.train_loader.data_info[1], self.train_loader.data_info[1], device=self.device)
        elif self.init_method == "uniform":
            uaps = torch.FloatTensor(self.k, self.train_loader.data_info[0], self.train_loader.data_info[1], self.train_loader.data_info[1]).uniform_(-self.xi, self.xi).to(self.device, non_blocking=True)
        else:
            assert False
        uaps.requires_grad_(True)
        return uaps

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

    def callback_save_ckpt(self):
        self.ckptio.save()
