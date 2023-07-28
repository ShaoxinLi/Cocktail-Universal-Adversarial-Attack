#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import torch
import warnings
import torch.nn.functional as F

from ..utils import seed_everything, save_record, CheckpointIO
from ..utils.time import convert_secs2time
from ..utils.plot import plot_metrics, plot_images
from ..metrics import AverageMeter, accuracy, LossConstructor

warnings.filterwarnings("ignore")


class NAGAttack():
    def __init__(self, exp_dir, xi, opt_alg, lr, n_epochs, logger, latent_dim=10, device=None, num_gpus=1, seed=None):

        self.exp_dir = exp_dir
        self.xi = xi / 255.
        self.opt_alg = opt_alg
        self.lr = lr
        self.n_epochs = n_epochs
        self.logger = logger
        self.latent_dim = latent_dim
        self.num_gpus = num_gpus
        self.seed = seed
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def fit(self, generator, net, train_loader):

        self.generator = generator
        self.net = net
        self.train_loader = train_loader

        # prepare the routine for fitting
        self.on_fit_start()

        for epoch in range(self.n_epochs):

            # train one epoch
            self.on_train_epoch_start(epoch)
            train_fooling_loss, train_diversity_loss, train_fr = self.train_epoch(epoch)
            self.on_train_epoch_end(epoch)

            # callbacks
            self.callback_save_record(
                fooling_loss=train_fooling_loss, diversity_loss=train_diversity_loss, train_fr=train_fr,
                epoch=epoch + 1)
            self.callback_save_ckpt()

        self.on_fit_end()

    def on_fit_start(self):

        # set the random seed
        seed_everything(self.seed)

        # parallel the network if possible
        self.net.to(self.device)
        self.generator.to(self.device)
        if self.device == torch.device("cuda") and torch.cuda.device_count() > 1 and self.num_gpus > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=list(range(self.num_gpus)))
            self.generator = torch.nn.DataParallel(self.generator, device_ids=list(range(self.num_gpus)))

        # configure the optimizer and the scheduler
        self.opt, self.sch = self.configure_optimizers(paras=self.generator.parameters(), opt_alg=self.opt_alg, lr=self.lr)

        # set the criterion
        self.criterion = LossConstructor(
            src_loss_fn_name="neg_logit_loss_nag", other_loss_fn_name="", src_classes=None,
            n_classes=self.train_loader.data_info[3], confidence=0, alpha=0, device=self.device
        ).to(self.device)

        # set the training history file path
        self.his_file = os.path.join(self.exp_dir, "train_history.csv")

        # set a timer
        self.epoch_start_time = time.time()
        self.epoch_time_meter = AverageMeter()

        # set the ckptio
        self.ckptio = CheckpointIO(
            ckpt_dir=self.exp_dir,
            multi_gpu=torch.cuda.device_count() > 1 and self.num_gpus > 1,
            opt_state=self.opt, generator=self.generator
        )

    def on_train_epoch_start(self, epoch):

        need_hour, need_mins, need_secs = convert_secs2time(self.epoch_time_meter.avg * (self.n_epochs - epoch))
        need_time = f"[Need: {need_hour:02d}:{need_mins:02d}:{need_secs:02d}]"
        self.logger.info(f"===========================> [Epoch={epoch + 1:03d} / {self.n_epochs:03d}] {need_time:s}")

        # init average meters
        self.train_fooling_loss_meter = AverageMeter()
        self.train_diversity_loss_meter = AverageMeter()
        self.train_acc_meter = AverageMeter()
        self.train_fr_meter = AverageMeter()

        # init timers
        self.batch_start_time = time.time()
        self.data_time_meter = AverageMeter()
        self.batch_time_meter = AverageMeter()

        # switch net mode
        self.net.eval()
        self.generator.train()

    def train_epoch(self, epoch):

        for i, (batch_inputs, batch_gt_targets) in enumerate(self.train_loader):

            batch_inputs, batch_gt_targets = batch_inputs.to(self.device), batch_gt_targets.to(self.device)

            # measure data loading time
            self.data_time_meter.update(time.time() - self.batch_start_time)

            # train one step
            batch_pert_logits, batch_clean_logits, fooling_loss, diversity_loss = self.train_step(batch_inputs, batch_gt_targets)

            # on train step end
            self.on_train_step_end(batch_pert_logits, batch_clean_logits, batch_gt_targets, fooling_loss, diversity_loss)

        return self.train_fooling_loss_meter.avg, self.train_diversity_loss_meter.avg, self.train_fr_meter.avg

    def train_step(self, batch_inputs, batch_gt_targets):

        # get the clean logits and the clean labels
        with torch.no_grad():
            batch_clean_logits = self.net(batch_inputs)
            batch_clean_softmax = F.softmax(batch_clean_logits, dim=1)

        self.opt.zero_grad()

        # generate a batch of perturbations
        latent_seed = 2 * torch.rand(batch_inputs.size(0), self.latent_dim, 1, 1, device=self.device) - 1
        batch_uaps = self.generator(latent_seed)

        # compute fooling loss
        batch_pert_inputs = batch_inputs + batch_uaps
        batch_pert_inputs = torch.clamp(batch_pert_inputs, 0.0, 1.0)
        batch_pert_logits = self.net(batch_pert_inputs)
        batch_pert_softmax = F.softmax(batch_pert_logits, dim=1)
        fooling_loss = torch.mean(self.criterion(batch_pert_softmax, batch_clean_softmax, batch_gt_targets))

        # compute diversity loss
        shuffled_idxs = torch.randperm(batch_inputs.size(0))
        batch_uaps_shuffled = batch_uaps[shuffled_idxs]
        batch_pert_inputs_shuffled = batch_inputs + batch_uaps_shuffled
        batch_pert_inputs_shuffled = torch.clamp(batch_pert_inputs_shuffled, 0.0, 1.0)
        batch_pert_logits_shuffled = self.net(batch_pert_inputs_shuffled)
        batch_pert_softmax_shuffled = F.softmax(batch_pert_logits_shuffled, dim=1)
        diversity_loss = torch.cosine_similarity(batch_pert_softmax, batch_pert_softmax_shuffled).mean()

        # backward
        if not torch.isinf(fooling_loss):
            total_loss = fooling_loss + diversity_loss
            total_loss.backward()
            self.opt.step()
        else:
            fooling_loss = torch.tensor(0.0)

        return batch_pert_logits, batch_clean_logits, fooling_loss.item(), diversity_loss.item()

    def on_train_step_end(self, batch_pert_logits, batch_clean_logits, batch_gt_targets, fooling_loss,
                          diversity_loss):

        with torch.no_grad():
            acc = accuracy(batch_pert_logits, batch_gt_targets)[0].item()
            fr = 1. - accuracy(batch_pert_logits, batch_clean_logits.argmax(dim=1))[0].item()

        # update metric meters
        self.train_fooling_loss_meter.update(fooling_loss, batch_pert_logits.size(0))
        self.train_diversity_loss_meter.update(diversity_loss, batch_pert_logits.size(0))
        self.train_acc_meter.update(acc, batch_pert_logits.size(0))
        self.train_fr_meter.update(fr, batch_pert_logits.size(0))

        # measure elapsed time
        self.batch_time_meter.update(time.time() - self.batch_start_time)
        self.batch_start_time = time.time()

    def on_train_epoch_end(self, epoch):

        self.logger.info(
            f"*********Train*********\tEpoch [{epoch + 1:03d}]  "
            f"Fooling Loss {self.train_fooling_loss_meter.avg:.3f}  "
            f"Diversity Lioss {self.train_diversity_loss_meter.avg:.3f}  "
            f"Accuracy {self.train_acc_meter.avg * 100.:.3f}  "
            f"Fooling ratio {self.train_fr_meter.avg * 100.:.3f}"
        )

        # use learning rate scheduler if it exists
        if self.sch is not None:
            self.sch.step()

    def on_fit_end(self):
        name_keys_map = {
            "fooling_loss": ["fooling_loss"],
            "diversity_loss": ["diversity_loss"],
            "train_fr": ["train_fr"]
        }
        for img_name, keys in name_keys_map.items():
            plot_metrics(
                csv_file=self.his_file, keys=keys, img_name=img_name,
                rolling_window_size=1, average_window_size=1
            )

        self.generator.eval()
        with torch.no_grad():
            latent_seed = 2 * torch.rand(10, self.latent_dim, 1, 1, device=self.device) - 1
            batch_uaps = self.generator(latent_seed)
        batch_uaps = torch.permute(batch_uaps.cpu(), (0, 2, 3, 1)).numpy()
        batch_uaps = (batch_uaps / self.xi + 1.) / 2.
        plot_images(batch_uaps, os.path.join(self.exp_dir, "uaps"))

        # set the testing result file path
        self.train_file = os.path.join(self.exp_dir, "train_result.csv")
        self.train_loader.shuffle = False
        with torch.no_grad():
            latent_seed = 2 * torch.rand(1, self.latent_dim, 1, 1, device=self.device) - 1
            self.uap = self.generator(latent_seed)
            assert torch.max(self.uap.data) <= self.xi
            assert torch.min(self.uap.data) >= -self.xi
        with torch.no_grad():
            for i, (batch_inputs, batch_gt_targets) in enumerate(self.train_loader):
                batch_inputs, batch_gt_targets = batch_inputs.to(self.device), batch_gt_targets.to(self.device)
                batch_clean_logits = self.net(batch_inputs)
                batch_pert_inputs = batch_inputs + self.uap
                batch_pert_inputs = torch.clamp(batch_pert_inputs, 0.0, 1.0)
                batch_pert_logits = self.net(batch_pert_inputs)
                for pert_logits, clean_logits, gt_target in zip(batch_pert_logits, batch_clean_logits, batch_gt_targets):
                    record = {
                        "gt_target": int(gt_target),
                        "clean_target": int(clean_logits.argmax()),
                        "clean_target_prob": float(clean_logits.max()),
                        "pert_target": int(pert_logits.argmax()),
                        "pert_target_prob": float(pert_logits.max())
                    }
                    save_record(self.train_file, **record)

    def test(self, generator, net, test_loader, auto_restore=False, uap=None, idx=None):

        self.generator = generator
        self.net = net
        self.test_loader = test_loader

        # on test start
        self.on_test_start(auto_restore, uap, idx)

        with torch.no_grad():
            for i, (batch_inputs, batch_gt_targets) in enumerate(test_loader):
                batch_inputs, batch_gt_targets = batch_inputs.to(self.device), batch_gt_targets.to(self.device)

                # test one step
                batch_pert_logits, batch_clean_logits = self.test_step(batch_inputs)

                # on test step end
                self.on_test_step_end(batch_pert_logits, batch_clean_logits, batch_gt_targets)

        # on test end
        self.on_test_end()

    def on_test_start(self, auto_restore, uap, idx):

        # auto restore the trained uap
        self.net.to(self.device)
        self.generator.to(self.device)
        if auto_restore:
            self.ckptio = CheckpointIO(
                ckpt_dir=self.exp_dir,
                multi_gpu=torch.cuda.device_count() > 1 and self.num_gpus > 1,
                generator=self.generator
            )
            self.ckptio.load()
            self.generator.eval()
            with torch.no_grad():
                latent_seed = 2 * torch.rand(1, self.latent_dim, 1, 1, device=self.device) - 1
                self.uap = self.generator(latent_seed)
        else:
            assert uap is not None
            self.uap = uap.to(self.device)
        assert torch.max(self.uap.data) <= self.xi
        assert torch.min(self.uap.data) >= -self.xi

        # parallel the network if possible
        if self.device == torch.device("cuda") and torch.cuda.device_count() > 1 and self.num_gpus > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=list(range(self.num_gpus)))
            self.generator = torch.nn.DataParallel(self.generator, device_ids=list(range(self.num_gpus)))

        # init metric meters
        self.test_acc_meter = AverageMeter()
        self.test_fr_meter = AverageMeter()

        # set the testing result file path
        self.test_file = os.path.join(self.exp_dir, f"test_result_uap{idx}.csv")

        # init distribution of perturbed targets
        self.pert_targets_dis = [0 for _ in range(self.test_loader.data_info[3])]

        # switch net mode
        self.net.eval()

    def test_step(self, batch_inputs):

        # get the clean logits
        batch_clean_logits = self.net(batch_inputs)
        batch_pert_inputs = batch_inputs + self.uap
        batch_pert_inputs = torch.clamp(batch_pert_inputs, 0.0, 1.0)

        # forward
        batch_pert_logits = self.net(batch_pert_inputs)

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

        # testing log
        self.logger.info(
            f"*********Test*********  "
            f"Accuracy {self.test_acc_meter.avg * 100.:.3f}  "
            f"Fooling ratio {self.test_fr_meter.avg * 100.:.3f}"
        )
        self.logger.info(
            f"The distribution of perturbed targets: \n{self.pert_targets_dis}"
        )

    def sample(self, k, generator, auto_restore=True):
        self.generator = generator
        # restore the best or latest checkpoint
        if auto_restore:
            self.ckptio = CheckpointIO(
                ckpt_dir=self.exp_dir,
                multi_gpu=torch.cuda.device_count() > 1 and self.num_gpus > 1,
                generator=self.generator
            )
            self.ckptio.load()

        self.generator.eval()
        self.generator.to(self.device)

        with torch.no_grad():
            latent_seed = 2 * torch.rand(k, self.latent_dim, 1, 1, device=self.device) - 1
            uaps = self.generator(latent_seed)

        assert torch.max(uaps.data) <= self.xi
        assert torch.min(uaps.data) >= -self.xi

        torch.save(uaps, os.path.join(self.exp_dir, f"sampled_{k}_UAPs.pth"))
        uaps = torch.permute(uaps.cpu(), (0, 2, 3, 1)).numpy()
        uaps = (uaps / self.xi + 1.) / 2.
        plot_images(uaps, os.path.join(self.exp_dir, f"sampled_{k}_UAPs"))

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