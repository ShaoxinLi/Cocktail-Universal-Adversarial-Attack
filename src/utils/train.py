#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import torch


class EarlyStopper():
    def __init__(self, patience=5, delta=0.,):

        self.patience = patience
        self.delta = delta
        self.reset()

    def reset(self):

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, loss):

        score = -loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


@torch.no_grad()
def seed_everything(seed):

    assert seed is not None, f"Please set seeds before proceeding"
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = True