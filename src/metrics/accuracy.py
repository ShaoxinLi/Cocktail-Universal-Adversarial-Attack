#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch


@torch.no_grad()
def accuracy(outputs, targets, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        _, pred = outputs.topk(maxk, 1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k / batch_size)
        return res

