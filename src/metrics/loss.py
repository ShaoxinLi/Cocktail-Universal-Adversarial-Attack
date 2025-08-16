#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F


class LossConstructor(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, src_loss_fn_name, other_loss_fn_name="", src_classes=None, n_classes=10, confidence=0.,
                 alpha=0.0, weight=None, size_average=None, reduce=None, reduction="none",
                 device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super(LossConstructor, self).__init__(weight, size_average, reduce, reduction)

        self.device = device
        self.n_classes = n_classes
        self.confidence = torch.tensor(confidence)
        self.alpha = alpha
        self.src_classes = list(range(self.n_classes)) if src_classes is None else src_classes

        if src_loss_fn_name == "neg_logits_ce":
            self.src_loss_fn = neg_logits_ce
        elif src_loss_fn_name == "neg_bounded_ce":
            self.src_loss_fn = neg_bounded_ce
        elif src_loss_fn_name == "neg_logit":
            self.src_loss_fn = neg_logit_loss
        elif src_loss_fn_name == "neg_bounded_logit":
            self.src_loss_fn = neg_bounded_logit_loss
        elif src_loss_fn_name == "neg_cosine_sim":
            self.src_loss_fn = neg_cosine_sim
        elif src_loss_fn_name == "neg_logit_loss_nag":
            self.src_loss_fn = neg_logit_loss_nag
        else:
            assert False, f"Loss {src_loss_fn_name} is not supported yet."

        if other_loss_fn_name == "":
            self.other_loss_fn = empty
        elif other_loss_fn_name == "logits_ce":
            self.other_loss_fn = logits_ce
        elif other_loss_fn_name == "ce":
            self.other_loss_fn = ce
        elif other_loss_fn_name == "logit":
            self.other_loss_fn = logit_loss
        elif other_loss_fn_name == "cosine_sim":
            self.other_loss_fn = cosine_sim
        else:
            assert False, f"Loss {other_loss_fn_name} is not supported yet."

    def forward(self, pert_logits, clean_logits, targets):

        num_repeat = int(pert_logits.size(0) / clean_logits.size(0)) if pert_logits.size(0) != clean_logits.size(0) else 1
        clean_logits = clean_logits.repeat_interleave(num_repeat, dim=0)
        targets = targets.repeat_interleave(num_repeat, dim=0)

        # compute the loss of source classes
        src_classes_mask = torch.tensor([i in self.src_classes for i in targets])
        if len(self.src_classes) == self.n_classes:
            assert sum(src_classes_mask) == pert_logits.size(0)
        if torch.sum(src_classes_mask) > 0:
            src_pert_logits = pert_logits[src_classes_mask]
            src_clean_logits_src = clean_logits[src_classes_mask]
            src_loss = self.src_loss_fn(
                src_pert_logits, src_clean_logits_src, n_classes=self.n_classes,
                confidence=self.confidence, device=self.device
            )
        else:
            src_loss = torch.tensor([], device=self.device)

        # compute the loss of other classes
        other_classes_mask = ~src_classes_mask
        if len(self.src_classes) == self.n_classes:
            assert sum(other_classes_mask) == 0
        if torch.sum(other_classes_mask) > 0 and self.alpha != 0.0:
            other_pert_logits = pert_logits[other_classes_mask]
            other_clean_logits = clean_logits[other_classes_mask]
            others_loss = self.other_loss_fn(
                other_pert_logits, other_clean_logits, n_classes=self.n_classes,
                confidence=self.confidence, device=self.device
            )
        else:
            others_loss = torch.tensor([], device=self.device)

        # total losses
        loss = torch.cat((src_loss, self.alpha * others_loss), dim=0)
        if len(loss) == 0:
            loss = torch.tensor([0.], requires_grad=True)
        return loss


def empty(pert_logits, clean_logits, n_classes=None, confidence=None, device=None):

    loss = torch.tensor([], requires_grad=True, device=device)
    return loss


def logits_ce(pert_logits, clean_logits, n_classes=None, confidence=None, device=None):

    loss = torch.nn.CrossEntropyLoss(reduction="none")(pert_logits, clean_logits)
    return loss


def neg_logits_ce(pert_logits, clean_logits, n_classes=None, confidence=None, device=None):

    loss = -torch.nn.CrossEntropyLoss(reduction="none")(pert_logits, clean_logits)
    return loss


def ce(pert_logits, clean_logits, n_classes=None, confidence=None, device=None):

    clean_targets = clean_logits.argmax(dim=1)
    loss = torch.nn.CrossEntropyLoss(reduction="none")(pert_logits, clean_targets)
    return loss


def neg_bounded_ce(pert_logits, clean_logits, n_classes=None, confidence=None, device=None):

    clean_targets = clean_logits.argmax(dim=1)

    # according to UA-training paper
    loss = -torch.min(torch.nn.CrossEntropyLoss(reduction="none")(pert_logits, clean_targets), confidence)
    return loss


def logit_loss(pert_logits, clean_logits, n_classes=None, confidence=None, device=None):

    clean_targets = clean_logits.argmax(dim=1)
    clean_targets_one_hot = F.one_hot(clean_targets, num_classes=n_classes)

    # compute logits of clean targets
    logits_of_clean_targets = (clean_targets_one_hot * pert_logits).sum(1)
    loss = -logits_of_clean_targets
    return loss


def neg_logit_loss(pert_logits, clean_logits, n_classes=None, confidence=None, device=None):

    clean_targets = clean_logits.argmax(dim=1)
    clean_targets_one_hot = F.one_hot(clean_targets, num_classes=n_classes)

    # compute logits of clean targets
    logits_of_clean_targets = (clean_targets_one_hot * pert_logits).sum(1)
    loss = logits_of_clean_targets
    return loss


def neg_logit_loss_nag(pert_logits, clean_logits, n_classes=None, confidence=None, device=None):
    # from NAG paper

    clean_targets = clean_logits.argmax(dim=1)
    clean_targets_one_hot = F.one_hot(clean_targets, num_classes=n_classes)

    # compute logits of clean targets
    logits_of_clean_targets = (clean_targets_one_hot * pert_logits).sum(1).mean()
    loss = -torch.log(1 - logits_of_clean_targets).unsqueeze(0)
    return loss


def neg_bounded_logit_loss(pert_logits, clean_logits, n_classes=None, confidence=None, device=None):

    clean_targets = clean_logits.argmax(dim=1)
    clean_targets_one_hot = F.one_hot(clean_targets, num_classes=n_classes)

    # compute logits of clean targets
    logits_of_clean_targets = (clean_targets_one_hot * pert_logits).sum(1)

    # for logits of non-clean labels, we don't have to compute the grad w.r.t. the logits again
    max_logits_of_non_clean_target = ((1 - clean_targets_one_hot) * pert_logits - clean_targets_one_hot * 100000.).max(1)[0].detach()

    # according to DT-UAPs paper
    loss = torch.clamp(logits_of_clean_targets - max_logits_of_non_clean_target, min=-confidence)
    return loss


def cosine_sim(pert_logits, clean_logits, n_classes=None, confidence=None, device=None):

    loss = -torch.nn.CosineSimilarity()(clean_logits, pert_logits)
    return loss


def neg_cosine_sim(pert_logits, clean_logits, n_classes=None, confidence=None, device=None):

    loss = torch.nn.CosineSimilarity()(clean_logits, pert_logits)
    return loss
