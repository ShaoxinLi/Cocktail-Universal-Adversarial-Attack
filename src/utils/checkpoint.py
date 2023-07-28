#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import copy
import torch
from .file import check_dir


class CheckpointIO(object):

    def __init__(self, ckpt_dir, multi_gpu=False, **kwargs):
        check_dir(ckpt_dir)
        self.ckpt_path = os.path.join(ckpt_dir, "checkpoint.pth.tar")
        self.module_dict = kwargs
        self.multi_gpu = multi_gpu

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save(self):
        out_dict = {}
        for name, module in self.module_dict.items():
            if isinstance(module, torch.nn.Module) or isinstance(module, torch.optim.Optimizer):
                if self.multi_gpu:
                    out_dict[name] = module.module.state_dict()
                else:
                    out_dict[name] = module.state_dict()
            elif isinstance(module, torch.Tensor):
                out_dict[name] = copy.deepcopy(module.detach())
            else:
                assert False
        torch.save(out_dict, self.ckpt_path)

    def load(self):
        return self.load_from_path(self.ckpt_path)

    def load_from_path(self, ckpt_path):
        assert os.path.exists(ckpt_path), f"{ckpt_path} does not exist!"
        if torch.cuda.is_available():
            module_dict = torch.load(ckpt_path)
        else:
            module_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
        for name, module in self.module_dict.items():
            if isinstance(module, torch.nn.Module) or isinstance(module, torch.optim.Optimizer):
                if self.multi_gpu:
                    module.module.load_state_dict(module_dict[name])
                else:
                    module.load_state_dict(module_dict[name])
            elif isinstance(module, torch.Tensor) or module is None:
                self.module_dict[name] = module_dict[name]
            else:
                assert False
        return self.module_dict