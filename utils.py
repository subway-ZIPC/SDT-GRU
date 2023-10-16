from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import logging
import os
import sys


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def get_logger(exp_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    path = os.path.join(exp_dir, 'log.txt')
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    if os.path.exists(path):
        os.remove(path)
    hdlr = logging.FileHandler(path)
    console = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    hdlr.setFormatter(formatter)
    console.setFormatter(formatter)

    logger.addHandler(hdlr)
    logger.addHandler(console)

    return logger


def move2device(x, device):
    if isinstance(x, list):
        y = []
        for item in x:
            y.append(move2device(item, device))
    elif isinstance(x, dict):
        y = {}
        for k, v in x.items():
            y[k] = move2device(v, device)
    elif x is None:
        y = None
    else:
        y = x.to(device)
    return y


class StepLR2(MultiStepLR):
    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 last_epoch=-1,
                 min_lr=2.0e-6,
                 warm_up=False,
                 warm_up_ep=0,
                 warm_up_lr=1e-4):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        self.warm_up = warm_up
        self.warm_up_ep = warm_up_ep
        self.warm_up_lr = warm_up_lr
        self.start = None
        super(StepLR2, self).__init__(optimizer, milestones, gamma)

    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()

        if self.last_epoch < self.warm_up_ep and self.warm_up:
            if isinstance(lr_candidate, list):
                if self.start is None:
                    self.start = lr_candidate[0]
                for i in range(len(lr_candidate)):
                    lr_candidate[i] = self.warm_up_lr
            else:
                if self.start is None:
                    self.start = lr_candidate
                lr_candidate = self.warm_up_lr

        if self.last_epoch > self.warm_up_ep and self.start is not None:
            if isinstance(lr_candidate, list):
                for i in range(len(lr_candidate)):
                    lr_candidate[i] = self.start
            else:
                lr_candidate = self.start
            self.start = None

        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])

        else:
            lr_candidate = max(self.min_lr, lr_candidate)

        return lr_candidate
