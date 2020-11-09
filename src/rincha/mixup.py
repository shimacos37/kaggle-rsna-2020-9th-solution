import torch
import torch.nn as nn
import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class CutMix(nn.Module):
    def __init__(self, p=1, beta=1):
        super(CutMix, self).__init__()
        self.p = p
        self.beta = beta

    def forward(self, input, target):
        r = np.random.rand(1)
        if r < self.p:
            lam = np.random.beta(self.beta, self.beta)
            rand_index = torch.randperm(input.size(0), device=input.device)
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            target = target_a * lam + target_b * (1 - lam)
        return input, target


class MixUp(nn.Module):
    def __init__(self, p=1, beta=1):
        super(MixUp, self).__init__()
        self.p = p
        self.beta = beta

    def forward(self, input, target):
        r = np.random.rand(1)
        if r < self.p:
            lam = np.random.beta(self.beta, self.beta)
            rand_index = torch.randperm(input.size(0), device=input.device)
            input_a, input_b = input, input[rand_index]
            target_a, target_b = target, target[rand_index]
            input = input_a * lam + input_b * (1 - lam)
            target = target_a * lam + target_b * (1 - lam)
        return input, target

