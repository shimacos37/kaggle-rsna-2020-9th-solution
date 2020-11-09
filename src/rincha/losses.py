import torch.nn as nn
import torch
from timm.loss import LabelSmoothingCrossEntropy


def get_criterion(name):
    if name == 'smooth':
        criterion = LabelSmoothingCrossEntropy(0.05)
    elif name == 'ce':
        criterion = nn.BCEWithLogitsLoss()
    return criterion


class MyLoss(nn.Module):
    def __init__(self, cfg, ignore_index):
        super(MyLoss, self).__init__()
        self.cfg = cfg
        self.ignore_index = ignore_index
        self.criterion = [nn.BCEWithLogitsLoss()]

    def forward(self, input, target):
        input, target = input[target != self.ignore_index], target[target != self.ignore_index]
        loss = 0
        for criterion in self.criterion:
            loss += criterion(input, target)
        return loss


class MetricLoss(nn.Module):
    def __init__(self):
        super(MetricLoss, self).__init__()
        self.weight = nn.Parameter(torch.tensor([0.0736196319,
                                                 0.09202453988,
                                                 0.1042944785,
                                                 0.1042944785,
                                                 0.1877300613,
                                                 0.06257668712,
                                                 0.06257668712,
                                                 0.2346625767,
                                                 0.0782208589]), requires_grad=False)
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, per_image_x, per_exam_x, per_image_target, per_exam_target, image_weight, ):
        per_exam_loss = self.bce(per_exam_x, per_exam_target)
        per_exam_loss = (per_exam_loss * self.weight).sum()
        per_image_loss = self.bce(per_image_x, per_image_target)
        per_image_loss = (per_image_loss * image_weight).sum()
        loss_denominator = per_exam_loss + per_image_loss
        loss_numerator = self.weight.sum() * len(per_exam_x) + image_weight.sum()
        loss = loss_denominator / loss_numerator
        return loss, loss_denominator, loss_numerator
