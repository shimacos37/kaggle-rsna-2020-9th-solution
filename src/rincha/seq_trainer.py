import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as ptl
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, CosineAnnealingLR
from torchcontrib.optim import SWA
import torch.distributed as dist
import torch_optimizer
from scheduler import CosineAnnealingWarmUpRestarts, WarmUpLR
from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F
from losses import MyLoss, MetricLoss
import numpy as np
from transformers.optimization import AdamW
from sklearn import metrics
from scipy.special import exp


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class SeqLightningModule(ptl.LightningModule):
    # モデルの定義(PyTorchと一緒)
    def __init__(self, model, cfg, train_dataset, valid_dataset, valid_result_path):
        super(SeqLightningModule, self).__init__()
        self.model = model
        self.cfg = cfg
        self.criterion = MyLoss(cfg, ignore_index=-1)
        self.per_exam_criterion = MyLoss(cfg, ignore_index=-1)
        self.per_image_criterion = MyLoss(cfg, ignore_index=-1)
        self.metric_loss = MetricLoss()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.valid_result_path = valid_result_path

    def setup(self, stage: str):
        try:
            self.size = dist.get_world_size()
            self.iter_per_epoch = int(int(
                len(self.train_dataset) / self.cfg.model.batch_size) / self.size / self.cfg.model.accumulate)
        except:
            print('no ddp')
            self.size = 1
            self.iter_per_epoch = int(int(
                len(self.train_dataset) / self.cfg.model.batch_size) / self.size / self.cfg.model.accumulate)

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_nb):
        per_exam_x, per_image_x = self.forward(batch)
        per_exam_target, per_image_target = batch['per_exam_target'], batch['per_image_target']
        image_weight = batch['image_weight']
        mask = batch['mask']
        per_image_x, per_image_target = per_image_x[mask], per_image_target[mask]
        image_weight = image_weight[mask]
        loss = self.metric_loss(per_image_x, per_exam_x, per_image_target, per_exam_target, image_weight)[0]
        if self.cfg.second.crf:
            loss += self.crf(per_image_x, per_image_target)
        tensorboard_logs = {}
        tensorboard_logs.update({'loss': loss, 'metric': loss})
        return {**tensorboard_logs, 'log': tensorboard_logs, 'step': self.current_epoch}

    def training_epoch_end(self, outputs):
        loss = torch.stack([output['loss'] for output in outputs]).mean()
        metric = torch.stack([output['metric'] for output in outputs]).mean()
        tensorboard_logs = {'train_loss': loss, 'step': self.current_epoch, 'train_metric': metric}
        return {'log': tensorboard_logs}

    def on_train_end(self) -> None:
        if self.cfg.optimizer.swa:
            if dist.get_rank() == 0:
                self.trainer.optimizers[0].swap_swa_sgd()
                self.trainer.save_checkpoint(os.path.join(self.trainer.checkpoint_callback.dirpath, "swa_model.ckpt"))
        return None

    # ミニバッチに対するバリデーションの関数
    def validation_step(self, batch, batch_nb):
        per_exam_x, per_image_x = self.forward(batch)
        per_image_x = self.crf.decode(per_image_x)
        per_exam_target, per_image_target = batch['per_exam_target'], batch['per_image_target']
        image_weight = batch['image_weight']
        mask = batch['mask']
        per_image_x, per_image_target = per_image_x[mask], per_image_target[mask]
        image_weight = image_weight[mask]
        loss, loss_denominator, loss_numerator = self.metric_loss(per_image_x, per_exam_x, per_image_target,
                                                                  per_exam_target, image_weight)
        tqdm_dict = {'val_loss': loss, 'val_metric': loss}
        out_log = tqdm_dict
        return {**out_log, 'loss_denominator': loss_denominator, 'loss_numerator': loss_numerator,
                'progress_bar': tqdm_dict}

    # バリデーションループが終わったときに実行される関数
    def validation_epoch_end(self, outputs):
        loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        loss_denominator = torch.stack([output['loss_denominator'] for output in outputs]).sum()
        loss_numerator = torch.stack([output['loss_numerator'] for output in outputs]).sum()
        metric = loss_denominator / loss_numerator
        tensorboard_logs = {'val_loss': metric, 'step': self.current_epoch, 'val_metric': metric}
        tqdm_dict = {'val_loss': metric}
        return {'val_loss': metric, 'log': tensorboard_logs, 'progress_bar': tqdm_dict}

    def test_step(self, batch, batch_nb):
        per_exam_x, per_image_x = self.model.predict(batch)
        return {'per_exam_x': F.sigmoid(per_exam_x).cpu().numpy(), 'per_image_x': F.sigmoid(per_image_x).cpu().numpy()}

    def test_epoch_end(self, outputs):
        per_exam_x = np.concatenate([output['per_exam_x'] for output in outputs])
        per_image_x = np.concatenate([output['per_image_x'] for output in outputs])
        return {'per_image_x': per_image_x, 'per_exam_x': per_exam_x}

    # 最適化アルゴリズムの指定
    def configure_optimizers(self):
        schedulers = []
        print('Number of iteration per epoch: {}'.format(self.iter_per_epoch))
        if self.cfg.optimizer.name == 'adam':
            optim = torch.optim.Adam(self.parameters(), lr=self.cfg.optimizer.lr)
        elif self.cfg.optimizer.name == 'radam':
            optim = torch_optimizer.RAdam(self.parameters(), lr=self.cfg.optimizer.lr)
        elif self.cfg.optimizer.name == 'novograd':
            optim = torch_optimizer.NovoGrad(self.parameters(), lr=self.cfg.optimizer.lr)
        elif self.cfg.optimizer.name == 'adamw':
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.weight', 'layer_norm.bias']
            optimizer_grouped_parameters = []
            weight_decay = 0.01
            for name, param in list(self.model.named_parameters()):
                if any(nd in name for nd in no_decay):
                    wd = 0
                else:
                    wd = weight_decay
                optimizer_grouped_parameters.append({'params': param, 'weight_decay': wd, 'lr': self.cfg.optimizer.lr})
            optim = AdamW(params=optimizer_grouped_parameters)
        if self.cfg.optimizer.lookahead:
            optim = torch_optimizer.Lookahead(optim)
        if self.cfg.optimizer.swa:
            optim = SWA(optim, swa_start=int(self.cfg.optimizer.num_epochs * 0.75 * self.iter_per_epoch),
                        swa_freq=self.iter_per_epoch)

        if self.cfg.optimizer.warmup:
            scheduler = WarmUpLR(optim, warmup_steps=self.iter_per_epoch * 4)
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
            schedulers.append(scheduler)
        if self.cfg.optimizer.scheduler == 'cosine_restart':
            scheduler = CosineAnnealingWarmUpRestarts(optim, last_epoch=-1, T_mult=2, T_0=self.iter_per_epoch * 2,
                                                      T_up=self.iter_per_epoch // 10, gamma=0.8, eta_max=1e-4)
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
            schedulers.append(scheduler)
        elif self.cfg.optimizer.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optim, last_epoch=-1,
                                          T_max=self.iter_per_epoch * self.cfg.optimizer.num_epochs, eta_min=5e-4)
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
            schedulers.append(scheduler)
        elif self.cfg.optimizer.scheduler == 'step':
            scheduler = MultiStepLR(optim, milestones=[7, 10], gamma=0.2)
            scheduler = {'scheduler': scheduler, 'interval': 'epoch'}
            schedulers.append(scheduler)
        elif self.cfg.optimizer.scheduler == 'reduce':
            scheduler = ReduceLROnPlateau(optim, mode='min', patience=1, verbose=True, factor=0.2)
            scheduler = {'scheduler': scheduler, 'interval': 'epoch', 'monitor': 'val_loss'}
            schedulers.append(scheduler)
        elif self.cfg.optimizer.scheduler == 'warm_linear':
            scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=self.iter_per_epoch * 0.1,
                                                        num_training_steps=self.iter_per_epoch * self.cfg.optimizer.num_epochs)
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
            schedulers.append(scheduler)
        elif self.cfg.optimizer.scheduler == 'warm_constant':
            scheduler = get_constant_schedule_with_warmup(optim, num_warmup_steps=self.iter_per_epoch * 4)
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
            schedulers.append(scheduler)

        else:
            return [optim]
        return [optim], schedulers

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.model.batch_size, shuffle=True, num_workers=12,
                          drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.cfg.model.batch_size, shuffle=True,
                          num_workers=12, pin_memory=True)
