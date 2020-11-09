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
from losses import MyLoss
import numpy as np
from transformers.optimization import AdamW
from sklearn import metrics


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class MyLightningModule(ptl.LightningModule):
    # モデルの定義(PyTorchと一緒)
    def __init__(self, model, cfg, train_dataset, valid_dataset, valid_result_path, df):
        super(MyLightningModule, self).__init__()
        self.model = model
        self.cfg = cfg
        self.df = df
        self.criterion = MyLoss(cfg, ignore_index=-1)
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
        x = self.forward(batch)
        target = batch['target']
        loss = self.criterion(x, target)
        tensorboard_logs = {}
        tensorboard_logs.update({'loss': loss})
        tqdm_dict = {'train_loss': loss}
        return {'loss': loss, **tensorboard_logs, 'log': tensorboard_logs, 'progress_bar': tqdm_dict,
                'step': self.current_epoch}

    def training_epoch_end(self, outputs):
        loss = torch.stack([output['loss'] for output in outputs]).mean()
        tensorboard_logs = {'train_loss': loss, 'step': self.current_epoch}
        return {'log': tensorboard_logs}

    def on_train_end(self) -> None:
        if self.cfg.optimizer.swa:
            if dist.get_rank() == 0:
                self.trainer.optimizers[0].swap_swa_sgd()
                self.trainer.save_checkpoint(os.path.join(self.trainer.checkpoint_callback.dirpath, "swa_model.ckpt"))
        return None

    # ミニバッチに対するバリデーションの関数
    def validation_step(self, batch, batch_nb):
        x = self.forward(batch)
        target = batch['target']
        loss = self.criterion(x, target)
        true = target
        pred = x
        tqdm_dict = {'val_loss': loss}
        out_log = tqdm_dict
        return {**out_log, 'progress_bar': tqdm_dict, 'pred': pred, 'true': true}

    # バリデーションループが終わったときに実行される関数
    def validation_epoch_end(self, outputs):
        loss = torch.stack([output['val_loss'] for output in outputs]).mean()
        pred = torch.cat([output['pred'] for output in outputs], dim=0)
        true = torch.cat([output['true'] for output in outputs], dim=0)
        val_pred_list = [torch.zeros_like(pred, dtype=pred.dtype).to(pred.device) for _ in
                         range(self.size)]
        val_true_list = [torch.zeros_like(true, dtype=true.dtype).to(true.device) for _ in
                         range(self.size)]
        dist.all_gather(val_pred_list, pred)
        dist.all_gather(val_true_list, true)
        pred = torch.cat(val_pred_list, dim=0).cpu().numpy().astype(np.float)[:, 0]
        true = torch.cat(val_true_list, dim=0).cpu().numpy().astype(np.int)[:, 0]
        pred, true = pred[true != -1], true[true != -1]
        try:
            auc = metrics.roc_auc_score(true, pred)
        except:
            auc = -1
        tensorboard_logs = {'val_loss': loss, 'step': self.current_epoch, 'val_pe_auc': auc}
        tqdm_dict = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs, 'progress_bar': tqdm_dict}

    def test_step(self, batch, batch_nb):
        x = self.model.get_feature(batch)
        index = batch['index']
        return {'result': x.cpu().numpy(), 'index': index.cpu().numpy()}

    def test_epoch_end(self, outputs):

        result = np.concatenate([output['result'] for output in outputs])
        index = np.concatenate([output['index'] for output in outputs])
        return {'result': result, 'index': index}

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
                                          T_max=self.iter_per_epoch * self.cfg.optimizer.num_epochs)
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
            scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=self.iter_per_epoch * 4,
                                                        num_training_steps=self.iter_per_epoch * self.cfg.optimizer.num_epochs)
            scheduler = {'scheduler': scheduler, 'interval': 'step'}
            schedulers.append(scheduler)
        elif self.cfg.optimizer.scheduler == 'warm_constant':
            scheduler = get_constant_schedule_with_warmup(optim, num_warmup_steps=self.iter_per_epoch * 0.2)
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
