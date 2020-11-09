import hashlib
import multiprocessing as mp
import os
import random
import re
import shutil
from glob import glob

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
from base_model import BaseModel
from lib.metrics import competition_score
from lib.sync_batchnorm import convert_model


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


def prepair_dir(config):
    """
    Logの保存先を作成
    """
    for path in [
        config.store.result_path,
        config.store.log_path,
        config.store.model_path,
        config.store.feature_path,
    ]:
        if (
            os.path.exists(path)
            and config.train.warm_start is False
            and config.data.is_train
        ):
            shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)


def set_up(config):
    # Setup
    prepair_dir(config)
    set_seed(config.train.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(config.base.gpu_id)


class GPUModel(BaseModel, object):
    def __init__(self, hparams):
        super(GPUModel, self).__init__(hparams)
        self.cpu_count = mp.cpu_count() // len(self.base_config.gpu_id)
        if self.data_config.is_train:
            self.num_train_optimization_steps = int(
                self.train_config.epoch
                * len(self.train_dataset)
                / (self.train_config.batch_size)
                / self.train_config.accumulation_steps
                / len(self.base_config.gpu_id)
            )
        else:
            self.num_train_optimization_steps = 100  # 適当
        self.model = convert_model(self.model)

    def train_dataloader(self):
        if self.use_ddp:
            if self.data_config.dataset_name == "single_image_dataset":
                sampler = get_sampler("weighted_sampler", dataset=self.train_dataset)
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    self.train_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True,
                )
        else:
            sampler = torch.utils.data.RandomSampler(self.train_dataset)
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            pin_memory=True,
            drop_last=True,
            num_workers=self.cpu_count,
            sampler=sampler,
        )
        return train_loader

    def val_dataloader(self):
        if self.use_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=False,
            )
        else:
            sampler = torch.utils.data.SequentialSampler(
                self.valid_dataset,
            )
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.test_config.batch_size,
            num_workers=self.cpu_count,
            pin_memory=True,
            sampler=sampler,
        )

        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_config.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.cpu_count,
        )
        return test_loader

    def on_fit_end(self):
        """
        Called at the very end of fit.
        If on DDP it is called on every process
        """
        if self.data_config.n_fold == 4 and self.data_config.is_train:
            dfs = pd.concat(
                [
                    pd.read_csv(
                        f"{self.store_config.root_path}/fold{i}/result/valid_result_all.csv"
                    )
                    for i in range(5)
                ],
                axis=0,
            )
            dfs.to_csv(
                f"{self.store_config.root_path}/{self.store_config.model_name}_train.csv",
                index=False,
            )
            score = competition_score(dfs)
            print("all_score:", score)


@hydra.main(config_path="yamls/nn.yaml")
def main(config):
    set_up(config)
    os.chdir(config.data.workdir)
    # Preparing for trainer
    monitor_metric = "avg_val_score"
    checkpoint_callback = ModelCheckpoint(
        config.store.model_path,
        monitor_metric,
        verbose=True,
        save_top_k=1,
        mode="min",
        prefix=config.store.model_name,
        save_weights_only=False,
    )
    hparams = {}
    for _, value in config.items():
        hparams.update(value)
    if config.store.wandb_project is not None:
        logger = WandbLogger(
            name=config.store.model_name + f"_fold{config.data.n_fold}",
            save_dir=config.store.log_path,
            project=config.store.wandb_project,
            version=hashlib.sha224(bytes(str(hparams), "utf8")).hexdigest()[:4],
            anonymous=True,
            group=config.store.model_name,
        )
    else:
        logger = None

    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=5,
        verbose=False,
        mode="min",
    )

    backend = "ddp" if len(config.base.gpu_id) > 1 else None
    if config.train.warm_start:
        checkpoint_path = sorted(
            glob(config.store.model_path + "/*epoch*"),
            key=lambda x: re.split("[=.]", x)[-2],
        )[-1]
        print(checkpoint_path)
    else:
        checkpoint_path = None

    model = GPUModel(config)
    params = {
        "logger": logger,
        "max_epochs": config.train.epoch,
        "checkpoint_callback": checkpoint_callback,
        "callbacks": [early_stop_callback],
        "accumulate_grad_batches": config.train.accumulation_steps,
        "amp_backend": "native",
        "amp_level": "O1",
        "gpus": len(config.base.gpu_id),
        "distributed_backend": backend,
        "limit_train_batches": 1.0,
        "check_val_every_n_epoch": 1,
        "limit_val_batches": 1.0,
        "limit_test_batches": 0.0,
        "num_sanity_val_steps": 5,
        "num_nodes": 1,
        "gradient_clip_val": 0.5,
        "log_every_n_steps": 10,
        "flush_logs_every_n_steps": 10,
        "profiler": True,
        "deterministic": True,
        "resume_from_checkpoint": checkpoint_path,
        "weights_summary": "top",
        "reload_dataloaders_every_epoch": True,
        "replace_sampler_ddp": False,
    }
    if config.data.is_train:
        trainer = Trainer(**params)
        trainer.fit(model)
    else:
        params.update(
            {
                "gpus": 1,
                "logger": None,
                "limit_train_batches": 0.0,
                "limit_val_batches": 0.0,
                "limit_test_batches": 1.0,
            }
        )
        trainer = Trainer(**params)
        trainer.test(model, model.test_dataloader())


if __name__ == "__main__":
    main()
