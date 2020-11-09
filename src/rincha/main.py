import hydra
import warnings
import os
import re
from model import ImageModel
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import random
from pathlib import Path
from dataproc import DICOMDataproc
from trainer import MyLightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
hydra.output_subdir = "null"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run(fold, cfg):
    warnings.filterwarnings('ignore')
    cwd = Path(hydra.utils.get_original_cwd())
    print(cfg.pretty())
    random.seed(cfg.exp.seed)
    np.random.seed(cfg.exp.seed)
    torch.manual_seed(cfg.exp.seed)
    torch.cuda.manual_seed(cfg.exp.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    dataproc = DICOMDataproc(cfg=cfg, data_root=cwd / 'input')
    out_path = cwd / 'output' / 'result_tb'
    trained_path = cwd / 'output' / 'result_model'
    valid_result_path = cwd / 'output' / 'result_valid'

    print('\n\n\nFold: {} training start!!!\n\n\n'.format(fold + 1))
    train_dataset, valid_dataset = dataproc.get_dataset(fold=fold)
    out_path = out_path / cfg.model.name
    trained_path = trained_path / cfg.model.name / cfg.exp.name / f'fold_{fold + 1}'
    valid_result_path = valid_result_path / cfg.model.name / cfg.exp.name / f'fold_{fold + 1}'
    tb_logger = TensorBoardLogger(save_dir=out_path, name=cfg.exp.name, version=f'fold_{fold + 1}')

    (out_path / cfg.exp.name).mkdir(parents=True, exist_ok=True)
    model = ImageModel(cfg=cfg)
    out_path.mkdir(parents=True, exist_ok=True)
    trained_path.mkdir(parents=True, exist_ok=True)
    valid_result_path.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=trained_path / "{epoch:02d}-{val_loss:.4f}",
        monitor='val_loss', mode='min', save_top_k=1, save_weights_only=False, save_last=True)
    lr_logger = LearningRateMonitor()
    model = MyLightningModule(model=model, cfg=cfg, train_dataset=train_dataset, valid_dataset=valid_dataset,
                              valid_result_path=valid_result_path, df=dataproc.df)
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=2, verbose=True, mode='min')
    resume_path = str(trained_path / 'last.ckpt') if cfg.exp.resume else None

    if cfg.exp.resume and not os.path.exists(resume_path):
        resume_path = None
    trainer = Trainer(max_epochs=cfg.optimizer.num_epochs, default_root_dir=os.getcwd(),
                      gpus=-1 if cfg.exp.cuda else None, checkpoint_callback=checkpoint_callback, logger=tb_logger,
                      callbacks=[lr_logger], accumulate_grad_batches=cfg.model.accumulate,
                      resume_from_checkpoint=resume_path,
                      distributed_backend='ddp' if cfg.exp.cuda else None,
                      precision=16 if cfg.exp.fp16 else 32,
                      amp_backend='native' if cfg.exp.fp16 else None,
                      amp_level='O1' if cfg.exp.fp16 else None,
                      check_val_every_n_epoch=10, sync_batchnorm=True)
    trainer.fit(model)
    del trainer, model
    import gc
    gc.collect()


@hydra.main(config_path='config.yaml')
def main(cfg: DictConfig):
    run(cfg.exp.fold, cfg=cfg)


if __name__ == '__main__':
    main()
