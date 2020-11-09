import hydra
import warnings
import os
import re
from model import SeqModel, DoubleHopModel, DeconvFeatureModel
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import random
from pathlib import Path
from dataproc import SeqDataproc, HopDataproc
from seq_trainer import SeqLightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

os.environ["TOKENIZERS_PARALLELISM"] = "false"

target_col = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']


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
    out_path = cwd / 'seq_result_tb'
    trained_path = cwd / 'seq_result_model'
    valid_result_path = cwd / 'seq_result_valid'
    if cfg.second.hop:
        feature_512_root = cwd / 'result_valid' / 'tf_efficientnet_b5_ns' / f'fold_{fold + 1}' / 'cnn_feature'
        feature_384_root = cwd / 'result_valid' / 'tf_efficientnet_b3_ns' / f'fold_{fold + 1}' / 'cnn_feature'
        dataproc = HopDataproc(cfg=cfg, feature_512_root=feature_512_root, feature_384_root=feature_384_root,
                               data_root=cwd / '../input')
        if cfg.second.input_double:
            model = DeconvFeatureModel(cfg=cfg)
        else:
            model = DoubleHopModel(cfg=cfg)
    else:
        feature_root = cwd / 'result_valid' / cfg.second.model_name / cfg.second.name / f'fold_{fold + 1}' / 'cnn_feature'
        dataproc = SeqDataproc(cfg=cfg, feature_root=feature_root, data_root=cwd / '../input')
        model = SeqModel(cfg=cfg)
    print('\n\n\nFold: {} training start!!!\n\n\n'.format(fold + 1))
    train_dataset, valid_dataset = dataproc.get_dataset(fold=fold)
    out_path = out_path / cfg.model.name
    trained_path = trained_path / cfg.model.name / cfg.exp.name / f'fold_{fold + 1}'
    valid_result_path = valid_result_path / cfg.model.name / f'fold_{fold + 1}'
    tb_logger = TensorBoardLogger(save_dir=out_path, name=cfg.exp.name, version=f'fold_{fold + 1}')
    out_path.mkdir(parents=True, exist_ok=True)
    trained_path.mkdir(parents=True, exist_ok=True)
    valid_result_path.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        filepath=trained_path / "{epoch:02d}-{val_loss:.4f}",
        monitor='val_loss', mode='min', save_top_k=1, save_weights_only=False, save_last=True)
    lr_logger = LearningRateMonitor()
    model = SeqLightningModule(model=model, cfg=cfg, train_dataset=train_dataset, valid_dataset=valid_dataset,
                               valid_result_path=valid_result_path)
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
                      amp_level='O1' if cfg.exp.fp16 else None)
    trainer.fit(model)
    del model, trainer
    import gc
    gc.collect()


@hydra.main(config_path='config.yaml')
def main(cfg: DictConfig):
    for i in range(cfg.exp.k_folds):
        run(i, cfg=cfg)
    cwd = Path(hydra.utils.get_original_cwd())
    OmegaConf.save(cfg, cwd / 'seq_result_model' / 'config.yaml')


if __name__ == '__main__':
    main()
