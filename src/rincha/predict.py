import hydra
import warnings
import os
import re
from model import ImageModel
from omegaconf import DictConfig
import numpy as np
import torch
import random
from pathlib import Path
from dataproc import DICOMDataproc
from trainer import MyLightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
hydra.output_subdir = "null"
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
    dataproc = DICOMDataproc(cfg=cfg, data_root=cwd / 'input')
    out_path = cwd / 'output' / 'result_tb'
    trained_path = cwd / 'output' / 'result_model'
    if cfg.exp.pred == 'valid':
        valid_result_path = cwd / 'output' / 'result_valid'
    else:
        valid_result_path = cwd / 'output' / 'result_test'
    train_dataset, valid_dataset = dataproc.get_dataset(fold=fold)
    out_path = out_path / cfg.model.name
    trained_path = trained_path / cfg.model.name / cfg.exp.name / f'fold_{fold + 1}'
    valid_result_path = valid_result_path / cfg.model.name / cfg.exp.name / f'fold_{fold + 1}'
    model = ImageModel(cfg=cfg)
    out_path.mkdir(parents=True, exist_ok=True)
    trained_path.mkdir(parents=True, exist_ok=True)
    valid_result_path.mkdir(parents=True, exist_ok=True)
    if cfg.exp.pred == 'valid':
        dataset, df = dataproc.get_all_dataset(fold=fold)
    else:
        dataset, df = dataproc.get_test_dataset(fold=fold)
    data_loader = DataLoader(dataset, batch_size=cfg.model.batch_size, shuffle=False, drop_last=False,
                             num_workers=12, pin_memory=True)

    model = MyLightningModule(model=model, cfg=cfg, train_dataset=train_dataset, valid_dataset=valid_dataset,
                              valid_result_path=valid_result_path, df=df)

    resume_path = trained_path / 'last.ckpt'
    state = torch.load(resume_path)['state_dict']
    print('Resume path: {}'.format(resume_path))
    model.load_state_dict(state)
    trainer = Trainer(max_epochs=cfg.optimizer.num_epochs, default_root_dir=os.getcwd(),
                      gpus=[0], accumulate_grad_batches=cfg.model.accumulate)
    result = trainer.test(model, verbose=False, test_dataloaders=data_loader)[0]
    num_list = df.groupby(['StudyInstanceUID', 'SeriesInstanceUID'])['InstanceNumber'].count()
    i = 0
    for idx, num, in num_list.iteritems():
        StudyInstanceUID, SeriesInstanceUID = idx
        cnn_feature = result['result'][i: i + num]
        (valid_result_path / 'cnn_feature' / StudyInstanceUID).mkdir(exist_ok=True, parents=True)
        np.save(valid_result_path / 'cnn_feature' / StudyInstanceUID / SeriesInstanceUID, cnn_feature)
        i += num
    del trainer, model


@hydra.main(config_path='config.yaml')
def main(cfg: DictConfig):
    run(cfg.exp.fold, cfg=cfg)


if __name__ == '__main__':
    main()
