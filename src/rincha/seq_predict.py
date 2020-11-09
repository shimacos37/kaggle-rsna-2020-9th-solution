import hydra
import warnings
import os
import re
from model import SeqModel, DoubleHopModel, DeconvFeatureModel
from omegaconf import DictConfig
import numpy as np
import torch
import pandas as pd
import random
from pathlib import Path
from dataproc import SeqDataproc, HopDataproc
from seq_trainer import SeqLightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        feature_root = cwd / 'result_valid' / cfg.second.model_name / f'fold_{fold + 1}' / 'cnn_feature'
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
    resume_path = list(trained_path.glob('epoch=*.ckpt'))
    pattern = r'([+-]?[0-9]+\.?[0-9]*)'
    resume_path = str(resume_path[np.argmin([float(re.findall(pattern, str(i.name))[-1]) for i in resume_path])])
    state = torch.load(resume_path)['state_dict']
    print('Resume path: {}'.format(resume_path))
    model.load_state_dict(state)

    trainer = Trainer(max_epochs=cfg.optimizer.num_epochs, default_root_dir=os.getcwd(),
                      gpus=[0],
                      resume_from_checkpoint=resume_path,
                      distributed_backend='ddp' if cfg.exp.cuda else None,
                      precision=16 if cfg.exp.fp16 else 32,
                      amp_backend='native' if cfg.exp.fp16 else None,
                      amp_level='O1' if cfg.exp.fp16 else None)
    data_loader = DataLoader(valid_dataset, batch_size=cfg.model.batch_size, shuffle=False, drop_last=False,
                             num_workers=12)
    df = dataproc.get_df(fold=fold)
    result = trainer.test(model, test_dataloaders=data_loader, verbose=False)[0]
    per_image_x, per_exam_x = result['per_image_x'], result['per_exam_x']
    num_list = df.groupby(['StudyInstanceUID', 'SeriesInstanceUID'])['InstanceNumber'].count().reset_index()
    label_col = ['negative_exam_for_pe', 'indeterminate', 'chronic_pe',
                 'acute_and_chronic_pe', 'central_pe', 'leftsided_pe',
                 'rightsided_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']
    out = pd.DataFrame()
    for idx in tqdm(range(len(num_list))):
        tmp = pd.DataFrame()
        data = num_list.iloc[idx]
        num = data['InstanceNumber']
        StudyInstanceUID, SeriesInstanceUID = data['StudyInstanceUID'], data['SeriesInstanceUID']
        SOPInstanceUID = df[df['StudyInstanceUID'] == StudyInstanceUID]['SOPInstanceUID']
        tmp['id'] = SOPInstanceUID
        tmp['SOPInstanceUID'] = SOPInstanceUID
        tmp['StudyInstanceUID'] = StudyInstanceUID
        tmp['SeriesInstanceUID'] = SeriesInstanceUID
        tmp['pe_present_on_image'] = per_image_x[idx, :num]
        tmp[label_col] = per_exam_x[idx]
        tmp['fold'] = fold
        out = pd.concat([out, tmp])
    out.to_csv(valid_result_path / 'valid.csv', index=False)
    return out


@hydra.main(config_path='config.yaml')
def main(cfg: DictConfig):
    cwd = Path(hydra.utils.get_original_cwd())
    valid_result_path = cwd / 'seq_result_valid'
    valid_result_path = valid_result_path / cfg.model.name
    result = []
    label_cols = ['pe_present_on_image', 'negative_exam_for_pe', 'indeterminate', 'chronic_pe',
                  'acute_and_chronic_pe', 'central_pe', 'leftsided_pe',
                  'rightsided_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']
    pred_cols = [f"{label_col}_pred" for label_col in label_cols]
    for i in range(cfg.exp.k_folds):
        result.append(run(i, cfg=cfg))
    result = pd.concat(result)
    result.to_csv(valid_result_path / 'valid.csv', index=False)

    result[pred_cols] = result[label_cols]
    train_df = pd.read_csv(cwd / '../input/train_clean.csv').sort_values('SOPInstanceUID').reset_index(drop=True)
    result = result.sort_values('SOPInstanceUID').reset_index(drop=True)
    result[label_cols] = train_df[label_cols]
    from metrics import competition_score
    print('Competition Score: {}'.format(competition_score(result)))


if __name__ == '__main__':
    main()
