import io
import os
import random
import shutil
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import yaml
from google.cloud import storage

from lib.factories import (
    get_dataset,
    get_loss,
    get_model,
    get_optimizer,
    get_scheduler,
)
from lib.metrics import competition_score, my_log_loss


try:
    import torch_xla.core.xla_model as xm
except ModuleNotFoundError:
    pass


plt.style.use("ggplot")


def mesh_reduce(tag, data, reduce_fn):
    """Performs an out-of-graph client mesh reduction.
    Args:
      tag (string): The name of the rendezvous to join.
      data: The data to be reduced. The `reduce_fn` callable will receive a list
        with the copies of the same data coming from all the mesh client processes
        (one per core).
      reduce_fn (callable): A function which receives a list of `data`-like
        objects and returns the reduced result.
    Returns:
      The reduced value.
    """
    cpu_data = xm._maybe_convert_to_cpu(data)
    bio = io.BytesIO()
    torch.save(cpu_data, bio)
    xdata = xm.rendezvous(tag, bio.getvalue())
    xldata = []
    for xd in xdata:
        xbio = io.BytesIO(xd)
        xldata.append(torch.load(xbio))
    return reduce_fn(xldata) if xldata else cpu_data


class BaseModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # Setting
        self.base_config = hparams.base
        self.data_config = hparams.data
        self.model_config = hparams.model
        self.train_config = hparams.train
        self.test_config = hparams.test
        self.store_config = hparams.store
        # load from factories
        self.model = get_model(hparams.model)
        if self.base_config.loss_name not in ["arcface", "adacos"]:
            self.loss = get_loss(loss_name=self.base_config.loss_name)
        else:
            self.loss = get_loss(
                loss_name=self.base_config.loss_name,
                in_features=self.model.in_features,
                num_classes=self.model_config.num_classes,
            )
        if self.data_config.is_train:
            self.train_dataset = get_dataset(data_config=self.data_config, mode="train")
            self.valid_dataset = get_dataset(data_config=self.data_config, mode="valid")
            self.num_train_optimization_steps = int(
                self.train_config.epoch
                * len(self.train_dataset)
                / (self.train_config.batch_size)
                / self.train_config.accumulation_steps
                / self.base_config.num_cores
            )
        else:
            if self.test_config.is_validation:
                self.test_dataset = get_dataset(
                    data_config=self.data_config, mode="valid"
                )
                self.prefix = "valid"
            else:
                self.test_dataset = get_dataset(
                    data_config=self.data_config, mode="test"
                )
                self.prefix = "test"
            self.num_train_optimization_steps = 100
        # path setting
        self.initialize_variables()
        self.save_flg = False
        self.refinement_step = False

    def configure_optimizers(self):
        optimizer = get_optimizer(
            opt_name=self.base_config.opt_name,
            params=filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.train_config.learning_rate,
        )
        scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.num_train_optimization_steps, eta_min=1e-6
        )
        scheduler = {
            "scheduler": get_scheduler(
                scheduler_name="warmup_scheduler",
                optimizer=optimizer,
                multiplier=10,
                total_epoch=int(self.num_train_optimization_steps * 0.01),
                after_scheduler=scheduler_cosine,
            ),
            "interval": "step",
        }
        scheduler["scheduler"].step(self.step)
        return [optimizer], [scheduler]

    def forward(self, **batch):
        return self.model(**batch)

    def __convert_to_prob(self, pred):
        if len(self.data_config.label_cols) > 3:
            softmax = torch.softmax(pred[..., :3], dim=-1)
            sigmoid = torch.sigmoid(pred[..., 3:])
            pred = torch.cat([softmax, sigmoid], dim=-1)
        elif len(self.data_config.label_cols) == 3:
            pred = torch.softmax(pred, dim=-1)
        else:
            pred = torch.sigmoid(pred)
        return pred

    def training_step(self, batch, batch_nb):

        pred = self.forward(**batch)
        # pred = self.__convert_to_prob(pred)
        loss = self.loss(pred, batch)
        metrics = {}
        metrics["loss"] = loss
        metrics["log"] = {
            "train_loss": loss,
        }
        metrics["progress_bar"] = {
            "lr": self.trainer.lr_schedulers[0]["scheduler"].optimizer.param_groups[0][
                "lr"
            ],
        }

        return metrics

    def validation_step(self, batch, batch_nb):
        if self.test_config.is_tta:
            batch["data"] = torch.cat(
                [batch["data"], batch["data"].flip(2), batch["data"].flip(1)], dim=0
            )  # horizontal flip

        pred = self.forward(**batch)
        label = batch["label"].float()
        if self.test_config.is_tta:
            pred = pred.reshape(3, pred.shape[0] // 3, pred.shape[-1]).mean(0)
        loss = self.loss(pred, batch)
        if self.base_config.loss_name == "rsna_split_loss":
            image_pred, exam_pred = pred
            n_repeat = image_pred.shape[1]
            exam_pred = exam_pred.unsqueeze(1).repeat(1, n_repeat, 1)
            pred = torch.cat([image_pred, exam_pred], dim=-1)
            pred = torch.sigmoid(pred)
        else:
            pred = self.__convert_to_prob(pred)

        pred = pred.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        metrics = {
            "id": batch["id"].detach().cpu().numpy(),
            "exam_id": batch["exam_id"].detach().cpu().numpy(),
            "preds": pred.reshape(pred.shape[0] * pred.shape[1], -1)
            if "single" not in self.data_config.dataset_name
            else pred,
            "label": label.reshape(label.shape[0] * label.shape[1], -1)
            if "single" not in self.data_config.dataset_name
            else label,
            "loss": loss,
        }
        if self.store_config.save_feature:
            feature = self.model.get_feature()
            metrics.update({"feature": feature.detach().cpu().numpy()})

        return metrics

    def test_step(self, batch, batch_nb):
        if self.test_config.is_tta:
            batch["data"] = torch.cat(
                [batch["data"], batch["data"].flip(2), batch["data"].flip(1)], dim=0
            )  # horizontal flip
        with torch.no_grad():
            pred = self.forward(**batch)
        if self.base_config.loss_name == "rsna_split_loss":
            image_pred, exam_pred = pred
            n_repeat = image_pred.shape[1]
            exam_pred = exam_pred.unsqueeze(1).repeat(1, n_repeat, 1)
            pred = torch.cat([image_pred, exam_pred], dim=-1)
            pred = torch.sigmoid(pred).detach().cpu().numpy()
        else:
            pred = self.__convert_to_prob(pred).detach().cpu().numpy()
        if self.test_config.is_tta:
            pred = pred.reshape(3, pred.shape[0] // 3, pred.shape[-1]).mean(0)
        exam_ids = batch["exam_id"].detach().cpu().numpy()
        ids = batch["id"].detach().cpu().numpy()
        metrics = {
            "id": ids,
            "exam_id": exam_ids,
            "preds": pred.reshape(pred.shape[0] * pred.shape[1], -1)
            if "single" not in self.data_config.dataset_name
            else pred,
        }
        if self.store_config.save_feature:
            feature = self.model.get_feature().detach().cpu().numpy()
            metrics.update({"feature": feature})
        if self.test_config.is_validation:
            label = batch["label"].float().detach().cpu().numpy()
            metrics.update(
                {
                    "label": label.reshape(label.shape[0] * label.shape[1], -1)
                    if "single" not in self.data_config.dataset_name
                    else label
                }
            )
        return metrics

    def _compose_result(self, outputs):

        preds = np.concatenate([x["preds"] for x in outputs], axis=0)
        ids = np.concatenate([x["id"] for x in outputs]).reshape(
            -1,
        )
        exam_ids = np.concatenate([x["exam_id"] for x in outputs]).reshape(
            -1,
        )
        if self.data_config.is_train:
            dataset = self.valid_dataset
        else:
            dataset = self.test_dataset
        label_cols = dataset.label_cols
        df_dict = {"SOPInstanceUID": ids, "StudyInstanceUID": exam_ids}
        for i, label_col in enumerate(label_cols):
            df_dict[f"{label_col}_pred"] = preds[:, i]
        if "label" in outputs[0].keys():
            label = np.concatenate([x["label"] for x in outputs], axis=0)
            for i, label_col in enumerate(label_cols):
                df_dict[label_col] = label[:, i]
        df = pd.DataFrame(df_dict)
        df = df.query("SOPInstanceUID != -1").reset_index(drop=True)
        le_dict = dataset.le_dict
        for col in ["SOPInstanceUID", "StudyInstanceUID"]:
            df[col] = le_dict[col].inverse_transform(df[col])
        # TTA部分の処理
        df = (
            df.groupby(["StudyInstanceUID", "SOPInstanceUID"])[
                [
                    col
                    for col in df.columns
                    if col not in ["StudyInstanceUID", "SOPInstanceUID"]
                ]
            ]
            .mean()
            .reset_index()
        )
        return df

    def validation_epoch_end(self, outputs):
        loss = np.mean([x["loss"].item() for x in outputs])
        if self.store_config.save_feature:
            feature = np.concatenate([x["feature"] for x in outputs], axis=0)
            rank = dist.get_rank()
            np.save(f"{self.store_config.feature_path}/{rank}_feature.npy", feature)
        df = self._compose_result(outputs)
        if self.use_tpu:
            rank = xm.get_ordinal()
            df.to_csv(
                os.path.join(self.store_config.result_path, f"valid_result_{rank}.csv"),
                index=False,
            )
            xm.wait_device_ops()
            loss = mesh_reduce("loss", torch.tensor(loss), torch.stack).mean()
        elif self.use_ddp:
            rank = dist.get_rank()
            df.to_csv(
                os.path.join(self.store_config.result_path, f"valid_result_{rank}.csv"),
                index=False,
            )
            dist.barrier()
            metrics = {"avg_loss": loss}
            print(rank, loss)
            world_size = dist.get_world_size()
            aggregated_metrics = {}
            for metric_name, metric_val in metrics.items():
                metric_tensor = torch.tensor(metric_val).to(f"cuda:{rank}")
                dist.barrier()
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                reduced_metric = metric_tensor.item() / world_size
                aggregated_metrics[metric_name] = reduced_metric
            loss = aggregated_metrics["avg_loss"]
        else:
            pass
        if self.use_tpu or self.use_ddp:
            paths = sorted(
                glob(
                    os.path.join(
                        self.store_config.result_path, "valid_result_[0-9].csv"
                    )
                )
            )
            df = pd.concat([pd.read_csv(path) for path in paths])
            df = df.drop_duplicates()
        if self.base_config.loss_name in [
            "rsna_loss",
            "rsna_split_loss",
            "normal_rsna_loss",
        ]:
            avg_score = competition_score(df)
        else:
            avg_score = np.mean(
                [
                    my_log_loss(df[label_col].values, df[f"{label_col}_pred"].values)
                    for label_col in self.data_config.label_cols
                ]
            )
        res = {}
        res["step"] = int(self.global_step)
        res["epoch"] = int(self.current_epoch)
        if avg_score <= self.best_score and df.shape[0] >= 2000:
            self.best_score = avg_score
            self.save_flg = True
            res["best_score"] = float(self.best_score)
            df.to_csv(
                os.path.join(self.store_config.result_path, "valid_result_all.csv"),
                index=False,
            )
            with open(
                os.path.join(self.store_config.log_path, "best_score.yaml"), "w"
            ) as f:
                yaml.dump(res, f, default_flow_style=False)
        metrics = {}
        metrics["progress_bar"] = {
            "val_loss": loss,
            "avg_val_score": avg_score,
            "best_score": self.best_score,
        }
        metrics["log"] = {
            "val_loss": loss,
            "avg_val_score": avg_score,
            "best_score": self.best_score,
        }
        return metrics

    def on_epoch_end(self):
        if self.save_flg:
            if self.store_config.gcs_project is not None:
                self.upload_directory()
            self.save_flg = False
        if self.current_epoch >= self.train_config.refinement_step:
            self.train_dataset.set_refinement_step()
        if "single" in self.data_config.dataset_name:
            self.train_dataset.resample_df()

    def test_epoch_end(self, outputs):
        df = self._compose_result(outputs)
        if self.store_config.save_feature:
            feature = np.concatenate([x["feature"] for x in outputs], axis=0)
            rank = dist.get_rank()
            np.save(f"{self.store_config.feature_path}/{rank}_feature.npy", feature)
        if self.use_ddp:
            rank = dist.get_rank()
            df.to_csv(
                os.path.join(
                    self.store_config.result_path,
                    f"{self.prefix}_result_{rank}.csv",
                ),
                index=False,
            )
            dist.barrier()
            paths = sorted(
                glob(
                    os.path.join(
                        self.store_config.result_path,
                        f"{self.prefix}_result_[0-9].csv",
                    )
                )
            )
            df = pd.concat([pd.read_csv(path) for path in paths])
            df = df.drop_duplicates()
        df.to_csv(
            os.path.join(self.store_config.result_path, f"{self.prefix}_result.csv"),
            index=False,
        )
        if not self.test_config.is_validation:
            result = {}
            if self.data_config.n_fold == 4:
                sub_dict = {}
                dfs = [
                    pd.read_csv(
                        f"{self.store_config.root_path}/fold{i}/result/test_result.csv"
                    )
                    for i in range(5)
                ]
                sub_dict["StudyInstanceUID"] = dfs[0]["StudyInstanceUID"].values
                sub_dict["SOPInstanceUID"] = dfs[0]["SOPInstanceUID"].values
                for label_col in self.data_config.label_cols:
                    sub_dict[label_col] = np.mean(
                        [df[f"{label_col}_pred"].values for df in dfs], axis=0
                    )
                pd.DataFrame(sub_dict).to_csv(
                    f"{self.store_config.root_path}/{self.store_config.model_name}_test.csv",
                    index=False,
                )
        else:
            score = competition_score(df)
            result = {f"score_fold{self.data_config.n_fold}": score}
            if self.data_config.n_fold == 4:
                dfs = pd.concat(
                    [
                        pd.read_csv(
                            f"{self.store_config.root_path}/fold{i}/result/valid_result.csv"
                        )
                        for i in range(5)
                    ],
                    axis=0,
                )
                dfs.to_csv(
                    f"{self.store_config.root_path}/{self.store_config.model_name}_train.csv",
                    index=False,
                )
                score = competition_score(df)
                result.update({"score_all": score})

            return result

    def get_progress_bar_dict(self):
        running_train_loss = self.trainer.train_loop.running_loss.mean()
        avg_training_loss = (
            running_train_loss.cpu().item()
            if running_train_loss is not None
            else float("NaN")
        )
        tqdm_dict = {"loss": "{:2.6g}".format(avg_training_loss)}

        if self.trainer.truncated_bptt_steps is not None:
            tqdm_dict["split_idx"] = self.trainer.split_idx

        if self.trainer.logger is not None and self.trainer.logger.version is not None:
            version = self.trainer.logger.version
            version = version[-4:] if isinstance(version, str) else version
            tqdm_dict["v_num"] = version

        return tqdm_dict

    def initialize_variables(self):
        self.step = 0
        self.best_score = np.inf
        if self.train_config.warm_start:
            with open(
                os.path.join(self.store_config.log_path, "best_score.yaml"), "r"
            ) as f:
                res = yaml.safe_load(f)
            if "best_score" in res.keys():
                self.best_score = res["best_score"]
            self.step = res["step"]

    def upload_directory(self):
        storage_client = storage.Client(self.store_config.gcs_project)
        bucket = storage_client.get_bucket(self.store_config.bucket_name)
        filenames = glob(
            os.path.join(self.store_config.save_path, "**"), recursive=True
        )
        for filename in filenames:
            if os.path.isdir(filename):
                continue
            destination_blob_name = os.path.join(
                self.store_config.gcs_path,
                filename.split(self.store_config.save_path)[-1][1:],
            )
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(filename)
