import logging
import os
import pickle
import random
from glob import glob
from itertools import combinations
from typing import Dict, List, Optional, Union, Tuple

import hydra
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google.cloud import storage
from omegaconf import DictConfig

from lib.io import load_train_data, load_test_data
from lib.metrics import competition_score, my_log_loss, satisfy_label_consistency

plt.style.use("ggplot")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepair_dir(config: DictConfig) -> None:
    """
    Logの保存先を作成
    """
    for path in [
        config.store.result_path,
        config.store.log_path,
        config.store.model_path,
    ]:
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    os.environ.PYTHONHASHSEED = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LGBMModel(object):
    """
    label_col毎にlightgbm modelを作成するようのクラス
    """

    def __init__(
        self,
        feature_cols: Union[List, np.ndarray],
        pred_cols: Union[List, np.ndarray],
        label_col: Union[List, np.ndarray],
        params: Dict,
        cat_cols: Optional[Union[List, np.ndarray]],
    ):
        self.feature_cols = feature_cols
        self.pred_cols = pred_cols
        self.cat_cols = cat_cols
        self.label_col = label_col
        self.params = params
        self.model_dicts: Dict[int, lgb.Booster] = {}

    def store_model(self, bst: lgb.Booster, n_fold: int) -> None:
        self.model_dicts[n_fold] = bst

    def store_importance(self, importance_df: pd.DataFrame) -> None:
        self.importance_df = importance_df

    def cv(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        importances = []
        for n_fold in range(5):
            bst = self.fit(train_df, n_fold, pseudo_df)
            valid = train_df.query("fold == @n_fold")
            train_df.loc[valid.index, f"{self.label_col}_pred"] = bst.predict(
                valid[self.feature_cols]
            )
            test_df[f"{self.label_col}_pred_fold{n_fold}"] = bst.predict(
                test_df[self.feature_cols]
            )
            self.store_model(bst, n_fold)
            importances.append(bst.feature_importance())
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        importance_df = pd.DataFrame(
            {"mean": importances_mean, "std": importances_std}, index=self.feature_cols
        ).sort_values(by="mean", ascending=False)
        self.store_importance(importance_df)
        return train_df, test_df

    def fit(
        self,
        train_df: pd.DataFrame,
        n_fold: int,
        pseudo_df: Optional[pd.DataFrame] = None,
    ) -> lgb.Booster:
        notnull_idx = train_df[self.label_col].notnull()
        if self.label_col == "pe_present_on_image":
            train_df_ = train_df.query("negative_exam_for_pe==0").reset_index(drop=True)
        else:
            train_df_ = train_df
        if pseudo_df is not None:
            X_train = pd.concat(
                [
                    train_df_.loc[notnull_idx].query("fold!=@n_fold")[
                        self.feature_cols
                    ],
                    pseudo_df[self.feature_cols],
                ]
            )
            y_train = pd.concat(
                [
                    train_df_.loc[notnull_idx].query("fold!=@n_fold")[self.label_col],
                    pseudo_df[self.label_col],
                ]
            )
        else:
            X_train = train_df_.loc[notnull_idx].query("fold!=@n_fold")[
                self.feature_cols
            ]
            y_train = train_df_.loc[notnull_idx].query("fold!=@n_fold")[self.label_col]

        X_valid = train_df_.loc[notnull_idx].query("fold==@n_fold")[self.feature_cols]
        y_valid = train_df_.loc[notnull_idx].query("fold==@n_fold")[self.label_col]

        lgtrain = lgb.Dataset(
            X_train,
            label=np.array(y_train),
            feature_name=self.feature_cols,
        )
        lgvalid = lgb.Dataset(
            X_valid,
            label=np.array(y_valid),
            feature_name=self.feature_cols,
        )
        bst = lgb.train(
            self.params,
            lgtrain,
            num_boost_round=50000,
            valid_sets=[lgtrain, lgvalid],
            valid_names=["train", "valid"],
            early_stopping_rounds=300,
            verbose_eval=1000,
            # feval=self.custom_metric,
            categorical_feature=self.cat_cols,
        )
        return bst

    def save_model(self) -> None:
        with open(f"./output/lightgbm_models/{self.label_col}.pkl", "wb") as f:
            pickle.dump(self.model_dicts, f)


def save_importance(
    importance_df: pd.DataFrame,
    label_col: str,
    store_config: DictConfig,
    suffix: str = "",
) -> None:
    importance_df.sort_values("mean").iloc[-50:].plot.barh(xerr="std", figsize=(10, 20))
    plt.tight_layout()
    plt.savefig(
        os.path.join(store_config.result_path, f"importance_{label_col + suffix}.png")
    )
    importance_df.name = "feature_name"
    importance_df = importance_df.reset_index().sort_values(by="mean", ascending=False)
    importance_df.to_csv(
        os.path.join(store_config.result_path, f"importance_{label_col + suffix}.csv"),
        index=False,
    )


def upload_directory(store_config: DictConfig) -> None:
    storage_client = storage.Client(store_config.gcs_project)
    bucket = storage_client.get_bucket(store_config.bucket_name)
    filenames = glob(os.path.join(store_config.save_path, "**"), recursive=True)
    for filename in filenames:
        if os.path.isdir(filename):
            continue
        destination_blob_name = os.path.join(
            store_config.gcs_path,
            filename.split(store_config.save_path)[-1][1:],
        )
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(filename)


@hydra.main(config_path="yamls/stacking.yaml")
def main(config: DictConfig) -> None:
    prepair_dir(config)
    set_seed(config.data.seed)
    label_cols = list(config.data.label_cols)
    meta_feature_cols = list(config.data.meta_feature_cols)
    raw_df_train = pd.read_csv(f"{config.store.workdir}/input/train_clean.csv")
    train_dfs, names = load_train_data(config.store.workdir)
    test_dfs, _ = load_test_data(config.store.workdir)
    params = {
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "num_leaves": 2,
        "feature_fraction": 0.8,
        "bagging_fraction": 1,
        "bagging_freq": 1,
        "min_child_samples": 10,
        "task": "train",
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": None,
        "max_depth": 7,
        "learning_rate": 0.01,
        "num_thread": -1,
        "max_bin": 256,
        "verbose": -1,
        "device": "cpu",
        "scale_pos_weight": 1,
    }
    train_ft_dict = {}
    test_ft_dict = {}
    pred_cols = []
    train_ft_dict["StudyInstanceUID"] = train_dfs[0]["StudyInstanceUID"]
    train_ft_dict["SOPInstanceUID"] = train_dfs[0]["SOPInstanceUID"]
    test_ft_dict["StudyInstanceUID"] = test_dfs[0]["StudyInstanceUID"]
    test_ft_dict["SOPInstanceUID"] = test_dfs[0]["SOPInstanceUID"]
    for label_col in label_cols:
        train_ft_dict[label_col] = train_dfs[0][label_col]
    for name, df in zip(names, train_dfs):
        for label_col in label_cols:
            if f"{label_col}_pred" in df.columns:
                train_ft_dict[f"{name}_{label_col}_pred"] = df[f"{label_col}_pred"]
                pred_cols += [f"{name}_{label_col}_pred"]

    for name, df in zip(names, test_dfs):
        for label_col in label_cols:
            for i in range(5):
                if label_col in df.columns:
                    test_ft_dict[f"{name}_{label_col}_pred"] = df[label_col]
    train_df = pd.DataFrame(train_ft_dict)
    test_df = pd.DataFrame(test_ft_dict)
    train_df = train_df.merge(
        raw_df_train[["SOPInstanceUID", "fold"]],
        on="SOPInstanceUID",
        how="left",
    )
    cat_cols = []
    feature_cols = pred_cols + cat_cols
    exam_label_cols = label_cols[1:]
    image_label_col = label_cols[0]
    exam_train_df = train_df.drop_duplicates(subset=["StudyInstanceUID"]).reset_index(
        drop=True
    )
    exam_test_df = test_df.drop_duplicates(subset=["StudyInstanceUID"]).reset_index(
        drop=True
    )
    pos_cols = [col for col in train_df.columns if "pe_present_on_image_pred" in col]
    feat_df = train_df.groupby(["StudyInstanceUID"])[pos_cols].agg(
        ["mean", "std", "min", "max"]
    )
    feat_df.columns = ["_".join(cols) for cols in feat_df.columns]
    exam_train_df = exam_train_df.merge(feat_df, on="StudyInstanceUID", how="left")
    feat_df = test_df.groupby(["StudyInstanceUID"])[pos_cols].agg(
        ["mean", "std", "min", "max"]
    )
    feat_df.columns = ["_".join(cols) for cols in feat_df.columns]
    exam_test_df = exam_test_df.merge(feat_df, on="StudyInstanceUID", how="left")
    exam_feature_cols = [
        col
        for col in exam_train_df.columns
        if col
        not in ["StudyInstanceUID", "SOPInstanceUID", "fold"]
        + label_cols
        + pos_cols
        + meta_feature_cols
    ]
    for label_col in exam_label_cols:
        model = LGBMModel(
            exam_feature_cols, pred_cols, label_col, params, cat_cols=cat_cols
        )
        exam_train_df, exam_test_df = model.cv(exam_train_df, exam_test_df)
        save_importance(model.importance_df, label_col, config.store)
        score = my_log_loss(
            exam_train_df[label_col].values,
            exam_train_df[f"{label_col}_pred"].values,
            reduction="mean",
        )
        logger.info(
            f"log_loss ({label_col}): {score}",
        )
        exam_test_df[label_col] = exam_test_df[
            [f"{label_col}_pred_fold{i}" for i in range(5)]
        ].mean(1)
        with open(f"{config.store.model_path}/stacking_{label_col}.pkl", "wb") as f:
            pickle.dump(model.model_dicts, f)
    with open(f"{config.store.model_path}/exam_feature_cols.pkl", "wb") as f:
        pickle.dump(exam_feature_cols, f)

    model = LGBMModel(
        feature_cols, pred_cols, image_label_col, params, cat_cols=cat_cols
    )
    train_df, test_df = model.cv(train_df, test_df)
    save_importance(model.importance_df, image_label_col, config.store)
    score = my_log_loss(
        train_df[image_label_col].values,
        train_df[f"{image_label_col}_pred"].values,
        reduction="mean",
    )
    logger.info(
        f"log_loss ({image_label_col}): {score}",
    )
    test_df[image_label_col] = test_df[
        [f"{image_label_col}_pred_fold{i}" for i in range(5)]
    ].mean(1)
    with open(f"{config.store.model_path}/stacking_{image_label_col}.pkl", "wb") as f:
        pickle.dump(model.model_dicts, f)
    with open(f"{config.store.model_path}/image_feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    train_df = train_df.merge(
        exam_train_df[
            ["StudyInstanceUID"] + [f"{col}_pred" for col in exam_label_cols]
        ],
        on=["StudyInstanceUID"],
        how="left",
    )
    test_df = test_df.merge(
        exam_test_df[["StudyInstanceUID"] + exam_label_cols],
        on=["StudyInstanceUID"],
        how="left",
    )
    score = competition_score(train_df)

    logger.info(f"meric : {score}")
    pred_cols = [f"{col}_pred" for col in label_cols]
    train_df[["StudyInstanceUID", "SOPInstanceUID"] + label_cols + pred_cols].to_csv(
        os.path.join(config.store.save_path, f"{config.store.model_name}_train.csv"),
        index=False,
    )
    test_df[["StudyInstanceUID", "SOPInstanceUID"] + label_cols].to_csv(
        os.path.join(config.store.save_path, f"{config.store.model_name}_test.csv"),
        index=False,
    )
    satisfy_label_consistency(train_df)


if __name__ == "__main__":
    main()
