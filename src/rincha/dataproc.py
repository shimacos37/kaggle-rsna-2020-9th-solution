import pandas as pd
import numpy as np
from dataset import ImageDataset, SeqDataset, DICOMDataset, HopDataset
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MinMaxScaler


class DICOMDataproc:
    def __init__(self, cfg, data_root: Path):
        self.fold = cfg.exp.fold
        self.cfg = cfg
        df = pd.read_csv(data_root / "train_clean.csv")
        df["image_path"] = str(data_root / "train") + "/" + df["image_id"] + ".dcm"
        df["InstanceNumberMax"] = df.groupby(["StudyInstanceUID"])[
            "InstanceNumber"
        ].transform("max")
        self.df = df

        test_df = pd.read_csv(data_root / "test_clean.csv")
        test_df["image_path"] = (
            str(data_root / "test") + "/" + test_df["image_id"] + ".dcm"
        )
        test_df["InstanceNumberMax"] = test_df.groupby(["StudyInstanceUID"])[
            "InstanceNumber"
        ].transform("max")
        self.test_df = test_df

    def get_dataset(self, fold):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        figsize = self.cfg.model.figsize
        prob = 0.5
        train_transform = A.Compose(
            [
                A.Resize(figsize, figsize, p=1),
                A.OneOf(
                    [A.GridDistortion(), A.ElasticTransform(), A.OpticalDistortion()],
                    prob,
                ),
                A.OneOf([A.RGBShift(), A.HueSaturationValue()], prob),
                A.OneOf([A.RandomBrightnessContrast(), A.RandomGamma()], prob),
                A.OneOf([A.GridDropout(), A.CoarseDropout()], prob),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=prob
                ),
                A.Normalize(mean, std, p=1),
                ToTensorV2(p=1.0),
            ]
        )

        valid_transform = A.Compose(
            [
                A.Resize(figsize, figsize, p=1),
                A.Normalize(mean, std, p=1),
                ToTensorV2(p=1.0),
            ]
        )

        train_data = self.df[self.df["fold"] != fold].reset_index(drop=True)
        valid_data = self.df[self.df["fold"] == fold].reset_index(drop=True)
        train_dataset = DICOMDataset(
            train_data, transforms=train_transform, predict=False, cfg=self.cfg
        )
        valid_dataset = DICOMDataset(
            valid_data, transforms=valid_transform, predict=False, cfg=self.cfg
        )
        return train_dataset, valid_dataset

    def get_all_dataset(self, fold):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        figsize = self.cfg.model.figsize
        valid_transform = A.Compose(
            [
                A.Resize(figsize, figsize, p=1),
                A.Normalize(mean, std, p=1),
                ToTensorV2(p=1.0),
            ]
        )
        data = self.df
        data = data.sort_values(["StudyInstanceUID", "InstanceNumber"]).reset_index(
            drop=True
        )
        dataset = DICOMDataset(
            data, transforms=valid_transform, predict=False, cfg=self.cfg
        )
        return dataset, data

    def get_test_dataset(self, fold):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        figsize = self.cfg.model.figsize
        valid_transform = A.Compose(
            [
                A.Resize(figsize, figsize, p=1),
                A.Normalize(mean, std, p=1),
                ToTensorV2(p=1.0),
            ]
        )
        data = self.test_df
        data = data.sort_values(["StudyInstanceUID", "InstanceNumber"]).reset_index(
            drop=True
        )
        dataset = DICOMDataset(
            data, transforms=valid_transform, predict=True, cfg=self.cfg
        )
        return dataset, data


class SeqDataproc:
    def __init__(self, cfg, data_root: Path, feature_root: Path):
        self.cfg = cfg
        df = pd.read_csv(data_root / "train_clean.csv")
        df["feature_path"] = (
            str(feature_root)
            + "/"
            + df["StudyInstanceUID"]
            + "/"
            + df["SeriesInstanceUID"]
            + ".npy"
        )
        df["ImagePositionPatient_2"] = df["ImagePositionPatient"].apply(
            lambda x: float(x[1:-1].split(",")[2].strip())
        )
        df["tick"] = df.groupby("StudyInstanceUID")["ImagePositionPatient_2"].apply(
            lambda x: np.abs(x.shift(1) - x)
        )
        df["tick"] = df["tick"].fillna(df["tick"].mode())

        self.df = df

    def get_df(self, fold):
        df = self.df
        if self.cfg.second.use_val:
            df["feature_path"] = df.apply(
                lambda x: x["feature_path"].replace(
                    "fold_{}".format(fold + 1), "fold_{}".format(int(x["fold"] + 1))
                ),
                axis=1,
            )
        valid_data = df[df["fold"] == fold].reset_index(drop=True)
        return valid_data

    def get_dataset(self, fold):
        df = self.df
        if self.cfg.second.use_val:
            df["feature_path"] = df.apply(
                lambda x: x["feature_path"].replace(
                    "fold_{}".format(fold + 1), "fold_{}".format(int(x["fold"] + 1))
                ),
                axis=1,
            )
        train_data = df[df["fold"] != fold].reset_index(drop=True)
        valid_data = df[df["fold"] == fold].reset_index(drop=True)
        scaler = MinMaxScaler()
        meta_col = [
            "XRayTubeCurrent",
            "KVP",
            "Exposure",
            "WindowCenter",
            "WindowWidth",
            "tick",
        ]
        train_data[meta_col] = scaler.fit_transform(train_data[meta_col])
        valid_data[meta_col] = scaler.transform(valid_data[meta_col])
        train_dataset = SeqDataset(train_data, predict=False)
        valid_dataset = SeqDataset(valid_data, predict=False)
        return train_dataset, valid_dataset


class HopDataproc:
    def __init__(
        self, cfg, data_root: Path, feature_384_root: Path, feature_512_root: Path
    ):
        self.cfg = cfg
        df = pd.read_csv(data_root / "train_clean.csv")
        df["feature_384_path"] = (
            str(feature_384_root)
            + "/"
            + df["StudyInstanceUID"]
            + "/"
            + df["SeriesInstanceUID"]
            + ".npy"
        )

        df["feature_512_path"] = (
            str(feature_512_root)
            + "/"
            + df["StudyInstanceUID"]
            + "/"
            + df["SeriesInstanceUID"]
            + ".npy"
        )

        df["ImagePositionPatient_x"] = df["ImagePositionPatient"].apply(
            lambda x: float(x[1:-1].split()[0])
        )
        df["ImagePositionPatient_y"] = df["ImagePositionPatient"].apply(
            lambda x: float(x[1:-1].split()[1])
        )
        df["ImagePositionPatient_z"] = df["ImagePositionPatient"].apply(
            lambda x: float(x[1:-1].split()[2])
        )
        self.df = df

    def get_df(self, fold):
        df = self.df
        valid_data = df[df["fold"] == fold].reset_index(drop=True)
        return valid_data

    def get_dataset(self, fold):
        df = self.df
        train_data = df[df["fold"] != fold].reset_index(drop=True)
        valid_data = df[df["fold"] == fold].reset_index(drop=True)
        train_dataset = HopDataset(train_data, mode="train")
        valid_dataset = HopDataset(valid_data, mode="valid")
        return train_dataset, valid_dataset
