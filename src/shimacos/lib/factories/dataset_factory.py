import pickle
import re
from glob import glob
import cv2
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from typing import Dict, Any
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pydicom
from albumentations import (
    GridDistortion,
    ElasticTransform,
    OpticalDistortion,
    Resize,
    RGBShift,
    HueSaturationValue,
    GridDropout,
    CoarseDropout,
    ShiftScaleRotate,
    RandomGamma,
    RandomBrightnessContrast,
    Compose,
    OneOf,
    Normalize,
)

from lib.io import load_train_data, load_test_data


def get_dicom_value(x, cast=int):
    if type(x) in [pydicom.multival.MultiValue, tuple]:
        return cast(x[0])
    else:
        return cast(x)


def cast(value):
    if type(value) is pydicom.valuerep.MultiValue:
        return tuple(value)
    return value


def get_dicom_raw(dicom):
    return {
        attr: cast(getattr(dicom, attr))
        for attr in dir(dicom)
        if attr[0].isupper() and attr not in ["PixelData"]
    }


def rescale_image(image, slope, intercept):
    return image * slope + intercept


def apply_window(image, center, width):
    image = image.copy()
    min_value = center - width // 2
    max_value = center + width // 2
    image[image < min_value] = min_value
    image[image > max_value] = max_value
    image = image - np.min(image)
    image = image / np.max(image)
    image = (image * 255.0).astype("uint8")
    return image


class RSNADataset(object):
    def __init__(self, data_config: DictConfig, mode: str = "train"):

        self.config = data_config
        self.mode = mode
        self.refinement_step = False
        self.le_dict = {}

        # Data load
        train_df = pd.read_csv(f"{data_config.workdir}/input/train_clean.csv")
        self.label_cols = data_config.label_cols
        feature = train_df.groupby(["StudyInstanceUID"])["pe_present_on_image"].apply(
            lambda x: x.sum() / len(x)
        )
        feature.name = "image_weight"
        feature = feature.reset_index()
        train_df = train_df.merge(feature, on="StudyInstanceUID", how="left")

        if mode == "train":
            self.df = train_df.query(f"fold!={data_config.n_fold}").reset_index(
                drop=True
            )
        elif mode == "valid":
            if data_config.is_train:
                self.df = train_df.query(f"fold=={data_config.n_fold}").reset_index(
                    drop=True
                )
            else:
                self.df = train_df
        elif mode == "test":
            self.df = pd.read_csv(f"{data_config.workdir}/input/test_clean.csv")

        for col in ["StudyInstanceUID", "SOPInstanceUID"]:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.le_dict[col] = le
        self.exam_df = (
            self.df[["StudyInstanceUID", "negative_exam_for_pe"]]
            .drop_duplicates()
            .reset_index()
        )
        self.exam_ids = self.exam_df["StudyInstanceUID"].unique()
        # if mode != "train":
        tmp = self.df.groupby(["SeriesInstanceUID"]).apply(self.culc_start_index)
        for col in tmp.columns:
            self.df[col] = tmp[col]
        self.id_indexs = self.df["id_index"].unique()

    def set_refinement_step(self):
        self.refinement_step = True

    def culc_start_index(self, df: pd.DataFrame) -> pd.DataFrame:
        df["index"] = np.arange(len(df))
        df["index"] = df["index"] // self.config.slide_size
        df["id_index"] = df["SeriesInstanceUID"] + "_" + df["index"].astype(str)
        return df[["index", "id_index"]]

    def _augmenation(self, p: float = 0.3) -> Compose:
        aug_list = []

        if self.mode == "train":
            if not self.refinement_step:
                # height = np.random.randint(height - height * 0.05, height)
                # width = np.random.randint(width - width * 0.05, width)
                aug_list.extend(
                    [
                        OneOf(
                            [GridDistortion(), ElasticTransform(), OpticalDistortion()],
                            p,
                        ),
                        # OneOf([RGBShift(), HueSaturationValue()], p),
                        OneOf([RandomBrightnessContrast(), RandomGamma()], p),
                        # OneOf([GridDropout(), CoarseDropout()], p),
                        ShiftScaleRotate(
                            shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=p
                        ),
                    ]
                )

        aug_list.extend(
            [
                Resize(
                    self.config.image_size,
                    self.config.image_size,
                    cv2.INTER_LINEAR,
                ),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), p=1),
            ]
        )
        return Compose(
            aug_list,
            additional_targets={f"image{i}": "image" for i in range(401)},
        )

    def _load_dicom(self, image_id: str) -> np.ndarray:
        if self.mode != "test":
            path = f"{self.config.workdir}/input/train/{image_id}.dcm"
        else:
            path = f"{self.config.workdir}/input/test/{image_id}.dcm"
        dicom = pydicom.dcmread(path)
        img = dicom.pixel_array
        img = rescale_image(img, dicom.RescaleSlope, dicom.RescaleIntercept)
        center = (
            int(dicom.WindowCenter[0])
            if type(dicom.WindowCenter) == pydicom.multival.MultiValue
            else int(dicom.WindowCenter)
        )
        width = (
            int(dicom.WindowWidth[0])
            if type(dicom.WindowWidth) == pydicom.multival.MultiValue
            else int(dicom.WindowWidth)
        )
        img1 = apply_window(img, 40, 400)  # normal
        img2 = apply_window(img, 100, 700)  # PE
        img3 = apply_window(img, -600, 1500)  # lung
        img = np.array([img1, img2, img3]).transpose(1, 2, 0)
        # for i in range(img.shape[-1]):
        #     img[..., i] = (img[..., i] - img[..., i].min()) / (
        #         img[..., i].max() - img[..., i].min()
        #     )
        return img

    def _load_jpg(self, image_id: str) -> np.ndarray:
        path = "/".join(image_id.split("/")[:2])
        name = image_id.split("/")[2]
        img_path = glob(f"{self.config.workdir}/input/train-jpegs/{path}/*{name}*")[0]
        img = cv2.imread(img_path).astype(np.float32)
        # for i in range(img.shape[-1]):
        #     img[..., i] = (img[..., i] - img[..., i].min()) / (
        #         img[..., i].max() - img[..., i].min()
        #     )
        return img

    def _load_jpg(self, image_id: str) -> np.ndarray:
        path = "/".join(image_id.split("/")[:2])
        name = image_id.split("/")[2]
        img_path = glob(f"{self.config.workdir}/input/train-jpegs/{path}/*{name}*")[0]
        img = cv2.imread(img_path).astype(np.float32)
        for i in range(img.shape[-1]):
            img[..., i] = (img[..., i] - img[..., i].min()) / (
                img[..., i].max() - img[..., i].min()
            )
        return img

    def _correct_label(self, label: np.ndarray) -> np.ndarray:
        if np.all(label[:, 0] != 1) and np.any(label[:, 2:] > 0):
            # negative sampleとして扱う
            label[np.where(label == 1)] = 0
            label[:, 1] = 1

        return label

    def __len__(self) -> int:
        if self.mode != "train":
            return len(self.id_indexs)
        else:
            return len(self.id_indexs)

    def padding(
        self, val: np.ndarray, max_seq_len: int, constant_value: int = -1
    ) -> np.ndarray:
        shape = val.shape
        if len(shape) == 1:
            val = np.pad(
                val,
                (0, max_seq_len - len(val)),
                mode="constant",
                constant_values=constant_value,
            )
        elif len(shape) == 2:
            val = np.pad(
                val,
                ((0, max_seq_len - len(val)), (0, 0)),
                mode="constant",
                constant_values=constant_value,
            )
        elif len(shape) == 3:
            val = np.pad(
                val,
                ((0, max_seq_len - len(val)), (0, 0), (0, 0)),
                mode="constant",
                constant_values=constant_value,
            )
        elif len(shape) == 4:
            val = np.pad(
                val,
                ((0, max_seq_len - len(val)), (0, 0), (0, 0), (0, 0)),
                mode="constant",
                constant_values=constant_value,
            )
        return val

    def random_sample_length_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) - self.config.max_frame > 1:
            start_idx = np.random.randint(0, len(df) - self.config.max_frame)
        else:
            start_idx = 0
        return df.iloc[start_idx : start_idx + self.config.max_frame]

    def _correct_label(self, label: np.ndarray) -> np.ndarray:
        if np.all(label[:, 0] == 0) and np.any(label[:, 3:] > 0):
            # 全ての画像にPEがない かつ PE系のラベルがついてる時は
            # negative sampleとして扱う
            label[np.where(label == 1)] = 0
            label[:, 1] = 1

        return label

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        imgs : [height, width, 3]
        """
        id_index = self.id_indexs[idx]
        exam_id, index = id_index.split("_")
        if self.mode == "train":
            # exam_id = self.exam_ids[idx]
            exam_df = self.df.query("SeriesInstanceUID == @exam_id")
            exam_df = self.random_sample_length_df(exam_df)
        else:
            index = int(index)
            exam_df = self.df.query(
                "SeriesInstanceUID == @exam_id & index >= @index"
            ).iloc[: self.config.max_frame]
        imgs = np.array(
            [self._load_dicom(img_id) for img_id in exam_df["image_id"].values]
        )  # [n_image, height, width, channel]
        imgs_dict = {f"image{n_frame}": img for n_frame, img in enumerate(imgs)}
        imgs_dict["image"] = imgs_dict["image0"]
        imgs_dict = self._augmenation()(**imgs_dict)
        imgs = np.array(
            [imgs_dict[f"image{i}"] for i in range(len(imgs_dict) - 1)]
        ).transpose(0, 3, 1, 2)
        if self.mode != "test":
            label = exam_df[self.label_cols].values
            if self.mode == "train":
                label = self._correct_label(label)

            return {
                "id": self.padding(
                    exam_df["SOPInstanceUID"].values, self.config.max_frame
                ),
                "exam_id": self.padding(
                    exam_df["StudyInstanceUID"].values, self.config.max_frame
                ),
                "data": self.padding(imgs, self.config.max_frame),
                "label": self.padding(label, self.config.max_frame),
                "seq_len": len(exam_df),
                "image_weight": self.padding(
                    exam_df["image_weight"].values, self.config.max_frame
                ),
            }
        else:
            return {
                "id": self.padding(
                    exam_df["SOPInstanceUID"].values, self.config.max_frame
                ),
                "data": self.padding(imgs, self.config.max_frame),
                "seq_len": len(exam_df),
                "image_weight": self.padding(
                    exam_df["image_weight"].values, self.config.max_frame
                ),
            }


class SingleImageDataset(RSNADataset):
    def __init__(self, data_config: DictConfig, mode: str = "train"):
        self.config = data_config
        self.mode = mode
        self.refinement_step = False
        self.le_dict = {}

        # Data load
        train_df = pd.read_csv(f"{data_config.workdir}/input/train_clean.csv")
        self.label_cols = data_config.label_cols
        feature = train_df.groupby(["StudyInstanceUID"])["pe_present_on_image"].apply(
            lambda x: x.sum() / len(x)
        )
        feature.name = "image_weight"
        feature = feature.reset_index()
        train_df = train_df.merge(feature, on="StudyInstanceUID", how="left")

        if mode == "train":
            self.df = train_df.query(f"fold!={data_config.n_fold}").reset_index(
                drop=True
            )
        elif mode == "valid":
            if data_config.is_train:
                self.df = train_df.query(f"fold=={data_config.n_fold}").reset_index(
                    drop=True
                )
            else:
                self.df = train_df
        elif mode == "test":
            self.df = pd.read_csv(f"{data_config.workdir}/input/test_clean.csv")

        for col in ["StudyInstanceUID", "SOPInstanceUID"]:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.le_dict[col] = le
        # if mode != "train":
        self.df["InstanceNumberMax"] = self.df.groupby(["StudyInstanceUID"])[
            "InstanceNumber"
        ].transform("max")
        self.resample_df()

    def resample_df(self):
        if self.mode == "train":
            # self.sample_df = (
            #     self.df.groupby(["StudyInstanceUID"])
            #     .apply(lambda x: x.sample(1))
            #     .reset_index(drop=True)
            # )
            self.sample_df = self.df
        else:
            self.sample_df = self.df

    def __len__(self):
        return len(self.sample_df)

    def _correct_label(self, label: np.ndarray) -> np.ndarray:
        if len(self.label_cols) > 3:
            if label[0] != 1 and np.any(label[3:] > 0):
                # negative sampleとして扱う
                label[np.where(label == 1)] = 0
                label[1] = 1
        else:
            if np.all(label == 0):
                label[1] = 1
        return label

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        imgs : [height, width, 3]
        """
        image_id = self.sample_df.loc[idx, "image_id"]
        if not self.config.use_jpeg:
            img = self._load_dicom(image_id)
        else:
            img = self._load_jpg(image_id)
        img_dict = self._augmenation()(image=img)
        img = img_dict["image"]
        if self.mode != "test":
            label = self.sample_df.loc[idx, self.label_cols].values.astype(np.float32)
            if self.mode == "train":
                label = self._correct_label(label)
            return {
                "id": self.sample_df.loc[idx, "SOPInstanceUID"],
                "exam_id": self.sample_df.loc[idx, "StudyInstanceUID"],
                "data": img,
                "label": label,
                "seq_len": 1,
            }
        else:
            return {
                "id": self.sample_df.loc[idx, "SOPInstanceUID"],
                "data": img,
                "seq_len": 1,
            }


class AdjImageDataset(RSNADataset):
    def __init__(self, data_config: DictConfig, mode: str = "train"):
        self.config = data_config
        self.mode = mode
        self.refinement_step = False
        self.le_dict = {}

        # Data load
        train_df = pd.read_csv(f"{data_config.workdir}/input/train_clean.csv")
        self.label_cols = data_config.label_cols
        feature = train_df.groupby(["StudyInstanceUID"])["pe_present_on_image"].apply(
            lambda x: x.sum() / len(x)
        )
        feature.name = "image_weight"
        feature = feature.reset_index()
        train_df = train_df.merge(feature, on="StudyInstanceUID", how="left")

        if mode == "train":
            self.df = train_df.query(f"fold!={data_config.n_fold}").reset_index(
                drop=True
            )
        elif mode == "valid":
            if data_config.is_train:
                self.df = train_df.query(f"fold=={data_config.n_fold}").reset_index(
                    drop=True
                )
            else:
                self.df = train_df
        elif mode == "test":
            self.df = pd.read_csv(f"{data_config.workdir}/input/test_clean.csv")

        for col in ["StudyInstanceUID", "SOPInstanceUID"]:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.le_dict[col] = le
        self.df["InstanceNumberMax"] = self.df.groupby(["StudyInstanceUID"])[
            "InstanceNumber"
        ].transform("max")
        self.resample_df()

    def resample_df(self):
        if self.mode == "train":
            self.sample_df = self.df
        else:
            self.sample_df = self.df

    def __len__(self):
        return len(self.sample_df)

    def _correct_label(self, label: np.ndarray) -> np.ndarray:
        if len(self.label_cols) > 3:
            if label[0] != 1 and np.any(label[3:] > 0):
                # negative sampleとして扱う
                label[np.where(label == 1)] = 0
                label[1] = 1
        else:
            if np.all(label == 0):
                label[1] = 1
        return label

    def _load_adjacent_img(self, idx: int):
        instance_num, instance_num_max = self.sample_df.loc[
            idx, ["InstanceNumber", "InstanceNumberMax"]
        ]
        if instance_num > 0:
            if instance_num < instance_num_max:
                idx_range = range(idx - 1, idx + 2)
            else:
                # instance_num_max の最小値は64
                idx_range = range(idx - 2, idx + 1)
        else:
            idx_range = range(idx, idx + 3)
        image_ids = self.sample_df.loc[idx_range, "image_id"].values
        if not self.config.use_jpeg:
            img = np.concatenate(
                [self._load_dicom(image_id) for image_id in image_ids], axis=-1
            )
        else:
            img = np.concatenate(
                [self._load_jpg(image_id) for image_id in image_ids], axis=-1
            )
        return img

    def _adj_augmentation(self, img):
        imgs_dict = {f"image{i}": img[..., i : i + 3] for i in range(0, 9, 3)}
        imgs_dict["image"] = imgs_dict["image0"]
        imgs_dict = self._augmenation()(**imgs_dict)
        img = np.concatenate([imgs_dict[f"image{i}"] for i in range(0, 9, 3)], axis=-1)
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        imgs : [height, width, 9]
        """
        img = self._load_adjacent_img(idx)
        img = self._adj_augmentation(img)
        if self.mode != "test":
            label = self.sample_df.loc[idx, self.label_cols].values.astype(np.float32)
            if self.mode == "train":
                label = self._correct_label(label)
            return {
                "id": self.sample_df.loc[idx, "SOPInstanceUID"],
                "exam_id": self.sample_df.loc[idx, "StudyInstanceUID"],
                "data": img,
                "label": label,
                "seq_len": 1,
            }


class FeatureDataset(RSNADataset):
    def __init__(self, data_config: DictConfig, mode: str = "train"):

        self.config = data_config
        self.mode = mode
        self.refinement_step = False
        self.le_dict = {}
        self.std_dict = {}
        # Data load
        train_df = pd.read_csv(f"{data_config.workdir}/input/train_clean.csv")
        test_df = pd.read_csv(f"{data_config.workdir}/input/test_clean.csv")
        self.label_cols = data_config.label_cols
        feature = train_df.groupby(["StudyInstanceUID"])["pe_present_on_image"].apply(
            lambda x: x.sum() / len(x)
        )
        feature.name = "image_weight"
        feature = feature.reset_index()
        train_df = train_df.merge(feature, on="StudyInstanceUID", how="left")
        for col in data_config.meta_feature_cols:
            std = StandardScaler()
            train_df[col] = std.fit_transform(train_df[[col]].values)
            test_df[col] = std.transform(test_df[[col]].values)
            train_df[col].fillna(train_df[col].mean(), inplace=True)
            test_df[col].fillna(test_df[col].mean(), inplace=True)
            self.std_dict[col] = std

        if mode == "train":
            self.df = train_df.query(f"fold!={data_config.n_fold}").reset_index(
                drop=True
            )
        elif mode == "valid":
            self.df = train_df.query(f"fold=={data_config.n_fold}").reset_index(
                drop=True
            )
        elif mode == "test":
            self.df = test_df

        for col in ["StudyInstanceUID", "SOPInstanceUID"]:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.le_dict[col] = le
        self.exam_ids = self.df["StudyInstanceUID"].unique()
        self.max_seq_len = 401

    def _load_feature(
        self, id_: str, series_id: str, image_size: int = 512
    ) -> np.ndarray:
        id_ = self.le_dict["StudyInstanceUID"].inverse_transform(np.array([id_]))[0]
        if self.mode != "test":
            root_dir = "result_valid"
        else:
            root_dir = "result_test"
        if image_size == 384:
            feature_path = f"{self.config.workdir}/output/{root_dir}/tf_efficientnet_b3_ns/384-b3/fold_{self.config.n_fold + 1}/cnn_feature/{id_}/{series_id}.npy"
        else:
            feature_path = f"{self.config.workdir}/output/{root_dir}/tf_efficientnet_b5_ns/512-b5/fold_{self.config.n_fold + 1}/cnn_feature/{id_}/{series_id}.npy"
        feature = np.load(feature_path)
        return feature

    def __len__(self) -> int:
        return len(self.exam_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        imgs : [height, width, 3]
        """
        exam_id = self.exam_ids[idx]
        exam_df = self.df.query("StudyInstanceUID == @exam_id")
        series_id = exam_df["SeriesInstanceUID"].iloc[0]
        if self.config.image_mode == "small":
            feature = self._load_feature(exam_id, series_id, image_size=384)
        elif self.config.image_mode == "large":
            feature = self._load_feature(exam_id, series_id, image_size=512)
        elif self.config.image_mode == "concat":
            feature_384 = self._load_feature(exam_id, series_id, image_size=384)
            feature_512 = self._load_feature(exam_id, series_id, image_size=512)
            feature = np.concatenate([feature_384, feature_512], axis=1)
        meta_feature = exam_df[self.config.meta_feature_cols].values
        if self.mode == "train":
            start_idx = np.random.randint(2)
            feature = feature[start_idx::2]
            meta_feature = meta_feature[start_idx::2]
        else:
            feature = feature[0::2]
            meta_feature = meta_feature[0::2]
        if self.mode != "test":
            label = exam_df[self.label_cols].values
            return {
                "id": self.padding(exam_df["SOPInstanceUID"].values, self.max_seq_len),
                "exam_id": self.padding(
                    exam_df["StudyInstanceUID"].values, self.max_seq_len
                ),
                "feature": self.padding(feature, 201, constant_value=0),
                "meta_feature": self.padding(meta_feature, 201, constant_value=0),
                "label": self.padding(label, self.max_seq_len),
                "seq_len": len(exam_df),
                "image_weight": self.padding(
                    exam_df["image_weight"].values, self.max_seq_len
                ),
            }
        else:
            return {
                "id": self.padding(exam_df["SOPInstanceUID"].values, self.max_seq_len),
                "exam_id": self.padding(
                    exam_df["StudyInstanceUID"].values, self.max_seq_len
                ),
                "feature": self.padding(feature, 201),
                "meta_feature": self.padding(meta_feature, 201, constant_value=0),
                "seq_len": len(exam_df),
            }


class StackingDataset(RSNADataset):
    def __init__(self, data_config: DictConfig, mode: str = "train"):

        self.config = data_config
        self.mode = mode
        self.refinement_step = False
        self.le_dict = {}
        # Data load
        raw_df_train = pd.read_csv(f"{data_config.workdir}/input/train_clean.csv")
        train_dfs, names = load_train_data(data_config.workdir)
        test_dfs, _ = load_test_data(data_config.workdir)
        self.label_cols = data_config.label_cols

        train_ft_dict = {}
        test_ft_dict = {}
        self.feature_cols = []
        train_ft_dict["StudyInstanceUID"] = train_dfs[0]["StudyInstanceUID"]
        train_ft_dict["SOPInstanceUID"] = train_dfs[0]["SOPInstanceUID"]
        test_ft_dict["StudyInstanceUID"] = test_dfs[0]["StudyInstanceUID"]
        test_ft_dict["SOPInstanceUID"] = test_dfs[0]["SOPInstanceUID"]
        for label_col in self.label_cols:
            train_ft_dict[label_col] = train_dfs[0][label_col]
        for name, df in zip(names, train_dfs):
            for label_col in self.label_cols:
                if f"{label_col}_pred" in df.columns:
                    train_ft_dict[f"{name}_{label_col}_pred"] = df[f"{label_col}_pred"]
                    self.feature_cols += [f"{name}_{label_col}_pred"]
        for name, df in zip(names, test_dfs):
            for label_col in self.label_cols:
                for i in range(5):
                    if label_col in df.columns:
                        test_ft_dict[f"{name}_{label_col}_pred"] = df[label_col]
        train_df = pd.DataFrame(train_ft_dict)
        test_df = pd.DataFrame(test_ft_dict)
        train_df = train_df.merge(
            raw_df_train[
                ["SOPInstanceUID", "fold"] + list(data_config.meta_feature_cols)
            ],
            on="SOPInstanceUID",
            how="left",
        )

        feature = train_df.groupby(["StudyInstanceUID"])["pe_present_on_image"].apply(
            lambda x: x.sum() / len(x)
        )
        feature.name = "image_weight"
        feature = feature.reset_index()
        train_df = train_df.merge(feature, on="StudyInstanceUID", how="left")
        std = StandardScaler()
        train_df[self.feature_cols] = std.fit_transform(
            train_df[self.feature_cols].values
        )
        test_df[self.feature_cols] = std.transform(test_df[self.feature_cols].values)
        # with open(f"{data_config.workdir}/output/stacking_std.pkl", "wb") as f:
        #     pickle.dump(std, f)
        if mode == "train":
            self.df = train_df.query(f"fold!={data_config.n_fold}").reset_index(
                drop=True
            )
        elif mode == "valid":
            self.df = train_df.query(f"fold=={data_config.n_fold}").reset_index(
                drop=True
            )
        elif mode == "test":
            self.df = test_df

        for col in ["StudyInstanceUID", "SOPInstanceUID"]:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.le_dict[col] = le
        self.exam_ids = self.df["StudyInstanceUID"].unique()
        self.max_seq_len = 401

    def __len__(self) -> int:
        return len(self.exam_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        imgs : [height, width, 3]
        """
        exam_id = self.exam_ids[idx]
        exam_df = self.df.query("StudyInstanceUID == @exam_id")
        feature = exam_df[self.feature_cols].values
        if self.mode != "test":
            label = exam_df[self.label_cols].values
            return {
                "id": self.padding(exam_df["SOPInstanceUID"].values, self.max_seq_len),
                "exam_id": self.padding(
                    exam_df["StudyInstanceUID"].values, self.max_seq_len
                ),
                "feature": self.padding(feature, self.max_seq_len, constant_value=0),
                "label": self.padding(label, self.max_seq_len),
                "seq_len": len(exam_df),
                "image_weight": self.padding(
                    exam_df["image_weight"].values, self.max_seq_len
                ),
            }
        else:
            return {
                "id": self.padding(exam_df["SOPInstanceUID"].values, self.max_seq_len),
                "exam_id": self.padding(
                    exam_df["StudyInstanceUID"].values, self.max_seq_len
                ),
                "feature": self.padding(feature, self.max_seq_len),
                "seq_len": len(exam_df),
            }


def get_rsna_dataset(data_config, mode):
    dataset = RSNADataset(data_config, mode)
    return dataset


def get_single_adj_dataset(data_config, mode):
    dataset = AdjImageDataset(data_config, mode)
    return dataset


def get_single_image_dataset(data_config, mode):
    dataset = SingleImageDataset(data_config, mode)
    return dataset


def get_feature_dataset(data_config, mode):
    dataset = FeatureDataset(data_config, mode)
    return dataset


def get_stacking_dataset(data_config, mode):
    dataset = StackingDataset(data_config, mode)
    return dataset


def get_dataset(data_config, mode):
    print("dataset name:", data_config.dataset_name)
    f = globals().get("get_" + data_config.dataset_name)
    return f(data_config, mode)
