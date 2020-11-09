import pandas as pd
import cv2
from torch.utils.data import Dataset
import numpy as np
import pydicom


class DICOMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cfg, transforms, predict=False):
        self.df = df
        self.cfg = cfg
        self.predict = predict
        self.transforms = transforms
        self.WL_list = [-600, 100, 40]
        self.WW_list = [1500, 700, 400]
        self.label_columns = ['pe_present_on_image', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
                              'leftsided_pe', 'chronic_pe', 'true_filling_defect_not_pe',
                              'rightsided_pe', 'acute_and_chronic_pe', 'central_pe']

    def make_target(self, x: pd.Series):
        x = x[['pe_present_on_image', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', 'leftsided_pe', 'chronic_pe',
               'true_filling_defect_not_pe', 'rightsided_pe', 'acute_and_chronic_pe', 'central_pe',
               'indeterminate',
               ]]
        if x['indeterminate'] == 1:
            out = pd.Series(np.ones(len(x)) * -1, index=x.index)
            out[['indeterminate']] = x[['indeterminate']]
        else:
            if x['pe_present_on_image'] == 1:
                out = x
            else:
                out = pd.Series(np.zeros(len(x)), index=x.index)
        return out.values.astype(np.float32)

    def slice_target(self, idx):
        instance_num, instance_num_max = self.df.loc[
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
        target = np.concatenate([self.make_target(self.df.iloc[idx]) for idx in idx_range])
        return target

    def window(self, img, WL=50, WW=350):
        upper, lower = WL + WW // 2, WL - WW // 2
        X = np.clip(img.copy(), lower, upper)
        X = X - np.min(X)
        X = X / np.max(X)
        X = (X * 255.0).astype('uint8')
        return X

    def single_convert_image(self, img, slice_idx):
        if self.cfg.model.slice_mode == 'SLICE':
            return self.window(img, self.WL_list[slice_idx], self.WW_list[slice_idx])
        else:
            if self.cfg.model.slice_mode == 'LUNG':
                return self.window(img, self.WL_list[0], self.WW_list[0])
            elif self.cfg.model.slice_mode == 'PE':
                return self.window(img, self.WL_list[1], self.WW_list[1])
            elif self.cfg.model.slice_mode == 'MEDIASTINAL':
                return self.window(img, self.WL_list[2], self.WW_list[2])

    def convert_image(self, img):
        return np.stack([self.window(img, WL, WW) for WL, WW in zip(self.WL_list, self.WW_list)], axis=2)

    def load_dicom(self, path, slice_idx=None):
        scan = pydicom.dcmread(path)
        M = float(scan.RescaleSlope)
        B = float(scan.RescaleIntercept)
        image = scan.pixel_array * M + B
        if self.cfg.model.slice:
            image = self.single_convert_image(image, slice_idx=slice_idx)
        else:
            image = self.convert_image(image)
        return image

    def _load_adjacent_img(self, idx: int) -> np.ndarray:
        instance_num, instance_num_max = self.df.loc[
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
        image_paths = self.df.loc[idx_range, "image_path"].values
        img = np.stack(
            [self.load_dicom(image_path, slice_idx=slice_idx) for slice_idx, image_path in enumerate(image_paths)],
            axis=2
        )
        return img

    def __getitem__(self, index):
        data = self.df.iloc[index]
        if self.cfg.model.slice:
            image = self._load_adjacent_img(index)
        else:
            image = self.load_dicom(data['image_path'])
        if self.transforms:
            sample = {'image': image}
            image = self.transforms(**sample)['image']
        out = {'image': image, 'index': index}
        if not self.predict:
            if self.cfg.model.slice:
                target = self.slice_target(index)
            else:
                target = self.make_target(data)
            out.update({'target': target})
        return out

    def __len__(self):
        return len(self.df)


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transforms, predict=False):
        self.df = df
        self.predict = predict
        self.transforms = transforms
        self.label_columns = ['pe_present_on_image', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
                              'leftsided_pe', 'chronic_pe', 'true_filling_defect_not_pe',
                              'rightsided_pe', 'acute_and_chronic_pe', 'central_pe']

    def make_target(self, x: pd.Series):
        x = x[['pe_present_on_image', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1', 'leftsided_pe', 'chronic_pe',
               'true_filling_defect_not_pe', 'rightsided_pe', 'acute_and_chronic_pe', 'central_pe',
               'indeterminate',
               ]]
        if x['indeterminate'] == 1:
            out = pd.Series(np.ones(len(x)) * -1, index=x.index)
            out[['indeterminate']] = x[['indeterminate']]
        else:
            if x['pe_present_on_image'] == 1:
                out = x
            else:
                out = pd.Series(np.zeros(len(x)), index=x.index)
        return out.values.astype(np.float32)

    def __getitem__(self, index):
        data = self.df.iloc[index]
        image = cv2.imread(str(data['image_path']))
        if self.transforms:
            sample = {'image': image}
            image = self.transforms(**sample)['image']
        out = {'image': image}
        if not self.predict:
            target = self.make_target(data)
            out.update({'target': target})
        return out

    def __len__(self):
        return len(self.df)


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, predict=False):
        self.df = df
        self.predict = predict
        self.label_columns = ['pe_present_on_image', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
                              'leftsided_pe', 'chronic_pe', 'true_filling_defect_not_pe',
                              'rightsided_pe', 'acute_and_chronic_pe', 'central_pe']
        self.StudyInstanceUID = self.df['StudyInstanceUID'].unique()

    def __getitem__(self, index):
        StudyInstanceUID = self.StudyInstanceUID[index]
        data = self.df[self.df['StudyInstanceUID'] == StudyInstanceUID]
        data = data.sort_values('InstanceNumber').reset_index(drop=True)
        tmp = np.load(data['feature_path'][0])
        cnn_feature = np.zeros([401, tmp.shape[1]], dtype=np.float32)
        cnn_feature[:tmp.shape[0]] = tmp
        feature = cnn_feature
        mask = np.zeros(401, dtype=np.bool)
        mask[:len(data)] = True
        out = {'feature': feature, 'mask': mask}
        if not self.predict:
            per_image_target = np.zeros([401, 1], dtype=np.float32)
            per_exam_target = data.iloc[0][['negative_exam_for_pe', 'indeterminate', 'chronic_pe',
                                            'acute_and_chronic_pe', 'central_pe', 'leftsided_pe',
                                            'rightsided_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']].values.astype(
                np.float32)
            per_image_target[: len(data)] = data[['pe_present_on_image']].values.astype(np.float32)
            image_weight = np.zeros([401, 1], dtype=np.float32)
            image_weight[:len(data), :] = data['pe_present_on_image'].sum() / len(data) * 0.07361963
            out.update({'per_exam_target': per_exam_target, 'per_image_target': per_image_target,
                        'image_weight': image_weight})
        return out

    def __len__(self):
        return len(self.StudyInstanceUID)


class HopDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode='train'):
        self.df = df
        self.mode = mode
        self.meta_columns = ["KVP",
                             "XRayTubeCurrent",
                             "Exposure",
                             "SliceThickness",
                             "ImagePositionPatient_x",
                             "ImagePositionPatient_y",
                             "ImagePositionPatient_z"]
        self.meta_mean = np.array(
            [
                114.08353157,
                419.09953533,
                108.65329098,
                1.0090654,
                -172.34524724,
                -141.6034326,
                -45.66431551,
            ]
        )
        self.meta_std = np.array(
            [
                1.09305001e01,
                1.92887125e02,
                4.97377464e02,
                2.65628402e-01,
                2.62277483e01,
                7.11791545e01,
                4.41005297e02,
            ]
        )
        self.label_columns = ['pe_present_on_image', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1',
                              'leftsided_pe', 'chronic_pe', 'true_filling_defect_not_pe',
                              'rightsided_pe', 'acute_and_chronic_pe', 'central_pe']
        self.StudyInstanceUID = self.df['StudyInstanceUID'].unique()

    def __getitem__(self, index):
        StudyInstanceUID = self.StudyInstanceUID[index]
        data = self.df[self.df['StudyInstanceUID'] == StudyInstanceUID]
        data = data.sort_values('InstanceNumber').reset_index(drop=True)
        feature_512 = np.load(data['feature_512_path'][0])[:401].astype(np.float32)
        feature_384 = np.load(data['feature_384_path'][0])[:401].astype(np.float32)
        seq_len = min(len(data), 401)
        meta_feature = data[self.meta_columns].values[:401].astype(np.float32)
        meta_feature = (meta_feature - self.meta_mean) / self.meta_std
        if self.mode == "train":
            start_idx = np.random.randint(2)
            feature_512 = feature_512[start_idx::2]
            meta_feature_512 = meta_feature[start_idx::2]
            feature_384 = feature_384[(start_idx + 1) % 2::2]
            meta_feature_384 = meta_feature[(start_idx + 1) % 2::2]
        else:
            feature_512 = feature_512[0::2]
            meta_feature_512 = meta_feature[0::2]
            feature_384 = feature_384[1::2]
            meta_feature_384 = meta_feature[1::2]

        out = {'feature_512': feature_512,
               'feature_384': feature_384,
               'meta_feature_512': meta_feature_512,
               'meta_feature_384': meta_feature_384}
        out = {key: np.pad(item, [(0, 201 - len(item)), (0, 0)]) for key, item in out.items()}
        mask = np.zeros(401, dtype=np.bool)
        mask[:len(data)] = True
        per_image_target = np.zeros([401, 1], dtype=np.float32)
        per_exam_target = data.iloc[0][['negative_exam_for_pe', 'indeterminate', 'chronic_pe',
                                        'acute_and_chronic_pe', 'central_pe', 'leftsided_pe',
                                        'rightsided_pe', 'rv_lv_ratio_gte_1', 'rv_lv_ratio_lt_1']].values.astype(
            np.float32)
        per_image_target[: len(data)] = data[['pe_present_on_image']].values.astype(np.float32)
        image_weight = np.zeros([401, 1], dtype=np.float32)
        image_weight[:len(data), :] = data['pe_present_on_image'].sum() / len(data) * 0.07361963
        out.update({'per_exam_target': per_exam_target,
                    'per_image_target': per_image_target,
                    'image_weight': image_weight,
                    'seq_len': seq_len,
                    'mask': mask})
        return out

    def __len__(self):
        return len(self.StudyInstanceUID)
