import pandas as pd
import pydicom
from tqdm import tqdm
from joblib import delayed, Parallel


def extract_meta_dataframe(path) -> pd.DataFrame:
    dicom = pydicom.dcmread(path)
    meta_dict = {
        col: int(getattr(dicom, col)[0])
        if type(getattr(dicom, col)) == pydicom.multival.MultiValue
        else int(getattr(dicom, col))
        for col in [
            "PixelSpacing",
            "SliceThickness",
            "KVP",
            "XRayTubeCurrent",
            "Exposure",
            "SeriesNumber",
            "InstanceNumber",
            "WindowCenter",
            "WindowWidth",
            "RescaleIntercept",
            "RescaleSlope",
        ]
    }
    meta_dict.update(
        {
            col: getattr(dicom, col)
            if type(getattr(dicom, col)) != pydicom.multival.MultiValue
            else str(getattr(dicom, col))
            for col in [
                "SOPInstanceUID",
                "Modality",
                "RotationDirection",
                "ImagePositionPatient",
                "ImageOrientationPatient",
            ]
        }
    )
    return pd.DataFrame(meta_dict, index=[0])


def add_image_id(df: pd.DataFrame) -> pd.DataFrame:
    df["image_id"] = (
        df["StudyInstanceUID"]
        + "/"
        + df["SeriesInstanceUID"]
        + "/"
        + df["SOPInstanceUID"]
    )
    return df


def main():
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
    train_df = add_image_id(train_df)
    test_df = add_image_id(test_df)
    image_ids = train_df["image_id"].values
    meta_dfs = Parallel(n_jobs=-1, verbose=3)(
        [
            delayed(extract_meta_dataframe)(f"./input/train/{image_id}.dcm")
            for image_id in tqdm(image_ids)
        ]
    )
    meta_dfs = pd.concat(meta_dfs, axis=0)
    meta_dfs = meta_dfs.sort_values(by=["SOPInstanceUID", "InstanceNumber"])
    meta_dfs.to_csv("./input/train_meta.csv", index=False)

    image_ids = test_df["image_id"].values
    meta_dfs = Parallel(n_jobs=-1, verbose=3)(
        [
            delayed(extract_meta_dataframe)(f"./input/test/{image_id}.dcm")
            for image_id in tqdm(image_ids)
        ]
    )
    meta_dfs = pd.concat(meta_dfs, axis=0)
    meta_dfs = meta_dfs.sort_values(by=["SOPInstanceUID", "InstanceNumber"])
    meta_dfs.to_csv("./input/test_meta.csv", index=False)


if __name__ == "__main__":
    main()
