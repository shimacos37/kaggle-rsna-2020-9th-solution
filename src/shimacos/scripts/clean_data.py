import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

if __name__ == "__main__":
    label_cols = [
        "pe_present_on_image",
        "negative_exam_for_pe",
        "indeterminate",
        "rv_lv_ratio_gte_1",
        "rv_lv_ratio_lt_1",
        "leftsided_pe",
        "rightsided_pe",
        "central_pe",
        "chronic_pe",
        "acute_and_chronic_pe",
    ]
    train_df = pd.read_csv("./input/train.csv")
    test_df = pd.read_csv("./input/test.csv")
    train_meta_df = pd.read_csv("./input/train_meta.csv")
    test_meta_df = pd.read_csv("./input/test_meta.csv")
    train_df = train_df.merge(train_meta_df, on="SOPInstanceUID", how="left")
    test_df = test_df.merge(test_meta_df, on="SOPInstanceUID", how="left")
    train_df = train_df.sort_values(["StudyInstanceUID", "InstanceNumber"]).reset_index(
        drop=True
    )
    test_df = test_df.sort_values(["StudyInstanceUID", "InstanceNumber"]).reset_index(
        drop=True
    )
    train_df["ImageOrientationPatient"] = train_df["ImageOrientationPatient"].apply(
        lambda x: [int(a) for a in eval(x)][0]
    )
    train_df["ImagePositionPatient_x"] = train_df["ImagePositionPatient"].apply(
        lambda x: [int(a) for a in eval(x)][0]
    )
    train_df["ImagePositionPatient_y"] = train_df["ImagePositionPatient"].apply(
        lambda x: [int(a) for a in eval(x)][1]
    )
    train_df["ImagePositionPatient_z"] = train_df["ImagePositionPatient"].apply(
        lambda x: [int(a) for a in eval(x)][2]
    )
    train_df["image_id"] = (
        train_df["StudyInstanceUID"]
        + "/"
        + train_df["SeriesInstanceUID"]
        + "/"
        + train_df["SOPInstanceUID"]
    )
    test_df["ImageOrientationPatient"] = test_df["ImageOrientationPatient"].apply(
        lambda x: [int(a) for a in eval(x)][0]
    )
    test_df["ImagePositionPatient_x"] = test_df["ImagePositionPatient"].apply(
        lambda x: [int(a) for a in eval(x)][0]
    )
    test_df["ImagePositionPatient_y"] = test_df["ImagePositionPatient"].apply(
        lambda x: [int(a) for a in eval(x)][1]
    )
    test_df["ImagePositionPatient_z"] = test_df["ImagePositionPatient"].apply(
        lambda x: [int(a) for a in eval(x)][2]
    )
    test_df["image_id"] = (
        test_df["StudyInstanceUID"]
        + "/"
        + test_df["SeriesInstanceUID"]
        + "/"
        + test_df["SOPInstanceUID"]
    )

    feature = train_df.groupby(["StudyInstanceUID"])["InstanceNumber"].min()
    feature.name = "InstanceNumberMin"
    train_df = train_df.merge(feature, on="StudyInstanceUID")
    train_df["InstanceNumber"] -= train_df["InstanceNumberMin"]
    train_df = train_df.query(
        "InstanceNumber <= 400 & ImageOrientationPatient==1"
    ).reset_index(drop=True)
    test_df = test_df.query(
        "InstanceNumber <= 400 & ImageOrientationPatient==1"
    ).reset_index(drop=True)

    tmp_df = (
        train_df[["StudyInstanceUID"] + label_cols[1:]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    y = np.zeros(len(tmp_df))
    for i, col in enumerate(label_cols[1:]):
        y += tmp_df[col] * 2 ** i
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=777)
    for k, (_, val_index) in enumerate(skf.split(tmp_df, y)):
        tmp_df.loc[val_index, "fold"] = k
    train_df = train_df.merge(
        tmp_df[["StudyInstanceUID", "fold"]], on="StudyInstanceUID", how="left"
    )

    train_df.to_csv("./input/train_clean.csv", index=False)
    test_df.to_csv("./input/test_clean.csv", index=False)
