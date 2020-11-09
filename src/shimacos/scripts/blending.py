import sys
from typing import List
import pandas as pd
import numpy as np

sys.path.append("./src/shimacos")

from lib.metrics import (
    competition_score,
    satisfy_label_consistency,
)


def make_submission(test_df: pd.DataFrame, label_cols: List[str]) -> pd.DataFrame:
    pred_cols = [f"{label_col}_pred" for label_col in label_cols]
    exam_pred = test_df[["StudyInstanceUID"] + pred_cols[1:]].drop_duplicates()
    submit_dict = {"id": [], "label": []}
    for label_col in label_cols[1:]:
        submit_dict["id"].extend(
            (exam_pred["StudyInstanceUID"] + f"_{label_col}").values.tolist()
        )
        submit_dict["label"].extend(exam_pred[f"{label_col}_pred"].values.tolist())
    submit_dict["id"].extend(test_df["SOPInstanceUID"].values.tolist())
    submit_dict["label"].extend(test_df["pe_present_on_image_pred"].values.tolist())
    submission = pd.DataFrame(submit_dict)
    return submission


def main():
    model_names = [
        "stacking_lgbm",
        "stacking_cnn",
        "stacking_gru",
    ]
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
    train_dfs = [
        pd.read_csv(f"./output/{name}/{name}_train.csv") for name in model_names
    ]
    test_dfs = [pd.read_csv(f"./output/{name}/{name}_test.csv") for name in model_names]

    result_train_df = {}
    result_test_df = {}
    result_train_df["StudyInstanceUID"] = train_dfs[0]["StudyInstanceUID"]
    result_train_df["SOPInstanceUID"] = train_dfs[0]["SOPInstanceUID"]
    result_test_df["StudyInstanceUID"] = test_dfs[0]["StudyInstanceUID"]
    result_test_df["SOPInstanceUID"] = test_dfs[0]["SOPInstanceUID"]

    for label_col in label_cols:
        result_train_df[label_col] = train_dfs[0][label_col]

    for label_col in label_cols:
        result_train_df[f"{label_col}_pred"] = np.average(
            [df[f"{label_col}_pred"].values for df in train_dfs],
            axis=0,
            weights=[2, 1, 2],
        )
        result_test_df[f"{label_col}_pred"] = np.average(
            [df[label_col].values for df in test_dfs],
            axis=0,
            weights=[2, 1, 2],
        )
    result_train_df = pd.DataFrame(result_train_df)
    result_test_df = pd.DataFrame(result_test_df)

    print("cv score: (before pp)", competition_score(result_train_df))
    result_train_df = satisfy_label_consistency(result_train_df)
    print("cv score: (after pp)", competition_score(result_train_df))
    result_test_df = satisfy_label_consistency(result_test_df)
    submission = make_submission(result_test_df, label_cols)
    submission.to_csv("./output/final_submission.csv", index=False)


if __name__ == "__main__":
    main()