import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def my_log_loss(
        y_true, y_pred, reduction="none", sample_weight=None, eps=1e-5
) -> float:
    y_pred = np.clip(y_pred, eps, 1 - eps)
    bce = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    if sample_weight is not None:
        if np.all(sample_weight) == 0:
            bce = 0
        else:
            bce = np.average(bce, weights=sample_weight)
    else:
        if reduction == "mean":
            bce = bce.mean()
        elif reduction == "sum":
            bce = bce.sum()
        else:
            pass
    return bce


def competition_score(df: pd.DataFrame) -> float:
    label_cols = [
        "pe_present_on_image",
        "negative_exam_for_pe",
        "indeterminate",
        "chronic_pe",
        "acute_and_chronic_pe",
        "central_pe",
        "leftsided_pe",
        "rightsided_pe",
        "rv_lv_ratio_gte_1",
        "rv_lv_ratio_lt_1",
    ]
    pred_cols = [f"{label_col}_pred" for label_col in label_cols]
    weights = [
        0.07361963,
        0.0736196319,
        0.09202453988,
        0.1042944785,
        0.1042944785,
        0.1877300613,
        0.06257668712,
        0.06257668712,
        0.2346625767,
        0.0782208589,
    ]

    target_exam = (
        df[["StudyInstanceUID"] + label_cols[1:]].groupby("StudyInstanceUID").mean()
    )
    probs_exam = (
        df[["StudyInstanceUID"] + pred_cols[1:]].groupby("StudyInstanceUID").mean()
    )

    score_exam = []
    for col, w in zip(label_cols[1:], weights[1:]):
        score = (
                my_log_loss(
                    target_exam[col].values,
                    probs_exam[f"{col}_pred"].values,
                    reduction="mean",
                )
                * w
        )
        score = score * target_exam.shape[0]  # calc sum, not mean
        score_exam.append(score)

    score_exam = np.sum(score_exam)  # sum, not mean

    image_weights = weights[0] * df.groupby("StudyInstanceUID")[
        label_cols[0]
    ].transform("mean")

    score_img = (
            my_log_loss(
                df[label_cols[0]].values,
                df[pred_cols[0]].values,
                reduction="none",
            )
            * image_weights
            * weights[0]
    ).sum()

    total_score = score_exam + score_img
    total_weights = np.sum(weights[1:]) * df.StudyInstanceUID.nunique() + np.sum(
        weights[0] * image_weights
    )
    return total_score / total_weights
