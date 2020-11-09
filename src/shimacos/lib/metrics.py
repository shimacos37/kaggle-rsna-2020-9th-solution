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


def softmax(x, axis):
    u = np.sum(np.exp(x), axis=axis, keepdims=True)
    return np.exp(x) / u


def postprocess(x, s=2.0):
    logit = np.log(x / (1 - x))
    logit = logit + s
    sigmoid = 1 / (1 + np.exp(-logit))
    return sigmoid


def satisfy_label_consistency(df, delta=1):
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
    pred_cols = [f"{col}_pred" for col in label_cols]

    rule_breaks = consistency_check(df).index
    print(rule_breaks)
    if len(rule_breaks) > 0:
        df["positive_exam_for_pe_pred"] = 1 - df["negative_exam_for_pe_pred"]
        df.loc[
            df.query("positive_exam_for_pe_pred <= pe_present_on_image_pred").index,
            "pe_present_on_image_pred",
        ] = df.loc[
            df.query("positive_exam_for_pe_pred <= pe_present_on_image_pred").index,
            "positive_exam_for_pe_pred",
        ]
        rule_breaks = consistency_check(df).index
        df["positive_images_in_exam"] = df["StudyInstanceUID"].map(
            df.groupby(["StudyInstanceUID"])["pe_present_on_image_pred"].max()
        )
        df_pos = df.query("positive_images_in_exam > 0.5")
        df_neg = df.query("positive_images_in_exam <= 0.5")
        if "1a" in rule_breaks:
            rv_filter = "rv_lv_ratio_gte_1_pred > 0.5 & rv_lv_ratio_lt_1_pred > 0.5"
            while len(df_pos.query(rv_filter)) > 0:
                df_pos.loc[df_pos.query(rv_filter).index, "rv_min"] = df_pos.query(
                    rv_filter
                )[pred_cols[8:]].min(1)
                for rv_col in pred_cols[8:]:
                    df_pos.loc[
                        df_pos.query(rv_filter + f" & {rv_col} == rv_min").index, rv_col
                    ] = postprocess(
                        df_pos.query(rv_filter + f" & {rv_col} == rv_min")[
                            rv_col
                        ].values,
                        s=-0.1,
                    )
            rv_filter = "rv_lv_ratio_gte_1_pred <= 0.5 & rv_lv_ratio_lt_1_pred <= 0.5"
            while len(df_pos.query(rv_filter)) > 0:
                df_pos.loc[df_pos.query(rv_filter).index, "rv_max"] = df_pos.query(
                    rv_filter
                )[pred_cols[8:]].max(1)
                for rv_col in pred_cols[8:]:
                    df_pos.loc[
                        df_pos.query(rv_filter + f" & {rv_col} == rv_max").index, rv_col
                    ] = postprocess(
                        df_pos.query(rv_filter + f" & {rv_col} == rv_max")[
                            rv_col
                        ].values,
                        s=0.1,
                    )
            df.loc[df_pos.index, pred_cols[8:]] = df_pos[pred_cols[8:]]
        if "1b" in rule_breaks:
            pe_filter = " & ".join([f"{col} <= 0.5" for col in pred_cols[5:8]])
            while "1b" in consistency_check(df).index:
                for col in pred_cols[5:8]:
                    df_pos.loc[df_pos.query(pe_filter).index, col] = postprocess(
                        df_pos.loc[df_pos.query(pe_filter).index, col], s=0.1
                    )
                df.loc[df_pos.index, pred_cols[5:8]] = df_pos[pred_cols[5:8]].values
        if "1c" in rule_breaks:
            chronic_filter = "chronic_pe_pred > 0.5 & acute_and_chronic_pe_pred > 0.5"
            df_pos.loc[df_pos.query(chronic_filter).index, pred_cols[3:5]] = softmax(
                df_pos.query(chronic_filter)[pred_cols[3:5]].values, axis=1
            )
            df.loc[df_pos.index, pred_cols[3:5]] = df_pos[pred_cols[3:5]]
        if "1d" in rule_breaks:
            neg_filter = "negative_exam_for_pe_pred > 0.5 | indeterminate_pred > 0.5"
            while "1d" in consistency_check(df).index:
                for col in pred_cols[1:3]:
                    df_pos.loc[df_pos.query(neg_filter).index, col] = postprocess(
                        df_pos.loc[df_pos.query(neg_filter).index, col], s=-0.1
                    )
                df.loc[df_pos.index, pred_cols[1:3]] = df_pos[pred_cols[1:3]].values
        if "2a" in rule_breaks:
            neg_filter = "negative_exam_for_pe_pred > 0.5 & indeterminate_pred > 0.5"
            while len(df_neg.query(neg_filter)) > 0:
                df_neg.loc[df_neg.query(neg_filter).index, "neg_min"] = df_neg.query(
                    neg_filter
                )[pred_cols[1:3]].min(1)
                for neg_col in pred_cols[1:3]:
                    df_neg.loc[
                        df_neg.query(neg_filter + f" & {neg_col} == neg_min").index,
                        neg_col,
                    ] = postprocess(
                        df_neg.query(neg_filter + f" & {neg_col} == neg_min")[
                            neg_col
                        ].values,
                        s=-0.1,
                    )
            neg_filter = "negative_exam_for_pe_pred <= 0.5 & indeterminate_pred <= 0.5"
            while len(df_neg.query(neg_filter)) > 0:
                df_neg.loc[df_neg.query(neg_filter).index, "neg_max"] = df_neg.query(
                    neg_filter
                )[pred_cols[1:3]].max(1)
                for neg_col in pred_cols[1:3]:
                    df_neg.loc[
                        df_neg.query(neg_filter + f" & {neg_col} == neg_max").index,
                        neg_col,
                    ] = postprocess(
                        df_neg.query(neg_filter + f" & {neg_col} == neg_max")[
                            neg_col
                        ].values,
                        s=0.1,
                    )
            df.loc[df_neg.index, pred_cols[1:3]] = df_neg[pred_cols[1:3]]
        if "2b" in rule_breaks:
            while "2b" in consistency_check(df).index:
                for col in pred_cols[3:]:
                    df_neg.loc[df_neg.query(f"{col} > 0.5").index, col] = postprocess(
                        df_neg.loc[df_neg.query(f"{col} > 0.5").index, col], s=-0.1
                    )
                df.loc[df_neg.index, pred_cols[3:]] = df_neg[pred_cols[3:]].values
    return df


def consistency_check(df):
    df["positive_images_in_exam"] = df["StudyInstanceUID"].map(
        df.groupby(["StudyInstanceUID"])["pe_present_on_image_pred"].max()
    )
    df_pos = df.loc[df.positive_images_in_exam > 0.5]
    df_neg = df.loc[df.positive_images_in_exam <= 0.5]
    rule1a = df_pos.loc[
        (
            (df_pos["rv_lv_ratio_lt_1_pred"] > 0.5)
            & (df_pos["rv_lv_ratio_gte_1_pred"] > 0.5)
        )
        | (
            (df_pos["rv_lv_ratio_lt_1_pred"] <= 0.5)
            & (df_pos["rv_lv_ratio_gte_1_pred"] <= 0.5)
        )
    ].reset_index(drop=True)
    rule1a["broken_rule"] = "1a"
    rule1b = df_pos.loc[
        (df_pos["central_pe_pred"] <= 0.5)
        & (df_pos["rightsided_pe_pred"] <= 0.5)
        & (df_pos["leftsided_pe_pred"] <= 0.5)
    ].reset_index(drop=True)
    rule1b["broken_rule"] = "1b"

    rule1c = df_pos.loc[
        (df_pos["acute_and_chronic_pe_pred"] > 0.5) & (df_pos["chronic_pe_pred"] > 0.5)
    ].reset_index(drop=True)
    rule1c["broken_rule"] = "1c"

    rule1d = df_pos.loc[
        (df_pos["indeterminate_pred"] > 0.5)
        | (df_pos["negative_exam_for_pe_pred"] > 0.5)
    ].reset_index(drop=True)
    rule1d["broken_rule"] = "1d"
    rule2a = df_neg.loc[
        (
            (df_neg["indeterminate_pred"] > 0.5)
            & (df_neg["negative_exam_for_pe_pred"] > 0.5)
        )
        | (
            (df_neg["indeterminate_pred"] <= 0.5)
            & (df_neg["negative_exam_for_pe_pred"] <= 0.5)
        )
    ].reset_index(drop=True)
    rule2a["broken_rule"] = "2a"

    rule2b = df_neg.loc[
        (df_neg["rv_lv_ratio_lt_1_pred"] > 0.5)
        | (df_neg["rv_lv_ratio_gte_1_pred"] > 0.5)
        | (df_neg["central_pe_pred"] > 0.5)
        | (df_neg["rightsided_pe_pred"] > 0.5)
        | (df_neg["leftsided_pe_pred"] > 0.5)
        | (df_neg["acute_and_chronic_pe_pred"] > 0.5)
        | (df_neg["chronic_pe_pred"] > 0.5)
    ].reset_index(drop=True)
    rule2b["broken_rule"] = "2b"
    errors = pd.concat([rule1a, rule1b, rule1c, rule1d, rule2a, rule2b], axis=0)
    return errors["broken_rule"].value_counts()


def competition_score(df: pd.DataFrame, eps: float = 1e-5, agg: str = "mean") -> float:
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
        df[["StudyInstanceUID"] + pred_cols[1:]].groupby("StudyInstanceUID").agg(agg)
    )

    score_exam = []
    for col, w in zip(label_cols[1:], weights[1:]):
        score = (
            my_log_loss(
                target_exam[col].values,
                probs_exam[f"{col}_pred"].values,
                reduction="mean",
                eps=eps,
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
            df[label_cols[0]].values, df[pred_cols[0]].values, reduction="none", eps=eps
        )
        * image_weights
        * weights[0]
    ).sum()

    total_score = score_exam + score_img
    total_weights = np.sum(weights[1:]) * df.StudyInstanceUID.nunique() + np.sum(
        weights[0] * image_weights
    )
    return total_score / total_weights


def exam_score(df: pd.DataFrame, eps: float = 1e-5, agg: str = "mean") -> float:
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
        df[["StudyInstanceUID"] + pred_cols[1:]].groupby("StudyInstanceUID").agg(agg)
    )

    score_exam = []
    for col, w in zip(label_cols[1:], weights[1:]):
        score = (
            my_log_loss(
                target_exam[col].values,
                probs_exam[f"{col}_pred"].values,
                reduction="mean",
                eps=eps,
            )
            * w
        )
        score = score * target_exam.shape[0]  # calc sum, not mean
        score_exam.append(score)

    score_exam = np.sum(score_exam)  # sum, not mean

    total_score = score_exam
    total_weights = np.sum(weights[1:]) * df.StudyInstanceUID.nunique()
    return total_score / total_weights


def image_score(df: pd.DataFrame, eps: float = 1e-5, agg: str = "mean") -> float:

    image_weights = 0.07361963 * df.groupby("StudyInstanceUID")[
        "pe_present_on_image"
    ].transform("mean")

    score_img = (
        my_log_loss(
            df["pe_present_on_image"].values,
            df["pe_present_on_image_pred"].values,
            reduction="none",
            eps=eps,
        )
        * image_weights
        * 0.07361963
    ).sum()

    total_score = score_img
    total_weights = np.sum(0.07361963 * image_weights)
    return total_score / total_weights
