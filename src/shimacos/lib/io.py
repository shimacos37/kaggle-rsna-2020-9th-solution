from typing import List, Tuple

import pandas as pd


def load_train_data(workdir: str) -> Tuple[List[pd.DataFrame], List[str]]:
    model_names = [
        "tf_efficientnet_b5_ns_feature_deconv_cnn",
        "tf_efficientnet_b5_ns_feature_deconv_lstm",
        "tf_efficientnet_b5_ns_feature_deconv_gru",
        "tf_efficientnet_b5_ns_feature_deconv_cnn_lstm",
        "tf_efficientnet_b3_ns_feature_deconv_cnn",
        "tf_efficientnet_b3_ns_feature_deconv_lstm",
        "tf_efficientnet_b3_ns_feature_deconv_gru",
        "tf_efficientnet_b3_ns_feature_deconv_cnn_lstm",
        "concat_512_384_cnn",
        "concat_512_384_lstm",
        "concat_512_384_gru",
        "concat_512_384_cnn_lstm",
    ]
    dfs = []
    dfs.extend(
        [
            pd.read_csv(f"{workdir}/output/{name}/{name}_train.csv")
            for name in model_names
        ]
    )
    return dfs, model_names


def load_test_data(workdir: str) -> Tuple[List[pd.DataFrame], List[str]]:
    model_names = [
        "tf_efficientnet_b5_ns_feature_deconv_cnn",
        "tf_efficientnet_b5_ns_feature_deconv_lstm",
        "tf_efficientnet_b5_ns_feature_deconv_gru",
        "tf_efficientnet_b5_ns_feature_deconv_cnn_lstm",
        "tf_efficientnet_b3_ns_feature_deconv_cnn",
        "tf_efficientnet_b3_ns_feature_deconv_lstm",
        "tf_efficientnet_b3_ns_feature_deconv_gru",
        "tf_efficientnet_b3_ns_feature_deconv_cnn_lstm",
        "concat_512_384_cnn",
        "concat_512_384_lstm",
        "concat_512_384_gru",
        "concat_512_384_cnn_lstm",
    ]
    dfs = []
    dfs.extend(
        [
            pd.read_csv(f"{workdir}/output/{name}/{name}_test.csv")
            for name in model_names
        ]
    )
    return dfs, model_names
