"""Categorical encoders for donor response models."""
from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder


def build_categorical_encoder(
    encoding: str = "onehot",
    handle_unknown: str = "ignore",
):
    """Create the categorical encoder used inside a model pipeline."""
    if encoding == "onehot":
        params = {"handle_unknown": handle_unknown}
        if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
            params["sparse_output"] = False
        else:
            params["sparse"] = False
        return OneHotEncoder(**params)
    if encoding == "ordinal":
        return OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
    if encoding == "passthrough":
        return FunctionTransformer(validate=False)
    if encoding == "target":
        raise NotImplementedError(
            "Target encoding is intentionally not included in the base scaffold. "
            "Add it later with leakage-safe cross-validation logic."
        )
    raise ValueError(
        "Unsupported encoding strategy. Choose from "
        "'onehot', 'ordinal', 'passthrough', or 'target'."
    )


def one_hot_encode(data: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """One-hot encode object and category columns in a DataFrame."""
    if columns is None:
        columns = data.select_dtypes(include=["object", "category"]).columns.tolist()

    if not columns:
        return data.copy()

    encoder = build_categorical_encoder("onehot")
    encoded = encoder.fit_transform(data[columns])
    feature_names = encoder.get_feature_names_out(columns)
    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=data.index)

    return pd.concat([data.drop(columns, axis=1), encoded_df], axis=1)


def label_encode(data: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Ordinally encode object and category columns in a DataFrame."""
    if columns is None:
        columns = data.select_dtypes(include=["object", "category"]).columns.tolist()

    if not columns:
        return data.copy()

    encoder = build_categorical_encoder("ordinal")
    encoded = encoder.fit_transform(data[columns])
    encoded_df = pd.DataFrame(encoded, columns=columns, index=data.index)

    result = data.copy()
    result[columns] = encoded_df
    return result
