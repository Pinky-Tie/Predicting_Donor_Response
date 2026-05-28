"""Numeric scalers for donor response models."""
from __future__ import annotations
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler

def _round_transformer():
    """Return a FunctionTransformer that rounds numeric arrays to 4 decimal places.

    This is used as the final step in numeric pipelines to keep numeric values
    consistent and avoid excessive floating point precision when storing or
    comparing transformed features.
    """
    return FunctionTransformer(lambda X: np.round(X, 4), validate=False)


def build_numeric_scaler(strategy: str = "standard"):
    """Create the numeric scaler used inside a model pipeline."""
    if strategy == "standard":
        scaler = StandardScaler()
    elif strategy == "minmax":
        scaler = MinMaxScaler()
    elif strategy == "robust":
        scaler = RobustScaler()
    elif strategy == "passthrough":
        return _round_transformer()
    else:
        raise ValueError(
            "Unsupported scaling strategy. Choose from "
            "'standard', 'minmax', 'robust', or 'passthrough'."
        )

    return make_pipeline(scaler, _round_transformer())
