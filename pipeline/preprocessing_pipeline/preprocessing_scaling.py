"""Numeric scalers for donor response models."""
from __future__ import annotations
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler
def build_numeric_scaler(strategy: str = "standard"):
    """Create the numeric scaler used inside a model pipeline."""
    if strategy == "standard":
        return StandardScaler()
    if strategy == "minmax":
        return MinMaxScaler()
    if strategy == "robust":
        return RobustScaler()
    if strategy == "passthrough":
        return FunctionTransformer(validate=False)

    raise ValueError(
        "Unsupported scaling strategy. Choose from "
        "'standard', 'minmax', 'robust', or 'passthrough'."
    )
