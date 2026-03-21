"""Categorical encoders for donor response models."""
from __future__ import annotations
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
