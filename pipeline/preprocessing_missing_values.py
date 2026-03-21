"""Imputation builders for numeric and categorical donor features."""
from __future__ import annotations
from sklearn.impute import SimpleImputer
def build_numeric_imputer(strategy: str = "median") -> SimpleImputer:
    """Return a numeric imputer suited for model pipelines."""
    valid_strategies = {"mean", "median", "most_frequent", "constant"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unsupported numeric imputation strategy '{strategy}'. "
            f"Choose one of {sorted(valid_strategies)}."
        )
    return SimpleImputer(strategy=strategy)
def build_categorical_imputer(
    strategy: str = "most_frequent",
    fill_value: str = "Missing",
) -> SimpleImputer:
    """Return a categorical imputer suited for model pipelines."""
    valid_strategies = {"most_frequent", "constant"}
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unsupported categorical imputation strategy '{strategy}'. "
            f"Choose one of {sorted(valid_strategies)}."
        )

    if strategy == "constant":
        return SimpleImputer(strategy=strategy, fill_value=fill_value)

    return SimpleImputer(strategy=strategy)
