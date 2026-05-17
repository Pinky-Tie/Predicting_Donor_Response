# """Imputation builders for numeric and categorical donor features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
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


def impute_columns(data: pd.DataFrame, dict: dict[str, str]) -> pd.DataFrame:
    """
    Impute missing values in specified columns using the given methods.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with missing values.
    dict : dict[str, str]
        Mapping of column name to imputation method. Supported methods:
            - "mean"           : replace with column mean
            - "median"         : replace with column median
            - "most_frequent"  : replace with most frequent value
            - "constant"       : replace with "Missing" (categorical)
            - "min"            : replace with column minimum value
            - "knn"            : k-NN imputation with k=5

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with imputed values in the specified columns.
    """
    NUMERIC_STRATEGIES = {"mean", "median", "most_frequent", "min", "knn"}
    CATEGORICAL_STRATEGIES = {"most_frequent", "constant"}
    ALL_STRATEGIES = NUMERIC_STRATEGIES | CATEGORICAL_STRATEGIES

    result = data.copy()

    for column, method in dict.items():
        if column not in result.columns:
            raise ValueError(f"Column '{column}' not found in dataframe.")
        if method not in ALL_STRATEGIES:
            raise ValueError(
                f"Unsupported method '{method}' for column '{column}'. "
                f"Choose one of {sorted(ALL_STRATEGIES)}."
            )

        col_data = result[[column]]

        if method == "min":
            min_val = result[column].min()
            imputer = SimpleImputer(strategy="constant", fill_value=min_val)
            imputed = imputer.fit_transform(col_data).ravel()

        elif method == "knn":
            imputer = KNNImputer(n_neighbors=5)
            imputed = imputer.fit_transform(col_data).ravel()

        elif method in {"mean", "median", "most_frequent"}:
            imputer = build_numeric_imputer(strategy=method)
            imputed = imputer.fit_transform(col_data).ravel()

        elif method == "constant":
            imputer = build_categorical_imputer(strategy="constant")
            imputed = imputer.fit_transform(col_data).ravel()

        if np.issubdtype(np.asarray(imputed).dtype, np.number):
            imputed = np.round(imputed, 4)

        result[column] = imputed

    return result
