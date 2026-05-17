
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from typing import Union

def detect_outlier_columns(data: pd.DataFrame, IQR_value: float = 1.5) -> list[str]:
    """
    Returns a list of numeric column names that contain at least one outlier,
    using the Tukey IQR fence:
        lower = Q1 - IQR_value * IQR
        upper = Q3 + IQR_value * IQR
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns

    outlier_cols = []
    for col in numeric_cols:
        series = data[col].dropna()
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - IQR_value * iqr, q3 + IQR_value * iqr
        if ((series < lower) | (series > upper)).any():
            outlier_cols.append(col)

    return outlier_cols

def rescale_outliers(data: pd.Series, method: str, bins: int = None) -> pd.Series:
    """
    Function that takes skewed or outlier prone columns and scales them with the chosen method.

    Available methods:
    - scale   : applies RobustScaler (median + IQR based, outlier-resistant)
    - transform : applies log1p if right-skewed (skew > 0), exp if left-skewed (skew < 0)
    - binning : creates broader ordinal categories (bins parameter required)

    Parameters:
        data  : pd.Series  — the column to transform
        method: str        — one of 'scale', 'transform', 'binning'
        bins  : int        — number of bins (required only for 'binning')

    Returns:
        pd.Series with the transformed values (same index as input)
    """
    supported = ('scale', 'transform', 'binning')
    if method not in supported:
        raise ValueError(f"Unknown method '{method}'. Choose from: {supported}")

    result = data.copy()

    if method == 'scale':
        scaler = RobustScaler()
        non_null_mask = result.notna()
        result.loc[non_null_mask] = scaler.fit_transform(
            result[non_null_mask].values.reshape(-1, 1)
        ).flatten()

    elif method == 'transform':
        skewness = data.dropna().skew()
        if skewness > 0:                         # right tail → compress with log
            non_null = result.dropna()
            if (non_null <= -1).any():
                min_val = non_null.min()
                print(f"Warning: log1p requires values > -1. Shifting data by adding {(-1 - min_val)}")
                shift = -1 - min_val
                result = result + shift

            result = np.log1p(result)
        elif skewness < 0:                       # left tail → pull up with exp
            result = np.exp(result)
        else:
            pass                                 # symmetric, no transform needed

    elif method == 'binning':
        if bins is None:
            raise ValueError("'binning' method requires the 'bins' parameter to be set.")
        if not isinstance(bins, int) or bins < 2:
            raise ValueError("'bins' must be an integer >= 2.")
        result = pd.cut(result, bins=bins, labels=False)
        result = result.astype('Int64')

    if pd.api.types.is_numeric_dtype(result.dtype):
        result = result.round(4)

    return result

def split_outlier_cluster(
    data: pd.DataFrame,
    split_by: list = [],
    evasive: bool = True,
    IRQ_value: float = 1.5,
) -> Union[tuple, dict]:
    """
    Splits a DataFrame based on whether values in `split_by` columns are outliers.

    Outlier detection uses the Tukey IQR fence:
        lower = Q1 - IRQ_value * IQR
        upper = Q3 + IRQ_value * IQR

    Parameters:
        data      : pd.DataFrame — source dataset
        split_by  : list[str]   — columns to evaluate for outliers
        evasive   : bool
            True  → returns a dict with one entry per column:
                     {'col': (common_df, outlier_df), ...}
            False → returns a single tuple (common_df, outlier_df)
                    where common has NO outliers in ANY split_by column
                    and outlier has at least one outlier in ANY split_by column
        IRQ_value : float — IQR multiplier (default 1.5; use 3.0 for extreme outliers)

    Returns:
        dict  if evasive=True  → {col: (common, skewed), ...}
        tuple if evasive=False → (common, skewed)
    """
    if not split_by:
        raise ValueError("'split_by' must contain at least one column name.")

    missing = [c for c in split_by if c not in data.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    def _outlier_mask(series: pd.Series) -> pd.Series:
        """Returns a boolean mask — True where the value IS an outlier."""
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - IRQ_value * iqr, q3 + IRQ_value * iqr
        return (series < lower) | (series > upper)

    if evasive:
        # One (common, skewed) pair per column — independent splits
        splits = {}
        for col in split_by:
            mask = _outlier_mask(data[col])
            splits[col] = (
                data[~mask].reset_index(drop=True),   # common
                data[mask].reset_index(drop=True),    # skewed / outliers
            )
        return splits

    else:
        # A row is "skewed" if it is an outlier in ANY of the split_by columns
        combined_outlier_mask = pd.Series(False, index=data.index)
        for col in split_by:
            combined_outlier_mask |= _outlier_mask(data[col])

        common = data[~combined_outlier_mask].reset_index(drop=True)
        skewed = data[combined_outlier_mask].reset_index(drop=True)
        return common, skewed