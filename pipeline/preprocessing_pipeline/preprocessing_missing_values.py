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

def treat_mnar(
    df: pd.DataFrame,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Apply targeted imputation for the three confirmed MNAR variables.

    A binary missingness indicator ({COL}_MISSING, int8) is created for each
    variable BEFORE imputation, so the full missing set is captured. These
    indicators are kept in the dataframe as features for downstream models.

    Variables treated
    -----------------
    WEALTH_RATING (46 % missing) — strong MNAR (AUC 0.896)
        Missing because the enrichment vendor cannot score low-value / short-
        tenure donors. Filled with 0 (valid domain minimum) to avoid inflating
        apparent wealth for the lowest-engagement segment.

    MONTHS_SINCE_LAST_PROM_RESP (3.28 % missing) — MNAR (AUC 0.702)
        Missingness associated with recent gift behaviour, consistent with
        CRM logging gaps for less-engaged donors. Filled with the median.

    DONOR_AGE (26 % missing) — conservative MNAR (AUC 0.613)
        The numeric logistic regression underestimates this signal because the
        primary driver (HOME_OWNER) is categorical. Manual analysis confirms a
        4× missingness gap between renters (44 %) and homeowners (11 %),
        consistent with structural absence from property records.
        Imputed with stratified median per HOME_OWNER x URBANICITY cell.
        Fallback chain: HOME_OWNER-only median → global median.

    Parameters
    ----------
    df : pd.DataFrame
        Frame after invalid value cleaning, before generic imputation.
    inplace : bool
        If False (default) operate on a copy.

    Returns
    -------
    pd.DataFrame
        Same frame with three _MISSING indicator columns added and the three
        MNAR columns imputed.
    """
    out = df if inplace else df.copy()

    # ------------------------------------------------------------------
    # WEALTH_RATING — fill with domain minimum + indicator
    # ------------------------------------------------------------------
    if "WEALTH_RATING" in out.columns:
        out["WEALTH_RATING_MISSING"] = out["WEALTH_RATING"].isna().astype("int8")
        out["WEALTH_RATING"] = out["WEALTH_RATING"].fillna(0)

    # ------------------------------------------------------------------
    # MONTHS_SINCE_LAST_PROM_RESP — fill with median + indicator
    # ------------------------------------------------------------------
    if "MONTHS_SINCE_LAST_PROM_RESP" in out.columns:
        out["MONTHS_SINCE_LAST_PROM_RESP_MISSING"] = (
            out["MONTHS_SINCE_LAST_PROM_RESP"].isna().astype("int8")
        )
        _median = out["MONTHS_SINCE_LAST_PROM_RESP"].dropna().median()
        out["MONTHS_SINCE_LAST_PROM_RESP"] = (
            out["MONTHS_SINCE_LAST_PROM_RESP"].fillna(_median)
        )

    # ------------------------------------------------------------------
    # DONOR_AGE — stratified median imputation + indicator
    # ------------------------------------------------------------------
    if "DONOR_AGE" in out.columns:
        out["DONOR_AGE_MISSING"] = out["DONOR_AGE"].isna().astype("int8")

        _age_medians = (
            out.dropna(subset=["DONOR_AGE"])
            .groupby(["HOME_OWNER", "URBANICITY"], observed=True)["DONOR_AGE"]
            .median()
        )
        _hw_medians = (
            out.dropna(subset=["DONOR_AGE"])
            .groupby("HOME_OWNER", observed=True)["DONOR_AGE"]
            .median()
        )
        _global_age = out["DONOR_AGE"].dropna().median()

        def _impute_age(row):
            key = (row.get("HOME_OWNER"), row.get("URBANICITY"))
            if key in _age_medians.index:
                return _age_medians[key]
            hw = _hw_medians.get(row.get("HOME_OWNER"))
            return hw if pd.notna(hw) else _global_age

        _mask = out["DONOR_AGE_MISSING"] == 1
        if _mask.any():
            out.loc[_mask, "DONOR_AGE"] = out[_mask].apply(_impute_age, axis=1)

    return out