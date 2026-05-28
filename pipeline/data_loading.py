"""Helpers to load donor response datasets and describe their schema."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from pandas.api.types import is_numeric_dtype

TARGET_COLUMN = "TARGET_B"
ID_COLUMN = "CONTROL_NUMBER"
DATA_DIR = Path("project_data")

def load_train_data(
    data_dir: str | Path = DATA_DIR,
    filename: str = "donors_train.csv",
) -> pd.DataFrame:
    """
    Load the training dataset from CSV.

    Parameters
    ----------
    data_dir : str | Path, optional
        Directory where the data file is located. Defaults to `DATA_DIR`.
    filename : str, optional
        Filename to load. Defaults to "donors_train.csv".

    Returns
    -------
    pd.DataFrame
        The loaded training dataframe.
    """
    return pd.read_csv(Path(data_dir) / filename)
def load_test_data(
    data_dir: str | Path = DATA_DIR,
    filename: str = "donors_test.csv",
) -> pd.DataFrame:
    """
    Load the test dataset from CSV.

    Parameters
    ----------
    data_dir : str | Path, optional
        Directory where the data file is located. Defaults to `DATA_DIR`.
    filename : str, optional
        Filename to load. Defaults to "donors_test.csv".

    Returns
    -------
    pd.DataFrame
        The loaded test dataframe.
    """
    return pd.read_csv(Path(data_dir) / filename)
def split_features_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    drop_id: bool = False,
    id_column: str = ID_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing both features and the target column.
    target_column : str, optional
        Name of the target column. Defaults to `TARGET_COLUMN`.
    drop_id : bool, optional
        If True and `id_column` exists, drop the id column from features.
    id_column : str, optional
        Column name for record identifier. Defaults to `ID_COLUMN`.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Tuple containing (X, y) where X is the features dataframe and y is the
        target series.

    Raises
    ------
    KeyError
        If `target_column` is not present in `df`.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' was not found in the dataframe.")
    X = df.drop(columns=[target_column]).copy()
    if drop_id and id_column in X.columns:
        X = X.drop(columns=[id_column])
    y = df[target_column].copy()
    return X, y
def infer_feature_types(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    id_column: str = ID_COLUMN,
) -> dict[str, list[str]]:
    """
    Infer and return lists of feature names grouped by type.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and optionally id/target columns.
    target_column : str, optional
        Name of the target column to exclude from feature lists.
    id_column : str, optional
        Name of the id column to exclude from feature lists.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with keys: 'all_features', 'numeric_features', 'categorical_features'.
    """
    excluded_columns = {target_column, id_column}
    feature_columns = [column for column in df.columns if column not in excluded_columns]
    numeric_features = [
        column for column in feature_columns if is_numeric_dtype(df[column])
    ]
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]
    return {
        "all_features": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
