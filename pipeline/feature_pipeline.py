"""Shared feature engineering, preprocessing, and evaluation helpers.

Centralizes the logic that was previously duplicated across the baseline-models
and feature-engineering notebooks so both can call the same canonical
implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# ---------------------------------------------------------------------------
# Constants describing preprocessing scenarios
# ---------------------------------------------------------------------------
HIGH_MISSING_THRESHOLD = 0.40
HIGH_MISSING_CANDIDATES: list[str] = ["WEALTH_RATING"]
DEFAULT_CODED_NUMERIC_FEATURES: list[str] = ["INCOME_GROUP", "WEALTH_RATING"]

PREPROCESSING_SCENARIOS: list[dict] = [
    {
        "scenario_name": "Drop WEALTH_RATING | INCOME_GROUP numeric",
        "drop_high_missing_features": True,
        "income_group_as_categorical": False,
    },
    {
        "scenario_name": "Keep WEALTH_RATING | INCOME_GROUP numeric",
        "drop_high_missing_features": False,
        "income_group_as_categorical": False,
    },
    {
        "scenario_name": "Drop WEALTH_RATING | INCOME_GROUP categorical",
        "drop_high_missing_features": True,
        "income_group_as_categorical": True,
    },
    {
        "scenario_name": "Keep WEALTH_RATING | INCOME_GROUP categorical",
        "drop_high_missing_features": False,
        "income_group_as_categorical": True,
    },
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while treating zero denominators as missing values."""
    return numerator / denominator.replace(0, np.nan)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create donor-level features using the cleaned original variables."""
    engineered = df.copy()

    # Ratio-based features linked to donation value and campaign exposure
    if {"LIFETIME_GIFT_AMOUNT", "LIFETIME_GIFT_COUNT"}.issubset(engineered.columns):
        engineered["AVG_GIFT_PER_DONATION"] = safe_divide(
            engineered["LIFETIME_GIFT_AMOUNT"],
            engineered["LIFETIME_GIFT_COUNT"],
        )

    if {"LIFETIME_GIFT_AMOUNT", "LIFETIME_PROM"}.issubset(engineered.columns):
        engineered["GIFT_AMOUNT_PER_PROMOTION"] = safe_divide(
            engineered["LIFETIME_GIFT_AMOUNT"],
            engineered["LIFETIME_PROM"],
        )

    if {"LIFETIME_GIFT_COUNT", "LIFETIME_PROM"}.issubset(engineered.columns):
        engineered["GIFT_COUNT_PER_PROMOTION"] = safe_divide(
            engineered["LIFETIME_GIFT_COUNT"],
            engineered["LIFETIME_PROM"],
        )

    if {"RECENT_RESPONSE_COUNT", "LIFETIME_PROM"}.issubset(engineered.columns):
        engineered["PROMOTION_RESPONSE_RATE"] = safe_divide(
            engineered["RECENT_RESPONSE_COUNT"],
            engineered["LIFETIME_PROM"],
        )

    # Donor-activity features linked to time and engagement intensity
    if {"MONTHS_SINCE_FIRST_GIFT", "MONTHS_SINCE_LAST_GIFT"}.issubset(engineered.columns):
        engineered["MONTHS_BETWEEN_FIRST_AND_LAST_GIFT"] = (
            engineered["MONTHS_SINCE_FIRST_GIFT"]
            - engineered["MONTHS_SINCE_LAST_GIFT"]
        )

    if {"LIFETIME_GIFT_COUNT", "MONTHS_SINCE_FIRST_GIFT"}.issubset(engineered.columns):
        engineered["GIFT_COUNT_PER_MONTH_ACTIVE"] = safe_divide(
            engineered["LIFETIME_GIFT_COUNT"],
            engineered["MONTHS_SINCE_FIRST_GIFT"],
        )

    if {"LIFETIME_PROM", "MONTHS_SINCE_FIRST_GIFT"}.issubset(engineered.columns):
        engineered["PROMOTIONS_PER_MONTH_ACTIVE"] = safe_divide(
            engineered["LIFETIME_PROM"],
            engineered["MONTHS_SINCE_FIRST_GIFT"],
        )

    # Recent-response features that compare card and overall engagement patterns
    if {"RECENT_RESPONSE_PROP", "RECENT_CARD_RESPONSE_PROP"}.issubset(engineered.columns):
        engineered["RESPONSES_PER_YEAR"] = (
            engineered["RECENT_RESPONSE_COUNT"]
            / 4
        )

    if {"RECENT_RESPONSE_PROP", "RECENT_CARD_RESPONSE_PROP"}.issubset(engineered.columns):
        engineered["RECENT_RESPONSE_RATIO_GAP"] = (
            engineered["RECENT_RESPONSE_PROP"]
            - engineered["RECENT_CARD_RESPONSE_PROP"]
        )

    if {"RECENT_RESPONSE_COUNT", "RECENT_CARD_RESPONSE_COUNT"}.issubset(engineered.columns):
        engineered["RECENT_RESPONSE_COUNT_GAP"] = (
            engineered["RECENT_RESPONSE_COUNT"]
            - engineered["RECENT_CARD_RESPONSE_COUNT"]
        )
        engineered["CARD_RESPONSE_SHARE"] = safe_divide(
            engineered["RECENT_CARD_RESPONSE_COUNT"],
            engineered["RECENT_RESPONSE_COUNT"],
        )

    if {"RECENT_CARD_RESPONSE_COUNT", "LIFETIME_CARD_PROM"}.issubset(engineered.columns):
        engineered["CARD_RESPONSE_RATE"] = safe_divide(
            engineered["RECENT_CARD_RESPONSE_COUNT"],
            engineered["LIFETIME_CARD_PROM"],
        )

    return engineered


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def prepare_modeling_frame(
    df: pd.DataFrame, income_group_as_categorical: bool = False
) -> pd.DataFrame:
    """Create a modeling dataframe for one preprocessing scenario."""
    prepared = df.copy()

    if income_group_as_categorical and "INCOME_GROUP" in prepared.columns:
        prepared["INCOME_GROUP"] = prepared["INCOME_GROUP"].astype("object")

    categorical_columns = prepared.select_dtypes(exclude=[np.number]).columns.tolist()
    for column in categorical_columns:
        prepared[column] = prepared[column].astype("object")
        prepared[column] = prepared[column].apply(
            lambda value: np.nan if pd.isna(value) else value
        )

    return prepared


def build_one_hot_encoder() -> OneHotEncoder:
    """Create a one-hot encoder compatible with different scikit-learn versions."""
    params: dict = {"handle_unknown": "ignore"}
    if "sparse_output" in OneHotEncoder.__init__.__code__.co_varnames:
        params["sparse_output"] = False
    else:
        params["sparse"] = False
    return OneHotEncoder(**params)


def get_feature_groups(
    df: pd.DataFrame,
    drop_high_missing_features: bool = True,
    coded_numeric_features=None,
) -> dict:
    """Organize the variables into preprocessing groups."""
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if coded_numeric_features is None:
        coded_numeric_features = DEFAULT_CODED_NUMERIC_FEATURES

    dropped_high_missing_features: list[str] = []
    if drop_high_missing_features:
        dropped_high_missing_features = [
            column
            for column in HIGH_MISSING_CANDIDATES
            if column in df.columns and df[column].isna().mean() >= HIGH_MISSING_THRESHOLD
        ]

    coded_numeric_features = [
        column
        for column in coded_numeric_features
        if column in numeric_features and column not in dropped_high_missing_features
    ]

    continuous_numeric_features = [
        column
        for column in numeric_features
        if column not in coded_numeric_features + dropped_high_missing_features
    ]

    return {
        "continuous_numeric_features": continuous_numeric_features,
        "coded_numeric_features": coded_numeric_features,
        "categorical_features": categorical_features,
        "dropped_high_missing_features": dropped_high_missing_features,
    }


def build_preprocessor(
    df: pd.DataFrame,
    model_family: str = "linear",
    drop_high_missing_features: bool = True,
):
    """Create a preprocessing pipeline adapted to the feature groups and the model family."""
    groups = get_feature_groups(
        df,
        drop_high_missing_features=drop_high_missing_features,
    )
    transformers: list = []

    continuous_numeric_steps: list = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ]
    coded_numeric_steps: list = [
        ("imputer", SimpleImputer(strategy="most_frequent", add_indicator=True)),
    ]

    if model_family == "linear":
        continuous_numeric_steps.append(("scaler", RobustScaler()))
        coded_numeric_steps.append(("scaler", RobustScaler()))

    if groups["continuous_numeric_features"]:
        transformers.append(
            (
                "continuous_numeric",
                Pipeline(steps=continuous_numeric_steps),
                groups["continuous_numeric_features"],
            )
        )

    if groups["coded_numeric_features"]:
        transformers.append(
            (
                "coded_numeric",
                Pipeline(steps=coded_numeric_steps),
                groups["coded_numeric_features"],
            )
        )

    if groups["categorical_features"]:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encoder", build_one_hot_encoder()),
                    ]
                ),
                groups["categorical_features"],
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, groups


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def compute_metrics_from_probabilities(
    y_true, y_proba, threshold: float = 0.50
) -> dict:
    """Convert predicted probabilities into class predictions and compute standard metrics."""
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def find_best_f1_threshold(y_true, y_proba) -> float:
    """Search for the probability threshold that maximizes the F1-score on the validation set."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    if len(thresholds) == 0:
        return 0.50

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(np.nanargmax(f1_scores))
    return float(thresholds[best_index])


def evaluate_classifier(
    model_pipeline,
    X_train_df,
    X_val_df,
    y_train_series,
    y_val_series,
    threshold: float = 0.50,
    return_train_metrics: bool = False,
):
    """Fit a pipeline on the training data and evaluate it on the validation set.

    When *return_train_metrics* is True the function returns a third element with
    the same metric dict computed on the training set, useful for diagnosing
    overfitting (train–val gap).
    """
    model_pipeline.fit(X_train_df, y_train_series)
    y_proba = model_pipeline.predict_proba(X_val_df)[:, 1]
    metrics = compute_metrics_from_probabilities(
        y_val_series, y_proba, threshold=threshold
    )

    if return_train_metrics:
        y_train_proba = model_pipeline.predict_proba(X_train_df)[:, 1]
        train_metrics = compute_metrics_from_probabilities(
            y_train_series, y_train_proba, threshold=threshold
        )
        return metrics, y_proba, train_metrics

    return metrics, y_proba


def evaluate_feature_set(
    feature_set_name,
    X_train_df,
    X_val_df,
    y_train_series,
    y_val_series,
    model_configs,
    preprocessing_scenarios,
):
    """Evaluate one feature representation under all preprocessing scenarios and candidate models."""
    rows: list[dict] = []
    validation_probabilities: dict = {}

    for scenario in preprocessing_scenarios:
        scenario_train = prepare_modeling_frame(
            X_train_df,
            income_group_as_categorical=scenario["income_group_as_categorical"],
        )
        scenario_val = prepare_modeling_frame(
            X_val_df,
            income_group_as_categorical=scenario["income_group_as_categorical"],
        )

        for model_name, config in model_configs.items():
            preprocessor, feature_groups = build_preprocessor(
                scenario_train,
                model_family=config["model_family"],
                drop_high_missing_features=scenario["drop_high_missing_features"],
            )

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", clone(preprocessor)),
                    ("model", clone(config["estimator"])),
                ]
            )

            metrics, y_proba = evaluate_classifier(
                model_pipeline=pipeline,
                X_train_df=scenario_train,
                X_val_df=scenario_val,
                y_train_series=y_train_series,
                y_val_series=y_val_series,
                threshold=0.50,
            )

            rows.append(
                {
                    "feature_set": feature_set_name,
                    "scenario_name": scenario["scenario_name"],
                    "model": model_name,
                    "model_family": config["model_family"],
                    "income_group_role": "categorical"
                    if scenario["income_group_as_categorical"]
                    else "numeric",
                    "wealth_rating_strategy": "drop_if_missing_ge_40pct"
                    if scenario["drop_high_missing_features"]
                    else "keep_with_imputation",
                    "n_continuous_numeric_features": len(
                        feature_groups["continuous_numeric_features"]
                    ),
                    "n_coded_numeric_features": len(
                        feature_groups["coded_numeric_features"]
                    ),
                    "n_categorical_features": len(feature_groups["categorical_features"]),
                    "dropped_high_missing": ", ".join(
                        feature_groups["dropped_high_missing_features"]
                    )
                    or "None",
                    **metrics,
                }
            )

            validation_probabilities[
                (feature_set_name, scenario["scenario_name"], model_name)
            ] = y_proba

    return pd.DataFrame(rows), validation_probabilities


__all__ = [
    "HIGH_MISSING_THRESHOLD",
    "HIGH_MISSING_CANDIDATES",
    "DEFAULT_CODED_NUMERIC_FEATURES",
    "PREPROCESSING_SCENARIOS",
    "safe_divide",
    "add_engineered_features",
    "prepare_modeling_frame",
    "build_one_hot_encoder",
    "get_feature_groups",
    "build_preprocessor",
    "compute_metrics_from_probabilities",
    "find_best_f1_threshold",
    "evaluate_classifier",
    "evaluate_feature_set",
]
