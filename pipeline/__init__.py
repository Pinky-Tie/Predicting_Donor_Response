"""Reusable pipeline components for donor response classification."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DEFAULT_ID_COLUMN": ".data_loading",
    "DEFAULT_TARGET_COLUMN": ".data_loading",
    "DonorFeatureEngineer": ".preprocessing_feature_engineering",
    "build_categorical_encoder": ".preprocessing_encoding",
    "build_categorical_imputer": ".preprocessing_missing_values",
    "build_numeric_imputer": ".preprocessing_missing_values",
    "build_numeric_scaler": ".preprocessing_scaling",
    "build_submission": ".submission",
    "compare_models": ".model_training",
    "evaluate_classification": ".evaluation",
    "evaluate_thresholds": ".evaluation",
    "find_best_threshold": ".evaluation",
    "get_baseline_models": ".model_training",
    "infer_feature_types": ".data_loading",
    "load_test_data": ".data_loading",
    "load_train_data": ".data_loading",
    "run_cross_validation": ".model_training",
    "save_submission": ".submission",
    "split_features_target": ".data_loading",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = import_module(_EXPORTS[name], __name__)
    return getattr(module, name)
