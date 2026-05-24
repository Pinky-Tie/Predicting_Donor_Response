"""Reusable pipeline components for donor response classification."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "DEFAULT_CODED_NUMERIC_FEATURES": ".feature_pipeline",
    "DEFAULT_ID_COLUMN": ".data_loading",
    "DEFAULT_TARGET_COLUMN": ".data_loading",
    "DonorFeatureEngineer": ".preprocessing_feature_engineering",
    "HIGH_MISSING_CANDIDATES": ".feature_pipeline",
    "HIGH_MISSING_THRESHOLD": ".feature_pipeline",
    "PREPROCESSING_SCENARIOS": ".feature_pipeline",
    "add_engineered_features": ".feature_pipeline",
    "build_categorical_encoder": ".preprocessing_encoding",
    "build_categorical_imputer": ".preprocessing_missing_values",
    "build_numeric_imputer": ".preprocessing_missing_values",
    "build_numeric_scaler": ".preprocessing_scaling",
    "build_one_hot_encoder": ".feature_pipeline",
    "build_preprocessor": ".feature_pipeline",
    "build_submission": ".submission",
    "compare_models": ".model_training",
    "compute_metrics_from_probabilities": ".feature_pipeline",
    "evaluate_classification": ".evaluation",
    "evaluate_classifier": ".feature_pipeline",
    "evaluate_feature_set": ".feature_pipeline",
    "evaluate_thresholds": ".evaluation",
    "find_best_f1_threshold": ".feature_pipeline",
    "find_best_threshold": ".evaluation",
    "get_baseline_models": ".model_training",
    "get_feature_groups": ".feature_pipeline",
    "infer_feature_types": ".data_loading",
    "load_test_data": ".data_loading",
    "load_train_data": ".data_loading",
    "prepare_modeling_frame": ".feature_pipeline",
    "run_cross_validation": ".model_training",
    "safe_divide": ".feature_pipeline",
    "save_submission": ".submission",
    "split_features_target": ".data_loading",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = import_module(_EXPORTS[name], __name__)
    return getattr(module, name)
