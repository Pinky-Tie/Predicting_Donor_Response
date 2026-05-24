"""Utility functions for EDA, metrics, and plotting."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "analyze_missingness_patterns": ".utils_eda",
    "classification_report_df": ".utils_metrics",
    "metric_table": ".utils_metrics",
    "plot_anova_f_scores": ".utils_plots",
    "plot_boxplots_numeric": ".utils_plots",
    "plot_categorical_frequencies": ".utils_plots",
    "plot_categorical_target_rate": ".utils_plots",
    "plot_confusion_matrix_heatmap": ".utils_plots",
    "plot_correlation_heatmap": ".utils_plots",
    "plot_invalid_value_counts": ".utils_plots",
    "plot_lasso_feature_importance": ".utils_plots",
    "plot_missing_overlap_heatmap": ".utils_plots",
    "plot_missing_percentages": ".utils_plots",
    "plot_missing_vs_target": ".utils_plots",
    "plot_outlier_boxplots": ".utils_plots",
    "plot_outlier_counts_iqr": ".utils_plots",
    "plot_outlier_histograms": ".utils_plots",
    "plot_outlier_percentages_iqr": ".utils_plots",
    "plot_pca_explained_variance": ".utils_plots",
    "plot_probability_distribution": ".utils_plots",
    "plot_rfe_score_curve": ".utils_plots",
    "plot_target_correlation": ".utils_plots",
    "plot_tree_feature_importance": ".utils_plots",
    "plot_variance_ranking": ".utils_plots",
    "reduce_memory_usage": ".utils_eda",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = import_module(_EXPORTS[name], __name__)
    return getattr(module, name)
