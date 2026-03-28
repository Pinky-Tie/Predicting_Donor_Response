"""Utility functions for EDA, metrics, and plotting."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "analyze_missingness_patterns": ".utils_eda",
    "classification_report_df": ".utils_metrics",
    "metric_table": ".utils_metrics",
    "plot_confusion_matrix_heatmap": ".utils_plots",
    "plot_probability_distribution": ".utils_plots",
    "reduce_memory_usage": ".utils_eda",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module = import_module(_EXPORTS[name], __name__)
    return getattr(module, name)
