"""Convenience wrappers for presenting classification metrics."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import classification_report


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    """Return sklearn's classification report as a dataframe."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).transpose()


def metric_table(metrics: dict[str, float | int]) -> pd.DataFrame:
    """Convert a metric dictionary into a tidy two-column dataframe."""
    return pd.DataFrame(
        {
            "metric": list(metrics.keys()),
            "value": list(metrics.values()),
        }
    )

