# """Evaluation helpers for binary donor response models."""
# from __future__ import annotations
# import numpy as np
# import pandas as pd
# from sklearn.metrics import (
#     accuracy_score,
#     confusion_matrix,
#     f1_score,
#     precision_score,
#     recall_score,
#     roc_auc_score,
# )
# def _get_positive_class_scores(y_proba) -> np.ndarray:
#     """Normalize predicted probabilities into a 1D array of positive scores."""
#     scores = np.asarray(y_proba)
#     if scores.ndim == 1:
#         return scores
#     if scores.ndim == 2 and scores.shape[1] >= 2:
#         return scores[:, 1]
#     raise ValueError("Predicted probabilities must be a 1D array or an Nx2 array.")
# def evaluate_classification(y_true, y_pred, y_proba=None) -> dict[str, float | int]:
#     """Return the core classification metrics used in the project."""
#     metrics = {
#         "accuracy": accuracy_score(y_true, y_pred),
#         "precision": precision_score(y_true, y_pred, zero_division=0),
#         "recall": recall_score(y_true, y_pred, zero_division=0),
#         "f1": f1_score(y_true, y_pred, zero_division=0),
#     }
#     if y_proba is not None:
#         try:
#             metrics["roc_auc"] = roc_auc_score(y_true, _get_positive_class_scores(y_proba))
#         except ValueError:
#             metrics["roc_auc"] = np.nan

#     tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#     metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
#     return metrics
# def evaluate_thresholds(
#     y_true,
#     y_proba,
#     thresholds: list[float] | None = None,
# ) -> pd.DataFrame:
#     """Evaluate several probability thresholds for the positive class."""
#     scores = _get_positive_class_scores(y_proba)
#     thresholds = thresholds or [round(value, 2) for value in np.arange(0.1, 0.95, 0.05)]

#     rows = []
#     for threshold in thresholds:
#         y_pred = (scores >= threshold).astype(int)
#         row = {"threshold": threshold}
#         row.update(evaluate_classification(y_true=y_true, y_pred=y_pred, y_proba=scores))
#         rows.append(row)

#     return pd.DataFrame(rows)
# def find_best_threshold(
#     y_true,
#     y_proba,
#     metric: str = "f1",
# ) -> dict[str, float]:
#     """Select the threshold that maximizes a chosen evaluation metric."""
#     threshold_results = evaluate_thresholds(y_true=y_true, y_proba=y_proba)
#     if metric not in threshold_results.columns:
#         raise KeyError(f"Metric '{metric}' was not found in the threshold evaluation results.")

#     best_row = threshold_results.loc[threshold_results[metric].idxmax()]
#     return {
#         "best_threshold": float(best_row["threshold"]),
#         metric: float(best_row[metric]),
#     }
