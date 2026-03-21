"""Plot helpers used across notebooks and model evaluation."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix_heatmap(
    y_true,
    y_pred,
    labels: tuple[str, str] = ("No Donation", "Donation"),
    ax=None,
):
    """Plot a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)
    ax.set_title("Confusion Matrix")
    return ax


def plot_probability_distribution(y_proba, ax=None, bins: int = 20):
    """Plot the distribution of positive-class predicted probabilities."""
    scores = np.asarray(y_proba)
    if scores.ndim == 2 and scores.shape[1] >= 2:
        scores = scores[:, 1]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.hist(scores, bins=bins, color="#2f6db2", alpha=0.85, edgecolor="white")
    ax.set_title("Predicted Probability Distribution")
    ax.set_xlabel("Positive Class Probability")
    ax.set_ylabel("Count")
    return ax

