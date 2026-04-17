"""Classification + calibration metrics for binary classifiers.

Calibration matters here: when the model says "0.9", we want that to mean
"90% chance this is a real planet" — not just "this is in the top decile of
my scores". A miscalibrated classifier with great AUC still produces
misleading candidate scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import calibration_curve as _sk_calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    brier: float


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> ClassificationMetrics:
    y_pred = (y_score >= threshold).astype(int)
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(y_true, y_score)),
        pr_auc=float(average_precision_score(y_true, y_score)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        brier=float(brier_score_loss(y_true, y_score)),
    )


def calibration_curve(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Reliability diagram (fraction-of-positives vs predicted-probability)."""
    return _sk_calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
