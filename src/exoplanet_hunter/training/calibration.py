"""Post-hoc temperature scaling for binary-sigmoid models (Guo et al. 2017).

Fit a single scalar T > 0 on validation logits by minimising negative
log-likelihood. Apply at inference as `sigmoid(logit / T)`.

Rank-preserving: ROC-AUC / PR-AUC are identical pre- and post-calibration
(monotonic transform on a ranking). Only Brier and reliability change.
T > 1 → model was overconfident; T < 1 → underconfident. ExoNet (Islam 2026)
reports T* = 1.573 for their dual-stream classifier.

The `TemperatureScaler` class mirrors the sklearn `IsotonicRegression`
interface (`.predict(scores) -> scores`) so the existing inference path
in `scripts/score_target.py` works unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-7


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    bounds: tuple[float, float] = (0.05, 20.0),
) -> float:
    """Minimise NLL(T) on validation `(scores, labels)` and return T*."""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    logits = _logit(scores)

    def nll(T: float) -> float:
        if T <= 0:
            return float("inf")
        p = _sigmoid(logits / T)
        p = np.clip(p, _EPS, 1.0 - _EPS)
        return -float(np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    return float(res.x)


class TemperatureScaler:
    """Sklearn-shaped wrapper around a learned scalar temperature."""

    def __init__(self, T: float = 1.0) -> None:
        self.T = float(T)

    @classmethod
    def from_validation(
        cls,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> TemperatureScaler:
        return cls(T=fit_temperature(scores, labels))

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64).ravel()
        return _sigmoid(_logit(scores) / self.T)

    def __repr__(self) -> str:
        return f"TemperatureScaler(T={self.T:.4f})"
