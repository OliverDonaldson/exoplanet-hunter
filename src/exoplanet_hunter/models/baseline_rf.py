"""Random Forest baseline.

Why RF? It's the natural baseline for this task and aligns with DATA 305 Week 2:

  * Bagging + random feature subsampling (the two core RF ideas) reduce variance.
  * Class-weighting handles imbalance (most TICs are NOT planets).
  * Out-of-bag error gives free validation; we still use a stratified k-fold
    here for an apples-to-apples comparison with the CNN.
  * SHAP gives interpretable feature importance — useful when you want to
    understand *why* the model thinks a candidate looks planet-y.
"""

from __future__ import annotations

from typing import Any

from hydra.utils import instantiate
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_random_forest(cfg: Any) -> Pipeline:
    """Build a (StandardScaler → RandomForestClassifier) pipeline from a Hydra cfg.

    The scaler is harmless for trees, but means the same fitted pipeline can be
    reused if we later swap RF for a logistic regression / SVM baseline.
    """
    estimator: RandomForestClassifier = instantiate(cfg.estimator)
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )
