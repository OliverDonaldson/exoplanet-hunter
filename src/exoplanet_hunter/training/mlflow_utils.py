"""Helpers for logging to MLflow.

Centralises the boilerplate so the training scripts stay focused on the model.
Every run logs:

  * Hyperparameters (Hydra config flattened).
  * Metrics (per-epoch via callback; final on test).
  * Plots (ROC, PR, confusion matrix, learning curves).
  * Model artifact (`.keras` for TF, joblib for sklearn).
  * Code version (git SHA if available).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def setup_mlflow(cfg: DictConfig) -> None:
    """Point MLflow at the configured tracking URI + experiment."""
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment)


def _git_sha() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def log_config(cfg: DictConfig) -> None:
    """Flatten an OmegaConf config into MLflow params (truncating long values)."""
    flat = _flatten_dict(OmegaConf.to_container(cfg, resolve=True))
    for k, v in flat.items():
        s = str(v)
        if len(s) > 250:
            s = s[:247] + "..."
        mlflow.log_param(k, s)
    if sha := _git_sha():
        mlflow.set_tag("git_sha", sha)


def _flatten_dict(d: Any, parent: str = "", sep: str = ".") -> dict[str, Any]:
    items: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{parent}{sep}{k}" if parent else k
            items.update(_flatten_dict(v, key, sep))
    else:
        items[parent] = d
    return items


# ---------------------------------------------------------------- artefacts


def log_classification_artifacts(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float = 0.5,
    out_dir: Path,
) -> None:
    """Log ROC, PR, and confusion-matrix plots + summary metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    y_pred = (y_score >= threshold).astype(int)

    # --- metrics ---------------------------------------------------------
    auc = roc_auc_score(y_true, y_score)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    mlflow.log_metrics(
        {
            "test_roc_auc": float(auc),
            "test_precision": float(precision),
            "test_recall": float(recall),
            "test_f1": float(f1),
        }
    )

    # --- ROC -------------------------------------------------------------
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax, name="model")
    ax.set_title(f"ROC — AUC = {auc:.3f}")
    fig.tight_layout()
    roc_path = out_dir / "roc.png"
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(roc_path))

    # --- PR --------------------------------------------------------------
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax, name="model")
    ax.set_title("Precision-Recall")
    fig.tight_layout()
    pr_path = out_dir / "pr.png"
    fig.savefig(pr_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(pr_path))

    # --- confusion matrix -----------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion @ thr={threshold}")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks([0, 1], labels=["non-planet", "planet"])
    ax.set_yticks([0, 1], labels=["non-planet", "planet"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(int(cm[i, j])),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(cm_path))


def log_history(history: dict[str, list[float]], out_dir: Path) -> None:
    """Plot Keras learning curves and log to MLflow."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("loss", "auc", "accuracy"):
        if key not in history:
            continue
        fig, ax = plt.subplots()
        ax.plot(history[key], label=f"train_{key}")
        if (vk := f"val_{key}") in history:
            ax.plot(history[vk], label=vk)
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.set_title(f"Learning curve — {key}")
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"history_{key}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        mlflow.log_artifact(str(path))


def keras_callbacks(cfg: DictConfig, ckpt_path: Path) -> list:
    """Build the standard Keras callback list from a Hydra training cfg."""
    import tensorflow as tf

    cb = cfg.callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=cb.early_stopping.monitor,
            mode=cb.early_stopping.mode,
            patience=cb.early_stopping.patience,
            restore_best_weights=cb.early_stopping.restore_best_weights,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor=cb.model_checkpoint.monitor,
            mode=cb.model_checkpoint.mode,
            save_best_only=cb.model_checkpoint.save_best_only,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=cb.reduce_lr.monitor,
            mode=cb.reduce_lr.mode,
            factor=cb.reduce_lr.factor,
            patience=cb.reduce_lr.patience,
            min_lr=cb.reduce_lr.min_lr,
        ),
    ]

    # MLflow autologging hooks into Keras model.fit() and logs metrics per epoch.
    os.environ.setdefault("MLFLOW_AUTOLOG", "1")
    mlflow.tensorflow.autolog(log_models=False)

    return callbacks
