"""Hydra-driven training entry point.

Usage (from repo root):

    python -m exoplanet_hunter.training.train                      # defaults
    python -m exoplanet_hunter.training.train model=random_forest  # swap model
    python -m exoplanet_hunter.training.train data=small           # swap dataset
    python -m exoplanet_hunter.training.train model=cnn_dualview train.epochs=5

Logs every run to MLflow. Saves best checkpoint to `models/`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
import joblib
import mlflow
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from exoplanet_hunter.features import extract_features
from exoplanet_hunter.models import (
    binary_focal_loss,
    build_cnn_dualview,
    build_random_forest,
)
from exoplanet_hunter.training.data_module import (
    LightcurveDataset,
    ViewArrays,
    load_views,
    slice_views,
    train_val_test_split,
)
from exoplanet_hunter.training.mlflow_utils import (
    keras_callbacks,
    log_classification_artifacts,
    log_config,
    log_history,
    setup_mlflow,
)
from exoplanet_hunter.utils import ProjectPaths, get_logger, set_global_seed

log = get_logger(__name__)


def run(cfg: DictConfig) -> float:
    """Train one model from a fully-resolved Hydra config.

    Decoupled from `@hydra.main` so it can be invoked from any entry-point
    script (e.g. `scripts/train_model.py`) without the script needing to
    re-resolve the config-path issue.
    """
    set_global_seed(int(cfg.seed))
    paths = ProjectPaths.from_cfg(cfg)
    setup_mlflow(cfg)

    views_path = paths.data_processed / "views.npz"
    if not views_path.exists():
        raise FileNotFoundError(
            f"{views_path} missing. Run `python scripts/build_dataset.py data={cfg.data.name}` first."
        )

    views = load_views(views_path)

    if cfg.model.type == "sklearn":
        train_v, val_v, test_v = train_val_test_split(
            views,
            train=float(cfg.data.split.train),
            val=float(cfg.data.split.val),
            test=float(cfg.data.split.test),
            seed=int(cfg.seed),
        )
        log.info(
            "[train] split sizes — train=%d  val=%d  test=%d",
            len(train_v.labels),
            len(val_v.labels),
            len(test_v.labels),
        )
        return _train_sklearn(cfg, paths, train_v, val_v, test_v)
    if cfg.model.type == "keras":
        return _train_keras(cfg, paths, views)
    raise ValueError(f"unknown model type: {cfg.model.type}")


@hydra.main(version_base="1.3", config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """Hydra entry-point for `python -m exoplanet_hunter.training.train`."""
    return run(cfg)


# ---------------------------------------------------------------- sklearn ----


def _train_sklearn(
    cfg: DictConfig,
    paths: ProjectPaths,
    train_v: ViewArrays,
    val_v: ViewArrays,
    test_v: ViewArrays,
) -> float:
    import shap
    from sklearn.metrics import classification_report

    # Concatenate train+val for k-fold CV; hold test out.
    train_views_all = np.concatenate([train_v.global_views, val_v.global_views])
    train_y_all = np.concatenate([train_v.labels, val_v.labels])
    test_views = test_v.global_views
    test_y = test_v.labels.astype(int)

    log.info("[train-rf] extracting handcrafted features ...")
    X_trainval = np.array([extract_features(v) for v in train_views_all])
    X_test = np.array([extract_features(v) for v in test_views])

    pipeline = build_random_forest(cfg.model)

    with mlflow.start_run(run_name=f"rf-{cfg.data.name}"):
        log_config(cfg)

        # Stratified k-fold CV on the train+val pool.
        skf = StratifiedKFold(
            n_splits=int(cfg.model.cross_validation.n_splits),
            shuffle=bool(cfg.model.cross_validation.shuffle),
            random_state=int(cfg.model.cross_validation.random_state),
        )
        cv_aucs: list[float] = []
        for fold, (tr, va) in enumerate(skf.split(X_trainval, train_y_all)):
            pipeline.fit(X_trainval[tr], train_y_all[tr])
            score = pipeline.predict_proba(X_trainval[va])[:, 1]
            auc = roc_auc_score(train_y_all[va], score)
            cv_aucs.append(auc)
            mlflow.log_metric(f"cv_auc_fold_{fold}", float(auc))
        mlflow.log_metric("cv_auc_mean", float(np.mean(cv_aucs)))
        mlflow.log_metric("cv_auc_std", float(np.std(cv_aucs)))
        log.info("[train-rf] CV AUC %.4f ± %.4f", np.mean(cv_aucs), np.std(cv_aucs))

        # Refit on full train+val and evaluate on test.
        pipeline.fit(X_trainval, train_y_all)
        test_score = pipeline.predict_proba(X_test)[:, 1]

        report = classification_report(test_y, (test_score >= 0.5).astype(int), zero_division=0)
        log.info("[train-rf] test classification report:\n%s", report)

        log_classification_artifacts(
            test_y,
            test_score,
            threshold=0.5,
            out_dir=paths.results / "rf",
        )

        # SHAP feature importance.
        if bool(cfg.model.feature_importance):
            try:
                explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
                shap_values = explainer.shap_values(
                    pipeline.named_steps["scaler"].transform(X_test)
                )
                if isinstance(shap_values, list):  # binary: list of two arrays
                    shap_values = shap_values[1]
                import matplotlib.pyplot as plt

                from exoplanet_hunter.features import FEATURE_NAMES

                plt.figure()
                shap.summary_plot(
                    shap_values,
                    pipeline.named_steps["scaler"].transform(X_test),
                    feature_names=FEATURE_NAMES,
                    show=False,
                )
                shap_path = paths.results / "rf" / "shap_summary.png"
                plt.tight_layout()
                plt.savefig(shap_path, dpi=120, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(str(shap_path))
            except Exception as exc:
                log.warning("[train-rf] SHAP failed: %s", exc)

        # Save the fitted pipeline.
        artifact = paths.models / "random_forest.joblib"
        joblib.dump(pipeline, artifact)
        mlflow.log_artifact(str(artifact))

        return float(np.mean(cv_aucs))


# ----------------------------------------------------------------- keras -----


def _train_keras(cfg: DictConfig, paths: ProjectPaths, views: ViewArrays) -> float:
    """Dispatch CNN training between single-split and k-fold CV modes.

    CV mode is selected when `cfg.model.cross_validation` exists and
    `n_splits >= 2`. Single-split mode preserves the legacy artifact path
    (`models/cnn_dualview.keras`) for downstream inference scripts.
    """
    cv_cfg = OmegaConf.select(cfg.model, "cross_validation")
    n_splits = int(cv_cfg.n_splits) if cv_cfg is not None else 1

    if n_splits < 2:
        train_v, val_v, test_v = train_val_test_split(
            views,
            train=float(cfg.data.split.train),
            val=float(cfg.data.split.val),
            test=float(cfg.data.split.test),
            seed=int(cfg.seed),
        )
        log.info(
            "[train-cnn] split sizes — train=%d  val=%d  test=%d",
            len(train_v.labels),
            len(val_v.labels),
            len(test_v.labels),
        )
        return _train_keras_single(cfg, paths, train_v, val_v, test_v)
    return _train_keras_cv(cfg, paths, views, cv_cfg)


def _train_keras_single(
    cfg: DictConfig,
    paths: ProjectPaths,
    train_v: ViewArrays,
    val_v: ViewArrays,
    test_v: ViewArrays,
) -> float:
    """Legacy single train/val/test split path. Saves to `models/cnn_dualview.keras`."""
    ckpt_path = paths.models / "cnn_dualview.keras"
    cal_path = paths.models / "cnn_calibrator.joblib"
    results_dir = paths.results / "cnn"
    with mlflow.start_run(run_name=f"cnn-{cfg.data.name}"):
        log_config(cfg)
        metrics = _run_keras_fold(
            cfg=cfg,
            train_v=train_v,
            val_v=val_v,
            test_v=test_v,
            ckpt_path=ckpt_path,
            cal_path=cal_path,
            results_dir=results_dir,
            history_dir=paths.results / "cnn",
            log_artifacts=True,
        )
        return float(metrics["test_roc_auc"])


def _train_keras_cv(
    cfg: DictConfig,
    paths: ProjectPaths,
    views: ViewArrays,
    cv_cfg: DictConfig,
) -> float:
    """Stratified group k-fold CV. Group = tic_id, stratify on label.

    Per fold:
      - outer split → fold-test (held out)
      - inner GroupShuffleSplit on the remainder → fold-train + fold-val
        (val drives EarlyStopping, threshold sweep, calibrator fit)
      - aux pipeline refit per fold; persisted alongside the fold checkpoint
      - per-fold classification artifacts logged in a nested MLflow run

    Returns the mean test ROC-AUC across folds (matches Hydra-sweep contract).
    """
    n_splits = int(cv_cfg.n_splits)
    shuffle = bool(cv_cfg.shuffle)
    random_state = int(cv_cfg.random_state)
    val_frac = float(cv_cfg.val_frac_within_fold)

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
    idx = np.arange(len(views.labels))
    y = views.labels.astype(int)
    groups = views.tic_ids

    parent_run_name = f"cnn-cv-{cfg.data.name}"
    with mlflow.start_run(run_name=parent_run_name) as parent:
        log_config(cfg)
        run_id = parent.info.run_id
        cv_root = paths.models / "cv" / run_id
        cv_root.mkdir(parents=True, exist_ok=True)
        results_root = paths.results / "cnn" / "cv" / run_id
        results_root.mkdir(parents=True, exist_ok=True)

        log.info(
            "[train-cnn-cv] %d folds, run_id=%s, artifacts→%s",
            n_splits,
            run_id,
            cv_root,
        )

        fold_rows: list[dict] = []
        for fold_idx, (trainval_idx, test_idx) in enumerate(sgkf.split(idx, y, groups)):
            inner_seed = int(cfg.seed) * 1000 + fold_idx
            inner = GroupShuffleSplit(
                n_splits=1,
                test_size=val_frac,
                random_state=inner_seed,
            )
            tr_rel, va_rel = next(inner.split(trainval_idx, y[trainval_idx], groups[trainval_idx]))
            train_idx = trainval_idx[tr_rel]
            val_idx = trainval_idx[va_rel]

            train_v = slice_views(views, train_idx)
            val_v = slice_views(views, val_idx)
            test_v = slice_views(views, test_idx)

            _log_fold_balance(fold_idx, train_v, val_v, test_v)

            fold_dir = cv_root / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_results = results_root / f"fold_{fold_idx}"

            log.info(
                "[train-cnn-cv] fold %d/%d  train=%d  val=%d  test=%d  inner_seed=%d",
                fold_idx,
                n_splits - 1,
                len(train_v.labels),
                len(val_v.labels),
                len(test_v.labels),
                inner_seed,
            )

            with mlflow.start_run(run_name=f"fold-{fold_idx}", nested=True):
                metrics = _run_keras_fold(
                    cfg=cfg,
                    train_v=train_v,
                    val_v=val_v,
                    test_v=test_v,
                    ckpt_path=fold_dir / "cnn_dualview.keras",
                    cal_path=fold_dir / "cnn_calibrator.joblib",
                    results_dir=fold_results,
                    history_dir=fold_results,
                    log_artifacts=True,
                )
            metrics["fold"] = fold_idx
            fold_rows.append(metrics)

        _aggregate_cv(fold_rows, cv_root)
        return float(np.mean([m["test_roc_auc"] for m in fold_rows]))


def _log_fold_balance(
    fold_idx: int, train_v: ViewArrays, val_v: ViewArrays, test_v: ViewArrays
) -> None:
    """Log positive/negative counts and class fraction for each split of a fold."""
    for name, v in (("train", train_v), ("val", val_v), ("test", test_v)):
        labels = v.labels.astype(int)
        n = int(len(labels))
        pos = int((labels == 1).sum())
        mlflow.log_metrics(
            {
                f"fold_{fold_idx}_{name}_n": float(n),
                f"fold_{fold_idx}_{name}_pos": float(pos),
                f"fold_{fold_idx}_{name}_pos_frac": float(pos / n) if n else 0.0,
            }
        )


def _aggregate_cv(fold_rows: list[dict], cv_root: Path) -> None:
    """Log mean/std across folds and write a summary table artifact."""
    keys = (
        "test_roc_auc",
        "test_pr_auc",
        "test_f1",
        "test_brier",
        "best_threshold",
        "temperature",
    )
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_rows], dtype=float)
        means[k] = float(np.mean(vals))
        stds[k] = float(np.std(vals))
        mlflow.log_metric(f"cv_{k}_mean", means[k])
        mlflow.log_metric(f"cv_{k}_std", stds[k])

    header = (
        f"{'fold':>4}  {'roc_auc':>8}  {'pr_auc':>8}  {'f1':>6}  "
        f"{'brier':>7}  {'thr':>5}  {'T*':>6}"
    )
    lines = [header, "-" * len(header)]
    for m in fold_rows:
        lines.append(
            f"{m['fold']:>4}  "
            f"{m['test_roc_auc']:>8.4f}  "
            f"{m['test_pr_auc']:>8.4f}  "
            f"{m['test_f1']:>6.3f}  "
            f"{m['test_brier']:>7.4f}  "
            f"{m['best_threshold']:>5.2f}  "
            f"{m['temperature']:>6.3f}"
        )
    lines.append("-" * len(header))
    lines.append(
        f"{'mean':>4}  "
        f"{means['test_roc_auc']:>8.4f}  "
        f"{means['test_pr_auc']:>8.4f}  "
        f"{means['test_f1']:>6.3f}  "
        f"{means['test_brier']:>7.4f}  "
        f"{'-':>5}  "
        f"{means['temperature']:>6.3f}"
    )
    lines.append(
        f"{'std':>4}  "
        f"{stds['test_roc_auc']:>8.4f}  "
        f"{stds['test_pr_auc']:>8.4f}  "
        f"{stds['test_f1']:>6.3f}  "
        f"{stds['test_brier']:>7.4f}  "
        f"{'-':>5}  "
        f"{stds['temperature']:>6.3f}"
    )
    summary_path = cv_root / "cv_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    mlflow.log_artifact(str(summary_path))
    log.info("[train-cnn-cv] %s", summary_path)
    log.info("[train-cnn-cv] summary:\n%s", "\n".join(lines))


def _log_se_attention(model: Any, test_ds: Any, results_dir: Path) -> None:
    """Log mean SE excitation weights per channel for each SE block.

    Branch-3 diagnostic — verifies the channel-attention layers are doing
    something non-trivial post-training. Saves a per-block JSON with the
    full per-channel weight vector + summary scalars (mean / std / min /
    max) to MLflow, and logs the summary scalars as run metrics.
    """
    import tensorflow as tf

    se_layers = [l for l in model.layers if l.name.endswith("_se_excite")]
    if not se_layers:
        return

    introspect = tf.keras.Model(
        inputs=model.inputs,
        outputs=[l.output for l in se_layers],
    )
    outputs = introspect.predict(test_ds, verbose=0)
    if not isinstance(outputs, list):
        outputs = [outputs]

    summary: dict[str, dict] = {}
    for layer, weights in zip(se_layers, outputs, strict=True):
        # weights shape: (N, C) — sigmoid activations per sample, per channel
        per_channel = np.asarray(weights).mean(axis=0)
        block_name = layer.name.replace("_se_excite", "")
        summary[block_name] = {
            "per_channel_mean": per_channel.tolist(),
            "mean": float(per_channel.mean()),
            "std": float(per_channel.std()),
            "min": float(per_channel.min()),
            "max": float(per_channel.max()),
            "n_channels": int(per_channel.size),
        }
        mlflow.log_metrics(
            {
                f"se_{block_name}_mean": summary[block_name]["mean"],
                f"se_{block_name}_std": summary[block_name]["std"],
                f"se_{block_name}_min": summary[block_name]["min"],
                f"se_{block_name}_max": summary[block_name]["max"],
            }
        )

    se_path = results_dir / "se_attention_weights.json"
    se_path.write_text(json.dumps(summary, indent=2))
    mlflow.log_artifact(str(se_path))


def _run_keras_fold(
    *,
    cfg: DictConfig,
    train_v: ViewArrays,
    val_v: ViewArrays,
    test_v: ViewArrays,
    ckpt_path: Path,
    cal_path: Path,
    results_dir: Path,
    history_dir: Path,
    log_artifacts: bool,
) -> dict[str, float]:
    """Train one CNN on a single (train, val, test) split.

    Caller is responsible for opening the surrounding MLflow run. Returns a
    dict of test metrics: roc_auc, pr_auc, f1, brier (calibrated), best_threshold.
    """
    import tensorflow as tf

    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    cal_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    use_aux = bool(getattr(cfg.model, "use_aux_features", False))
    aux_dim = (
        train_v.aux_features.shape[1] if use_aux and train_v.aux_features is not None else None
    )

    # Aux pipeline: median impute → standardise. Fit on train only; reuse
    # for val/test/inference. Without scaling, raw Teff (~5800) and log_period
    # (~1) sit on incompatible scales and the dense head's gradients explode
    # (the `loss=nan` failure we saw in earlier runs).
    aux_pipeline = None
    if use_aux and aux_dim is not None:
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        assert train_v.aux_features is not None
        assert val_v.aux_features is not None
        assert test_v.aux_features is not None
        aux_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        train_v.aux_features = aux_pipeline.fit_transform(train_v.aux_features).astype(np.float32)
        val_v.aux_features = aux_pipeline.transform(val_v.aux_features).astype(np.float32)
        test_v.aux_features = aux_pipeline.transform(test_v.aux_features).astype(np.float32)
        log.info(
            "[train-cnn] aux pipeline fitted — impute medians=%s  scale means=%s",
            aux_pipeline.named_steps["impute"].statistics_.tolist(),
            aux_pipeline.named_steps["scale"].mean_.tolist(),
        )

    model = build_cnn_dualview(
        cfg.model,
        global_input_length=train_v.global_views.shape[1],
        local_input_length=train_v.local_views.shape[1],
        aux_input_dim=aux_dim,
    )

    optimizer = instantiate(cfg.train.optimizer)

    if cfg.train.loss.type == "binary_crossentropy":
        loss = tf.keras.losses.BinaryCrossentropy()
    elif cfg.train.loss.type == "focal":
        loss = binary_focal_loss(
            gamma=float(cfg.train.loss.focal_gamma),
            alpha=float(cfg.train.loss.focal_alpha),
        )
    else:
        raise ValueError(f"unknown loss: {cfg.train.loss.type}")

    metrics_map = {
        "accuracy": "accuracy",
        "auc": tf.keras.metrics.AUC(name="auc"),
        "precision": tf.keras.metrics.Precision(name="precision"),
        "recall": tf.keras.metrics.Recall(name="recall"),
    }
    metrics = [metrics_map[m] for m in cfg.train.metrics if m in metrics_map]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary(print_fn=log.info)

    train_ds = LightcurveDataset(
        train_v,
        batch_size=int(cfg.train.batch_size),
        shuffle=True,
        augment=bool(cfg.preprocess.augmentation.enabled),
        time_shift_frac=float(cfg.preprocess.augmentation.time_shift_frac),
        noise_std=float(cfg.preprocess.augmentation.noise_std),
        scale_range=float(cfg.preprocess.augmentation.get("scale_range", 0.0)),
        mask_prob=float(cfg.preprocess.augmentation.get("mask_prob", 0.0)),
        use_aux=use_aux,
        seed=int(cfg.seed),
    ).to_tf_dataset()
    val_ds = LightcurveDataset(
        val_v,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        augment=False,
        use_aux=use_aux,
        seed=int(cfg.seed),
    ).to_tf_dataset()
    test_ds = LightcurveDataset(
        test_v,
        batch_size=int(cfg.train.batch_size),
        shuffle=False,
        augment=False,
        use_aux=use_aux,
        seed=int(cfg.seed),
    ).to_tf_dataset()

    # Class weights from training labels (auto by default).
    # Focal loss already rebalances via alpha; stacking sklearn class_weight on
    # top double-counts the minority class and can push the model into a
    # pathological regime. Skip class_weight whenever focal loss is active.
    class_weight = None
    if cfg.train.loss.type == "focal":
        log.info(
            "[train-cnn] focal loss active — skipping class_weight "
            "(rebalancing via focal alpha=%.3f instead)",
            float(cfg.train.loss.focal_alpha),
        )
    elif str(cfg.train.class_weight) == "auto":
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.array([0, 1])
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=train_v.labels.astype(int),
        )
        class_weight = dict(zip(classes.tolist(), weights.tolist(), strict=False))
        log.info("[train-cnn] class_weight=%s", class_weight)

    callbacks = keras_callbacks(cfg.train, ckpt_path)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(cfg.train.epochs),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=2,
    )
    log_history(history.history, history_dir)

    test_score = model.predict(test_ds).squeeze()
    test_y = test_v.labels.astype(int)

    # --- Optimal threshold sweep on validation set -------------------
    val_score = model.predict(val_ds).squeeze()
    val_y = val_v.labels.astype(int)
    thresholds = np.arange(0.05, 0.96, 0.01)
    f1s = [f1_score(val_y, (val_score >= t).astype(int), zero_division=0) for t in thresholds]
    best_threshold = float(thresholds[int(np.argmax(f1s))])
    mlflow.log_metric("best_threshold", best_threshold)
    log.info("[train-cnn] optimal threshold (val F1): %.2f", best_threshold)

    # --- Temperature scaling (Guo 2017; ExoNet 2026) ----------------
    # Rank-preserving post-hoc calibration: ROC-AUC unchanged, Brier improves.
    from exoplanet_hunter.training.calibration import TemperatureScaler

    calibrator = TemperatureScaler.from_validation(val_score, val_y)
    test_score_cal = calibrator.predict(test_score)
    T_star = float(calibrator.T)
    mlflow.log_metric("temperature_T_star", T_star)
    log.info("[train-cnn] temperature scaling — T* = %.4f", T_star)

    # Persist alongside model. Keep "calibrator" key (sklearn-shaped) so
    # scripts/score_target.py:129 keeps working without changes.
    joblib.dump(
        {
            "calibrator": calibrator,
            "temperature": T_star,
            "threshold": best_threshold,
            "aux_pipeline": aux_pipeline,
            "aux_dim": aux_dim,
        },
        cal_path,
    )

    if log_artifacts:
        log_classification_artifacts(
            test_y,
            test_score_cal,
            threshold=best_threshold,
            out_dir=results_dir,
        )
        mlflow.log_artifact(str(ckpt_path))
        mlflow.log_artifact(str(cal_path))
        _log_se_attention(model, test_ds, results_dir)

    # Compute return metrics. log_classification_artifacts already logged
    # test_roc_auc/test_pr_auc/test_f1/test_brier on the active run, but we
    # also recompute for the dict that the CV aggregator consumes — keeps the
    # aggregator independent of MLflow's metric-history API.
    test_roc_auc = float(roc_auc_score(test_y, test_score_cal))
    test_pr_auc = float(average_precision_score(test_y, test_score_cal))
    test_brier = float(brier_score_loss(test_y, test_score_cal))
    test_f1 = float(
        f1_score(test_y, (test_score_cal >= best_threshold).astype(int), zero_division=0)
    )
    return {
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
        "test_f1": test_f1,
        "test_brier": test_brier,
        "best_threshold": best_threshold,
        "temperature": T_star,
    }


if __name__ == "__main__":
    main()
