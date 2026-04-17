"""Hydra-driven training entry point.

Usage (from repo root):

    python -m exoplanet_hunter.training.train                      # defaults
    python -m exoplanet_hunter.training.train model=random_forest  # swap model
    python -m exoplanet_hunter.training.train data=small           # swap dataset
    python -m exoplanet_hunter.training.train model=cnn_dualview train.epochs=5

Logs every run to MLflow. Saves best checkpoint to `models/`.
"""

from __future__ import annotations

from pathlib import Path

import hydra
import joblib
import mlflow
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from exoplanet_hunter.features import extract_features
from exoplanet_hunter.models import (
    binary_focal_loss,
    build_cnn_dualview,
    build_random_forest,
)
from exoplanet_hunter.training.data_module import (
    LightcurveDataset,
    load_views,
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


@hydra.main(version_base="1.3", config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    set_global_seed(int(cfg.seed))
    paths = ProjectPaths.from_cfg(cfg)
    setup_mlflow(cfg)

    views_path = paths.data_processed / "views.npz"
    if not views_path.exists():
        raise FileNotFoundError(
            f"{views_path} missing. Run `python scripts/build_dataset.py data={cfg.data.name}` first."
        )

    views = load_views(views_path)
    train_v, val_v, test_v = train_val_test_split(
        views,
        train=float(cfg.data.split.train),
        val=float(cfg.data.split.val),
        test=float(cfg.data.split.test),
        seed=int(cfg.seed),
    )
    log.info(
        "[train] split sizes — train=%d  val=%d  test=%d",
        len(train_v.labels), len(val_v.labels), len(test_v.labels),
    )

    if cfg.model.type == "sklearn":
        return _train_sklearn(cfg, paths, train_v, val_v, test_v)
    if cfg.model.type == "keras":
        return _train_keras(cfg, paths, train_v, val_v, test_v)
    raise ValueError(f"unknown model type: {cfg.model.type}")


# ---------------------------------------------------------------- sklearn ----


def _train_sklearn(cfg, paths, train_v, val_v, test_v) -> float:                # noqa: ANN001
    import shap
    from sklearn.metrics import classification_report

    # Concatenate train+val for k-fold CV; hold test out.
    train_views_all = np.concatenate([train_v.global_views, val_v.global_views])
    train_y_all = np.concatenate([train_v.labels, val_v.labels])
    test_views = test_v.global_views
    test_y = test_v.labels.astype(int)

    log.info("[train-rf] extracting handcrafted features ...")
    X_trainval = np.array([extract_features(v) for v in train_views_all])
    X_test     = np.array([extract_features(v) for v in test_views])

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
        mlflow.log_metric("cv_auc_std",  float(np.std(cv_aucs)))
        log.info("[train-rf] CV AUC %.4f ± %.4f", np.mean(cv_aucs), np.std(cv_aucs))

        # Refit on full train+val and evaluate on test.
        pipeline.fit(X_trainval, train_y_all)
        test_score = pipeline.predict_proba(X_test)[:, 1]

        report = classification_report(test_y, (test_score >= 0.5).astype(int), zero_division=0)
        log.info("[train-rf] test classification report:\n%s", report)

        log_classification_artifacts(
            test_y, test_score, threshold=0.5, out_dir=paths.results / "rf",
        )

        # SHAP feature importance.
        if bool(cfg.model.feature_importance):
            try:
                explainer = shap.TreeExplainer(pipeline.named_steps["clf"])
                shap_values = explainer.shap_values(pipeline.named_steps["scaler"].transform(X_test))
                if isinstance(shap_values, list):  # binary: list of two arrays
                    shap_values = shap_values[1]
                from exoplanet_hunter.features import FEATURE_NAMES
                import matplotlib.pyplot as plt
                plt.figure()
                shap.summary_plot(
                    shap_values, pipeline.named_steps["scaler"].transform(X_test),
                    feature_names=FEATURE_NAMES, show=False,
                )
                shap_path = paths.results / "rf" / "shap_summary.png"
                plt.tight_layout()
                plt.savefig(shap_path, dpi=120, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(str(shap_path))
            except Exception as exc:                                            # noqa: BLE001
                log.warning("[train-rf] SHAP failed: %s", exc)

        # Save the fitted pipeline.
        artifact = paths.models / "random_forest.joblib"
        joblib.dump(pipeline, artifact)
        mlflow.log_artifact(str(artifact))

        return float(np.mean(cv_aucs))


# ----------------------------------------------------------------- keras -----


def _train_keras(cfg, paths, train_v, val_v, test_v) -> float:                  # noqa: ANN001
    import tensorflow as tf

    use_aux = bool(getattr(cfg.model, "use_aux_features", False))
    aux_dim = (
        train_v.aux_features.shape[1]
        if use_aux and train_v.aux_features is not None
        else None
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
        "accuracy":  "accuracy",
        "auc":       tf.keras.metrics.AUC(name="auc"),
        "precision": tf.keras.metrics.Precision(name="precision"),
        "recall":    tf.keras.metrics.Recall(name="recall"),
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
        flip_prob=float(cfg.preprocess.augmentation.flip_prob),
        use_aux=use_aux,
    ).to_tf_dataset()
    val_ds = LightcurveDataset(
        val_v, batch_size=int(cfg.train.batch_size),
        shuffle=False, augment=False, use_aux=use_aux,
    ).to_tf_dataset()
    test_ds = LightcurveDataset(
        test_v, batch_size=int(cfg.train.batch_size),
        shuffle=False, augment=False, use_aux=use_aux,
    ).to_tf_dataset()

    # Class weights from training labels (auto by default).
    class_weight = None
    if str(cfg.train.class_weight) == "auto":
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.array([0, 1])
        weights = compute_class_weight(
            class_weight="balanced", classes=classes, y=train_v.labels.astype(int),
        )
        class_weight = dict(zip(classes.tolist(), weights.tolist(), strict=False))
        log.info("[train-cnn] class_weight=%s", class_weight)

    ckpt_path = paths.models / "cnn_dualview.keras"

    with mlflow.start_run(run_name=f"cnn-{cfg.data.name}"):
        log_config(cfg)
        callbacks = keras_callbacks(cfg.train, ckpt_path)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=int(cfg.train.epochs),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=2,
        )
        log_history(history.history, paths.results / "cnn")

        # Evaluate on test.
        test_score = model.predict(test_ds).squeeze()
        test_y = test_v.labels.astype(int)
        log_classification_artifacts(
            test_y, test_score, threshold=0.5, out_dir=paths.results / "cnn",
        )

        # Log final model.
        mlflow.log_artifact(str(ckpt_path))
        return float(roc_auc_score(test_y, test_score))


if __name__ == "__main__":
    main()
