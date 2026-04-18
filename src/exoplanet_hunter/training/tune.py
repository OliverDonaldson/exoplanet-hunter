"""Optuna hyperparameter search for the dual-view CNN.

Wraps `train.main` so each Optuna trial is a Hydra-style run. Each trial logs
to MLflow as a nested run inside a parent "tune" run, so you can compare
trials in the MLflow UI.

Usage:

    python -m exoplanet_hunter.training.tune --config-name=config train=tune
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import mlflow
import optuna
from omegaconf import DictConfig, OmegaConf

from exoplanet_hunter.training.mlflow_utils import setup_mlflow
from exoplanet_hunter.utils import get_logger, set_global_seed

log = get_logger(__name__)


def _suggest(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Translate a YAML search-space entry into an Optuna `suggest_*` call."""
    kind = spec["type"]
    if kind == "loguniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
    if kind == "uniform":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]))
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    raise ValueError(f"unknown search-space type: {kind}")


@hydra.main(version_base="1.3", config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    if cfg.train.name != "tune":
        raise SystemExit("tune entry point requires train=tune")

    set_global_seed(int(cfg.seed))
    setup_mlflow(cfg)

    pruner = hydra.utils.instantiate(cfg.train.optuna.pruner)
    study = optuna.create_study(
        direction=str(cfg.train.optuna.direction),
        pruner=pruner,
        study_name=f"{cfg.project_name}-{cfg.model.name}-{cfg.data.name}",
    )

    parent_run = mlflow.start_run(run_name=f"tune-{cfg.model.name}-{cfg.data.name}")

    def objective(trial: optuna.Trial) -> float:
        # Apply suggested overrides to a copy of the cfg.
        trial_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
        for key, spec in cfg.train.search_space.items():
            value = _suggest(trial, key, spec)
            OmegaConf.update(trial_cfg, key, value, force_add=True)

        log.info("[tune] trial %d → %s", trial.number, trial.params)
        with mlflow.start_run(nested=True, run_name=f"trial-{trial.number}"):
            mlflow.log_params({f"trial.{k}": v for k, v in trial.params.items()})
            from exoplanet_hunter.training.train import main as train_main

            score = float(train_main(trial_cfg))
            mlflow.log_metric(str(cfg.train.optuna.metric), score)
            return score

    try:
        study.optimize(
            objective,
            n_trials=int(cfg.train.optuna.n_trials),
            timeout=int(cfg.train.optuna.timeout) if cfg.train.optuna.timeout else None,
        )
        log.info("[tune] best score = %.4f", study.best_value)
        log.info("[tune] best params = %s", study.best_params)
        for k, v in study.best_params.items():
            mlflow.log_param(f"best.{k}", v)
        mlflow.log_metric("best_value", float(study.best_value))

        # Dump the study for later inspection.
        out_dir = Path("results/tune")
        out_dir.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_parquet(out_dir / "trials.parquet", index=False)
        return float(study.best_value)
    finally:
        mlflow.end_run()
        if parent_run:
            mlflow.end_run()


if __name__ == "__main__":
    main()
