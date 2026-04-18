"""Train a model — Hydra entry point.

Usage:

    python scripts/train_model.py model=random_forest data=small
    python scripts/train_model.py model=cnn_dualview  data=small
    python scripts/train_model.py model=cnn_dualview_stellar data=default

Each invocation creates one MLflow run under the configured tracking URI.
The actual training logic lives in `exoplanet_hunter.training.train.run` —
this script is just the Hydra wrapper with the correct relative `config_path`.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from exoplanet_hunter.training.train import run


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    return run(cfg)


if __name__ == "__main__":
    main()
