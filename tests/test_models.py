"""Smoke tests for model construction and a single training step.

We don't aim for accuracy here — just that the model builds with the right
shapes and that one optimisation step runs cleanly. Real performance is
exercised on real data via the training scripts.
"""

from __future__ import annotations

import numpy as np
import pytest
from omegaconf import OmegaConf

tf = pytest.importorskip("tensorflow")


@pytest.fixture
def cnn_cfg():
    return OmegaConf.create(
        {
            "name": "cnn_dualview",
            "type": "keras",
            "use_aux_features": False,
            "global_view": {
                "conv_blocks": [8, 16],
                "conv_per_block": 1,
                "kernel_size": 5,
                "pool_size": 2,
            },
            "local_view": {
                "conv_blocks": [8],
                "conv_per_block": 1,
                "kernel_size": 5,
                "pool_size": 2,
            },
            "head": {"fc_units": [32, 32], "dropout": 0.2},
            "output": {"activation": "sigmoid", "units": 1},
        }
    )


def test_cnn_dualview_builds_and_runs(cnn_cfg) -> None:
    from exoplanet_hunter.models import build_cnn_dualview

    model = build_cnn_dualview(
        cnn_cfg,
        global_input_length=2001,
        local_input_length=201,
        aux_input_dim=None,
    )
    g = np.random.default_rng(0).normal(size=(4, 2001, 1)).astype(np.float32)
    l = np.random.default_rng(1).normal(size=(4, 201, 1)).astype(np.float32)
    out = model({"global_view": g, "local_view": l}).numpy()
    assert out.shape == (4, 1)
    assert ((out >= 0.0) & (out <= 1.0)).all()


def test_cnn_dualview_trains_one_step(cnn_cfg) -> None:
    from exoplanet_hunter.models import build_cnn_dualview

    model = build_cnn_dualview(
        cnn_cfg,
        global_input_length=2001,
        local_input_length=201,
        aux_input_dim=None,
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    g = np.random.default_rng(0).normal(size=(8, 2001, 1)).astype(np.float32)
    l = np.random.default_rng(1).normal(size=(8, 201, 1)).astype(np.float32)
    y = np.random.default_rng(2).integers(0, 2, size=(8,)).astype(np.float32)
    history = model.fit(
        {"global_view": g, "local_view": l},
        y,
        epochs=1,
        batch_size=4,
        verbose=0,
    )
    assert "loss" in history.history


def test_cnn_with_aux_features(cnn_cfg) -> None:
    from exoplanet_hunter.models import build_cnn_dualview

    cfg = OmegaConf.create(OmegaConf.to_container(cnn_cfg, resolve=True))
    cfg.use_aux_features = True
    model = build_cnn_dualview(
        cfg,
        global_input_length=2001,
        local_input_length=201,
        aux_input_dim=4,
    )
    g = np.random.default_rng(0).normal(size=(2, 2001, 1)).astype(np.float32)
    l = np.random.default_rng(1).normal(size=(2, 201, 1)).astype(np.float32)
    a = np.random.default_rng(3).normal(size=(2, 4)).astype(np.float32)
    out = model({"global_view": g, "local_view": l, "aux_features": a}).numpy()
    assert out.shape == (2, 1)
