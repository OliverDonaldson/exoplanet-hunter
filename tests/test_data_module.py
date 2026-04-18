"""Tests for the tf.data dataset builder."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tensorflow")


def test_load_views_split_and_dataset(synthetic_views_dataset) -> None:
    from exoplanet_hunter.training.data_module import (
        LightcurveDataset,
        ViewArrays,
        train_val_test_split,
    )

    g, l, y = synthetic_views_dataset
    views = ViewArrays(
        global_views=g,
        local_views=l,
        labels=y,
        tic_ids=np.arange(len(y), dtype=np.int64),
        aux_features=None,
    )

    train, val, test = train_val_test_split(views, train=0.6, val=0.2, test=0.2, seed=0)
    assert len(train.labels) + len(val.labels) + len(test.labels) == len(y)

    ds = LightcurveDataset(train, batch_size=4, augment=False).to_tf_dataset()
    batch = next(iter(ds))
    inputs, labels = batch
    assert "global_view" in inputs
    assert "local_view" in inputs
    assert labels.shape[0] == 4
