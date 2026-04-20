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


def test_no_tic_leakage_across_splits(synthetic_views_dataset) -> None:
    """TIC IDs must be disjoint across train/val/test to prevent data leakage."""
    from exoplanet_hunter.training.data_module import (
        ViewArrays,
        train_val_test_split,
    )

    g, l, y = synthetic_views_dataset

    # Simulate multi-planet system: give several samples the same TIC ID
    tic_ids = np.arange(len(y), dtype=np.int64)
    tic_ids[1] = tic_ids[0]  # same star, two TCEs
    tic_ids[17] = tic_ids[16]

    views = ViewArrays(
        global_views=g,
        local_views=l,
        labels=y,
        tic_ids=tic_ids,
        aux_features=None,
    )

    train, val, test = train_val_test_split(views, train=0.6, val=0.2, test=0.2, seed=0)

    train_tics = set(train.tic_ids.tolist())
    val_tics = set(val.tic_ids.tolist())
    test_tics = set(test.tic_ids.tolist())

    assert train_tics.isdisjoint(val_tics), f"train∩val leak: {train_tics & val_tics}"
    assert train_tics.isdisjoint(test_tics), f"train∩test leak: {train_tics & test_tics}"
    assert val_tics.isdisjoint(test_tics), f"val∩test leak: {val_tics & test_tics}"
