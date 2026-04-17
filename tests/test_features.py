"""Tests for the hand-crafted feature extractor."""

from __future__ import annotations

import numpy as np

from exoplanet_hunter.features import FEATURE_NAMES, extract_features


def test_extract_features_shape(synthetic_view: np.ndarray) -> None:
    feats = extract_features(synthetic_view)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert feats.dtype == np.float32
    assert np.all(np.isfinite(feats))


def test_transit_has_higher_depth_than_quiet(
    synthetic_view: np.ndarray, synthetic_quiet_view: np.ndarray
) -> None:
    feats_transit = extract_features(synthetic_view)
    feats_quiet   = extract_features(synthetic_quiet_view)
    depth_idx = FEATURE_NAMES.index("depth")
    snr_idx   = FEATURE_NAMES.index("depth_snr")
    assert feats_transit[depth_idx] > feats_quiet[depth_idx]
    assert feats_transit[snr_idx]   > feats_quiet[snr_idx]
