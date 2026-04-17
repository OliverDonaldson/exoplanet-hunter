"""Shared pytest fixtures.

The synthetic-transit fixtures here let us test preprocessing + features
without hitting MAST. A planet with a known depth/period is injected into
clean white noise and we assert the recovered features are sensible.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synthetic_view() -> np.ndarray:
    """1D phase-folded view with a clean transit at the centre.

    Length 2001, baseline 0, transit dip -1 over the central 5% of the array.
    Mimics what `preprocess.views.build_views` would produce for a real
    confirmed planet: median-subtracted, depth-divided, transit at phase 0.
    """
    n = 2001
    view = np.zeros(n, dtype=np.float32)
    centre = n // 2
    half = int(0.025 * n)
    view[centre - half : centre + half] = -1.0
    view += np.random.default_rng(42).normal(scale=0.02, size=n).astype(np.float32)
    return view


@pytest.fixture
def synthetic_quiet_view() -> np.ndarray:
    """No-transit phase-folded view: pure noise around 0."""
    return np.random.default_rng(7).normal(scale=0.02, size=2001).astype(np.float32)


@pytest.fixture
def synthetic_views_dataset(
    synthetic_view: np.ndarray, synthetic_quiet_view: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tiny dataset (16 transits + 16 quiet) for end-to-end smoke tests."""
    rng = np.random.default_rng(0)
    g_views = []
    l_views = []
    labels: list[int] = []
    for i in range(16):
        v = synthetic_view + rng.normal(scale=0.005, size=synthetic_view.size).astype(np.float32)
        g_views.append(v)
        l_views.append(_local_window(v))
        labels.append(1)
    for i in range(16):
        v = synthetic_quiet_view + rng.normal(scale=0.005, size=synthetic_quiet_view.size).astype(np.float32)
        g_views.append(v)
        l_views.append(_local_window(v))
        labels.append(0)

    return (
        np.stack(g_views).astype(np.float32),
        np.stack(l_views).astype(np.float32),
        np.asarray(labels, dtype=np.int8),
    )


def _local_window(global_view: np.ndarray, n_local: int = 201) -> np.ndarray:
    n = global_view.size
    half = n_local // 2
    centre = n // 2
    return global_view[centre - half : centre + half + 1].copy()
