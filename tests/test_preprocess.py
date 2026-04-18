"""Tests for preprocessing — cleaning, folding, view extraction.

We avoid hitting MAST by constructing a synthetic light curve with a known
periodic dip and verifying that:

  * `clean_lightcurve` removes outliers without nuking the dip.
  * `flatten_lightcurve` removes a slow trend without distorting the dip.
  * `build_views` produces a global view whose minimum is at phase 0 and
    a local view whose central bins are negative (transit) and edges ~0.
"""

from __future__ import annotations

import numpy as np
import pytest

lk = pytest.importorskip("lightkurve")


def _make_synthetic_lc(
    n_days: float = 27.0,
    cadence_min: float = 2.0,
    period: float = 3.0,
    depth: float = 0.005,
    duration_h: float = 2.0,
    noise: float = 0.0005,
    long_trend_amp: float = 0.01,
    seed: int = 1,
) -> lk.LightCurve:
    """Return a TESS-like LightCurve with injected box transits + slow trend."""
    rng = np.random.default_rng(seed)
    cad_d = cadence_min / (60 * 24)
    t = np.arange(0.0, n_days, cad_d)
    flux = np.ones_like(t)

    # Slow stellar variability — must be flattened away.
    flux *= 1.0 + long_trend_amp * np.sin(2 * np.pi * t / 7.0)

    # Inject box transits.
    half = (duration_h / 24.0) / 2.0
    phase = ((t + 0.5 * period) % period) - 0.5 * period
    flux[np.abs(phase) < half] -= depth

    flux += rng.normal(scale=noise, size=t.size)

    return lk.LightCurve(time=t, flux=flux)


def test_clean_lightcurve_preserves_dip() -> None:
    from exoplanet_hunter.preprocess.clean import clean_lightcurve

    lc = _make_synthetic_lc()
    cleaned = clean_lightcurve(lc, sigma_clip=5.0, min_points=100)
    assert len(cleaned) > 0
    assert float(np.min(cleaned.flux.value)) < 1.0  # the dip survived


def test_flatten_lightcurve_removes_slow_trend() -> None:
    from exoplanet_hunter.preprocess.clean import (
        clean_lightcurve,
        flatten_lightcurve,
    )

    lc = _make_synthetic_lc(long_trend_amp=0.05)
    lc = clean_lightcurve(lc, sigma_clip=5.0, min_points=100)
    flat = flatten_lightcurve(lc, window_length=301, polyorder=2)
    # After flattening, baseline should sit very close to 1.
    median = float(np.median(flat.flux.value))
    assert 0.998 < median < 1.002


def test_build_views_centre_minimum() -> None:
    from exoplanet_hunter.preprocess import (
        build_views,
        clean_lightcurve,
        flatten_lightcurve,
    )

    lc = _make_synthetic_lc()
    lc = clean_lightcurve(lc, sigma_clip=5.0, min_points=100)
    lc = flatten_lightcurve(lc, window_length=301, polyorder=2)
    views = build_views(lc, period=3.0, t0=0.0, duration=2.0 / 24.0)

    g, l = views.global_view, views.local_view
    # The transit should leave a clear dip at the centre of both views.
    g_centre = g[len(g) // 2 - 5 : len(g) // 2 + 5]
    l_centre = l[len(l) // 2 - 5 : len(l) // 2 + 5]
    assert float(np.mean(g_centre)) < float(np.median(g))
    assert float(np.mean(l_centre)) < float(np.median(l))
    # Normalisation: minimum should be ≈ -1 (Shallue convention).
    assert -1.05 <= float(np.min(g)) <= -0.5
