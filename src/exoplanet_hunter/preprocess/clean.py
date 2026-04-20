"""Light-curve cleaning + detrending.

The two operations here happen on raw light curves before any phase-folding:

  * **clean_lightcurve** — drop NaNs and sigma-clip bright outliers (cosmic
    rays, momentum-dump artefacts, jumps). One-sided on the upper tail so
    deep transit dips aren't clipped as negative outliers.
  * **flatten_lightcurve** — fit and divide out long-term stellar variability
    via a Savitzky-Golay filter, with in-transit cadences masked out of the
    fit so the filter doesn't learn to interpolate through the very dip we
    want to preserve.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import lightkurve as lk


def clean_lightcurve(
    lc: lk.LightCurve,
    sigma_clip: float = 5.0,
    min_points: int = 1000,
) -> lk.LightCurve:
    """Drop NaNs and sigma-clip upper outliers only.

    Two-sided sigma clipping (the lightkurve default) would treat deep
    transit dips as negative outliers and delete them. We clip only the
    *upper* tail (cosmic rays, scattered-light spikes, pointing jumps);
    anything real on the lower tail is kept and handled by the flattening
    + masking step downstream.

    Parameters
    ----------
    lc          : input lightkurve LightCurve.
    sigma_clip  : reject points more than this many sigma above the rolling median.
    min_points  : raise ValueError if fewer good points remain.
    """
    cleaned = lc.remove_nans().remove_outliers(sigma_upper=sigma_clip, sigma_lower=np.inf)
    if len(cleaned) < min_points:
        raise ValueError(
            f"only {len(cleaned)} good cadences after cleaning (required ≥{min_points})"
        )
    return cleaned


def _transit_mask(
    time: np.ndarray,
    period: float,
    t0: float,
    duration: float,
    pad: float = 1.0,
) -> np.ndarray:
    """Boolean mask: True inside a transit window, False in baseline.

    `duration` is the full transit duration in *days*. `pad` widens the
    in-transit window as a multiple of duration so ingress/egress tails are
    fully excluded from the out-of-transit fit.
    """
    half = 0.5 * pad * duration
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    return np.abs(phase) <= half


def flatten_lightcurve(
    lc: lk.LightCurve,
    window_length: int = 301,
    polyorder: int = 2,
    *,
    period: float | None = None,
    t0: float | None = None,
    duration: float | None = None,
    mask_pad: float = 1.0,
) -> lk.LightCurve:
    """Remove long-term stellar variability with a Savitzky-Golay filter.

    `window_length` is in cadences, not days. For 2-min cadence (30 / hour),
    window 301 ≈ 10 hours — comfortably wider than typical short-period
    transits (1-6 h) so the transit dip is preserved.

    If `period`, `t0`, and `duration` are supplied, the in-transit cadences
    are masked out of the fit so the spline doesn't flatten the dip itself
    (the classic "filter learns the transit" failure mode). This requires
    knowing the ephemeris up-front — typically from the TCE / exoplanet
    archive catalog. Without an ephemeris, falls back to unmasked flattening.

    Returns a new LightCurve with the trend divided out.
    """
    mask = None
    if period is not None and t0 is not None and duration is not None:
        time = np.asarray(lc.time.value, dtype=float)
        # lightkurve convention: `mask=True` means "exclude from the fit".
        mask = _transit_mask(time, period=period, t0=t0, duration=duration, pad=mask_pad)

    return lc.flatten(window_length=window_length, polyorder=polyorder, mask=mask)
