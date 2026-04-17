"""Light-curve cleaning + detrending.

The two operations here happen on raw light curves before any phase-folding:

  * **clean_lightcurve** — drop NaNs and sigma-clip outliers (cosmic rays,
    momentum-dump artefacts, jumps).
  * **flatten_lightcurve** — fit and divide out long-term stellar variability
    via a Savitzky-Golay filter. The window must be much wider than the
    transit duration, or the transit itself gets filtered out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lightkurve as lk


def clean_lightcurve(
    lc: "lk.LightCurve",
    sigma_clip: float = 5.0,
    min_points: int = 1000,
) -> "lk.LightCurve":
    """Drop NaNs and sigma-clip outliers.

    Parameters
    ----------
    lc          : input lightkurve LightCurve.
    sigma_clip  : reject points more than this many sigma from the rolling median.
    min_points  : raise ValueError if fewer good points remain.
    """
    cleaned = lc.remove_nans().remove_outliers(sigma=sigma_clip)
    if len(cleaned) < min_points:
        raise ValueError(
            f"only {len(cleaned)} good cadences after cleaning "
            f"(required ≥{min_points})"
        )
    return cleaned


def flatten_lightcurve(
    lc: "lk.LightCurve",
    window_length: int = 301,
    polyorder: int = 2,
) -> "lk.LightCurve":
    """Remove long-term stellar variability with a Savitzky-Golay filter.

    `window_length` is in cadences, not days. For 2-min cadence (30 / hour),
    window 301 ≈ 10 hours — comfortably wider than typical short-period
    transits (1–6 h) so the transit dip is preserved.

    Returns a new LightCurve with the trend divided out (median-normalised).
    """
    return lc.flatten(window_length=window_length, polyorder=polyorder)
