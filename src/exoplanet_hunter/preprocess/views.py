"""Global + local view extraction (Shallue & Vanderburg 2018).

For each (light curve, period, t0, duration) we produce two arrays:

  * **global view** — full phase-folded light curve at low resolution
    (default 2001 bins). Captures the planet's overall orbital phase
    relative to the star, including any secondary eclipse signature.
  * **local view** — zoomed-in window around phase 0 spanning
    ±N transit durations (default 3). Captures the transit shape at
    high resolution.

Both views are median-normalised so flux=0 corresponds to the out-of-transit
baseline and the transit dip is negative.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from exoplanet_hunter.preprocess.fold import fold_and_bin

if TYPE_CHECKING:
    import lightkurve as lk


@dataclass(frozen=True)
class Views:
    global_view: np.ndarray
    local_view: np.ndarray


def _normalise(view: np.ndarray) -> np.ndarray:
    """Median-subtract and depth-divide.

    Subtracting the median puts the baseline at 0; dividing by |min - median|
    rescales the deepest dip to -1, regardless of absolute transit depth.
    This makes the model see *transit shape*, not *transit magnitude*.

    NaN handling: uses nan-aware median/min so that empty bins (NaN, from
    long data gaps after fold-and-bin) don't poison the whole view. Any
    remaining NaNs after normalisation are filled with 0 (the baseline) —
    the model treats them as "no flux deviation here", which is the right
    inductive bias for a missing observation.

    Raises ValueError if the entire input is NaN — this signals a
    fundamentally bad target that build_dataset.py should skip and count as
    preprocess_error (rather than silently shipping an all-NaN row that
    poisons gradients during training).
    """
    if not np.isfinite(view).any():
        raise ValueError("view is entirely NaN — no usable cadences after folding")
    med = float(np.nanmedian(view))
    centred = view - med
    depth = float(np.abs(np.nanmin(centred)))
    if depth < 1.0e-8:
        return np.nan_to_num(centred, nan=0.0)
    return np.nan_to_num(centred / depth, nan=0.0)


def build_views(
    lc: lk.LightCurve,
    period: float,
    t0: float,
    duration: float,
    *,
    global_bins: int = 2001,
    local_bins: int = 201,
    local_durations: float = 3.0,
) -> Views:
    """Build the global + local views for a single (lc, period, t0, duration).

    Parameters
    ----------
    lc              : flattened, cleaned light curve.
    period          : orbital period [days].
    t0              : transit midpoint epoch (BJD - 2457000) [days].
    duration        : full transit duration [days] (NOT hours).
    global_bins     : number of bins spanning the full phase.
    local_bins      : number of bins spanning ±local_durations of the transit.
    local_durations : half-width of the local window in transit durations.
    """
    if not np.isfinite(period) or period <= 0:
        raise ValueError(f"invalid period: {period}")
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid duration: {duration}")

    # ----- global ---------------------------------------------------------
    _, gview = fold_and_bin(
        lc,
        period=period,
        t0=t0,
        n_bins=global_bins,
        phase_min=-0.5,
        phase_max=0.5,
    )

    # ----- local ----------------------------------------------------------
    half = local_durations * duration / period  # half-window in phase units
    half = float(min(max(half, 1e-3), 0.5))  # clamp to a sane range

    _, lview = fold_and_bin(
        lc,
        period=period,
        t0=t0,
        n_bins=local_bins,
        phase_min=-half,
        phase_max=+half,
    )

    return Views(
        global_view=_normalise(gview).astype(np.float32),
        local_view=_normalise(lview).astype(np.float32),
    )
