"""Box Least Squares (BLS) period search.

Standard transit-finding algorithm: fits a flat-bottomed "box" dip at every
trial period and reports the strongest signal. Fast and robust, but assumes
the transit is rectangular — for U-shaped transits with limb-darkening
ingress/egress, see `tls.py` (TLS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import lightkurve as lk


@dataclass(frozen=True)
class PeriodSearchResult:
    period: float
    t0: float
    duration: float
    power: float
    snr: float


def bls_period_search(
    lc: "lk.LightCurve",
    *,
    period_min: float = 0.5,
    period_max: float = 15.0,
    duration_grid: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
) -> PeriodSearchResult:
    """Run BLS over a period range and return the strongest peak."""
    from astropy.timeseries import BoxLeastSquares

    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    mask = np.isfinite(time) & np.isfinite(flux)
    bls = BoxLeastSquares(time[mask], flux[mask])
    result = bls.autopower(
        list(duration_grid),
        minimum_period=period_min,
        maximum_period=period_max,
    )
    best = int(np.argmax(result.power))
    period = float(result.period[best])
    t0     = float(result.transit_time[best])
    dur    = float(result.duration[best])
    power  = float(result.power[best])

    # Crude SNR estimate: peak power / median power.
    snr = power / float(np.median(result.power) + 1e-12)
    return PeriodSearchResult(period=period, t0=t0, duration=dur, power=power, snr=snr)
