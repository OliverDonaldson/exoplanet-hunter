"""Transit Least Squares (TLS) period search.

Modern improvement on BLS (Hippke & Heller 2019). Fits a *physical* transit
shape with limb darkening rather than a box, which gives:

  * ~17% better detection efficiency on small planets (per the paper).
  * Built-in odd/even transit comparison — a strong false-positive flag for
    eclipsing binaries.

Slower than BLS — use BLS for a first pass on lots of stars, TLS for
candidates that pass BLS triage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from exoplanet_hunter.search.bls import PeriodSearchResult

if TYPE_CHECKING:
    import lightkurve as lk


def tls_period_search(
    lc: lk.LightCurve,
    *,
    period_min: float = 0.5,
    period_max: float = 15.0,
) -> PeriodSearchResult:
    """Run TLS and return the strongest peak.

    NOTE: requires `transitleastsquares` (in environment.yml). The library
    is opinionated about input format — we adapt by passing time/flux arrays.
    """
    import numpy as np
    from transitleastsquares import transitleastsquares

    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    mask = np.isfinite(time) & np.isfinite(flux)

    model = transitleastsquares(time[mask], flux[mask])
    result = model.power(period_min=period_min, period_max=period_max)

    return PeriodSearchResult(
        period=float(result.period),
        t0=float(result.T0),
        duration=float(result.duration),
        power=float(result.SDE),
        snr=float(result.snr),
    )
