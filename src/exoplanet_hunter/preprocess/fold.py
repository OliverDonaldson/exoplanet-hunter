"""Phase-folding and binning for transit search.

Once a (period, t0) is known (or estimated by BLS/TLS), we phase-fold the
light curve so all transits overlay on top of each other. This amplifies the
transit signal and makes it visible to a human and a model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import lightkurve as lk


def fold_and_bin(
    lc: lk.LightCurve,
    period: float,
    t0: float,
    n_bins: int,
    phase_min: float = -0.5,
    phase_max: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase-fold and bin a light curve.

    Returns
    -------
    bin_centers : phase values [phase_min, phase_max] of bin centres.
    binned_flux : median flux in each bin (NaN if empty).
    """
    folded = lc.fold(period=period, epoch_time=t0)
    phase = np.asarray(folded.time.value, dtype=float)
    flux = np.asarray(folded.flux.value, dtype=float)

    # Restrict to the requested phase window and drop NaNs.
    mask = (phase >= phase_min) & (phase <= phase_max) & np.isfinite(flux)
    phase = phase[mask]
    flux = flux[mask]

    edges = np.linspace(phase_min, phase_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # np.digitize is 1-indexed; convert to 0-indexed.
    idx = np.digitize(phase, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    binned = np.full(n_bins, np.nan, dtype=float)
    for b in range(n_bins):
        sel = flux[idx == b]
        if sel.size > 0:
            binned[b] = np.median(sel)

    # Linear interpolate over empty bins to keep the array dense.
    if np.isnan(binned).any():
        valid = ~np.isnan(binned)
        if valid.any():
            binned = np.interp(centers, centers[valid], binned[valid])

    return centers, binned
