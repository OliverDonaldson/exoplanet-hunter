"""Centroid-shift features for false-positive vetting.

A genuine transit is a small dip with no measurable shift in the photo-centre
of the target pixel. A *blended eclipsing binary* — a deep dip on a faint
background star that contaminates the aperture — produces a clear centroid
shift during the dip. This module computes simple statistics on the SPOC
centroid columns to flag that signature.

This is scaffolded — left as a stretch goal for Oliver to flesh out alongside
real centroid data inspection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import lightkurve as lk


def extract_centroid_features(
    lc: lk.LightCurve,
    period: float,
    t0: float,
    duration: float,
) -> dict[str, float]:
    """Compute centroid-shift statistics during transit vs out-of-transit.

    Returns a dict with at least:
      - centroid_shift_x : in-transit median minus out-of-transit median (x)
      - centroid_shift_y : same for y
      - centroid_snr     : magnitude of the shift divided by its scatter

    NOTE — TODO(Oliver): SPOC products carry centroid columns under various
    names depending on data product version. Inspect `lc.columns` and
    pick the right pair (commonly `mom_centr1`, `mom_centr2`).
    """
    cx_col = next((c for c in ("mom_centr1", "centroid_col") if c in lc.columns), None)
    cy_col = next((c for c in ("mom_centr2", "centroid_row") if c in lc.columns), None)
    if cx_col is None or cy_col is None:
        return {
            "centroid_shift_x": float("nan"),
            "centroid_shift_y": float("nan"),
            "centroid_snr": float("nan"),
        }

    folded = lc.fold(period=period, epoch_time=t0)
    phase = np.asarray(folded.time.value, dtype=float)
    cx = np.asarray(folded[cx_col].value, dtype=float)
    cy = np.asarray(folded[cy_col].value, dtype=float)

    half = (duration / period) / 2.0
    in_transit = (np.abs(phase) < half) & np.isfinite(cx) & np.isfinite(cy)
    out_transit = (np.abs(phase) > 3 * half) & np.isfinite(cx) & np.isfinite(cy)

    if not in_transit.any() or not out_transit.any():
        return {
            "centroid_shift_x": float("nan"),
            "centroid_shift_y": float("nan"),
            "centroid_snr": float("nan"),
        }

    dx = float(np.median(cx[in_transit]) - np.median(cx[out_transit]))
    dy = float(np.median(cy[in_transit]) - np.median(cy[out_transit]))
    sigma = float(np.std(cx[out_transit]) + np.std(cy[out_transit]) + 1e-10)
    return {
        "centroid_shift_x": dx,
        "centroid_shift_y": dy,
        "centroid_snr": (dx * dx + dy * dy) ** 0.5 / sigma,
    }
