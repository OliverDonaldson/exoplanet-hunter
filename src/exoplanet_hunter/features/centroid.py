"""Centroid-shift features for false-positive vetting.

A genuine transit is a small dip with no measurable shift in the photo-centre
of the target pixel. A *background eclipsing binary* (BEB) — a deep dip on a
faint star inside the photometric aperture — produces a clear centroid
shift during the dip. This module measures that shift after detrending the
raw `MOM_CENTR1/2` columns for the systematics that swamp the signal in raw
data: Kepler's quarterly 90° spacecraft rolls (tens-of-pixels jumps every
~93 days) and the per-quarter thermal/pointing drift.

Pipeline (per Ansdell 2018; Kepler-centroid detrending notes):

  1. Outlier rejection at 5σ MAD (cosmic rays, argabrightening events).
  2. Per-segment median subtraction (Kepler quarters / TESS sectors).
     Segment boundaries are detected from time gaps > 0.5 d, which works
     for both missions and survives `LightCurveCollection.stitch()`.
  3. Rolling-median detrend with a 1-day time-based window (per segment).
  4. Phase-fold the detrended series, mask in/out of transit windows.
  5. SNR per axis = S / (σ_oot_robust / √N_itr); 2D SNR = √(SNR_x² + SNR_y²).
     σ_oot_robust = 1.4826 · MAD of out-of-transit cadences.

Without these corrections, raw MOM_CENTR phase-folds capture the
inter-quarter pixel jumps rather than the intra-transit shift, producing
SNRs of the wrong order of magnitude (~19 arcsec on Kepler instead of the
expected 0.1–1 arcsec).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import median_filter

if TYPE_CHECKING:
    import lightkurve as lk

_MAD_TO_STD = 1.4826
_DEFAULT_GAP_DAYS = 0.5
_DEFAULT_ROLL_DAYS = 1.0
_DEFAULT_OUTLIER_SIGMA = 5.0


def _robust_std(x: np.ndarray) -> float:
    """1.4826 · MAD — robust estimator of σ for a Gaussian sample."""
    finite = np.isfinite(x)
    if finite.sum() < 3:
        return float("nan")
    xf = x[finite]
    return float(_MAD_TO_STD * np.median(np.abs(xf - np.median(xf))))


def _segment_by_time_gaps(
    t: np.ndarray, gap_days: float = _DEFAULT_GAP_DAYS
) -> list[tuple[int, int]]:
    """Return [(lo, hi), …] for each time-contiguous segment.

    Detects Kepler quarter / TESS sector boundaries from time gaps. Survives
    `LightCurveCollection.stitch()` which loses the `.quarter` metadata.
    """
    if t.size < 2:
        return [(0, t.size)]
    gaps = np.where(np.diff(t) > gap_days)[0]
    bounds = [0, *(gaps + 1).tolist(), t.size]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def _detrend_axis(
    t: np.ndarray,
    c: np.ndarray,
    *,
    gap_days: float = _DEFAULT_GAP_DAYS,
    rolling_window_days: float = _DEFAULT_ROLL_DAYS,
    outlier_sigma: float = _DEFAULT_OUTLIER_SIGMA,
) -> np.ndarray:
    """Detrend a single centroid axis (MOM_CENTR1 *or* MOM_CENTR2).

    Steps per segment: 5σ MAD outlier rejection → median subtraction →
    time-based rolling-median detrend. Returns NaN for masked or
    too-short segments — the caller masks these out before in/out
    transit statistics.
    """
    out = np.full_like(c, np.nan, dtype=float)
    for lo, hi in _segment_by_time_gaps(t, gap_days):
        seg_t = t[lo:hi]
        seg_c = c[lo:hi].astype(float).copy()
        finite = np.isfinite(seg_c)
        if finite.sum() < 5:
            continue

        # 1. outlier rejection (5σ MAD)
        seg_med = float(np.median(seg_c[finite]))
        seg_std = _robust_std(seg_c)
        if np.isfinite(seg_std) and seg_std > 0:
            seg_c[np.abs(seg_c - seg_med) > outlier_sigma * seg_std] = np.nan
            finite = np.isfinite(seg_c)
        if finite.sum() < 5:
            continue

        # 2. per-segment median subtraction
        seg_c -= float(np.median(seg_c[finite]))

        # 3. rolling-median detrend via scipy.ndimage.median_filter (C-impl).
        # Window size derived from the segment's median cadence — Kepler
        # 30-min cadence with 1-day window → ~49 samples; TESS 2-min cadence
        # → ~721 samples. Both ~50× faster than pandas time-based rolling
        # for the segment lengths we see in practice (60–90k cadences).
        # NaN-safe: fill with segment median before filter, restore mask after.
        # Sort first — stitched multi-quarter LCs may have minor non-monotonic
        # cadences that would skew the median window.
        order = np.argsort(seg_t, kind="stable")
        inv_order = np.argsort(order)
        seg_t_sorted = seg_t[order]
        seg_c_sorted = seg_c[order]

        diffs = np.diff(seg_t_sorted)
        diffs = diffs[diffs > 0]
        cadence_days = float(np.median(diffs)) if diffs.size else 0.0208  # ≈30 min
        if not np.isfinite(cadence_days) or cadence_days <= 0:
            cadence_days = 0.0208
        window_n = max(5, round(rolling_window_days / cadence_days))
        window_n = min(window_n, seg_c_sorted.size)
        if window_n % 2 == 0:
            window_n += 1  # odd for symmetric centring

        finite_sorted = np.isfinite(seg_c_sorted)
        if finite_sorted.sum() < 5:
            continue
        seg_filled = np.where(
            finite_sorted, seg_c_sorted, float(np.median(seg_c_sorted[finite_sorted]))
        )
        baseline = median_filter(seg_filled, size=window_n, mode="nearest")
        detrended_sorted = seg_c_sorted - baseline
        detrended_sorted[~finite_sorted] = np.nan
        out[lo:hi] = detrended_sorted[inv_order]
    return out


def extract_centroid_features(
    lc: lk.LightCurve,
    period: float,
    t0: float,
    duration: float,
) -> dict[str, float]:
    """Compute centroid-shift statistics during transit vs out-of-transit.

    Returns a dict with three keys:
      - centroid_shift_x : column shift, pixels (after detrend)
      - centroid_shift_y : row shift, pixels (after detrend)
      - centroid_snr     : √(SNR_x² + SNR_y²), dimensionless. Each axis's
                           SNR is `S_axis / (σ_oot / √N_itr)`. Genuine
                           on-target transits give values < ~3; BEBs give
                           values ≳ 3.

    Returns NaN for all three when MOM_CENTR1/2 are absent, or when the
    detrended in/out-of-transit masks are too small (< 3 / < 10 cadences)
    or σ_oot collapses to zero. NaNs flow to the build pipeline's median
    imputer downstream.
    """
    cx_col = next((c for c in ("mom_centr1", "centroid_col") if c in lc.columns), None)
    cy_col = next((c for c in ("mom_centr2", "centroid_row") if c in lc.columns), None)
    nan_dict = {
        "centroid_shift_x": float("nan"),
        "centroid_shift_y": float("nan"),
        "centroid_snr": float("nan"),
    }
    if cx_col is None or cy_col is None:
        return nan_dict

    t = np.asarray(lc.time.value, dtype=float)
    cx_raw = np.asarray(lc[cx_col].value, dtype=float)
    cy_raw = np.asarray(lc[cy_col].value, dtype=float)

    # Detrend in time order (folding here would break the rolling window).
    cx = _detrend_axis(t, cx_raw)
    cy = _detrend_axis(t, cy_raw)

    # Phase-fold inline: φ ∈ [−0.5, 0.5], same convention as lk.LightCurve.fold.
    phase = ((t - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)

    half = (duration / period) / 2.0
    finite = np.isfinite(cx) & np.isfinite(cy)
    in_transit = (np.abs(phase) < half) & finite
    out_transit = (np.abs(phase) > 3 * half) & finite

    n_itr = int(in_transit.sum())
    n_oot = int(out_transit.sum())
    if n_itr < 3 or n_oot < 10:
        return nan_dict

    s_x = float(np.median(cx[in_transit]) - np.median(cx[out_transit]))
    s_y = float(np.median(cy[in_transit]) - np.median(cy[out_transit]))

    sigma_x = _robust_std(cx[out_transit])
    sigma_y = _robust_std(cy[out_transit])
    if not (np.isfinite(sigma_x) and sigma_x > 0 and np.isfinite(sigma_y) and sigma_y > 0):
        return {
            "centroid_shift_x": s_x,
            "centroid_shift_y": s_y,
            "centroid_snr": float("nan"),
        }

    n_sqrt = float(n_itr) ** 0.5
    snr_x = s_x / (sigma_x / n_sqrt)
    snr_y = s_y / (sigma_y / n_sqrt)
    return {
        "centroid_shift_x": s_x,
        "centroid_shift_y": s_y,
        "centroid_snr": float(np.hypot(snr_x, snr_y)),
    }


def extract_centroid_offset(
    lc: lk.LightCurve,
    period: float,
    t0: float,
    duration: float,
) -> float:
    """Single-scalar wrapper of `extract_centroid_features` for the aux vector.

    Returns `centroid_snr` — the magnitude of the in-transit centroid shift
    in units of σ, from the Ansdell-style normalisation in
    `extract_centroid_features`. Genuine on-target transits produce values
    near zero (< ~3); background eclipsing binaries produce values ≳ 3.

    Returns NaN when `MOM_CENTR1/2` are missing from the FITS, the in-/out-
    of-transit masks are too small, or σ_oot collapses to zero. The build
    pipeline's median imputer fills NaNs at training time.
    """
    return float(extract_centroid_features(lc, period, t0, duration)["centroid_snr"])
