"""Hand-crafted scalar features extracted from a phase-folded view.

These feed the Random Forest baseline (DATA 305 Week 2 — ensembles). The set
is deliberately small and interpretable: a domain expert can read each
feature and explain why it should help separate transits from false positives.

Compare to the CNN, which learns its own features from the raw view.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

FEATURE_NAMES: list[str] = [
    # --- transit shape ---
    "depth",                # baseline - min(centre)
    "depth_snr",            # depth / std(wings)
    "duration_frac",        # fraction of bins below baseline - 1 sigma
    "min_flux",             # global min of view
    "centre_mean",
    "centre_std",
    "wing_mean",
    "wing_std",
    # --- distribution shape ---
    "skewness",
    "kurtosis",
    # --- asymmetry / V-shape (eclipsing-binary signal) ---
    "asymmetry",            # |mean(left half) - mean(right half)| in centre
    "v_shape_score",        # how V-shaped vs U-shaped the dip is
    # --- broad statistics ---
    "n_below_2sigma",       # fraction of points more than 2σ below baseline
    "n_below_3sigma",
]


def extract_features(view: np.ndarray) -> np.ndarray:
    """Extract a fixed-length feature vector from a normalised phase-folded view.

    Input is expected to be a 1D array of length N where N is even-ish; we
    treat the central 20% as "transit window" and the outer 40% as "wings".
    """
    if view.ndim != 1:
        raise ValueError(f"expected 1D view, got shape {view.shape}")

    n = view.size
    centre_slice = slice(int(0.4 * n), int(0.6 * n))
    left_wing    = view[: int(0.2 * n)]
    right_wing   = view[int(0.8 * n):]
    wings        = np.concatenate([left_wing, right_wing])
    centre       = view[centre_slice]

    sigma = float(np.std(wings) + 1e-10)
    baseline = float(np.median(wings))

    # Transit metrics — depth measured from the wing baseline.
    depth = baseline - float(np.min(centre))
    depth_snr = depth / sigma
    duration_frac = float(np.mean(view < baseline - sigma))

    # V-shape vs U-shape: U is flat at the bottom, V comes to a sharp point.
    # Compare central std to a true V's expected std.
    centre_min = float(np.min(centre))
    if depth > 1e-6:
        v_shape_score = float(np.std(centre)) / (depth + 1e-10)
    else:
        v_shape_score = 0.0

    asymmetry = abs(float(np.mean(centre[: len(centre) // 2]))
                    - float(np.mean(centre[len(centre) // 2 :])))

    return np.array(
        [
            depth,
            depth_snr,
            duration_frac,
            float(np.min(view)),
            float(np.mean(centre)),
            float(np.std(centre)),
            float(np.mean(wings)),
            float(np.std(wings)),
            float(stats.skew(view)),
            float(stats.kurtosis(view)),
            asymmetry,
            v_shape_score,
            float(np.mean(view < baseline - 2 * sigma)),
            float(np.mean(view < baseline - 3 * sigma)),
        ],
        dtype=np.float32,
    )
