"""Per-candidate vetting plots — scaffolded.

When the model flags a candidate, the next step is human vetting. Build a
single-page diagnostic figure showing the standard checks that astronomers
use to triage a TOI:

  1. Full-mission stitched + flattened light curve.
  2. Phase-folded global view at the candidate's best (period, t0).
  3. Phase-folded local view (zoomed on transit).
  4. Odd vs even transit comparison — large depth difference suggests an
     eclipsing-binary contaminant.
  5. Secondary-eclipse check at phase 0.5.
  6. Centroid-shift diagnostic (if pixel data available).

TODO(Oliver): wire this up after `search/bls.py` and `search/tls.py` exist
so a candidate object carries (period, t0, duration) ready to plot.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lightkurve as lk


@dataclass(frozen=True)
class CandidateReport:
    tic_id: int
    period: float
    t0: float
    duration: float
    score: float
    score_std: float


def vetting_figure(
    lc: "lk.LightCurve",
    report: CandidateReport,
    out_path: Path,
) -> Path:
    """Generate and save a six-panel vetting figure.

    NOT IMPLEMENTED YET — this is the highest-leverage thing to build next
    once you have the full pipeline producing candidates. Skeleton:

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        # 1. raw + flattened
        # 2. global view
        # 3. local view
        # 4. odd vs even
        # 5. secondary eclipse window
        # 6. centroid (if available)
        fig.savefig(out_path, dpi=150)
        return out_path
    """
    raise NotImplementedError("vetting_figure is scaffolded — see TODO in vetting.py")
