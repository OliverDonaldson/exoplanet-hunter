"""Quick look at the live bulk-scoring progress and the top candidates so far.

Reads ``results/candidates_scored.parquet`` (which the scoring script writes
atomically every 25 rows, so any read is a consistent snapshot) and prints:

  - scoring progress (total / ok / errored, by mission)
  - probability distribution (quantiles + counts above thresholds)
  - the top-N candidates sorted by ``prob_mean``

Usage:
    python scripts/show_progress.py                # top 20 by prob_mean
    python scripts/show_progress.py --top 50       # top 50
    python scripts/show_progress.py --min 0.95     # only prob_mean >= 0.95
    python scripts/show_progress.py --mission Kepler
    python scripts/show_progress.py --sort fold_disagree --asc  # most agreed
    python scripts/show_progress.py --live                  # join current ExoFOP TFOPWG
    python scripts/show_progress.py --live --refresh        # force registry re-fetch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

SCORED = PROJECT / "results" / "candidates_scored.parquet"
CANDIDATES = PROJECT / "data" / "labels" / "candidates.parquet"

TOP_COLS = [
    "tic_id",
    "toi",
    "name",
    "mission",
    "disposition",
    "period",
    "depth",
    "centroid_snr",
    "prob_mean",
    "prob_std",
    "fold_disagree",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=20, help="rows to display")
    ap.add_argument("--min", type=float, default=0.0, help="minimum prob_mean")
    ap.add_argument("--mission", choices=["TESS", "Kepler"], help="filter by mission")
    ap.add_argument("--sort", default="prob_mean", help="column to sort by")
    ap.add_argument("--asc", action="store_true", help="ascending sort")
    ap.add_argument("--live", action="store_true", help="join current ExoFOP TFOPWG disposition")
    ap.add_argument("--refresh", action="store_true", help="force registry re-fetch (with --live)")
    args = ap.parse_args()

    if not SCORED.exists():
        print(f"no scored parquet yet at {SCORED}")
        return 1

    scored = pd.read_parquet(SCORED)
    total_target = len(pd.read_parquet(CANDIDATES)) if CANDIDATES.exists() else None

    # ─── progress summary ──────────────────────────────────────────────
    ok = scored[scored.status == "ok"]
    errored = scored[scored.status != "ok"]
    print("\n=== scoring progress ===")
    if total_target:
        pct = 100 * len(scored) / total_target
        print(f"rows scored: {len(scored):,} / {total_target:,}  ({pct:.1f} %)")
    else:
        print(f"rows scored: {len(scored):,}")
    print(f"  ok      : {len(ok):,}")
    print(f"  errored : {len(errored):,}")
    print(f"  missions: {scored.mission.value_counts().to_dict()}")
    if "scored_at" in scored.columns and len(scored):
        print(f"  last scored_at: {scored.scored_at.max()}")

    # ─── probability distribution ─────────────────────────────────────
    if len(ok):
        q = ok.prob_mean.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
        print(f"\n=== prob_mean distribution (n={len(ok):,}) ===")
        for p, v in q.items():
            print(f"  p{int(p * 100):>3}: {v:.4f}")
        for thr in (0.5, 0.7, 0.9, 0.95, 0.99):
            n = (ok.prob_mean >= thr).sum()
            print(f"  prob_mean >= {thr:.2f} : {n:,}  ({100 * n / len(ok):.1f} %)")

    # ─── top-N table ───────────────────────────────────────────────────
    view = ok if len(ok) else scored
    if args.mission:
        view = view[view.mission == args.mission]
    view = view[view.prob_mean >= args.min]
    if args.sort not in view.columns:
        print(f"\nunknown --sort column: {args.sort!r}", file=sys.stderr)
        return 2
    view = view.sort_values(args.sort, ascending=args.asc).head(args.top)

    # Optionally join the current ExoFOP TFOPWG disposition.
    if args.live and len(view):
        from exoplanet_hunter.data.registries import fetch_exofop_tois, status_summary

        toi = fetch_exofop_tois(refresh=args.refresh)
        toi["TOI_f"] = toi["TOI"].astype(float)
        lookup = toi.set_index("TOI_f")[["TFOPWG Disposition", "TESS Disposition"]]
        view = view.copy()
        view["toi_f"] = view["toi"].astype(float)
        view = view.join(lookup, on="toi_f").drop(columns=["toi_f"])
        view = view.rename(
            columns={
                "TFOPWG Disposition": "tfopwg_now",
                "TESS Disposition": "tess_now",
            }
        )
        TOP_COLS.extend(["tfopwg_now", "tess_now"])
        s = status_summary()["exofop_tois"]
        age = s.get("age_hours")
        print(f"\n[live] exofop_tess_tois_live.csv  age={age}h  stale={s.get('stale')}")

    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}" if abs(x) < 1e4 else f"{x:.2e}")

    cols = [c for c in TOP_COLS if c in view.columns]
    title = f"=== top {min(args.top, len(view))} by {args.sort}"
    if args.min > 0:
        title += f", prob_mean >= {args.min:.2f}"
    if args.mission:
        title += f", mission={args.mission}"
    title += " ==="
    print(f"\n{title}")
    if len(view) == 0:
        print("  (no rows match the filter)")
    else:
        print(view[cols].to_string(index=False))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
