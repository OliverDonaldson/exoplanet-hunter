"""Force-refresh the live ExoFOP TOI + NEA PS caches.

Use when you want the freshest data on disk before running discovery_shortlist
or show_progress --live.

Usage:
    python scripts/refresh_registries.py
    python scripts/refresh_registries.py --only exofop   # just one source
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "src"))

from exoplanet_hunter.data.registries import (  # noqa: E402  (after sys.path insert)
    fetch_exofop_tois,
    fetch_nea_ps,
    status_summary,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", choices=["exofop", "nea"], help="only refresh one source")
    args = ap.parse_args()

    if args.only != "nea":
        print("fetching ExoFOP TOI table…")
        tois = fetch_exofop_tois(refresh=True)
        print(f"  ok: {len(tois):,} rows, {len(tois.columns)} cols")

    if args.only != "exofop":
        print("fetching NEA PS table (confirmed planets, default_flag=1)…")
        ps = fetch_nea_ps(refresh=True)
        print(f"  ok: {len(ps):,} rows, {len(ps.columns)} cols")

    print("\ncache state:")
    for label, info in status_summary().items():
        print(f"  {label}: age={info['age_hours']}h  stale={info['stale']}  path={info['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
