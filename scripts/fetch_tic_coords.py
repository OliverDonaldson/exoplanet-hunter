"""Fetch RA/Dec (+ Galactic coords, Tmag, Teff) from MAST TIC for all candidates.

Outputs ``data/external/tic_coords.parquet`` with columns:
    [tic_id, ra, dec, gallong, gallat, Tmag, Teff, queried_at]

Idempotent: skips TIC IDs already cached on disk. Safe to re-run.

Usage:
    python scripts/fetch_tic_coords.py
    python scripts/fetch_tic_coords.py --batch 200 --sleep 0.25
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.mast import Catalogs

PROJECT = Path(__file__).resolve().parent.parent
CANDIDATES = PROJECT / "data" / "labels" / "candidates.parquet"
OUT = PROJECT / "data" / "external" / "tic_coords.parquet"

KEEP_COLS = ["ID", "ra", "dec", "gallong", "gallat", "Tmag", "Teff"]


def load_targets() -> list[int]:
    candidates = pd.read_parquet(CANDIDATES)
    return sorted(set(candidates["tic_id"].astype(int).tolist()))


def load_cache() -> pd.DataFrame:
    if OUT.exists():
        return pd.read_parquet(OUT)
    return pd.DataFrame(
        columns=["tic_id", "ra", "dec", "gallong", "gallat", "Tmag", "Teff", "queried_at"]
    )


def save_cache(df: pd.DataFrame) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(OUT)


def query_batch(ids: list[int]) -> pd.DataFrame:
    res = Catalogs.query_criteria(catalog="Tic", ID=ids)
    present = [c for c in KEEP_COLS if c in res.colnames]
    coords = res[present].to_pandas()
    coords = coords.rename(columns={"ID": "tic_id"})
    coords["tic_id"] = coords["tic_id"].astype(np.int64)
    for c in ("ra", "dec", "gallong", "gallat", "Tmag", "Teff"):
        if c not in coords.columns:
            coords[c] = np.nan
    coords["queried_at"] = dt.datetime.now(dt.UTC).isoformat()
    return coords[["tic_id", "ra", "dec", "gallong", "gallat", "Tmag", "Teff", "queried_at"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=100, help="TIC IDs per MAST query")
    ap.add_argument("--sleep", type=float, default=0.4, help="seconds between batches")
    ap.add_argument("--checkpoint-every", type=int, default=500, help="rows between disk flushes")
    args = ap.parse_args()

    targets = load_targets()
    cache = load_cache()
    have = set(cache["tic_id"].astype(np.int64).tolist()) if len(cache) else set()
    todo = [t for t in targets if t not in have]

    print(f"targets: {len(targets):,} | cached: {len(have):,} | to fetch: {len(todo):,}")
    if not todo:
        print("all targets already cached. nothing to do.")
        return 0

    pending: list[pd.DataFrame] = []
    pending_n = 0
    n_batches = (len(todo) + args.batch - 1) // args.batch
    t_start = time.time()

    for bi in range(n_batches):
        chunk = todo[bi * args.batch : (bi + 1) * args.batch]
        try:
            batch_df = query_batch(chunk)
        except Exception as exc:
            print(
                f"  batch {bi + 1}/{n_batches}: ERROR {exc!r} (asked {len(chunk)} IDs) — skipping"
            )
            time.sleep(args.sleep * 4)
            continue
        pending.append(batch_df)
        pending_n += len(batch_df)
        done = len(have) + pending_n
        elapsed = time.time() - t_start
        rate = pending_n / max(elapsed, 1e-6)
        eta = (len(todo) - pending_n) / max(rate, 1e-6)
        print(
            f"  batch {bi + 1}/{n_batches}: got {len(batch_df)}/{len(chunk)} | "
            f"total {done}/{len(targets)} | {rate:.1f} rows/s | ETA {eta / 60:.1f} min"
        )

        if pending_n >= args.checkpoint_every:
            cache = pd.concat([cache, *pending], ignore_index=True).drop_duplicates(
                subset="tic_id", keep="last"
            )
            save_cache(cache)
            print(f"    checkpoint -> {OUT.relative_to(PROJECT)} ({len(cache):,} rows)")
            pending = []
            pending_n = 0

        time.sleep(args.sleep)

    if pending:
        cache = pd.concat([cache, *pending], ignore_index=True).drop_duplicates(
            subset="tic_id", keep="last"
        )
        save_cache(cache)

    missed = sorted(set(targets) - set(cache["tic_id"].astype(np.int64).tolist()))
    print(f"done. {len(cache):,} rows in {OUT.relative_to(PROJECT)}")
    if missed:
        print(f"  warning: {len(missed)} TIC IDs not returned by MAST (first 10: {missed[:10]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
