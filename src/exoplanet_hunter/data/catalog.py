"""Build the labelled catalogue from NASA Exoplanet Archive + TOI tables.

Two sources, both queried via the public TAP service:

  * **PS** — Confirmed planets. We filter to those discovered (or co-discovered)
    by TESS and require non-null transit depth + period.
  * **TOI** — TESS Objects of Interest, with the `tfopwg_disp` disposition
    column. We map dispositions to integer labels:

        CP, KP            -> 1   (confirmed / known planet — positive)
        FP, FA            -> 0   (false positive / false alarm — negative)
        PC                -> -1  (unconfirmed candidate — held out, used for inference)
        APC, anything else -> dropped
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from exoplanet_hunter.utils.logging import get_logger

log = get_logger(__name__)

TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Disposition → integer label.
DISPOSITION_LABELS: dict[str, int] = {
    "CP":  1,   # confirmed planet
    "KP":  1,   # known planet (typically pre-TESS confirmation)
    "FP":  0,   # false positive
    "FA":  0,   # false alarm (instrumental)
    "PC": -1,   # planet candidate — held out for inference
}

# Kepler KOI dispositions use different strings.
KEPLER_DISPOSITION_LABELS: dict[str, int] = {
    "CONFIRMED":      1,
    "FALSE POSITIVE": 0,
    "CANDIDATE":     -1,
}


@dataclass(frozen=True)
class CatalogRequest:
    n_confirmed: int
    n_false_pos: int
    n_confirmed_kepler: int = 0
    n_false_pos_kepler: int = 0
    seed: int = 42


def _tap_query(adql: str, fmt: str = "csv") -> pd.DataFrame:
    """Run a TAP query against the NASA Exoplanet Archive."""
    log.info("[catalog] querying TAP — %s", adql.split("from")[1].split()[0])
    r = requests.get(TAP_URL, params={"query": adql, "format": fmt}, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _query_confirmed_planets() -> pd.DataFrame:
    """Confirmed planets observed by TESS, with transit parameters.

    ``pl_tranmid`` is stored in the archive as full BJD (~2,458,000+), but
    TESS light curves use BTJD = BJD − 2457000. We subtract the offset at
    query time so downstream phase-folding works: a finite-precision period
    accumulates many days of error across ~700k cycles, so keeping the
    (t − t0) magnitude small is essential, not cosmetic.
    """
    adql = (
        "select pl_name, tic_id, hostname, "
        "       pl_orbper, pl_tranmid - 2457000.0 as pl_tranmid, "
        "       pl_trandep, pl_trandur, "
        "       st_teff, st_rad, st_logg, sy_tmag "
        "from ps "
        "where tic_id is not null "
        "  and pl_trandep is not null "
        "  and pl_orbper is not null "
        "  and pl_tranmid is not null "
        "  and disc_facility like '%TESS%'"
    )
    df = _tap_query(adql)
    # ps.tic_id comes back as e.g. "TIC 142937186"; normalise to bare integer
    # so it lines up with toi.tid (already an int) for the later concat/dedupe.
    df["tic_id"] = (
        df["tic_id"].astype(str).str.replace("TIC ", "", regex=False).str.strip().astype("int64")
    )
    df = df.drop_duplicates(subset="tic_id").reset_index(drop=True)
    df["disposition"] = "CP"
    df["label"] = 1
    df["mission"] = "TESS"
    return df.rename(
        columns={
            "pl_orbper":   "period",
            "pl_tranmid":  "t0",
            "pl_trandep":  "depth",
            "pl_trandur":  "duration",
            "st_teff":     "teff",
            "st_rad":      "radius",
            "st_logg":     "logg",
            "sy_tmag":     "tmag",
        }
    )


def _query_toi() -> pd.DataFrame:
    """All TOIs with their disposition — includes both candidates and false positives.

    Note: TOI's `pl_trandurh` is in **hours** while `ps.pl_trandur` (in
    `_query_confirmed_planets`) is in **days**. We convert to days here so
    the combined catalogue has consistent units throughout — downstream
    code (build_dataset.py, score_target.py, build_views) all expect days.

    ``pl_tranmid`` from the TOI table is full BJD (~2,458,000+). TESS light
    curves use BTJD = BJD − 2457000, so we subtract the offset at query time
    (same as ``_query_confirmed_planets``).
    """
    adql = (
        "select toi, tid as tic_id, "
        "       pl_orbper as period, pl_tranmid - 2457000.0 as t0, "
        "       pl_trandep as depth, pl_trandurh / 24.0 as duration, "
        "       tfopwg_disp as disposition, "
        "       st_teff as teff, st_rad as radius, "
        "       st_logg as logg, st_tmag as tmag "
        "from toi "
        "where tfopwg_disp is not null "
        "  and pl_orbper is not null "
        "  and pl_tranmid is not null"
    )
    df = _tap_query(adql)
    df["label"] = df["disposition"].map(DISPOSITION_LABELS)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["mission"] = "TESS"
    return df.drop_duplicates(subset="tic_id").reset_index(drop=True)


def _query_koi() -> pd.DataFrame:
    """Kepler Objects of Interest from the ``cumulative`` archive table.

    Unit normalisation (matches the TESS path):
      * ``koi_time0bk`` is BKJD (BJD − 2454833). Downstream code uses
        (t − t0) mod period, so the absolute epoch offset cancels.
      * ``koi_duration`` is **hours** → converted to days.
      * ``koi_depth`` is **ppm** → converted to fractional depth.
    """
    adql = (
        "select kepoi_name as name, "
        "       kepid as target_id, "
        "       koi_period as period, "
        "       koi_time0bk as t0, "
        "       koi_depth / 1.0e6 as depth, "
        "       koi_duration / 24.0 as duration, "
        "       koi_model_snr as snr, "
        "       koi_disposition as disposition, "
        "       koi_steff as teff, koi_srad as radius, "
        "       koi_slogg as logg, koi_kepmag as tmag "
        "from cumulative "
        "where koi_disposition is not null "
        "  and koi_period is not null "
        "  and koi_time0bk is not null"
    )
    df = _tap_query(adql)
    df["label"] = df["disposition"].map(KEPLER_DISPOSITION_LABELS)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["mission"] = "Kepler"
    # Rename target_id → tic_id for schema compatibility (it's actually a KIC ID).
    df = df.rename(columns={"target_id": "tic_id"})
    return df.drop_duplicates(subset="tic_id").reset_index(drop=True)


def build_label_catalog(req: CatalogRequest, out_dir: Path) -> pd.DataFrame:
    """Build the combined labelled catalogue and persist to parquet.

    Parameters
    ----------
    req     : sampling request — how many of each class to pull.
    out_dir : directory where `labels.parquet` will be written.

    Returns
    -------
    The combined dataframe with one row per TIC and columns
    `tic_id, period, t0, depth, duration, snr, disposition, label, teff, radius, logg, tmag`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    confirmed = _query_confirmed_planets()
    toi       = _query_toi()

    pos = pd.concat(
        [confirmed, toi[toi["label"] == 1]],
        ignore_index=True,
    ).drop_duplicates(subset="tic_id")

    neg = toi[toi["label"] == 0]
    pc  = toi[toi["label"] == -1]                # held-out candidates

    log.info(
        "[catalog] sources: confirmed=%d, TOI=%d (CP=%d, FP=%d, PC=%d)",
        len(confirmed), len(toi),
        len(toi[toi["label"] == 1]),
        len(toi[toi["label"] == 0]),
        len(pc),
    )

    # Subsample to requested counts, with deterministic seed.
    pos = pos.sample(min(req.n_confirmed, len(pos)), random_state=req.seed)
    neg = neg.sample(min(req.n_false_pos, len(neg)), random_state=req.seed)

    parts = [pos, neg]

    # --- Kepler / KOI targets (optional) --------------------------------
    if req.n_confirmed_kepler > 0 or req.n_false_pos_kepler > 0:
        koi = _query_koi()
        koi_pos = koi[koi["label"] == 1]
        koi_neg = koi[koi["label"] == 0]
        koi_pc  = koi[koi["label"] == -1]

        log.info(
            "[catalog] KOI sources: confirmed=%d, FP=%d, PC=%d",
            len(koi_pos), len(koi_neg), len(koi_pc),
        )

        koi_pos = koi_pos.sample(
            min(req.n_confirmed_kepler, len(koi_pos)), random_state=req.seed,
        )
        koi_neg = koi_neg.sample(
            min(req.n_false_pos_kepler, len(koi_neg)), random_state=req.seed,
        )
        parts.extend([koi_pos, koi_neg])

        # Persist Kepler held-out candidates alongside TESS candidates.
        pc = pd.concat([pc, koi_pc], ignore_index=True)

    catalog = pd.concat(parts, ignore_index=True)
    catalog["tic_id"] = catalog["tic_id"].astype("int64")

    out_path = out_dir / "labels.parquet"
    catalog.to_parquet(out_path, index=False)

    log.info("[catalog] wrote %d rows → %s", len(catalog), out_path)
    log.info(
        "[catalog]   pos=%d  neg=%d  candidates(held-out)=%d",
        (catalog["label"] == 1).sum(),
        (catalog["label"] == 0).sum(),
        len(pc),
    )

    # Persist held-out PCs separately — they're for inference, not training.
    pc_path = out_dir / "candidates.parquet"
    pc.to_parquet(pc_path, index=False)
    log.info("[catalog] wrote %d held-out candidates → %s", len(pc), pc_path)

    return catalog
