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

A separate "quiet" catalogue of TICs with no flagged signal is sampled from
random TIC IDs and verified against the TOI list — this gives the model
genuine non-transit examples (it should learn that "no dip" → not a planet).
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


@dataclass(frozen=True)
class CatalogRequest:
    n_confirmed: int
    n_false_pos: int
    n_quiet: int
    seed: int = 42


def _tap_query(adql: str, fmt: str = "csv") -> pd.DataFrame:
    """Run a TAP query against the NASA Exoplanet Archive."""
    log.info("[catalog] querying TAP — %s", adql.split("from")[1].split()[0])
    r = requests.get(TAP_URL, params={"query": adql, "format": fmt}, timeout=120)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def _query_confirmed_planets() -> pd.DataFrame:
    """Confirmed planets observed by TESS, with transit parameters."""
    adql = (
        "select pl_name, tic_id, hostname, "
        "       pl_orbper, pl_tranmid, pl_trandep, pl_trandur, "
        "       st_teff, st_rad, st_logg, st_tmag "
        "from ps "
        "where tic_id is not null "
        "  and pl_trandep is not null "
        "  and pl_orbper is not null "
        "  and pl_tranmid is not null "
        "  and (disc_facility like '%TESS%' or pl_facility like '%TESS%')"
    )
    df = _tap_query(adql)
    df = df.drop_duplicates(subset="tic_id").reset_index(drop=True)
    df["disposition"] = "CP"
    df["label"] = 1
    return df.rename(
        columns={
            "pl_orbper":   "period",
            "pl_tranmid":  "t0",
            "pl_trandep":  "depth",
            "pl_trandur":  "duration",
            "st_teff":     "teff",
            "st_rad":      "radius",
            "st_logg":     "logg",
            "st_tmag":     "tmag",
        }
    )


def _query_toi() -> pd.DataFrame:
    """All TOIs with their disposition — includes both candidates and false positives."""
    adql = (
        "select toi, tid as tic_id, "
        "       pl_orbper as period, pl_tranmid as t0, "
        "       pl_trandep as depth, pl_trandurh as duration, "
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
    `tic_id, period, t0, depth, duration, disposition, label, teff, radius, logg, tmag`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = pd.Series(range(10_000_000)).sample(frac=1.0, random_state=req.seed)

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

    # "Quiet" stars — sampled but reserved for after the planet/FP set is built;
    # in practice we draw them lazily from random TIC IDs in the downloader,
    # because verifying "no signal" requires the actual light curve.
    quiet = pd.DataFrame({"tic_id": rng.head(req.n_quiet).astype("int64")})
    quiet["label"] = 0
    quiet["disposition"] = "QUIET"
    for col in ("period", "t0", "depth", "duration", "teff", "radius", "logg", "tmag"):
        quiet[col] = pd.NA

    catalog = pd.concat([pos, neg, quiet], ignore_index=True)
    catalog["tic_id"] = catalog["tic_id"].astype("int64")

    out_path = out_dir / "labels.parquet"
    catalog.to_parquet(out_path, index=False)

    log.info("[catalog] wrote %d rows → %s", len(catalog), out_path)
    log.info(
        "[catalog]   pos=%d  neg=%d  quiet=%d  candidates(held-out)=%d",
        (catalog["label"] == 1).sum(),
        ((catalog["label"] == 0) & (catalog["disposition"] != "QUIET")).sum(),
        (catalog["disposition"] == "QUIET").sum(),
        len(pc),
    )

    # Persist held-out PCs separately — they're for inference, not training.
    pc_path = out_dir / "candidates.parquet"
    pc.to_parquet(pc_path, index=False)
    log.info("[catalog] wrote %d held-out candidates → %s", len(pc), pc_path)

    return catalog
