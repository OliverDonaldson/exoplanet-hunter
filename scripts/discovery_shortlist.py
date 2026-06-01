"""Cross-reference scored candidates against ExoFOP catalogs.

Joins ``results/candidates_scored.parquet`` with the ExoFOP TOI / CTOI / KOI
snapshots in ``data/external/exofop/`` and writes
``results/discovery_shortlist.parquet`` with:

  - Current TFOPWG / TESS Disposition (may differ from the stale PC label
    inside our ``candidates.parquet``).
  - Total follow-up observation count per candidate (Time Series + Imaging
    + Spectroscopy) — your "community attention" proxy.
  - Master priority + Comments from the TFOP working group.
  - A scalar ``discovery_score`` = prob × (1 − fold_disagree) × follow_up_penalty
    that ranks high-confidence under-investigated candidates highest.

Usage:
    # Top 30 ranked by discovery_score, prob ≥ 0.8
    python scripts/discovery_shortlist.py

    # Stricter
    python scripts/discovery_shortlist.py prob_min=0.9 followup_max=3 top_k=50

    # Also show cases where our model disagrees with ExoFOP (model=planet, ExoFOP=FP)
    python scripts/discovery_shortlist.py show_discrepancies=true
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from exoplanet_hunter.utils import ProjectPaths, get_logger, set_global_seed

log = get_logger(__name__)


# ---------- catalog loaders --------------------------------------------------


def _load_tois(path: Path) -> pd.DataFrame:
    tois = pd.read_csv(path, skiprows=1, low_memory=False)
    tois["toi_str"] = tois["TOI"].astype(str)
    tois["n_followup"] = (
        tois["Time Series Observations"].fillna(0)
        + tois["Imaging Observations"].fillna(0)
        + tois["Spectroscopy Observations"].fillna(0)
    ).astype(int)
    return tois


def _load_kois(path: Path) -> pd.DataFrame:
    kois = pd.read_csv(path, skiprows=1, low_memory=False)
    kois["n_followup"] = (
        kois["Time Series Observations"].fillna(0)
        + kois["Imaging Observations"].fillna(0)
        + kois["Spectroscopy Observations"].fillna(0)
    ).astype(int)
    return kois


def _load_ps(exofop_dir: Path) -> pd.DataFrame | None:
    """Load the most recent NEA Planetary Systems snapshot.

    Returns a DataFrame with one row per published planet parameter set
    (we filter to default_flag=1 elsewhere). Returns None if no PS file
    is present.
    """
    ps_files = sorted(exofop_dir.glob("PS_*.csv"))
    if not ps_files:
        return None
    ps = pd.read_csv(ps_files[-1], comment="#", low_memory=False)
    ps["releasedate"] = pd.to_datetime(ps["releasedate"], errors="coerce")
    # Extract TOI base number from pl_name (e.g. "TOI-1011 b" → 1011, "TOI-1011.01" → 1011)
    ps["toi_base"] = ps["pl_name"].str.extract(r"TOI[-\s]*(\d+)", expand=False)
    ps["toi_base"] = pd.to_numeric(ps["toi_base"], errors="coerce")
    return ps


def _add_since_confirmed(
    enriched: pd.DataFrame, ps: pd.DataFrame, training_cutoff: str = "2025-01-01"
) -> pd.DataFrame:
    """Tag candidates whose TOI matches a PS confirmation released after the
    training-data cutoff. Match key: TOI base number AND period within 2%.

    Adds columns:
      - confirmed_after_training : bool
      - confirmed_pl_name        : str (e.g. "TOI-1011 b")
      - confirmed_releasedate    : datetime
    """
    recent = ps[(ps["releasedate"] >= training_cutoff) & ps["toi_base"].notna()].copy()
    if "default_flag" in recent.columns:
        recent = recent[recent["default_flag"] == 1]

    out = enriched.copy()
    out["toi_base"] = out["toi"].apply(lambda x: int(x) if pd.notna(x) else None)
    out["confirmed_after_training"] = False
    out["confirmed_pl_name"] = None
    out["confirmed_releasedate"] = pd.NaT

    # Join + period filter (within 2%)
    j = out.merge(
        recent[["toi_base", "pl_name", "pl_orbper", "releasedate"]],
        on="toi_base",
        how="left",
        suffixes=("", "_ps"),
    )
    # period agreement within 2% (cand period vs PS period)
    mask = (
        j["pl_orbper"].notna()
        & j["period"].notna()
        & ((j["period"] - j["pl_orbper"]).abs() / j["period"].abs() < 0.02)
    )
    j.loc[mask, "confirmed_after_training"] = True
    j.loc[mask, "confirmed_pl_name"] = j.loc[mask, "pl_name"]
    j.loc[mask, "confirmed_releasedate"] = j.loc[mask, "releasedate"]
    j = j.drop(columns=["toi_base", "pl_name", "pl_orbper", "releasedate"], errors="ignore")
    # A multi-planet TOI shares one `toi_base` across its sibling planets, so the
    # left-merge above fans a single candidate into one row per sibling. The 2%
    # period mask flags only the period-matched planet, leaving the sibling rows
    # with confirmed_after_training=False. Collapse back to one row per candidate,
    # keeping the matched (confirmed) row when present.
    if "candidate_idx" in j.columns:
        j = (
            j.sort_values("confirmed_after_training", ascending=False, kind="stable")
            .drop_duplicates(subset="candidate_idx", keep="first")
            .sort_index()
        )
    return j


# ---------- merges -----------------------------------------------------------


def _merge_tess(scored: pd.DataFrame, tois: pd.DataFrame) -> pd.DataFrame:
    sc = scored.copy()
    sc["toi_str"] = sc["toi"].astype(str)
    keep = [
        "TIC ID",
        "toi_str",
        "TFOPWG Disposition",
        "TESS Disposition",
        "Time Series Observations",
        "Imaging Observations",
        "Spectroscopy Observations",
        "n_followup",
        "Master priority",
        "Comments",
        "Planet Radius (R_Earth)",
        "Planet Eq Temp (K)",
        "Planet Insolation (Earth flux)",
        "ESM",
        "TSM",
        "Stellar Distance (pc)",
    ]
    keep = [c for c in keep if c in tois.columns]
    merged = sc.merge(
        tois[keep],
        left_on=["tic_id", "toi_str"],
        right_on=["TIC ID", "toi_str"],
        how="left",
    )
    return merged.drop(columns=["toi_str", "TIC ID"], errors="ignore")


def _merge_kepler(scored: pd.DataFrame, kois: pd.DataFrame) -> pd.DataFrame:
    sc = scored.copy()
    keep = [
        "TIC ID",
        "KOI",
        "Kepler Name",
        "Disposition",
        "Time Series Observations",
        "Imaging Observations",
        "Spectroscopy Observations",
        "n_followup",
    ]
    keep = [c for c in keep if c in kois.columns]
    merged = sc.merge(
        kois[keep],
        left_on="tic_id",
        right_on="TIC ID",
        how="left",
    )
    merged = merged.rename(columns={"Disposition": "TFOPWG Disposition"})
    return merged.drop(columns=["TIC ID"], errors="ignore")


def _discovery_score(df: pd.DataFrame) -> pd.Series:
    """High when model is confident + folds agree + community attention is low.

    score = prob_mean · max(0, 1 − fold_disagree) · 1/(1 + n_followup/3)
    """
    prob = df["prob_mean"].fillna(0.0)
    fold_d = df["fold_disagree"].fillna(1.0)
    nfo = df["n_followup"].fillna(0.0)
    follow_pen = 1.0 / (1.0 + nfo / 3.0)
    return prob * np.clip(1.0 - fold_d, 0.0, 1.0) * follow_pen


# ---------- main -------------------------------------------------------------


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    paths = ProjectPaths.from_cfg(cfg)

    scored_path = paths.root / str(getattr(cfg, "out_path", "results/candidates_scored.parquet"))
    if not scored_path.exists():
        log.error(
            "[shortlist] no scored parquet at %s — run score_candidates.py first", scored_path
        )
        sys.exit(2)

    prob_min = float(getattr(cfg, "prob_min", 0.8))
    followup_max = getattr(cfg, "followup_max", None)
    top_k = int(getattr(cfg, "top_k", 30))
    show_discrepancies = bool(getattr(cfg, "show_discrepancies", False))

    out_path = paths.root / "results" / "discovery_shortlist.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    scored = pd.read_parquet(scored_path)
    log.info(
        "[shortlist] scored parquet: %d rows  ok=%d",
        len(scored),
        int((scored.status == "ok").sum()),
    )
    scored = scored[scored.status == "ok"].copy()

    exofop_dir = paths.root / "data" / "external" / "exofop"
    tois = _load_tois(exofop_dir / "exofop_tess_tois.csv")
    kois = _load_kois(exofop_dir / "exofop_tess_kois.csv")
    log.info("[shortlist] catalogs: %d TOIs  %d KOIs", len(tois), len(kois))

    tess_scored = scored[scored.mission == "TESS"]
    kep_scored = scored[scored.mission == "Kepler"]

    frames: list[pd.DataFrame] = []
    if len(tess_scored):
        frames.append(_merge_tess(tess_scored, tois))
    if len(kep_scored):
        frames.append(_merge_kepler(kep_scored, kois))
    if not frames:
        log.error("[shortlist] no rows to enrich")
        sys.exit(2)
    enriched = pd.concat(frames, ignore_index=True)
    enriched["discovery_score"] = _discovery_score(enriched)

    # Optional: tag candidates that have been *confirmed since training* via PS snapshot.
    # These are the cleanest blind-validation cases.
    ps = _load_ps(exofop_dir)
    if ps is not None:
        enriched = _add_since_confirmed(enriched, ps)
        n_conf = int(enriched["confirmed_after_training"].sum())
        log.info("[shortlist] %d candidates have been CONFIRMED since training", n_conf)

    enriched = enriched.sort_values("discovery_score", ascending=False).reset_index(drop=True)
    enriched.to_parquet(out_path, index=False)
    log.info("[shortlist] wrote enriched parquet (%d rows) → %s", len(enriched), out_path)

    # ---- summary --------------------------------------------------------
    if "TFOPWG Disposition" in enriched.columns:
        log.info(
            "[shortlist] TFOPWG disposition counts (current ExoFOP truth):\n%s",
            enriched["TFOPWG Disposition"].value_counts(dropna=False).to_dict(),
        )
    log.info(
        "[shortlist] follow-up count distribution: %s", enriched["n_followup"].describe().to_dict()
    )

    # ---- top-K discovery shortlist --------------------------------------
    sl = enriched[enriched["prob_mean"] >= prob_min].copy()
    if followup_max is not None:
        sl = sl[sl["n_followup"].fillna(99) <= int(followup_max)]
    # exclude known FPs / false alarms (we want discovery candidates)
    if "TFOPWG Disposition" in sl.columns:
        sl = sl[~sl["TFOPWG Disposition"].isin(["FP", "FA"])]
    sl = sl.head(top_k)

    if len(sl):
        cols = [
            "tic_id",
            "toi",
            "mission",
            "period",
            "prob_mean",
            "fold_disagree",
            "n_followup",
            "TFOPWG Disposition",
            "discovery_score",
        ]
        if "Planet Radius (R_Earth)" in sl.columns:
            cols.append("Planet Radius (R_Earth)")
        cols = [c for c in cols if c in sl.columns]
        log.info(
            "\n[TOP %d DISCOVERY CANDIDATES — prob ≥ %.2f%s]\n%s",
            len(sl),
            prob_min,
            f", n_followup ≤ {followup_max}" if followup_max is not None else "",
            sl[cols].to_string(index=False),
        )
    else:
        log.info(
            "[shortlist] no candidates pass prob ≥ %.2f%s",
            prob_min,
            f", n_followup ≤ {followup_max}" if followup_max is not None else "",
        )

    # ---- discrepancies (model says planet, ExoFOP says FP/FA) ----------
    if show_discrepancies and "TFOPWG Disposition" in enriched.columns:
        disc = enriched[
            (enriched["prob_mean"] > 0.7) & (enriched["TFOPWG Disposition"].isin(["FP", "FA"]))
        ]
        if len(disc):
            cols = [
                "tic_id",
                "toi",
                "mission",
                "prob_mean",
                "fold_disagree",
                "centroid_snr",
                "TFOPWG Disposition",
                "Comments",
            ]
            cols = [c for c in cols if c in disc.columns]
            log.info(
                "\n[DISCREPANCIES — model > 0.7 but ExoFOP says FP/FA]\n%s",
                disc[cols].to_string(index=False),
            )
        else:
            log.info("[discrepancies] none above prob > 0.7")


if __name__ == "__main__":
    main()
