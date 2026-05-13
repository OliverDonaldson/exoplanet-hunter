"""Preprocess already-downloaded light curves into views.npz — no network.

One-shot helper for the case where Stage 2 (download) is partially complete
and you want to train on whatever's on disk without waiting for the rest.

Logic: load the catalog and the downloader manifest; keep only rows where
the manifest reports ``success=True``; run Stage 3 (clean, flatten, fold,
build views) on those rows only; write ``data/processed/views.npz``.

Usage:
    python scripts/preprocess_only.py
    python scripts/preprocess_only.py data=large
"""

from __future__ import annotations

import json
import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm

from exoplanet_hunter.data.catalog import CatalogRequest, build_label_catalog
from exoplanet_hunter.data.stellar import fetch_stellar_params
from exoplanet_hunter.features.centroid import extract_centroid_offset
from exoplanet_hunter.preprocess import build_views, clean_lightcurve, flatten_lightcurve
from exoplanet_hunter.utils import ProjectPaths, get_logger, set_global_seed

log = get_logger(__name__)


def _load_success_keys(paths: ProjectPaths) -> set[tuple[str, int]]:
    """Read all manifests and return {(mission, target_id)} with success=True."""
    keys: set[tuple[str, int]] = set()
    for manifest_path in [
        paths.data_raw / "manifest.json",
        (paths.data_raw_kepler / "manifest.json") if paths.data_raw_kepler else None,
    ]:
        if not manifest_path or not manifest_path.exists():
            continue
        m = json.loads(manifest_path.read_text())
        for k, v in m.items():
            if not v.get("success"):
                continue
            if ":" in k:
                mission, tid = k.split(":", 1)
                keys.add((mission, int(tid)))
            else:
                keys.add(("TESS", int(k)))
    return keys


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    paths = ProjectPaths.from_cfg(cfg)

    # Safety check: if the config will request Kepler targets, the Kepler FITS
    # cache must actually be reachable. The KEPLER_RAW_DIR env var is the
    # canonical pointer; without it the path falls back to data/raw_kepler/
    # (typically empty), which silently produces a TESS-only dataset and looks
    # like missing_fits=2496 at the end of a 75-minute run. Fail loudly instead.
    n_kepler_total = int(cfg.data.get("n_confirmed_kepler", 0)) + int(
        cfg.data.get("n_false_pos_kepler", 0)
    )
    if n_kepler_total > 0:
        kepler_path = paths.data_raw_kepler
        if not kepler_path.exists() or not any(kepler_path.glob("kic_*.fits")):
            log.error(
                "[preprocess-only] config requests %d Kepler targets but the Kepler "
                "cache at %s is empty or missing.\n"
                "  Set KEPLER_RAW_DIR to the directory containing your kic_*.fits "
                "files before running, e.g.:\n"
                "    export KEPLER_RAW_DIR=/Volumes/SANDISK/exoplanet_kepler",
                n_kepler_total,
                kepler_path,
            )
            sys.exit(2)

    catalog = build_label_catalog(
        CatalogRequest(
            n_confirmed=int(cfg.data.n_confirmed),
            n_false_pos=int(cfg.data.n_false_pos),
            n_confirmed_kepler=int(cfg.data.get("n_confirmed_kepler", 0)),
            n_false_pos_kepler=int(cfg.data.get("n_false_pos_kepler", 0)),
            seed=int(cfg.data.seed),
        ),
        out_dir=paths.data_labels,
    )
    if "mission" not in catalog.columns:
        catalog["mission"] = "TESS"

    successes = _load_success_keys(paths)
    log.info("[preprocess-only] %d successful downloads across all manifests", len(successes))

    keep = catalog.apply(
        lambda r: (str(r["mission"]), int(r["tic_id"])) in successes,
        axis=1,
    )
    catalog = catalog[keep].reset_index(drop=True)
    log.info(
        "[preprocess-only] catalog filtered to %d rows with cached FITS on disk",
        len(catalog),
    )

    kepler_dir = paths.data_raw_kepler if paths.data_raw_kepler != paths.data_raw else None

    import lightkurve as lk

    g_views: list[np.ndarray] = []
    l_views: list[np.ndarray] = []
    labels: list[int] = []
    tic_ids: list[int] = []
    aux: list[list[float]] = []
    skips = {"missing_ephemeris": 0, "missing_fits": 0, "preprocess_error": 0}

    for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc="processing"):
        tic = int(row["tic_id"])
        mission = row.get("mission", "TESS")
        period, t0, duration = row.get("period"), row.get("t0"), row.get("duration")
        if any(v is None or np.isnan(v) for v in (period, t0, duration)):
            skips["missing_ephemeris"] += 1
            continue

        if mission == "Kepler":
            path = (kepler_dir or paths.data_raw) / f"kic_{tic}.fits"
        else:
            path = paths.data_raw / f"tic_{tic}.fits"
        if not path.exists():
            skips["missing_fits"] += 1
            continue

        try:
            lc = lk.read(str(path))
            # Compute centroid offset from the RAW lightkurve before clean/flatten
            # touches the MOM_CENTR1/2 columns. Wrap in try/except so a centroid
            # failure doesn't kill the whole row — NaN flows to the imputer.
            try:
                centroid_snr = extract_centroid_offset(
                    lc, float(period), float(t0), float(duration)
                )
            except Exception as cexc:
                log.debug("[preprocess-only] TIC %d centroid extract failed: %s", tic, cexc)
                centroid_snr = float("nan")
            lc = clean_lightcurve(lc, sigma_clip=float(cfg.preprocess.cleaning.sigma_clip))
            lc = flatten_lightcurve(
                lc,
                window_length=int(cfg.preprocess.flatten.window_length),
                polyorder=int(cfg.preprocess.flatten.polyorder),
                period=float(period),
                t0=float(t0),
                duration=float(duration),
            )
            views = build_views(
                lc,
                period=float(period),
                t0=float(t0),
                duration=float(duration),
                global_bins=int(cfg.preprocess.views.global_bins),
                local_bins=int(cfg.preprocess.views.local_bins),
                local_durations=float(cfg.preprocess.views.local_durations),
            )
        except Exception as exc:
            log.warning("[preprocess-only] TIC %d: %s", tic, exc)
            skips["preprocess_error"] += 1
            continue

        # Aux feature vector (9 dims):
        #   [teff, radius, logg, tmag,  depth, duration, log_period, snr, centroid_snr]
        #   stellar context (4) + transit shape (4) + centroid BEB diagnostic (1)
        # depth/duration/period come from the catalog (already computed).
        # SNR is available for Kepler (koi_model_snr); TESS targets use
        # the local-view peak depth as a cheap proxy until SNR is added
        # to the TESS catalog query.
        # centroid_snr is the 2D in-transit centroid shift in units of σ,
        # detrended for Kepler quarterly rolls (see features/centroid.py).
        log_period = np.log(float(period)) if float(period) > 0 else np.nan
        depth_val = float(row["depth"]) if pd.notna(row.get("depth")) else np.nan
        dur_val = float(row["duration"]) if pd.notna(row.get("duration")) else np.nan
        if mission == "Kepler":
            snr_val = float(row["snr"]) if pd.notna(row.get("snr")) else np.nan
            aux.append(
                [
                    float(row["teff"]) if pd.notna(row.get("teff")) else np.nan,
                    float(row["radius"]) if pd.notna(row.get("radius")) else np.nan,
                    float(row["logg"]) if pd.notna(row.get("logg")) else np.nan,
                    float(row["tmag"]) if pd.notna(row.get("tmag")) else np.nan,
                    depth_val,
                    dur_val,
                    log_period,
                    snr_val,
                    centroid_snr,
                ]
            )
        else:
            sp = fetch_stellar_params(tic)
            aux.append(
                [
                    sp.teff if sp.teff is not None else np.nan,
                    sp.radius if sp.radius is not None else np.nan,
                    sp.logg if sp.logg is not None else np.nan,
                    sp.tmag if sp.tmag is not None else np.nan,
                    depth_val,
                    dur_val,
                    log_period,
                    np.nan,  # transit SNR not yet in TESS catalog query
                    centroid_snr,
                ]
            )
        g_views.append(views.global_view)
        l_views.append(views.local_view)
        labels.append(int(row["label"]))
        tic_ids.append(tic)

    log.info(
        "[preprocess-only] kept=%d  missing_ephemeris=%d  missing_fits=%d  preprocess_error=%d",
        len(g_views),
        skips["missing_ephemeris"],
        skips["missing_fits"],
        skips["preprocess_error"],
    )
    if not g_views:
        log.error("[preprocess-only] no usable targets")
        sys.exit(1)

    out = paths.data_processed / "views.npz"
    np.savez_compressed(
        out,
        global_views=np.stack(g_views),
        local_views=np.stack(l_views),
        labels=np.asarray(labels, dtype=np.int8),
        tic_ids=np.asarray(tic_ids, dtype=np.int64),
        aux_features=np.asarray(aux, dtype=np.float32),
    )
    log.info(
        "[preprocess-only] wrote %d → %s  (pos=%d  neg=%d)",
        len(labels),
        out,
        int(np.sum(np.asarray(labels) == 1)),
        int(np.sum(np.asarray(labels) == 0)),
    )


if __name__ == "__main__":
    main()
