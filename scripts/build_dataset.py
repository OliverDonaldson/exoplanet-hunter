"""End-to-end dataset build script.

Three stages:
  1. Build / refresh the labelled catalogue.
  2. Download light curves for every TIC.
  3. Clean, flatten, fold, and extract global+local views into a single
     `data/processed/views.npz`.

Idempotent — safe to re-run; downloads + processed views are cached.

Hydra entry point. Usage:

    python scripts/build_dataset.py                  # full dataset
    python scripts/build_dataset.py data=small       # tiny smoke set
"""

from __future__ import annotations

import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm

from exoplanet_hunter.data.catalog import CatalogRequest, build_label_catalog
from exoplanet_hunter.data.download import LightCurveDownloader
from exoplanet_hunter.data.stellar import fetch_stellar_params
from exoplanet_hunter.preprocess import build_views, clean_lightcurve, flatten_lightcurve
from exoplanet_hunter.utils import ProjectPaths, get_logger, set_global_seed

log = get_logger(__name__)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    paths = ProjectPaths.from_cfg(cfg)

    # --- Stage 1 — labelled catalogue ----------------------------------
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

    # Ensure every row has a mission column (backward compat with old catalogs).
    if "mission" not in catalog.columns:
        catalog["mission"] = "TESS"

    # --- Stage 2 — download light curves -------------------------------
    kepler_dir = paths.data_raw_kepler if paths.data_raw_kepler != paths.data_raw else None
    downloader = LightCurveDownloader(
        cache_dir=paths.data_raw,
        kepler_cache_dir=kepler_dir,
        author=str(cfg.data.author),
        cadence=int(cfg.data.cadence) if cfg.data.cadence else None,
    )
    results = downloader.download_many(
        catalog["tic_id"].tolist(),
        missions=catalog["mission"].tolist(),
    )
    success_ids = {(r.mission, r.target_id) for r in results if r.success}
    log.info("[build] %d/%d targets downloaded successfully", len(success_ids), len(catalog))

    # --- Stage 3 — preprocess into views -------------------------------
    import lightkurve as lk

    g_views: list[np.ndarray] = []
    l_views: list[np.ndarray] = []
    labels: list[int] = []
    tic_ids: list[int] = []
    aux: list[list[float]] = []

    # Track *why* targets get dropped — invaluable on a long overnight build.
    # Catalog units are now days throughout (see catalog._query_toi docstring).
    skips = {
        "no_download": 0,
        "missing_ephemeris": 0,
        "missing_fits": 0,
        "preprocess_error": 0,
    }

    for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc="processing"):
        tic = int(row["tic_id"])
        mission = row.get("mission", "TESS")
        if (mission, tic) not in success_ids:
            skips["no_download"] += 1
            continue
        period = row.get("period")
        t0 = row.get("t0")
        duration = row.get("duration")
        if (
            period is None
            or np.isnan(period)
            or t0 is None
            or np.isnan(t0)
            or duration is None
            or np.isnan(duration)
        ):
            skips["missing_ephemeris"] += 1
            continue

        # Resolve the FITS path based on mission.
        if mission == "Kepler":
            path = (kepler_dir or paths.data_raw) / f"kic_{tic}.fits"
        else:
            path = paths.data_raw / f"tic_{tic}.fits"
        if not path.exists():
            skips["missing_fits"] += 1
            continue
        try:
            lc = lk.read(str(path))
            lc = clean_lightcurve(lc, sigma_clip=float(cfg.preprocess.cleaning.sigma_clip))
            # Ephemeris is known from the catalog row; mask transits when
            # fitting the flattening spline so the dip itself survives.
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
            log.warning("[build] TIC %d: preprocessing failed — %s", tic, exc)
            skips["preprocess_error"] += 1
            continue

        # Aux feature vector (8 dims):
        #   [teff, radius, logg, tmag,  depth, duration, log_period, snr]
        # Stellar params come from the KOI catalog row for Kepler targets and
        # from a TIC lookup for TESS targets. SNR is only populated for KOI
        # rows (koi_model_snr); TESS rows leave it NaN until the TESS catalog
        # query is extended.
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
                    np.nan,
                ]
            )
        g_views.append(views.global_view)
        l_views.append(views.local_view)
        labels.append(int(row["label"]))
        tic_ids.append(tic)

    log.info(
        "[build] kept %d targets;  skipped: no_download=%d  missing_ephemeris=%d  "
        "missing_fits=%d  preprocess_error=%d",
        len(g_views),
        skips["no_download"],
        skips["missing_ephemeris"],
        skips["missing_fits"],
        skips["preprocess_error"],
    )

    if not g_views:
        log.error("[build] no usable targets — check downloads and label catalogue")
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
        "[build] wrote %d examples → %s  (pos=%d  neg=%d)",
        len(labels),
        out,
        int(np.sum(np.asarray(labels) == 1)),
        int(np.sum(np.asarray(labels) == 0)),
    )


if __name__ == "__main__":
    main()
