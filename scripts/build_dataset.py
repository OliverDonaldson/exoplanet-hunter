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
    python scripts/build_dataset.py data.n_quiet=0   # skip quiet stars
"""

from __future__ import annotations

import sys

import hydra
import numpy as np
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
            n_quiet=int(cfg.data.n_quiet),
            seed=int(cfg.data.seed),
        ),
        out_dir=paths.data_labels,
    )

    # --- Stage 2 — download light curves -------------------------------
    downloader = LightCurveDownloader(
        cache_dir=paths.data_raw,
        author=str(cfg.data.author),
        cadence=int(cfg.data.cadence) if cfg.data.cadence else None,
    )
    results = downloader.download_many(catalog["tic_id"].tolist())
    success_tics = {r.tic_id for r in results if r.success}
    log.info("[build] %d/%d TICs downloaded successfully", len(success_tics), len(catalog))

    # --- Stage 3 — preprocess into views -------------------------------
    import lightkurve as lk

    g_views: list[np.ndarray] = []
    l_views: list[np.ndarray] = []
    labels: list[int] = []
    tic_ids: list[int] = []
    aux: list[list[float]] = []

    for _, row in tqdm(catalog.iterrows(), total=len(catalog), desc="processing"):
        tic = int(row["tic_id"])
        if tic not in success_tics:
            continue
        period = row.get("period")
        t0 = row.get("t0")
        # duration in TOI table is hours; convert. PS table is days. Coerce
        # anything we don't trust to a nominal 0.1 day (won't be used for
        # training quiet stars — we skip them when no period is present).
        duration_raw = row.get("duration")
        if (period is None or np.isnan(period)) or (t0 is None or np.isnan(t0)):
            continue
        if duration_raw is None or np.isnan(duration_raw):
            continue
        # heuristic: if duration > 1 we assume hours (TOI), else days (PS).
        duration = float(duration_raw) / 24.0 if float(duration_raw) > 1.0 else float(duration_raw)

        path = paths.data_raw / f"tic_{tic}.fits"
        if not path.exists():
            continue
        try:
            lc = lk.read(str(path))
            lc = clean_lightcurve(lc, sigma_clip=float(cfg.preprocess.cleaning.sigma_clip))
            lc = flatten_lightcurve(
                lc,
                window_length=int(cfg.preprocess.flatten.window_length),
                polyorder=int(cfg.preprocess.flatten.polyorder),
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
            continue

        # Stellar features (best-effort; NaN if unavailable).
        sp = fetch_stellar_params(tic)
        aux.append(
            [
                sp.teff if sp.teff is not None else np.nan,
                sp.radius if sp.radius is not None else np.nan,
                sp.logg if sp.logg is not None else np.nan,
                sp.tmag if sp.tmag is not None else np.nan,
            ]
        )
        g_views.append(views.global_view)
        l_views.append(views.local_view)
        labels.append(int(row["label"]))
        tic_ids.append(tic)

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
