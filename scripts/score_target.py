"""Score a single TIC ID with a saved model.

Examples:

    # Score with the dual-view CNN (with MC Dropout uncertainty).
    python scripts/score_target.py tic_id=307210830

    # Force a re-download.
    python scripts/score_target.py tic_id=307210830 force_download=true

    # Score with the RF baseline instead.
    python scripts/score_target.py tic_id=307210830 model_type=rf

If no (period, t0, duration) is supplied via the command line, BLS is used
to estimate one from the cleaned light curve.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

from exoplanet_hunter.data.download import LightCurveDownloader
from exoplanet_hunter.preprocess import build_views, clean_lightcurve, flatten_lightcurve
from exoplanet_hunter.search import bls_period_search
from exoplanet_hunter.utils import ProjectPaths, get_logger, set_global_seed

log = get_logger(__name__)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    paths = ProjectPaths.from_cfg(cfg)

    # CLI-only fields, with defaults pulled from cfg.
    tic_id = int(getattr(cfg, "tic_id", 0))
    if not tic_id:
        log.error("usage: scripts/score_target.py tic_id=<TIC>")
        sys.exit(2)
    model_type = str(getattr(cfg, "model_type", "cnn"))  # "cnn" | "rf"
    force_dl = bool(getattr(cfg, "force_download", False))
    period = getattr(cfg, "period", None)
    t0 = getattr(cfg, "t0", None)
    duration_h = getattr(cfg, "duration_h", None)
    n_mc = int(getattr(cfg, "n_mc", 50))

    # --- Download + clean ----------------------------------------------
    import lightkurve as lk

    dl = LightCurveDownloader(paths.data_raw, author="SPOC", cadence=120)
    res = dl.download_one(tic_id, force=force_dl)
    if not res.success or res.path is None:
        log.error("[score] no SPOC light curve for TIC %d (%s)", tic_id, res.reason)
        sys.exit(1)

    lc = lk.read(str(res.path))
    lc = clean_lightcurve(lc, sigma_clip=float(cfg.preprocess.cleaning.sigma_clip))
    lc = flatten_lightcurve(
        lc,
        window_length=int(cfg.preprocess.flatten.window_length),
        polyorder=int(cfg.preprocess.flatten.polyorder),
    )

    # --- Period search if needed ---------------------------------------
    if period is None or t0 is None or duration_h is None:
        log.info("[score] running BLS period search ...")
        bls = bls_period_search(lc)
        period = float(bls.period)
        t0 = float(bls.t0)
        duration = float(bls.duration)
        log.info(
            "[score] BLS best: P=%.4f d  t0=%.4f  dur=%.3f d  SNR=%.2f",
            period,
            t0,
            duration,
            bls.snr,
        )
    else:
        period = float(period)
        t0 = float(t0)
        duration = float(duration_h) / 24.0

    views = build_views(
        lc,
        period=period,
        t0=t0,
        duration=duration,
        global_bins=int(cfg.preprocess.views.global_bins),
        local_bins=int(cfg.preprocess.views.local_bins),
        local_durations=float(cfg.preprocess.views.local_durations),
    )

    # --- Score ----------------------------------------------------------
    if model_type == "cnn":
        import tensorflow as tf

        from exoplanet_hunter.models.uncertainty import mc_dropout_predict

        ckpt = Path(paths.models / "cnn_dualview.keras")
        if not ckpt.exists():
            log.error("[score] no model at %s — run training first", ckpt)
            sys.exit(1)
        model = tf.keras.models.load_model(str(ckpt), compile=False)

        inputs = {
            "global_view": views.global_view[None, :, None].astype(np.float32),
            "local_view": views.local_view[None, :, None].astype(np.float32),
        }
        result = mc_dropout_predict(model, inputs, n_samples=n_mc)
        log.info(
            "[score] TIC %d  P=%.4f d  →  prob = %.3f ± %.3f  (MC dropout n=%d)",
            tic_id,
            period,
            float(result.mean),
            float(result.std),
            n_mc,
        )
    elif model_type == "rf":
        import joblib

        from exoplanet_hunter.features import extract_features

        ckpt = Path(paths.models / "random_forest.joblib")
        if not ckpt.exists():
            log.error("[score] no model at %s — run RF training first", ckpt)
            sys.exit(1)
        pipeline = joblib.load(ckpt)
        feats = extract_features(views.global_view).reshape(1, -1)
        prob = float(pipeline.predict_proba(feats)[0, 1])
        log.info("[score] TIC %d  P=%.4f d  →  prob = %.3f  (random forest)", tic_id, period, prob)
    else:
        log.error("[score] unknown model_type=%s (use 'cnn' or 'rf')", model_type)
        sys.exit(2)


if __name__ == "__main__":
    main()
