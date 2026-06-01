"""Render attention diagnostics (SE + MHA) for top-K candidates.

Companion to ``scripts/render_vetting.py``. Same candidate selection, but
instead of the astronomer six-panel figure this loads the trained dual-view CNN,
extracts the Squeeze-and-Excitation channel gates and Multi-Head Attention
scores at inference time, and saves one attention figure per candidate to
``results/attention/<tic_id>_<period>.png``.

Reads the enriched ``results/discovery_shortlist.parquet`` if present, else
``results/candidates_scored.parquet`` (from score_candidates.py).

Usage:
    # Default: top 30 by prob_mean, status='ok', single final model
    python scripts/render_attention.py

    # Custom threshold + cap (keys exist in conf/config.yaml — no '+')
    python scripts/render_attention.py prob_threshold=0.95 top_k=50

    # Specific targets, bypassing prob filters
    python scripts/render_attention.py 'tic_ids=[150428135,307210830]'

    # Use a specific fold model instead of the final single model
    python scripts/render_attention.py +attn_model=models/cv/<HASH>/fold_0/cnn_dualview.keras
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm

from exoplanet_hunter.data.download import LightCurveDownloader
from exoplanet_hunter.eval.attention_diagnostic import attention_figure, extract_attention
from exoplanet_hunter.eval.vetting import CandidateReport
from exoplanet_hunter.preprocess import build_views, clean_lightcurve, flatten_lightcurve
from exoplanet_hunter.utils import ProjectPaths, get_logger, set_global_seed

log = get_logger(__name__)

# The single ``models/cnn_dualview.keras`` is a stale pre-SE/MHA baseline — it
# has no attention layers to extract. The branch-3 architecture (SE + MHA) lives
# in the CV fold bundles that score_candidates.py uses, so default to fold_0 of
# the same ensemble. Override with ``+attn_model=models/cv/<HASH>/fold_k/...``.
DEFAULT_CV_DIR = "models/cv/58570d85f1dd4f68a7e888988c88eeab"
DEFAULT_ATTN_MODEL = f"{DEFAULT_CV_DIR}/fold_0/cnn_dualview.keras"


def _local_phase_half(duration: float, period: float, local_durations: float) -> float:
    """Half-width of the local view in phase units — matches preprocess.build_views."""
    half = local_durations * duration / period
    return float(min(max(half, 1e-3), 0.5))


def _select(scored: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Same filtering/ordering as render_vetting.py so the two figure sets line up."""
    top_k = int(getattr(cfg, "top_k", 30))
    prob_threshold = float(getattr(cfg, "prob_threshold", 0.85))
    prob_max = getattr(cfg, "prob_max", None)
    prob_max = float(prob_max) if prob_max is not None else None
    ascending = bool(getattr(cfg, "ascending", False))
    tic_ids = [int(t) for t in (getattr(cfg, "tic_ids", None) or [])]
    fold_disagree_max = getattr(cfg, "fold_disagree_max", None)
    fold_disagree_max = float(fold_disagree_max) if fold_disagree_max is not None else None
    centroid_max = getattr(cfg, "centroid_max", None)
    centroid_max = float(centroid_max) if centroid_max is not None else None

    ok = scored[scored.status == "ok"].copy()
    if tic_ids:
        ok = ok[ok.tic_id.isin(tic_ids)]
        missing = sorted(set(tic_ids) - set(ok.tic_id.astype(int).tolist()))
        if missing:
            log.warning(
                "[render-attention] %d TIC IDs not in scored set: %s", len(missing), missing
            )
    else:
        ok = ok[ok.prob_mean >= prob_threshold]
        if prob_max is not None:
            ok = ok[ok.prob_mean <= prob_max]
    if fold_disagree_max is not None:
        ok = ok[ok.fold_disagree <= fold_disagree_max]
    if centroid_max is not None:
        ok = ok[(ok.centroid_snr.isna()) | (ok.centroid_snr <= centroid_max)]
    return ok.sort_values("prob_mean", ascending=ascending).head(top_k).reset_index(drop=True)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    paths = ProjectPaths.from_cfg(cfg)

    # Heavy import deferred until after CLI parse, so --help etc. stay snappy.
    import tensorflow as tf

    shortlist_path = paths.root / "results" / "discovery_shortlist.parquet"
    scored_path = paths.root / str(getattr(cfg, "out_path", "results/candidates_scored.parquet"))
    if shortlist_path.exists():
        scored_path = shortlist_path
        log.info("[render-attention] using enriched shortlist: %s", scored_path)
    if not scored_path.exists():
        log.error(
            "[render-attention] no scored parquet at %s — run score_candidates.py first",
            scored_path,
        )
        sys.exit(2)

    out_dir = paths.results / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    scored = pd.read_parquet(scored_path)
    log.info("[render-attention] loaded %d scored rows", len(scored))
    ok = _select(scored, cfg)
    log.info("[render-attention] selected %d candidates to render", len(ok))
    if len(ok) == 0:
        log.warning("[render-attention] nothing to render")
        return

    # ----- model -------------------------------------------------------------
    model_arg = str(getattr(cfg, "attn_model", DEFAULT_ATTN_MODEL))
    model_path = Path(model_arg) if Path(model_arg).is_absolute() else paths.root / model_arg
    if not model_path.exists():
        log.error("[render-attention] model not found: %s", model_path)
        sys.exit(2)
    log.info("[render-attention] loading attention model: %s", model_path)
    model = tf.keras.models.load_model(str(model_path), compile=False)

    # ----- preprocessing knobs (match score_candidates.py) -------------------
    sigma_clip = float(cfg.preprocess.cleaning.sigma_clip)
    flat_win = int(cfg.preprocess.flatten.window_length)
    flat_poly = int(cfg.preprocess.flatten.polyorder)
    global_bins = int(cfg.preprocess.views.global_bins)
    local_bins = int(cfg.preprocess.views.local_bins)
    local_durations = float(cfg.preprocess.views.local_durations)

    tess_dl = LightCurveDownloader(paths.data_raw, author="SPOC", cadence=120)
    kepler_dl = LightCurveDownloader(
        paths.data_raw, kepler_cache_dir=paths.data_raw_kepler, author="Kepler", cadence=None
    )

    import lightkurve as lk

    rendered: list[dict] = []
    for _, row in tqdm(ok.iterrows(), total=len(ok), desc="attention"):
        tic = int(row["tic_id"])
        mission = str(row.get("mission", "TESS"))
        period = float(row["period"])
        t0 = float(row["t0"])
        duration = float(row["duration"])
        dl = kepler_dl if mission == "Kepler" else tess_dl

        res = dl.download_one(tic, mission=mission)
        if not res.success or res.path is None:
            log.warning("[render-attention] tic=%d skipped — %s", tic, res.reason)
            continue
        try:
            raw = lk.read(str(res.path))
            cleaned = clean_lightcurve(raw, sigma_clip=sigma_clip)
            flat = flatten_lightcurve(
                cleaned,
                window_length=flat_win,
                polyorder=flat_poly,
                period=period,
                t0=t0,
                duration=duration,
            )
            views = build_views(
                flat,
                period=period,
                t0=t0,
                duration=duration,
                global_bins=global_bins,
                local_bins=local_bins,
                local_durations=local_durations,
            )
        except Exception as exc:
            log.warning("[render-attention] tic=%d preprocess failed: %s", tic, exc)
            continue

        try:
            maps = extract_attention(
                model,
                views.global_view,
                views.local_view,
                local_phase_half=_local_phase_half(duration, period, local_durations),
            )
            report = CandidateReport(
                tic_id=tic,
                period=period,
                t0=t0,
                duration=duration,
                score=float(row["prob_mean"]),
                score_std=float(row.get("prob_std", float("nan"))),
            )
            disposition = row.get("TFOPWG Disposition")
            if isinstance(disposition, float) and np.isnan(disposition):
                disposition = None
            out_file = out_dir / f"tic_{tic}_p{period:.3f}.png"
            attention_figure(
                maps,
                report,
                out_file,
                global_view=views.global_view,
                local_view=views.local_view,
                mission=mission,
                disposition=disposition,
            )
            rendered.append(
                {
                    "tic_id": tic,
                    "mission": mission,
                    "prob_mean": float(row["prob_mean"]),
                    "path": str(out_file),
                }
            )
        except Exception as exc:
            log.exception("[render-attention] tic=%d attention/render failed: %s", tic, exc)

    log.info("[render-attention] DONE  rendered=%d → %s", len(rendered), out_dir)
    if rendered:
        log.info("\n" + pd.DataFrame(rendered).to_string(index=False))


if __name__ == "__main__":
    main()
