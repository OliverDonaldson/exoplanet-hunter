"""Render six-panel vetting figures for top-K candidates from scored parquet.

Reads ``results/candidates_scored.parquet`` (produced by score_candidates.py),
filters to high-confidence candidates, and saves one PNG per candidate to
``results/vetting/<tic_id>_<period>.png``.

Usage:
    # Default: top 30 by prob_mean, status='ok' only
    python scripts/render_vetting.py

    # Custom threshold + cap
    python scripts/render_vetting.py +prob_threshold=0.95 +top_k=50

    # Stricter: require fold-agreement too
    python scripts/render_vetting.py +prob_threshold=0.9 +fold_disagree_max=0.05
"""

from __future__ import annotations

import sys

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from tqdm.auto import tqdm

from exoplanet_hunter.data.download import LightCurveDownloader
from exoplanet_hunter.eval.vetting import CandidateReport, vetting_figure
from exoplanet_hunter.utils import ProjectPaths, get_logger, set_global_seed

log = get_logger(__name__)


@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    paths = ProjectPaths.from_cfg(cfg)

    # Prefer the enriched discovery shortlist (has TFOPWG disposition + follow-ups).
    # Fall back to the raw scored parquet if shortlist hasn't been built yet.
    shortlist_path = paths.root / "results" / "discovery_shortlist.parquet"
    scored_path = paths.root / str(getattr(cfg, "out_path", "results/candidates_scored.parquet"))
    if shortlist_path.exists():
        scored_path = shortlist_path
        log.info("[render-vetting] using enriched shortlist: %s", scored_path)
    out_dir = paths.results / "vetting"
    out_dir.mkdir(parents=True, exist_ok=True)

    top_k = int(getattr(cfg, "top_k", 30))
    prob_threshold = float(getattr(cfg, "prob_threshold", 0.85))
    prob_max = getattr(cfg, "prob_max", None)
    prob_max = float(prob_max) if prob_max is not None else None
    ascending = bool(getattr(cfg, "ascending", False))
    tic_ids_cfg = getattr(cfg, "tic_ids", None) or []
    tic_ids = [int(t) for t in tic_ids_cfg]
    fold_disagree_max = getattr(cfg, "fold_disagree_max", None)
    fold_disagree_max = float(fold_disagree_max) if fold_disagree_max is not None else None
    centroid_max = getattr(cfg, "centroid_max", None)
    centroid_max = float(centroid_max) if centroid_max is not None else None

    if not scored_path.exists():
        log.error(
            "[render-vetting] no scored parquet at %s — run score_candidates.py first", scored_path
        )
        sys.exit(2)

    scored = pd.read_parquet(scored_path)
    log.info("[render-vetting] loaded %d scored rows", len(scored))

    # Filter
    ok = scored[scored.status == "ok"].copy()
    if tic_ids:
        # Explicit TIC list — bypass prob filters; still apply fold/centroid if set.
        ok = ok[ok.tic_id.isin(tic_ids)]
        missing = sorted(set(tic_ids) - set(ok.tic_id.astype(int).tolist()))
        if missing:
            log.warning("[render-vetting] %d TIC IDs not in scored set: %s", len(missing), missing)
    else:
        ok = ok[ok.prob_mean >= prob_threshold]
        if prob_max is not None:
            ok = ok[ok.prob_mean <= prob_max]
    if fold_disagree_max is not None:
        ok = ok[ok.fold_disagree <= fold_disagree_max]
    if centroid_max is not None:
        ok = ok[(ok.centroid_snr.isna()) | (ok.centroid_snr <= centroid_max)]
    ok = ok.sort_values("prob_mean", ascending=ascending).head(top_k).reset_index(drop=True)
    if tic_ids:
        log.info(
            "[render-vetting] filtered to %d candidates by explicit tic_ids (sort=%s)",
            len(ok),
            "asc" if ascending else "desc",
        )
    else:
        log.info(
            "[render-vetting] filtered to %d candidates (%.2f ≤ prob%s%s%s, sort=%s)",
            len(ok),
            prob_threshold,
            f" ≤ {prob_max:.2f}" if prob_max is not None else "",
            f", fold_sigma <= {fold_disagree_max}" if fold_disagree_max else "",
            f", centroid ≤ {centroid_max}" if centroid_max else "",
            "asc" if ascending else "desc",
        )

    if len(ok) == 0:
        log.warning("[render-vetting] nothing to render")
        return

    tess_dl = LightCurveDownloader(paths.data_raw, author="SPOC", cadence=120)
    kepler_dl = LightCurveDownloader(
        paths.data_raw,
        kepler_cache_dir=paths.data_raw_kepler,
        author="Kepler",
        cadence=None,
    )

    import lightkurve as lk

    rendered: list[dict] = []
    for _, row in tqdm(ok.iterrows(), total=len(ok), desc="rendering"):
        tic = int(row["tic_id"])
        mission = str(row.get("mission", "TESS"))
        dl = kepler_dl if mission == "Kepler" else tess_dl

        res = dl.download_one(tic, mission=mission)
        if not res.success or res.path is None:
            log.warning("[render-vetting] tic=%d skipped — %s", tic, res.reason)
            continue

        try:
            raw = lk.read(str(res.path))
        except Exception as exc:
            log.warning("[render-vetting] tic=%d read failed: %s", tic, exc)
            continue

        report = CandidateReport(
            tic_id=tic,
            period=float(row["period"]),
            t0=float(row["t0"]),
            duration=float(row["duration"]),
            score=float(row["prob_mean"]),
            score_std=float(row["prob_std"]),
        )

        fold_means = row.get("fold_means")
        fold_means = list(fold_means) if isinstance(fold_means, list | np.ndarray) else None

        out_file = out_dir / f"tic_{tic}_p{report.period:.3f}.png"
        # Pull enrichment from shortlist if present.
        disposition = row.get("TFOPWG Disposition")
        if isinstance(disposition, float) and np.isnan(disposition):
            disposition = None
        n_followup = row.get("n_followup")
        n_followup = int(n_followup) if n_followup is not None and pd.notna(n_followup) else None
        try:
            vetting_figure(
                raw,
                report,
                out_file,
                fold_means=fold_means,
                prob_p10=row.get("prob_p10"),
                prob_p90=row.get("prob_p90"),
                fold_disagree=row.get("fold_disagree"),
                mc_disagree=row.get("mc_disagree"),
                mission=mission,
                disposition=disposition,
                n_followup=n_followup,
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
            log.exception("[render-vetting] tic=%d render failed: %s", tic, exc)

    log.info("[render-vetting] DONE  rendered=%d → %s", len(rendered), out_dir)
    if rendered:
        summary = pd.DataFrame(rendered)
        log.info("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
