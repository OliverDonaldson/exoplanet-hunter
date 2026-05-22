"""Per-candidate vetting plots.

Six-panel diagnostic figure that astronomers use to triage a TOI:

  1. Phase-folded global view (full phase range).
  2. Phase-folded local view (zoomed on transit).
  3. Odd vs even transit overlay (large Δdepth → eclipsing binary).
  4. BLS periodogram (look for harmonics, dominant alternate periods).
  5. Centroid shift (large shift → background eclipsing binary, BEB).
  6. Ensemble probability with MC-Dropout + per-fold uncertainty.

Inputs are a raw lightkurve object (must still contain MOM_CENTR1/2 for the
centroid panel) and a CandidateReport carrying ephemeris + ensemble score.
Optional uncertainty fields (fold_means, prob_p10/p90, fold_disagree,
mc_disagree) come from ``scripts/score_candidates.py`` output.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lightkurve import LightCurve
from matplotlib.axes import Axes

from exoplanet_hunter.features.centroid import extract_centroid_features
from exoplanet_hunter.preprocess import clean_lightcurve, flatten_lightcurve


@dataclass(frozen=True)
class CandidateReport:
    tic_id: int
    period: float
    t0: float
    duration: float
    score: float
    score_std: float


# ---------- helpers ----------------------------------------------------------


def _phase_fold(
    time: np.ndarray, flux: np.ndarray, period: float, t0: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase ∈ [-0.5, 0.5], same convention as lk.LightCurve.fold."""
    mask = np.isfinite(time) & np.isfinite(flux)
    t, f = time[mask], flux[mask]
    phase = ((t - t0) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    n_transit = np.round((t - t0) / period).astype(int)
    return phase, f, n_transit


def _bin_median(
    phase: np.ndarray, flux: np.ndarray, n_bins: int, lo: float = -0.5, hi: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(phase, edges) - 1
    out = np.full(n_bins, np.nan)
    for i in range(n_bins):
        sel = idx == i
        if sel.any():
            out[i] = float(np.median(flux[sel]))
    return centers, out


# ---------- panels -----------------------------------------------------------


def _panel_global(
    ax: Axes, phase: np.ndarray, flux: np.ndarray, period: float, duration: float
) -> None:
    ax.scatter(phase, flux, s=1, alpha=0.10, color="gray", rasterized=True)
    centers, binned = _bin_median(phase, flux, n_bins=201)
    ax.plot(centers, binned, color="C0", linewidth=1.4)
    half = duration / period / 2.0
    ax.axvspan(-half, half, color="red", alpha=0.08)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalised flux")
    ax.set_title("Phase-folded global view")


def _panel_local(
    ax: Axes, phase: np.ndarray, flux: np.ndarray, period: float, duration: float
) -> None:
    half = duration / period / 2.0
    local_w = max(4 * half, 0.005)
    sel = np.abs(phase) < local_w
    ax.scatter(phase[sel], flux[sel], s=3, alpha=0.30, color="gray", rasterized=True)
    centers, binned = _bin_median(phase[sel], flux[sel], n_bins=60, lo=-local_w, hi=local_w)
    ax.plot(centers, binned, color="C0", linewidth=1.8)
    ax.axvspan(-half, half, color="red", alpha=0.08)
    ax.set_xlim(-local_w, local_w)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalised flux")
    ax.set_title("Local view (±2 durations)" if half > 0 else "Local view")


def _panel_odd_even(
    ax: Axes,
    phase: np.ndarray,
    flux: np.ndarray,
    n_transit: np.ndarray,
    period: float,
    duration: float,
) -> None:
    half = duration / period / 2.0
    local_w = max(4 * half, 0.005)
    is_odd = (n_transit % 2 == 1) & (np.abs(phase) < local_w)
    is_even = (n_transit % 2 == 0) & (np.abs(phase) < local_w)
    ax.scatter(
        phase[is_odd],
        flux[is_odd],
        s=3,
        alpha=0.5,
        color="C1",
        label=f"odd (n={int(is_odd.sum())})",
    )
    ax.scatter(
        phase[is_even],
        flux[is_even],
        s=3,
        alpha=0.5,
        color="C2",
        label=f"even (n={int(is_even.sum())})",
    )
    in_odd = is_odd & (np.abs(phase) < half)
    in_even = is_even & (np.abs(phase) < half)
    depth_odd = 1.0 - float(np.median(flux[in_odd])) if in_odd.any() else float("nan")
    depth_even = 1.0 - float(np.median(flux[in_even])) if in_even.any() else float("nan")
    if np.isfinite(depth_odd):
        ax.axhline(1 - depth_odd, color="C1", ls="--", alpha=0.7)
    if np.isfinite(depth_even):
        ax.axhline(1 - depth_even, color="C2", ls="--", alpha=0.7)
    delta = (
        abs(depth_odd - depth_even)
        if np.isfinite(depth_odd) and np.isfinite(depth_even)
        else float("nan")
    )
    ax.axvspan(-half, half, color="red", alpha=0.06)
    ax.set_xlim(-local_w, local_w)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalised flux")
    ax.set_title(f"Odd vs even  (Δdepth = {delta:.4f})" if np.isfinite(delta) else "Odd vs even")
    ax.legend(loc="lower right", fontsize=8)


def _panel_bls(
    ax: Axes, time: np.ndarray, flux: np.ndarray, duration: float, candidate_period: float
) -> None:
    from astropy.timeseries import BoxLeastSquares

    finite = np.isfinite(time) & np.isfinite(flux)
    bls = BoxLeastSquares(time[finite], flux[finite])
    baseline = float(time[finite].max() - time[finite].min())
    pmin = 0.5
    pmax = min(30.0, max(2.0, baseline / 3.0))
    periods = np.geomspace(pmin, pmax, 4000)
    try:
        result = bls.power(periods, duration)
        power = np.asarray(result.power, dtype=float)
    except Exception:
        ax.text(
            0.5, 0.5, "BLS failed", transform=ax.transAxes, ha="center", va="center", color="gray"
        )
        ax.set_title("BLS periodogram")
        return
    ax.plot(periods, power, color="C0", linewidth=0.8)
    ax.axvline(
        candidate_period,
        color="red",
        ls="--",
        alpha=0.7,
        label=f"candidate  P={candidate_period:.4f} d",
    )
    for harm, name in [(0.5, "½×"), (2.0, "2×"), (3.0, "3×")]:  # noqa: RUF001  (figure labels)
        ph = candidate_period * harm
        if pmin <= ph <= pmax:
            ax.axvline(ph, color="red", ls=":", alpha=0.3)
            ax.text(
                ph,
                ax.get_ylim()[1] * 0.95,
                name,
                color="red",
                alpha=0.5,
                fontsize=8,
                ha="center",
                va="top",
            )
    ax.set_xscale("log")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("BLS power")
    ax.set_title("BLS periodogram")
    ax.legend(loc="upper right", fontsize=8)


def _panel_centroid(
    ax: Axes, raw_lc: LightCurve, period: float, t0: float, duration: float
) -> None:
    try:
        cf = extract_centroid_features(raw_lc, period, t0, duration)
        cs_x = float(cf["centroid_shift_x"])
        cs_y = float(cf["centroid_shift_y"])
        cs_snr = float(cf["centroid_snr"])
    except Exception:
        cs_x = cs_y = cs_snr = float("nan")

    if not (np.isfinite(cs_x) and np.isfinite(cs_y)):
        ax.text(
            0.5,
            0.5,
            "Centroid data unavailable\n(MOM_CENTR missing or transit too short)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Centroid shift")
        return

    extent = max(0.05, 1.5 * np.hypot(cs_x, cs_y))
    ax.scatter([0], [0], marker="*", color="C0", s=180, label="Target", zorder=3)
    ax.scatter([cs_x], [cs_y], marker="x", color="C3", s=80, label="In-transit centroid", zorder=3)
    ax.annotate(
        "", xy=(cs_x, cs_y), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="C3", alpha=0.6)
    )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("column shift (pixels)")
    ax.set_ylabel("row shift (pixels)")
    flag = "  ⚠ BEB candidate" if cs_snr > 3.0 else ""
    ax.set_title(f"Centroid shift  ·  SNR = {cs_snr:.2f}{flag}")
    ax.legend(loc="upper right", fontsize=8)


def _panel_probability(
    ax: Axes,
    report: CandidateReport,
    fold_means: Sequence[float] | None,
    prob_p10: float | None,
    prob_p90: float | None,
    fold_disagree: float | None,
    mc_disagree: float | None,
    threshold: float,
) -> None:
    score = report.score
    sigma = report.score_std

    ax.errorbar(
        [0],
        [score],
        yerr=[sigma if np.isfinite(sigma) else 0.0],
        fmt="o",
        color="C0",
        markersize=14,
        capsize=10,
        lw=2,
        label=f"Ensemble mean = {score:.3f}",
    )

    if fold_means is not None and len(fold_means) > 0:
        xs = np.linspace(-0.2, 0.2, len(fold_means))
        ax.scatter(
            xs,
            list(fold_means),
            marker="x",
            color="C2",
            s=80,
            lw=2,
            label=f"Per-fold (k={len(fold_means)})",
        )

    if (
        prob_p10 is not None
        and prob_p90 is not None
        and np.isfinite(prob_p10)
        and np.isfinite(prob_p90)
    ):
        ax.fill_between(
            [-0.45, 0.45],
            prob_p10,
            prob_p90,
            color="C0",
            alpha=0.10,
            label=f"p10–p90 ({prob_p10:.2f}–{prob_p90:.2f})",  # noqa: RUF001
        )

    ax.axhline(threshold, color="red", ls="--", alpha=0.6, label=f"threshold = {threshold:.2f}")

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_ylabel("P(planet)")
    bits = [f"prob = {score:.3f}"]
    if np.isfinite(sigma):
        bits.append(f"σ = {sigma:.3f}")  # noqa: RUF001
    if fold_disagree is not None and np.isfinite(fold_disagree):
        bits.append(f"fold σ = {fold_disagree:.3f}")  # noqa: RUF001
    if mc_disagree is not None and np.isfinite(mc_disagree):
        bits.append(f"mc σ = {mc_disagree:.3f}")  # noqa: RUF001
    ax.set_title("  ·  ".join(bits))
    ax.legend(loc="lower left", fontsize=8)


# ---------- main entry -------------------------------------------------------


def vetting_figure(
    lc: LightCurve,
    report: CandidateReport,
    out_path: Path,
    *,
    flat_lc: LightCurve | None = None,
    fold_means: Sequence[float] | None = None,
    prob_p10: float | None = None,
    prob_p90: float | None = None,
    fold_disagree: float | None = None,
    mc_disagree: float | None = None,
    threshold: float = 0.5,
    mission: str = "TESS",
    disposition: str | None = None,
    n_followup: int | None = None,
    sigma_clip: float = 5.0,
    flatten_window: int = 301,
    flatten_polyorder: int = 3,
) -> Path:
    """Generate and save a six-panel vetting figure.

    Parameters
    ----------
    lc
        Raw lightkurve object (must still carry MOM_CENTR1/2 for the
        centroid panel). Cleaned + flattened internally for the phase-folded
        panels unless ``flat_lc`` is supplied.
    report
        CandidateReport with ephemeris + ensemble score.
    out_path
        Where to save the PNG.
    flat_lc
        Pre-flattened lightcurve. If supplied, we skip clean+flatten here.
    fold_means, prob_p10, prob_p90, fold_disagree, mc_disagree
        Uncertainty fields from ``scripts/score_candidates.py`` parquet.
    threshold
        Decision boundary line drawn on the probability panel.
    mission
        Label for the figure title.
    """
    import matplotlib.pyplot as plt

    period = float(report.period)
    t0 = float(report.t0)
    duration = float(report.duration)

    if flat_lc is None:
        cleaned = clean_lightcurve(lc, sigma_clip=sigma_clip)
        flat = flatten_lightcurve(
            cleaned,
            window_length=flatten_window,
            polyorder=flatten_polyorder,
            period=period,
            t0=t0,
            duration=duration,
        )
    else:
        flat = flat_lc

    time = np.asarray(flat.time.value, dtype=float)
    flux = np.asarray(flat.flux.value, dtype=float)
    phase, flux_f, n_transit = _phase_fold(time, flux, period, t0)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    (ax_g, ax_l), (ax_oe, ax_bls), (ax_cen, ax_prob) = axes

    _panel_global(ax_g, phase, flux_f, period, duration)
    _panel_local(ax_l, phase, flux_f, period, duration)
    _panel_odd_even(ax_oe, phase, flux_f, n_transit, period, duration)
    _panel_bls(ax_bls, time, flux, duration, period)
    _panel_centroid(ax_cen, lc, period, t0, duration)
    _panel_probability(
        ax_prob, report, fold_means, prob_p10, prob_p90, fold_disagree, mc_disagree, threshold
    )

    title = f"TIC {report.tic_id}"
    if mission:
        title += f"  ({mission})"
    title += f"  ·  P = {period:.4f} d  ·  T0 = {t0:.3f}  ·  duration = {duration * 24:.2f} h"
    subtitle_bits: list[str] = []
    if disposition:
        subtitle_bits.append(f"TFOPWG = {disposition}")
    if n_followup is not None:
        subtitle_bits.append(f"follow-ups = {int(n_followup)}")
    if subtitle_bits:
        title += "\n" + "  ·  ".join(subtitle_bits)
    fig.suptitle(title, fontsize=12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
