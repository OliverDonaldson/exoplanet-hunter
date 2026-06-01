"""Attention-map diagnostics for the dual-view CNN (SE + bilateral MHA).

The branch-3 architecture (``models/cnn_dualview.py``) carries two interpretable
attention mechanisms that are normally invisible at inference:

  * **Squeeze-and-Excitation** channel gates — one sigmoid weight per conv
    channel, per block, per tower (``{tower}_b{block}_se_excite``). Reading
    these shows *which feature channels the model amplifies vs. suppresses*;
    the hope is that high-frequency stellar-noise channels are systematically
    down-weighted.
  * **Multi-Head Attention** scores — the full (heads, query, key) softmax map
    at the end of each conv tower (``{tower}_mha``). Reading these shows *where
    along phase the model looks*; the hope is that temporal attention spikes at
    transit ingress/egress.

Neither is exposed by a forward pass — the model only returns P(planet). This
module rebuilds a lightweight extractor that taps the SE excitation tensors and
re-runs each ``MultiHeadAttention`` layer with ``return_attention_scores=True``
on the *same trained weights*, then renders a six-panel diagnostic that mirrors
``eval/vetting.py`` in spirit.

Aux stellar features are irrelevant here: SE and MHA both live entirely inside
the conv towers, upstream of the aux concatenation, so the extractor only needs
the global + local views (aux can be omitted).

Design note: kept separate from ``vetting.py`` on purpose. The six-panel vetting
figure is an *astronomer* triage tool that works from a raw light curve and
never loads the network; this is an *ML interpretability* tool that needs the
trained model. Keeping them apart avoids coupling the triage path to TensorFlow.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from exoplanet_hunter.eval.vetting import CandidateReport

if TYPE_CHECKING:  # avoid importing TF/Keras at module import time
    import tensorflow as tf


# ---------- data containers --------------------------------------------------


@dataclass(frozen=True)
class TowerAttention:
    """Extracted attention for one conv tower (``global`` or ``local``)."""

    tower: str
    se: dict[int, np.ndarray]  # block_idx -> (C,) sigmoid channel gates in [0, 1]
    mha_scores: np.ndarray  # (heads, T_query, T_key) softmax weights
    phase_min: float  # phase at the left edge of this view
    phase_max: float  # phase at the right edge of this view

    @property
    def feat_len(self) -> int:
        """Temporal length T of the pre-MHA feature map (post-pooling)."""
        return int(self.mha_scores.shape[-1])

    @property
    def n_heads(self) -> int:
        return int(self.mha_scores.shape[0])

    @property
    def head_mean(self) -> np.ndarray:
        """Head-averaged attention matrix, shape (T_query, T_key)."""
        return self.mha_scores.mean(axis=0)

    @property
    def attention_received(self) -> np.ndarray:
        """How much each key position is attended to, averaged over heads and
        queries. Shape (T_key,). This is the 'temporal importance' profile."""
        return self.mha_scores.mean(axis=(0, 1))

    def phase_centers(self) -> np.ndarray:
        """Phase coordinate of each of the T feature-map bins (bin centres)."""
        edges = np.linspace(self.phase_min, self.phase_max, self.feat_len + 1)
        return 0.5 * (edges[:-1] + edges[1:])


@dataclass(frozen=True)
class AttentionMaps:
    global_tower: TowerAttention
    local_tower: TowerAttention


# ---------- extraction -------------------------------------------------------

# Cache the tap model per network so re-rendering a top-K list doesn't rebuild
# the functional graph for every candidate. Keyed by id(model) within a process.
_EXTRACTOR_CACHE: dict[int, tuple[Any, dict[str, dict[int, str]]]] = {}

_SE_RE = re.compile(r"(global|local)_b(\d+)_se_excite$")


def clear_extractor_cache() -> None:
    """Drop cached tap models (call if you reload a model under the same id)."""
    _EXTRACTOR_CACHE.clear()


def _input_by_substr(model: tf.keras.Model, sub: str) -> Any:
    """Return the model input tensor whose name (or producing layer) matches."""
    sub = sub.lower()
    for t in model.inputs:
        if sub in getattr(t, "name", "").lower():
            return t
    for t in model.inputs:  # fallback via keras history
        try:
            if sub in t._keras_history[0].name.lower():
                return t
        except Exception:
            continue
    raise ValueError(f"no model input matching {sub!r}; have {[t.name for t in model.inputs]}")


def _mha_feature_tensor(model: tf.keras.Model, mha_name: str) -> Any:
    """Tensor feeding a MultiHeadAttention layer.

    The layer is called as ``mha(x, x, x)`` so Keras records ``.input`` as a
    list of three identical query/value/key tensors — any one is the conv-tower
    feature map (B, T, C) we want to re-run attention on.
    """
    inp = model.get_layer(mha_name).input
    return inp[0] if isinstance(inp, list | tuple) else inp


def _build_extractor(model: tf.keras.Model) -> tuple[Any, dict[str, dict[int, str]]]:
    """Build (and memoise) a tap model exposing SE gates + pre-MHA feature maps.

    Returns ``(extractor, se_map)`` where ``se_map[tower][block] = layer_name``.
    The extractor takes ``[global_view, local_view]`` (aux is not needed — it
    only enters downstream of the towers) and returns a dict of named tensors.
    """
    from tensorflow.keras import Model

    key = id(model)
    if key in _EXTRACTOR_CACHE:
        return _EXTRACTOR_CACHE[key]

    g_in = _input_by_substr(model, "global")
    l_in = _input_by_substr(model, "local")

    outputs: dict[str, Any] = {
        "global_feat": _mha_feature_tensor(model, "global_mha"),
        "local_feat": _mha_feature_tensor(model, "local_mha"),
    }
    se_map: dict[str, dict[int, str]] = {"global": {}, "local": {}}
    for layer in model.layers:
        m = _SE_RE.match(layer.name)
        if m:
            tower, block = m.group(1), int(m.group(2))
            se_map[tower][block] = layer.name
            outputs[layer.name] = layer.output

    extractor = Model([g_in, l_in], outputs, name="attention_extractor")
    _EXTRACTOR_CACHE[key] = (extractor, se_map)
    return extractor, se_map


def _as_model_input(view: np.ndarray, expected_len: int | None = None) -> np.ndarray:
    """Coerce a 1-D view (L,) or (L,1) or (1,L,1) to a (1, L, 1) float32 batch."""
    arr = np.asarray(view, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"expected a 1-D view, got shape {np.asarray(view).shape}")
    if expected_len is not None and arr.shape[0] != expected_len:
        raise ValueError(f"view length {arr.shape[0]} != model expects {expected_len}")
    return arr[None, :, None]


def extract_attention(
    model: tf.keras.Model,
    global_view: np.ndarray,
    local_view: np.ndarray,
    *,
    local_phase_half: float,
    global_phase: tuple[float, float] = (-0.5, 0.5),
    training: bool = False,
) -> AttentionMaps:
    """Extract SE channel gates + MHA attention scores for one candidate.

    Parameters
    ----------
    model
        Trained dual-view CNN (``build_cnn_dualview``). Loaded once and reused;
        the internal tap model is cached per ``id(model)``.
    global_view, local_view
        The exact arrays fed to the network at inference — the median-normalised
        phase-folded views from ``preprocess.build_views`` (lengths 2001 / 201
        by default). 1-D; reshaped to ``(1, L, 1)`` internally.
    local_phase_half
        Half-width of the local view in *phase* units, i.e.
        ``local_durations * duration / period`` (clamped to [1e-3, 0.5]). The
        local view spans ``[-local_phase_half, +local_phase_half]``.
    global_phase
        Phase span of the global view (always the full fold by default).
    training
        Forward-pass mode for the re-run MHA layers. ``False`` gives the
        deterministic attention map (MHA dropout off); leave it False for
        diagnostics.

    Returns
    -------
    AttentionMaps with a ``TowerAttention`` for each tower.
    """
    extractor, se_map = _build_extractor(model)

    gv = _as_model_input(global_view)
    lv = _as_model_input(local_view)

    feats = extractor([gv, lv], training=False)

    def _np(x: Any) -> np.ndarray:
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

    g_feat = feats["global_feat"]
    l_feat = feats["local_feat"]

    g_mha = model.get_layer("global_mha")
    l_mha = model.get_layer("local_mha")
    _, g_scores = g_mha(g_feat, g_feat, g_feat, return_attention_scores=True, training=training)
    _, l_scores = l_mha(l_feat, l_feat, l_feat, return_attention_scores=True, training=training)

    def _tower(name: str, scores: Any, phase_min: float, phase_max: float) -> TowerAttention:
        se = {blk: _np(feats[layer_name])[0] for blk, layer_name in sorted(se_map[name].items())}
        return TowerAttention(
            tower=name,
            se=se,
            mha_scores=_np(scores)[0],  # drop batch dim -> (heads, Tq, Tk)
            phase_min=phase_min,
            phase_max=phase_max,
        )

    half = float(min(max(local_phase_half, 1e-3), 0.5))
    return AttentionMaps(
        global_tower=_tower("global", g_scores, float(global_phase[0]), float(global_phase[1])),
        local_tower=_tower("local", l_scores, -half, half),
    )


# ---------- panels -----------------------------------------------------------


def _panel_temporal(
    ax: Any,
    tower: TowerAttention,
    view: np.ndarray,
    *,
    title: str,
    transit_half_phase: float,
    show_ingress: bool,
) -> None:
    """Flux view (faint) + temporal attention-received profile (bold, twin axis).

    Shades the transit window and, for the local view, marks ingress/egress so
    you can read off whether attention concentrates on the transit.
    """
    view = np.squeeze(np.asarray(view, dtype=float))
    view_phase = np.linspace(tower.phase_min, tower.phase_max, view.shape[0])

    ax.plot(view_phase, view, color="0.6", linewidth=1.0, label="flux view")
    ax.set_xlim(tower.phase_min, tower.phase_max)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Normalised flux", color="0.4")
    ax.tick_params(axis="y", labelcolor="0.4")

    th = abs(transit_half_phase)
    if np.isfinite(th) and th > 0:
        ax.axvspan(-th, th, color="red", alpha=0.08, zorder=0)
        if show_ingress:
            for x, lab in ((-th, "ingress"), (th, "egress")):
                ax.axvline(x, color="red", ls="--", alpha=0.6, linewidth=1.0)
                ax.text(
                    x,
                    ax.get_ylim()[1],
                    lab,
                    color="red",
                    fontsize=7,
                    ha="center",
                    va="bottom",
                    alpha=0.8,
                )

    received = tower.attention_received
    centers = tower.phase_centers()
    axr = ax.twinx()
    axr.plot(
        centers,
        received,
        color="C3",
        linewidth=1.8,
        marker="o",
        markersize=2.5,
        label="attention received",
    )
    axr.axhline(1.0 / tower.feat_len, color="C3", ls=":", alpha=0.5, linewidth=1.0)
    axr.set_ylabel("Attention received (mean over heads, queries)", color="C3")
    axr.tick_params(axis="y", labelcolor="C3")
    axr.set_ylim(bottom=0.0)

    # Quantify concentration: attention mass in-transit vs the uniform expectation.
    concentration = np.nan
    if np.isfinite(th) and th > 0:
        in_tr = np.abs(centers) <= th
        width_frac = float(in_tr.mean())
        if width_frac > 0 and received.sum() > 0:
            concentration = float(received[in_tr].sum() / received.sum() / width_frac)

    suffix = (
        f"  ·  transit concentration {concentration:.2f}×"  # noqa: RUF001  (figure label)
        if np.isfinite(concentration)
        else ""
    )
    ax.set_title(title + suffix)


def _panel_mha_matrix(ax: Any, tower: TowerAttention, *, title: str) -> Any:
    """Head-averaged (query × key) attention matrix as a heatmap, axes in phase."""
    mat = tower.head_mean
    ext = [tower.phase_min, tower.phase_max, tower.phase_max, tower.phase_min]
    im = ax.imshow(mat, aspect="auto", cmap="viridis", extent=ext, origin="upper")
    ax.set_xlabel("Key phase (attended to)")
    ax.set_ylabel("Query phase (attending)")
    ax.set_title(f"{title}  ·  {tower.n_heads} heads, T={tower.feat_len}")
    return im


def _panel_se(ax: Any, tower: TowerAttention, *, title: str) -> Any:
    """Per-block SE channel gates as stacked colour strips (channel index → x)."""
    blocks = sorted(tower.se.keys())
    last_im = None
    for row, blk in enumerate(blocks):
        gates = np.asarray(tower.se[blk], dtype=float)
        last_im = ax.imshow(
            gates[None, :],
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            extent=[0.0, 1.0, row, row + 1],
            origin="lower",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0, max(len(blocks), 1))
    ax.set_yticks([r + 0.5 for r in range(len(blocks))])
    ax.set_yticklabels([f"block {blk}\nC={tower.se[blk].shape[0]}" for blk in blocks], fontsize=8)
    ax.set_xlabel("Channel index (normalised)")
    means = ", ".join(f"b{blk}={tower.se[blk].mean():.2f}" for blk in blocks)
    ax.set_title(f"{title}  ·  mean gate {means}")
    return last_im


# ---------- main entry -------------------------------------------------------


def attention_figure(
    maps: AttentionMaps,
    report: CandidateReport,
    out_path: Path | str,
    *,
    global_view: np.ndarray,
    local_view: np.ndarray,
    mission: str = "TESS",
    disposition: str | None = None,
) -> Path:
    """Render the six-panel attention diagnostic and save it.

    Layout (mirrors the vetting figure's 3×2 grid):

        [ local temporal attention ]   [ local MHA matrix ]
        [ global temporal attention]   [ global MHA matrix ]
        [ local SE channel gates   ]   [ global SE channel gates ]

    Parameters
    ----------
    maps
        Output of :func:`extract_attention`.
    report
        Candidate ephemeris + score (reused from ``eval.vetting``).
    global_view, local_view
        The same normalised views passed to :func:`extract_attention`, used as
        the faint flux backdrop on the temporal panels.
    """
    import matplotlib.pyplot as plt

    period = float(report.period)
    duration = float(report.duration)
    transit_half_phase = (duration / period / 2.0) if period > 0 else float("nan")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    (ax_lt, ax_lm), (ax_gt, ax_gm), (ax_lse, ax_gse) = axes

    _panel_temporal(
        ax_lt,
        maps.local_tower,
        local_view,
        title="Local view · MHA temporal attention",
        transit_half_phase=transit_half_phase,
        show_ingress=True,
    )
    im_lm = _panel_mha_matrix(ax_lm, maps.local_tower, title="Local MHA attention")
    fig.colorbar(im_lm, ax=ax_lm, fraction=0.046, pad=0.04, label="attention weight")

    _panel_temporal(
        ax_gt,
        maps.global_tower,
        global_view,
        title="Global view · MHA temporal attention",
        transit_half_phase=transit_half_phase,
        show_ingress=False,
    )
    im_gm = _panel_mha_matrix(ax_gm, maps.global_tower, title="Global MHA attention")
    fig.colorbar(im_gm, ax=ax_gm, fraction=0.046, pad=0.04, label="attention weight")

    im_lse = _panel_se(ax_lse, maps.local_tower, title="Local SE channel gates")
    fig.colorbar(im_lse, ax=ax_lse, fraction=0.046, pad=0.04, label="SE gate (sigmoid)")
    im_gse = _panel_se(ax_gse, maps.global_tower, title="Global SE channel gates")
    fig.colorbar(im_gse, ax=ax_gse, fraction=0.046, pad=0.04, label="SE gate (sigmoid)")

    title = f"Attention diagnostic · TIC {report.tic_id}"
    if mission:
        title += f"  ({mission})"
    title += (
        f"  ·  P = {period:.4f} d  ·  T0 = {report.t0:.3f}"
        f"  ·  duration = {duration * 24:.2f} h  ·  P(planet) = {report.score:.3f}"
    )
    if disposition:
        title += f"\nTFOPWG = {disposition}"
    fig.suptitle(title, fontsize=12)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path
