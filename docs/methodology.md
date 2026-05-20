# Methodology

> Portfolio write-up for the Exoplanet Hunter project. Companion to the
> top-level [README](../README.md), which covers usage; this document covers
> *why* the system is built the way it is. For headline results,
> ablation ladders, and bibliography see
> [research_report_draft.md](research_report_draft.md).
>
> **Status:** Updated 2026-05-19 to reflect the branch-3 final architecture
> (5-fold group-stratified CV, SE + MHA + residual fusion, temperature
> scaling, 9-dim aux with `centroid_snr`).

## 1. Problem statement

When a planet passes between us and its host star, it blocks a tiny fraction
of starlight — a **transit**. The dip is small (≈ 1% for a Jupiter-sized
planet, ≈ 0.01% for an Earth-sized one), short (hours), and noisy, but
periodic. NASA's TESS satellite monitors hundreds of millions of stars and
produces a *light curve* — brightness vs time — for each one.

Most of those stars have never been individually reviewed. The opportunity:
build a classifier that can tell a real transit apart from the many things
that mimic one — eclipsing binary stars, instrumental glitches, starspot
modulation, background blends — and run it on unreviewed targets.

This is the same problem Shallue & Vanderburg (2018) attacked with the
original AstroNet on Kepler data, which led to the discovery of Kepler-90 i.
Subsequent work — Ansdell et al. (2018) added stellar context and
centroid-shift features; Yu et al. (2019) extended the architecture to TESS;
Dattilo et al. (2019) transferred it to K2; Valizadegan et al. (2022, 2025)
built the multi-branch ExoMiner family; Xie et al. (2025) added
Squeeze-and-Excitation channel attention plus a residual head; Islam (2026)
introduced trimodal late fusion with multi-head attention and post-hoc
temperature scaling — has refined this baseline. The architecture chosen
here is the AstroNet skeleton extended with the ideas from Xie et al. and
Islam.

## 2. Data sources

| Source                          | Used for                                  |
|---------------------------------|-------------------------------------------|
| **MAST** (via `lightkurve`)     | TESS SPOC + Kepler stitched light curves  |
| **NASA Exoplanet Archive** (PS) | Confirmed planet labels + transit params  |
| **NASA Exoplanet Archive** (TOI / KOI) | Candidate / FP / FA dispositions   |
| **TIC v8 / Gaia DR3**           | Stellar parameters (Teff, R*, log g, Tmag)|
| **ExoFOP-TESS**                 | Current TFOPWG dispositions + follow-up counts (cross-reference only) |

All five are free, no auth needed, and queryable from Python. The full
catalogue build is `src/exoplanet_hunter/data/catalog.py`. Per Christiansen
et al. (2025) the NEA + ExoFOP services are the canonical references for
this entire data pipeline.

### Label scheme

| Source / disposition | Label  | Use                                |
|----------------------|--------|------------------------------------|
| TOI / PS `CP`, `KP`  | 1      | training positive                  |
| KOI `CONFIRMED`      | 1      | training positive                  |
| TOI `FP`, `FA`       | 0      | training negative                  |
| KOI `FALSE POSITIVE` | 0      | training negative                  |
| TOI `PC`             | -1     | held out for inference / discovery |
| KOI `CANDIDATE`      | -1     | held out for inference / discovery |

An earlier version of the catalogue also included a "QUIET" supplement —
random TIC IDs phase-folded at a *synthesised* period as a no-signal anchor.
This was **retired** in `fix/training-stability`. Folding a flat baseline at
an arbitrary period produces an arbitrary view that the model cannot
generalise from, and TOI/KOI false positives provide enough negative
examples without the artefact.

## 3. Preprocessing pipeline

The pipeline (`src/exoplanet_hunter/preprocess/`) is deliberately close to
the one used in published transit-detection ML work, so model performance
is comparable to the literature.

1. **Clean** — drop NaNs, *one-sided* upper sigma clip at 5σ. The default
   lightkurve two-sided clip would treat deep transit dips as negative
   outliers and delete them; the lower bound is left at +∞.
2. **Flatten** — Savitzky-Golay filter (`window_length=301` for 2-min
   cadence ≈ 10 hours, `polyorder=3`). The window must be much wider than
   the transit duration or the transit gets filtered out alongside the
   stellar variability. *Crucially*, in-transit cadences are masked out of
   the fit using the known ephemeris from the catalogue row; otherwise the
   spline interpolates through the transit and erases the very signal we
   want to preserve. This is the classic "filter learns the transit"
   failure mode.
3. **Phase-fold and bin** — collapse the entire time series onto a single
   orbit using the catalogue period and epoch, then bin into two views:
   - **Global view** — 2,001 bins covering full phase. Carries information
     about secondary eclipses (warm Jupiters), out-of-transit baseline
     variability, and any *additional* transit dips at other phases.
   - **Local view** — 201 bins covering ±2 transit durations around phase 0.
     Captures the transit *shape* at high resolution: U-shape (planet) vs
     V-shape (eclipsing binary).

Both views are median-subtracted and divided by their absolute minimum so
the baseline sits at 0 and the deepest dip is −1. This forces the model to
see *transit shape*, not *transit magnitude* — a 1% dip and a 0.01% dip
with the same shape look identical to the model.

The output is a single compressed numpy archive (`data/processed/views.npz`)
containing `global_views`, `local_views`, `labels`, `tic_ids`, and a
**9-dimensional `aux_features` vector** per target:

```
[T_eff, R_*, log g, T_mag, depth, duration, log P, SNR, centroid_snr]
   --- stellar (4) ---     --- transit shape (4) ---     BEB (1)
```

The centroid-shift feature `centroid_snr` (added in branch 3, after
Ansdell et al. 2018) is the magnitude of the in-transit photocentre offset
in units of σ, after detrending the raw `MOM_CENTR1/2` columns for Kepler
quarterly rolls and per-segment drift. Genuine on-target transits give
values ≲ 3; background eclipsing binaries give values ≳ 3. Implementation
in `src/exoplanet_hunter/features/centroid.py`. The feature is **log1p
transformed** before standard-scaling, because its FP distribution is
heavy-tailed (q90 = 423, max ≈ 10,000) and feeding it raw to
`StandardScaler` causes the scaler to centre on the tail rather than the
cohort body.

## 4. Models

### 4.1 Random Forest baseline (`models/baseline_rf.py`)

The classical-ML baseline, on 14 hand-crafted features extracted from the
global view. Why RF specifically:

- Bagging + random feature subsampling reduce variance — the two ideas
  from DATA 305 Week 2.
- Class-weighted training handles imbalance.
- SHAP gives interpretable feature importance.
- Trains in seconds, so it's a useful sanity check before spending GPU
  on the CNN.

### 4.2 Dual-view 1D CNN (`models/cnn_dualview.py`)

The headline model. Architecture is Shallue & Vanderburg (2018)'s AstroNet
extended with Squeeze-and-Excitation channel attention (Hu et al. 2018;
placement per Xie et al. 2025), bilateral multi-head self-attention and
residual late fusion (Islam 2026), and LeakyReLU(α=0.1) in the head
(Xie et al. 2025 §2.2):

```
global_view (2001,) ─► Conv tower (3 blocks 16,32,64 + SE per block)
                       └► MHA(8 heads) + LayerNorm + residual ─► GAP ──┐
                                                                       │
local_view  (201,)  ─► Conv tower (2 blocks 16,32 + SE per block)      │
                       └► MHA(8 heads) + LayerNorm + residual ─► GAP ──┤
                                                                       ├─► concat
aux_features (9,) ─► StandardScaler + median impute ───────────────────┘
                                              (Wide & Deep path)         │
                                                                         ▼
                                              Residual fusion head:
                                              FC(256) → LeakyReLU → BN → Dropout
                                              FC(128) → LeakyReLU → BN → Dropout
                                              + linear shortcut [concat → 128]
                                              FC(1)  → sigmoid
```

Key design choices:

- **Two towers of different depths.** The global view (2,001 bins) carries
  more structure and gets the deeper tower; the local view (201 bins) is
  zoomed on the transit so a shallower tower preserves resolution.
- **Squeeze-and-Excitation (Hu et al. 2018).** A GAP → FC(C/r) → ReLU →
  FC(C) → Sigmoid → channel-scale block re-weights the conv channels by
  global context. Branch 3 placement is after each conv block, before the
  MaxPool, per Xie et al. (2025) Fig. 1.
- **Multi-Head Attention (Islam 2026 §III.D).** 8-head self-attention with
  residual + LayerNorm at the end of each conv tower, applied to the
  `(B, T, C)` feature map *before* GlobalAveragePool. Attention lets the
  model upweight ingress/egress cadences (where the morphology lives) and
  downweight the flat baseline.
- **Wide & Deep auxiliary path (Géron Ch. 10).** A 1% dip on a giant star
  implies a stellar companion, not a planet. Letting `T_eff / R_* / log g /
  T_mag / depth / duration / log P / SNR / centroid_snr` bypass the conv
  layers — concatenated directly with the pooled tower embeddings — is the
  Wide & Deep pattern. Short-circuits the convolutional bottleneck for
  features that aren't time series.
- **Residual fusion head (Islam 2026 §III.D).** Two-layer MLP (256, 128
  units) with LeakyReLU + BatchNorm + Dropout(p=0.4), wrapped in a linear
  shortcut from the concatenated embeddings to the head's last layer
  dimension. The shortcut prevents gradient stagnation in the fusion path
  before the encoders converge.
- **Dropout stays on at inference** (`training=True` in the head). Enables
  MC-Dropout uncertainty quantification (Gal & Ghahramani 2016): a
  candidate's "I'm 0.95 confident" is meaningful only with its
  spread across 30+ stochastic forward passes.
- **Focal loss option** (`models/losses.py`) for severe class imbalance,
  with γ = 2 and α = 0.75 tunable. When focal is active, `class_weight`
  is disabled to prevent double-counting.

### 4.3 Why a CNN at all?

- The transit shape contains physics that hand-crafted features partly
  throw away — limb-darkening makes the bottom of a real transit slightly
  curved, asymmetric ingress/egress betrays gravity darkening (Szabó et al.
  2020 on Kepler-13Ab), and a CNN can pick that up where a depth/duration
  feature cannot.
- The global view contains structure beyond the central dip (secondary
  eclipses, additional planets), which a feature-vector summary loses.
- The 1D CNN is the natural application of translation invariance: a
  transit shape is the same wherever it falls in phase.

## 5. Training

`scripts/train_model.py` is the single entry point, driven by Hydra. One
command swaps datasets and models:

```bash
python scripts/train_model.py model=cnn_dualview data=default
python scripts/train_model.py model=baseline_rf
```

### Split: 5-fold group-stratified cross-validation

- `sklearn.model_selection.StratifiedGroupKFold` with **group = `tic_id`**
  and stratify-on-label. Multi-planet systems and re-observed TICs are
  kept entirely within one fold. Without this, test AUC was inflated by
  2-5 points through "seen this star before" leakage.
- Within each outer fold, an inner 88/12 `GroupShuffleSplit` carves
  training from validation. The inner validation drives `EarlyStopping`,
  the F1-optimal decision threshold sweep, and the temperature-scaling
  fit.
- The earlier single 70/15/15 split (branch 1 / branch 2) is retained
  in the comparison table for historical context only — the move to k-fold
  is a *correctness* fix, not a tuning lever.

### Optimisation

- **Adam, lr = 5×10⁻⁴, with `clipnorm = 1.0`.** Gradient norm clipping
  was added after early runs collapsed to `loss = NaN`, traced to
  unscaled stellar inputs (T_eff ≈ 5,800 vs log_period ≈ 1) producing
  exploding gradients in the dense head.
- **Aux pipeline:** `sklearn.Pipeline([SimpleImputer(strategy='median'),
  StandardScaler])`. Fitted on training-fold data only, never refit on
  val/test/inference. Persisted alongside the model so `score_target.py`
  reproduces the exact training-time preprocessing.
- **Loss:** binary cross-entropy by default; binary focal loss (γ=2,
  α=0.75) optionally available (Lin et al. 2017).

### Augmentation

- Gaussian noise (σ = 5×10⁻⁴), small phase shifts (±0.5%), random depth
  scaling (±5%), 2% random bin masking.
- **Time-flip augmentation was removed** (`fix/training-stability`) on
  the basis of Szabó et al. (2020): real transits are not perfectly
  symmetric — gravity darkening on rapidly rotating stars produces
  asymmetric ingress/egress, and flipping these mislabels the geometry.

### Callbacks

`EarlyStopping(monitor='val_auc', patience=25, restore_best=True)`,
`ModelCheckpoint(monitor='val_auc')`, `ReduceLROnPlateau(monitor='val_loss',
factor=0.5, patience=8)`. Standard Keras callback toolbox from
Géron Ch. 11.

### Calibration

Post-hoc **temperature scaling** (Guo et al. 2017; ExoNet 2026). A single
scalar T > 0 fitted on validation logits by negative-log-likelihood
minimisation, applied at inference as `sigmoid(logit / T)`. Rank-preserving:
ROC-AUC and PR-AUC are identical pre- and post-calibration, but Brier and
reliability improve. T > 1 means the model was overconfident; T < 1, the
opposite. Branch-3 final fold T* values: 1.275 ± 0.250 across folds.

Branch 2 used `sklearn.isotonic.IsotonicRegression` instead. It's
functionally equivalent on accuracy but introduces ~3 dozen learnable
knots per fold against temperature's one, so generalises slightly worse
out of sample. The bundle interface — `calibrator.predict(scores)` —
is unchanged, so `score_target.py` works against either.

Decision threshold selected by sweeping ∈ [0.05, 0.95] on the validation
set and choosing the value that maximises F1.

### Tooling

| Tool | Purpose |
|---|---|
| **Hydra** | Composable YAML configs (`model=`, `data=`, `train=`) |
| **MLflow** | Every hyperparameter, metric, plot, model artefact |
| **Optuna** | Bayesian hyperparameter search with median pruning (planned) |
| **`lightkurve`** | MAST querying, downloading, stitching |
| **`astroquery`** | TIC v8 / Gaia DR3 stellar-parameter lookups |
| **Ruff + mypy + pytest** | Pre-commit linting, type checking, fixture tests |

## 6. Evaluation

| Metric            | Why it matters here                                            |
|-------------------|----------------------------------------------------------------|
| **ROC-AUC**       | Canonical binary classifier comparison.                        |
| **PR-AUC**        | More informative than ROC under class imbalance.               |
| **F1 @ τ\***      | At F1-optimal threshold; the threshold the deployed model uses.|
| **Brier score**   | Calibration — does prob 0.9 mean a 90% chance?                 |
| **Reliability**   | Reliability diagram — bin probabilities and check fractions.   |
| **Confusion @ τ** | Inspect the false-positive vs false-negative trade-off.        |

Calibration matters more than usual here: the *score* is the deliverable.
A miscalibrated model with great AUC still produces unreliable candidate
rankings. This is also why temperature scaling is fitted per-fold rather
than once on a held-out set.

## 7. Discovery (branch 4, in progress)

Once the model is trained, the discovery loop runs through
`scripts/score_candidates.py` and `scripts/render_vetting.py`:

1. **Score** every row in `data/labels/candidates.parquet` (6,200 held-out
   TOI + Kepler PCs) with the branch-3 final 5-fold ensemble:
   - For each candidate: download FITS → clean → flatten → build views →
     extract centroid → 9-dim aux → score with each of 5 fold-models →
     30 MC-Dropout samples per fold → calibrate each sample with the
     fold's temperature scaler.
   - Aggregate to ensemble mean probability, ensemble std, p10/p90,
     between-fold disagreement, within-fold dropout disagreement.
   - Write `results/candidates_scored.parquet` (resumable; atomic
     tmp+rename checkpoints every 25 rows).

2. **Cross-reference** with `scripts/discovery_shortlist.py`. Joins the
   scored parquet with the ExoFOP TOI/CTOI/KOI catalogs and the NEA PS
   table snapshot to add:
   - Current TFOPWG disposition (may differ from the 2024 PC label).
   - Total follow-up observation count per candidate.
   - `confirmed_after_training` flag — set if the TOI has been confirmed
     by a published paper since the training-data cutoff. These are the
     clean blind-validation cases.
   - Scalar `discovery_score = prob × (1 − fold_disagree) ×
     1/(1 + n_followup/3)` — high-confidence under-investigated candidates
     surface to the top.

3. **Render** vetting figures for the top-K via
   `src/exoplanet_hunter/eval/vetting.py`. Six panels per candidate:
   phase-folded global view, phase-folded local view, odd-vs-even
   transit overlay (EB check), BLS periodogram with harmonics, centroid
   shift diagram, ensemble probability with MC-Dropout uncertainty bands
   and per-fold dots. Title carries the current TFOPWG disposition and
   community follow-up count.

4. **Manual review** against ExoFOP-TESS. Any high-confidence,
   under-investigated candidate that survives the six-panel triage gets
   shortlisted for the discovery write-up. Anything novel could be
   submitted as a Community TOI (CTOI).

## 8. Limitations

- **Two-minute SPOC cadence** misses many short-period planets in fainter
  stars (which only have 30-min FFI data). A future version should add a
  separate pipeline for FFI light curves via `eleanor` or `tica`.
- **No transit timing variation (TTV) modelling** — TTVs are a strong
  indicator of multi-planet systems and aren't a feature.
- **Single-planet phase folding** — if a TIC has more than one planet, the
  pipeline picks the strongest signal and the others contaminate the wings.
- **Class-label drift** — TOI dispositions change as follow-up data
  arrives. Some "FP" dispositions are later upgraded to "PC" or "CP",
  and vice versa. Periodic catalogue refreshes are the only mitigation;
  the project's training catalogue is refreshed via `data/catalog.py`.
- **TFOPWG lag vs published-paper truth** — the TFOPWG disposition table
  updates more slowly than the NEA PS table. 125 of the 6,200 PCs in
  our discovery pool already have peer-reviewed confirmation papers in
  the PS snapshot — useful as a blind validation set.
- **Stellar parameter coverage** is uneven; many fainter TICs have only
  rough estimates from photometry, and metallicity is unavailable for
  most TESS targets.
- **Centroid SNR thresholding** — the SNR > 3 BEB flag is statistically
  meaningful but can produce false alarms on clean isolated targets
  where the OOT centroid noise is tiny. Combining the SNR with an
  absolute-shift threshold is on the future-work list.

## 9. References

Cited in this document; see [research_report_draft.md](research_report_draft.md)
for the full bibliography.

- Ansdell, M., et al. (2018). *ApJL* 869, L7.
- Christiansen, J. L., et al. (2025). *arXiv:* 2506.03299. (NEA + ExoFOP)
- Dattilo, A., et al. (2019). *AJ* 157, 169.
- Gal, Y., & Ghahramani, Z. (2016). *ICML* 33.
- Géron, A. (2019). *Hands-on Machine Learning*, 2nd ed.
- Guo, C., et al. (2017). *ICML* 34. (Temperature scaling)
- Hippke, M., et al. (2019). *AJ* 158, 143. (WOTAN detrending)
- Hippke, M., & Heller, R. (2019). *A&A* 623, A39. (TLS)
- Hu, J., Shen, L., & Sun, G. (2018). *CVPR.* (Squeeze-and-Excitation)
- Islam, M. R. (2026). *arXiv:* 2604.15560. (ExoNet)
- Lin, T.-Y., et al. (2017). *ICCV.* (Focal loss)
- Marsland, S. (2014). *Machine Learning: An Algorithmic Perspective*, 2nd ed.
- Shallue, C., & Vanderburg, A. (2018). *AJ* 155, 94. (AstroNet)
- Szabó, Gy. M., et al. (2020). *MNRASL* 492, L17. (Kepler-13Ab; replaces an
  earlier Howarth & Morello citation in this project after PDF audit.)
- Valizadegan, H., et al. (2022). *ApJ* 926, 120. (ExoMiner)
- Valizadegan, H., et al. (2025). *ApJ*, in press. (ExoMiner++)
- Xie, D., et al. (2025). *Research in Astronomy and Astrophysics* 25, 104004.
- Yu, L., et al. (2019). *AJ* 158, 25. (AstroNet-Triage / TESS)
