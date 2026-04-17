# Methodology

> Portfolio write-up for the Exoplanet Hunter project. Companion to the
> top-level [README](../README.md), which covers usage; this document covers
> *why* the system is built the way it is.

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

## 2. Data sources

| Source                          | Used for                                  |
|---------------------------------|-------------------------------------------|
| **MAST** (via `lightkurve`)     | TESS SPOC light curves                    |
| **NASA Exoplanet Archive** PS   | Confirmed planet labels + transit params  |
| **NASA Exoplanet Archive** TOI  | Candidate / FP / FA dispositions          |
| **TIC v8 / Gaia DR3**           | Stellar parameters (Teff, R*, log g)      |

All four are free, no auth needed, and queryable from Python. The full
catalogue build is `src/exoplanet_hunter/data/catalog.py`.

### Label scheme

| TFOPWG disposition | Label  | Use                              |
|--------------------|--------|----------------------------------|
| CP (confirmed)     | 1      | training positive                |
| KP (known)         | 1      | training positive                |
| FP (false pos.)    | 0      | training negative                |
| FA (false alarm)   | 0      | training negative                |
| PC (candidate)     | -1     | held out for inference / discovery |
| Random TIC         | 0 (Q)  | "quiet" star supplement          |

The "quiet" supplement matters: without it, the model learns "is there a dip"
rather than "is there a planet" — and a dip from an eclipsing binary will
trigger it.

## 3. Preprocessing pipeline

The pipeline (`src/exoplanet_hunter/preprocess/`) is deliberately the same
one used in published transit-detection ML work, so model performance is
comparable to the literature.

1. **Clean** — drop NaNs, sigma-clip outliers (cosmic rays, momentum-dump
   artefacts). 5σ is the standard threshold; tighter clips start eating
   the transits themselves.
2. **Flatten** — Savitzky-Golay filter (`window_length=301` for 2-min
   cadence ≈ 10 hours). The window must be much wider than the transit
   duration or the transit gets filtered out alongside the stellar
   variability.
3. **Phase-fold** — collapse the entire time series onto a single orbit
   using the known (or BLS-estimated) period and epoch.
4. **Bin to two views**:
   - **Global view** — 2001 bins covering full phase. Carries information
     about secondary eclipses (warm Jupiters), out-of-transit baseline
     variability, and any *additional* transit dips at other phases.
   - **Local view** — 201 bins covering ±3 transit durations around phase 0.
     Captures the transit *shape* at high resolution: U-shape (planet) vs
     V-shape (eclipsing binary).

Both views are median-subtracted and depth-divided so the baseline sits at
0 and the deepest dip is -1. This makes the model see *transit shape*, not
*transit magnitude* — a 1% dip and a 0.01% dip with the same shape look
identical to the model.

## 4. Models

### 4.1 Random Forest baseline (`models/baseline_rf.py`)

The classical-ML baseline, on 14 hand-crafted features extracted from the
global view. Why RF specifically:

- Bagging + random feature subsampling reduce variance — the two ideas from
  DATA 305 Week 2.
- Class-weighted training handles imbalance.
- SHAP gives interpretable feature importance — which features actually
  drive a prediction.
- Trains in seconds, so it's a useful sanity check before spending GPU on
  the CNN.

### 4.2 Dual-view 1D CNN (`models/cnn_dualview.py`)

The headline model — Shallue & Vanderburg (2018) architecture, which is what
Google's AstroNet uses.

```
global_view (2001,) ──► Conv tower (5 blocks, 16→256) ──► flatten ──┐
                                                                     ├─► concat
local_view  (201,)  ──► Conv tower (2 blocks, 16→32)  ──► flatten ──┤
                                                                     │
                          aux_features (n,) ─────────────────────────┘
                                              (Wide & Deep path)
                                                                     │
                                                                     ▼
                                                        FC × 4 (512) → sigmoid
```

Key design choices:

- **Two towers** of different depths because the inputs have different sizes
  (10× difference) and want different receptive fields. The global tower
  has more pooling stages and finishes with deeper feature maps because
  the global view is longer; the local tower stays shallower so it doesn't
  over-pool the 201-bin signal.
- **Wide & Deep auxiliary path** for stellar features: a 1% dip on a giant
  star implies a stellar companion, not a planet. Letting Teff / R* /
  log g bypass the conv layers is the **Wide & Deep** pattern from the
  DATA 305 Week 5 Functional API notes — short-circuits the convolutional
  bottleneck for features that aren't time series.
- **Dropout always on** in the FC head: enables MC-Dropout uncertainty
  quantification at inference (Gal & Ghahramani 2016). Critical when
  claiming "this is a planet candidate" — high mean *and* low std.
- **Focal loss option** (`models/losses.py`) for severe class imbalance,
  with γ tunable.

### 4.3 Why a CNN at all?

- The transit shape contains physics that hand-crafted features partly throw
  away — limb-darkening makes the bottom of a real transit slightly curved,
  and a CNN can pick that up where a depth/duration feature can't.
- The global view contains structure beyond the central dip (secondary
  eclipses, additional planets), which a feature vector summary loses.
- The 1D CNN is a natural application of the convolutional idea (translation
  invariance: a transit shape is the same wherever it falls), and it's
  the standard architecture for time-series classification.

## 5. Training

`src/exoplanet_hunter/training/train.py` is the single entry point, driven
by Hydra. One command swaps datasets and models:

```bash
python scripts/train_model.py model=cnn_dualview data=small
python scripts/train_model.py model=cnn_dualview_stellar data=default
```

What the script does:

- Stratified train/val/test split (70/15/15 by default).
- `tf.data` pipelines with augmentation: random phase shift, magnitude
  jitter, left-right flip (transit shape is symmetric in time).
- Class weights computed from the training set.
- Standard Keras callbacks (these come straight from the Week 5 notes):
  `EarlyStopping(restore_best_weights=True)`, `ModelCheckpoint`,
  `ReduceLROnPlateau`.
- Every run logged to MLflow: hyperparams (full config flattened),
  per-epoch metrics, learning curves, ROC + PR + confusion matrix,
  fitted model artifact, git SHA.

Optuna search (`training/tune.py`) sweeps LR, dropout, loss type, focal-γ,
and batch size with `MedianPruner` to kill bad trials early.

## 6. Evaluation

| Metric            | Why it matters here                                            |
|-------------------|----------------------------------------------------------------|
| **ROC-AUC**       | Canonical binary classifier comparison.                        |
| **PR-AUC**        | More informative than ROC under class imbalance.               |
| **Brier score**   | Calibration — does prob 0.9 mean a 90% chance?                 |
| **Reliability**   | Reliability diagram — bin probabilities and check fractions.   |
| **Confusion @ τ** | Inspect the false-positive vs false-negative trade-off.        |

Calibration matters more than usual here: the *score* is the deliverable.
A miscalibrated model with great AUC still produces unreliable candidate
rankings.

## 7. Discovery (planned)

Once the model is trained, the discovery loop (notebook 05, `eval/vetting.py`):

1. For each TOI-PC (candidate disposition) or random unreviewed TIC:
   - Download → clean → flatten → BLS/TLS period search → build views.
   - Score with MC-Dropout (50 samples).
2. Filter by `mean > 0.9` and `std < 0.1` — confident detections only.
3. Generate a six-panel vetting figure:
   - Full-mission flattened light curve.
   - Global view at best (period, t0).
   - Local view at best (period, t0).
   - Odd vs even transit overlay (eclipsing-binary check).
   - Phase-0.5 secondary-eclipse check.
   - Centroid-shift diagnostic (background-EB check).
4. Cross-reference with ExoFOP-TESS to see if anyone has flagged it.
5. If nothing turns up, submit as a Community TOI (CTOI).

## 8. Limitations

- **Two-min SPOC cadence** misses many short-period planets in fainter stars
  (which only have 30-min FFI data). A future version should add a separate
  pipeline for FFI light curves via `eleanor` or `tica`.
- **No transit timing variation** modelling — TTVs are a strong indicator of
  multi-planet systems and aren't currently a feature.
- **Single-planet phase folding** — if a TIC has more than one planet, the
  pipeline picks the strongest signal and the others contaminate the wings.
- **Class label noise** — TOI dispositions change as follow-up data arrives.
  Some "FP" dispositions are later upgraded to "PC" or "CP", and vice versa.
  Periodic refreshes of the catalogue are the only mitigation.
- **Stellar parameter coverage** is uneven; many fainter TICs have only
  rough estimates from photometry.

## 9. References

- Shallue, C. & Vanderburg, A. (2018). *Identifying Exoplanets with Deep Learning*. **AJ** 155, 94.
- Hippke, M. & Heller, R. (2019). *Optimised Transit Detection — Transit Least Squares*. **A&A** 623, A39.
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*. **ICML**.
- Lin, T-Y. et al. (2017). *Focal Loss for Dense Object Detection*. **ICCV**.
- Marsland, S. (2014). *Machine Learning: An Algorithmic Perspective*, 2nd ed. Chapman & Hall/CRC.
- Chollet, F. (2021). *Deep Learning with Python*, 2nd ed. Manning. (Functional API + Wide & Deep + callbacks.)
- LeCun, Y., Bengio, Y. & Hinton, G. (2015). *Deep learning*. **Nature** 521, 436–444.
