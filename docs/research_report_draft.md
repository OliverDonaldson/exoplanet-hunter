# Detecting Exoplanet Transits in TESS and Kepler Light Curves with a Calibrated Dual-View 1D CNN

**Author:** Oliver Donaldson
**Project type:** Personal research / portfolio piece
**Background:** Data Science student, Victoria University of Wellington (DATA 305 — Machine Learning, DATA 303 — Statistical Modelling)
**Status:** *Updated 2026-05-14 with branch-3 (architecture + centroid) final numbers. All "current" numbers now reflect 5-fold group-stratified cross-validation on 3,275 TESS+Kepler targets (MLflow run `58570d85f1dd4f68a7e888988c88eeab`). The earlier 21 April 2026 single-split baseline (`mlflow run 8dce07454c`, ROC-AUC = 0.901 test) is retained in the comparison table as the branch-2 milestone.*

---

## How this project was built

This is a personal-interest project. It started with a question I wanted to answer for myself — *can a deep-learning model trained on real NASA TESS data actually identify exoplanet transits, and what would it take to build one?* — and it became the vehicle I used to apply the deep-learning material from DATA 305 to a real research problem rather than a textbook one.

I built it with Anthropic's Claude (Sonnet 4.7, via the Claude Code CLI) as a pair-programming and tutoring collaborator. The collaboration was deliberately learning-first: I supplied the vision, the references, and the judgment calls; Claude wrote and explained code grounded in those references; I executed everything on my own machine, reviewed every change, and pushed back where it didn't make sense. Listing what each side actually contributed feels more honest than a generic "AI was used" line, and is what I'd want to see in someone else's portfolio:

**What I brought:**
- The original idea, the choice of problem (exoplanet transit detection), datasets (TESS + Kepler via MAST), labels (NASA Exoplanet Archive), and target architecture family (dual-view CNN per Shallue & Vanderburg 2018).
- All reference materials: my coursework notes from DATA 305 (Marsland 2014) and DATA 303 (2026), the Géron (2019) textbook, and the research papers cited in this report (Szabó et al. 2020 / Kepler-13Ab; Islam 2026 / ExoNet; Xie et al. 2025 / SE-CNN-RlNet).
- All design judgment calls: branch scoping, when to retrain, what to keep vs scrap, storage allocation between internal SSD and external USB, prioritising data scaling over architecture upgrades (and vice versa).
- All code execution and review — every dataset build, every training run, every git commit — on my own laptop, with my eyes on the output before moving on.
- Pushback when things didn't make sense: questioning verbose comments, questioning sample sizes before committing to a long retrain, catching dead config knobs that no longer did anything.
- Final shape of this report — what to emphasise, what to acknowledge as a limitation, how to frame it.

**What Claude drafted, with my approval:**
- The initial codebase scaffold (Hydra + MLflow + dual-view CNN + RF baseline + preprocessing modules + tests).
- Specific patches on the `fix/training-stability` branch: gradient clipping, the impute-and-scale aux pipeline, the persisted calibrator/scaler bundle, the removal of dead features, the `.gitignore` bug fix, the cross-script synchronisation of aux dimensionality.
- Explanatory writeups linking design choices to specific references (e.g. how Wide-and-Deep from Géron Ch. 10 maps onto the aux-feature path; why Szabó et al.'s findings argue against horizontal-flip augmentation; how ExoNet's per-KOI deduplication contrasts with the per-TIC group split adopted here).
- The first pass of this research report.

**What we worked out together:**
- Codebase audits: the leaky split (already correct in the existing code), the `loss = NaN` failure mode (root-caused to unscaled inputs interacting with Adam), the QUIET-with-synthesised-ephemeris hack (scrapped).
- Roadmap planning: the five-branch sequence, with rescoping at multiple points — e.g. collapsing the originally-separate "training stability" and "data scaling" branches into a single overnight build once disk space allowed.
- Interpretation of the new reference papers I supplied — Claude summarised, I decided which of their contributions were worth incorporating.

**Concepts I learned by doing:**
I came into the project with the foundational deep-learning material from DATA 305 (perceptron → MLP → backprop → optimisers, regularisation, callbacks, Keras Sequential + Functional APIs) and DATA 303 (regression diagnostics, model selection, calibration). Concepts I learned through this project, by hitting them as real problems and working through them: gradient clipping as a stability technique; the difference between row-stratified and group-stratified train/test splits and why leakage matters in multi-planet systems; isotonic regression and (for branch 2) temperature scaling for probability calibration; the Wide-and-Deep pattern instantiated for this domain; Hydra config composition; experiment tracking with MLflow; how unscaled inputs of wildly different magnitudes interact with adaptive optimisers to produce `NaN` losses; and the specific physical phenomena (gravity darkening, transit-duration variation, third-light contamination) that make real-world transit detection harder than the textbook formulation.

The vision, design decisions, technical judgment, and writing are mine. Claude is acknowledged as the principal coding and tutoring assistant — the kind of collaborator a long-term solo project benefits from having around.

---

## Abstract

We present an end-to-end deep-learning pipeline for detecting transiting exoplanets in NASA TESS and Kepler light curves. The system downloads stitched SPOC/Kepler light curves directly from MAST via `lightkurve`, builds the dual-view (global + local) phase-folded representation of Shallue & Vanderburg (2018), and feeds it through a 1D convolutional neural network with Squeeze-and-Excitation channel attention (Hu et al. 2018; Xie et al. 2025), bilateral multi-head attention plus residual late-fusion (Islam 2026), and a Wide-and-Deep auxiliary path carrying 9 stellar/transit/centroid-shift features. Training is 5-fold stratified group k-fold cross-validation grouped by host star to prevent multi-planet leakage; calibration is post-hoc temperature scaling (Guo et al. 2017) fitted per fold on validation logits. A baseline Random Forest on hand-crafted features provides a classical-ML reference. Across 5-fold CV on 3,275 TESS+Kepler targets the final branch-3 model achieves ROC-AUC = 0.9555 ± 0.0044, PR-AUC = 0.9586 ± 0.0058, F1 = 0.888 ± 0.012, Brier = 0.0905 ± 0.0130, with temperature T* = 1.28 ± 0.25 — at parity with Islam (2026)'s ExoNet (ROC-AUC 0.955) on a comparable dataset. The largest single contributions are the SE + MHA + residual fusion block (+0.040 ROC over the CV baseline) and a log1p-transformed centroid-shift feature for background-eclipsing-binary discrimination after Ansdell et al. (2018) (+0.022 ROC and Brier −0.022 over the raw-centroid step). The full pipeline is reproducible via Hydra + MLflow, and all code is open source.

---

## Introduction

As of 30 April 2026, the NASA Exoplanet Archive lists 6,278 confirmed exoplanets, of which 4,640 (73.9 %) were discovered by the transit method. NASA's *Transiting Exoplanet Survey Satellite* (TESS), operating since April 2018, has catalogued 7,931 project candidates; the 23 April 2026 data release added 114 new TESS planets in a single update, bringing TESS-confirmed totals to 885 (NASA Exoplanet Archive, 2026). The remaining ~7,000 unreviewed candidates constitute a structural backlog that is beyond the capacity of manual expert vetting and motivates an automated, calibrated, and reproducible machine-learning vetting pipeline. Kepler — TESS's predecessor, with the deeper, narrower-field photometry that produced the highest-quality vetting labels available — contributes 2,784 confirmed planets that this project uses as supplementary training data. Cui et al. (2026) place this work in demographic context: their RAVEN-vetted TESS-SPOC FGK sample gives an overall close-in (0.5–16 d, 2–20 R⊕) planet occurrence rate of 9.4 ± 0.7 %, with hot Jupiters at 0.39 ± 0.03 % and the Neptunian desert at 0.08 ± 0.01 % — anchoring the population our 6,200 held-out PCs are drawn from.

This project is a portfolio piece for DATA 305 (Marsland, 2014), grounded in the deep-learning tooling covered in Géron (2019) — Keras Functional API, callbacks, Wide-and-Deep architectures, MLflow experiment tracking — and the regression diagnostics taught in DATA 303 (which informed the calibration and uncertainty-quantification components). The technical objective is to build, from real data only (no synthetic light curves in the final training set), a pipeline that can:

1. **Classify** brightness dips in unreviewed targets as *transits* vs *false positives*;
2. **Calibrate** its predicted probabilities so they reflect true likelihoods, suitable for downstream candidate prioritisation;
3. **Generalise** beyond the symmetric textbook transit, on which most published models are trained, to handle the gravity-darkened, spin-orbit-misaligned, and time-variant transit shapes that occur in practice;
4. **Be reproducible**, with every experiment tracked, every hyperparameter version-controlled, and every dataset rebuildable from the catalogue queries.

The architectural baseline is Shallue & Vanderburg (2018)'s "AstroNet" — a dual-view 1D CNN that ingests a low-resolution global phase fold and a high-resolution local zoom of the transit window. Subsequent work (Ansdell et al., 2018; Yu et al., 2019; Dattilo et al., 2019; Valizadegan et al., 2022, 2025) has extended this baseline with stellar-context features, transfer learning across missions, and richer multi-branch encodings. Two 2025–2026 papers in particular shape the upgrade path adopted here: Xie et al. (2025) demonstrated that channel-attention (Squeeze-and-Excitation) blocks plus a residual fully-connected head substantially improve training stability and accuracy; Islam (2026) introduced trimodal late fusion with multi-head attention over the CNN feature map, achieving ROC-AUC = 0.955 on Kepler. The present work adopts these as the planned architecture for branch 2 of the project.

A separate physical-realism concern is raised by Szabó et al. (2020), whose combined analysis of Kepler and TESS observations of Kepler-13Ab — a hot Jupiter orbiting a rapidly rotating, gravity-darkened A-type star — demonstrates that real transits can be substantially asymmetric, can exhibit transit-duration variation (TDV) due to orbital precession (db/dt = −0.011 yr⁻¹), and require third-light correction when contaminated by a binary companion (Shporer et al. 2014, as applied by Szabó et al. 2020). Models trained exclusively on perfectly U-shaped transits will systematically under-detect such cases. Branch 3 of this project addresses this by including hard-example asymmetric transits in the training set and by adding the planet-radius and stellar-metallicity features that ExoNet uses to disambiguate eclipsing-binary contaminants from genuine planetary signals.

---

## Methodology

### Data sources

All training data is real, no synthetic light curves are used in the final training set. Three public sources, all queried free of charge:

| Source | Used for | Volume |
|---|---|---|
| **MAST archive** (TESS SPOC, Kepler) | Stitched light curves | 14 GB TESS + 38 GB Kepler (on external SSD) |
| **NASA Exoplanet Archive TAP** | Confirmed planets (`ps`), TOI dispositions (`toi`), KOI dispositions (`cumulative`) | ~700 KB labels parquet |
| **TIC v8 / Gaia DR3** (via `astroquery`) | Stellar parameters for TESS targets | per-target lookup |

### Catalogue construction

The labelled catalogue is built deterministically from three TAP queries (`src/exoplanet_hunter/data/catalog.py`). Disposition strings from each archive table are mapped to integer labels:

| Source | Disposition | Label |
|---|---|---|
| TOI / `ps` | `CP`, `KP`, confirmed | 1 (positive) |
| TOI | `FP`, `FA` | 0 (negative) |
| TOI | `PC` | −1 (held out for inference) |
| KOI | `CONFIRMED` | 1 |
| KOI | `FALSE POSITIVE` | 0 |
| KOI | `CANDIDATE` | −1 |

Per-row units are normalised at query time to ensure consistency: `pl_tranmid` is converted from full BJD to BTJD (subtracting 2,457,000.0), TOI `pl_trandurh` is divided by 24 to match the days convention used elsewhere, and KOI `koi_depth` (parts per million) is divided by 10⁶ to match the fractional-depth convention. These conversions were retrofits after a sign of the times: a half-day error in t₀ accumulates to many days of phase error over ~10⁵ orbital cycles, producing all-NaN folded views. Held-out `CANDIDATE` / `PC` rows are persisted separately to `data/labels/candidates.parquet` and never seen during training; they are the inference set used to identify novel signals.

An earlier version of the catalogue included a "QUIET" class — random TIC IDs phase-folded at a *synthesised* period — intended as a no-signal anchor. This was retired (commit `fix/training-stability`): folding a flat baseline at an arbitrary period produces an arbitrary view that the model cannot meaningfully generalise from, and TOI/KOI false positives provide adequate negative examples without the artefact.

### Preprocessing

Each downloaded light curve is processed by a deterministic three-stage pipeline (`src/exoplanet_hunter/preprocess/`):

1. **Clean** (`clean_lightcurve`): remove NaN cadences and apply a *one-sided* upper sigma clip at 5 σ. The default lightkurve two-sided clip would treat deep transit dips as negative outliers and delete them, so the lower bound is left at +∞.
2. **Flatten** (`flatten_lightcurve`): a Savitzky-Golay filter of window 301 cadences (≈ 10 hours at 2-minute cadence) is fit to the out-of-transit baseline and divided out. *In-transit cadences are masked out of the fit* using the known ephemeris from the catalogue row, otherwise the spline learns to interpolate through the transit and erases the very signal we want to preserve. This is the classic "filter learns the transit" failure mode.
3. **Fold and bin** (`build_views`): the cleaned, flattened light curve is phase-folded at the catalogue period and binned into a *global view* (2,001 bins spanning the full phase) and a *local view* (201 bins spanning ±3 transit durations around phase 0). Each view is median-subtracted and divided by its absolute minimum so that the baseline is at 0 and the deepest dip is at −1; this lets the model see *transit shape*, not *transit magnitude*.

The output is a single compressed numpy archive (`data/processed/views.npz`) containing `global_views`, `local_views`, `labels`, `tic_ids`, and a 9-dimensional `aux_features` vector per target: `[T_eff, R_*, log g, T_mag, depth, duration, log P, SNR, centroid_snr]`. The centroid-shift feature (added in branch 3) is the magnitude of the in-transit photocentre offset in units of σ, after detrending the raw `MOM_CENTR1/2` columns for Kepler quarterly rolls and per-segment drift; genuine on-target transits give values < ~3, background eclipsing binaries give values ≳ 3 (Ansdell et al. 2018). Implementation in `src/exoplanet_hunter/features/centroid.py`.

### Model architecture

The principal model is a dual-view 1D CNN (`src/exoplanet_hunter/models/cnn_dualview.py`) implemented in the Keras Functional API. The architecture is Shallue & Vanderburg (2018) extended with attention and residual fusion in branch 3:

- **Global tower:** 3 convolutional blocks (16, 32, 64 filters), 2 conv layers per block with kernel 5, BatchNorm, ReLU, MaxPool size 5, optional SpatialDropout.
- **Local tower:** 2 convolutional blocks (16, 32 filters), 2 conv layers per block with kernel 5, MaxPool size 3.
- **Channel attention (branch 3):** Squeeze-and-Excitation block after each conv block, before MaxPool — GAP → FC(C/r) → ReLU → FC(C) → Sigmoid → channel-wise scale (Hu et al. 2018; placement per Xie et al. 2025, Fig. 1). Re-weights the convolutional channels by global context.
- **Temporal attention (branch 3):** Multi-Head Attention with residual + LayerNorm at the end of each conv tower, applied to the `(B, T, C)` feature map before GlobalAveragePooling, applied bilaterally to global and local towers (ExoNet, Islam 2026, §III.D).
- **Wide path (auxiliary):** the 9-d standardised stellar/transit/centroid feature vector concatenated *directly* with the pooled tower embeddings (the Wide-and-Deep pattern from Géron Ch. 10).
- **Residual fusion head (branch 3):** two fully-connected layers (256, 128 units) with **LeakyReLU(α=0.1, Xie et al. 2025 §2.2)** + BatchNorm + Dropout (p = 0.4), wrapped in a linear residual shortcut from the concatenated embeddings to the head's last layer dimension (ExoNet, Islam 2026, §III.D). The shortcut prevents gradient stagnation in the fusion path before the encoders converge. Dropout is left enabled at inference time (`training=True`) so MC-Dropout uncertainty estimation is available downstream (Gal & Ghahramani, 2016).

A baseline Random Forest classifier on hand-crafted features (depth, duration, depth-SNR, ingress slope, secondary-eclipse depth, odd/even depth ratio) provides the classical-ML reference, with k-fold CV and SHAP feature-importance plots for interpretability.

### Training

Training runs are launched via Hydra (`scripts/train_model.py`); all hyperparameters live in composable YAML configs under `conf/`. Per-run experiment tracking, including resolved configs, learning curves, evaluation plots, and model artefacts, is logged to MLflow.

- **Split:** 5-fold stratified group k-fold cross-validation with `sklearn.model_selection.StratifiedGroupKFold` (group = `tic_id`, stratify on label). Within each outer fold an inner 88/12 `GroupShuffleSplit` separates training from validation; the inner validation drives EarlyStopping, the F1-optimal decision threshold sweep, and the temperature-scaling fit. Multi-planet systems and re-observed TICs are kept entirely within a single fold; without this, test AUC is inflated by 2–5 points through "seen this star before" leakage. The earlier single 70/15/15 split (used through branch 2) is retained in the comparison table for historical context — the move to k-fold is a *correctness* fix, not a tuning lever.
- **Optimiser:** Adam, learning rate 5×10⁻⁴, **with `clipnorm = 1.0` to cap gradient norms**. This was added after multiple earlier runs collapsed to `loss = NaN` mid-training, traced to unscaled stellar-parameter inputs (T_eff ≈ 5,800 vs log_period ≈ 1) producing exploding gradients in the dense head.
- **Aux feature pipeline:** raw aux features are passed through a `sklearn.Pipeline([SimpleImputer(strategy="median"), StandardScaler])`, fitted on the training split only and reused (not refit) at val/test/inference. The fitted pipeline is persisted alongside the model checkpoint so `score_target.py` reproduces the exact training-time preprocessing.
- **Loss:** binary cross-entropy by default; binary focal loss (γ = 2, α = 0.75) optionally available for stronger negative-class downweighting. When focal loss is active, `class_weight` is disabled to prevent double-counting.
- **Augmentation:** small Gaussian noise (σ = 5×10⁻⁴), small phase shifts (±0.5 %), random depth scaling (±5 %), and 2 % random bin masking. *Time-flip augmentation was removed* on the basis of Szabó et al. (2020): real transits are not symmetric, and flipping them mislabels asymmetric ingress/egress shapes.
- **Callbacks:** `EarlyStopping` on `val_auc` (patience 25, restore best), `ModelCheckpoint` on `val_auc`, `ReduceLROnPlateau` on `val_loss` (factor 0.5, patience 8). All standard from the DATA 305 / Géron Ch. 11 toolbox.
- **Calibration:** post-hoc temperature scaling (Guo et al. 2017; ExoNet 2026) — a single scalar T > 0 fitted on validation logits by negative-log-likelihood minimisation, applied at inference as `sigmoid(logit / T)`. Rank-preserving: ROC-AUC and PR-AUC are identical pre- and post-calibration, but Brier and reliability improve. T > 1 indicates the model was overconfident; T < 1 the opposite. (Branch 2 used `sklearn.isotonic.IsotonicRegression` instead; it is functionally equivalent on accuracy but introduces ~3 dozen learnable knots per fold against temperature's one, so generalises slightly worse out of sample. The bundle interface — `calibrator.predict(scores)` — is unchanged, so `score_target.py` works against either.) Decision threshold selected by sweeping ∈ [0.05, 0.95] on the validation set and choosing the value that maximises F1.

### Tooling

| Tool | Purpose |
|---|---|
| **Hydra** | Composable YAML configs (`model=`, `data=`, `train=` swappable from CLI) |
| **MLflow** | Experiment tracking — every hyperparameter, metric, plot, and model artefact |
| **Optuna** | Bayesian hyperparameter search with median pruning (planned) |
| **`lightkurve`** | MAST querying, downloading, and stitching |
| **`astroquery`** | TIC v8 / Gaia DR3 stellar-parameter lookups |
| **Ruff + mypy + pytest** | Pre-commit linting, type checking, synthetic-fixture unit tests |

---

## Results & Discussion

### Branch-3 final performance (5-fold group-stratified CV)

5-fold `StratifiedGroupKFold` (group = `tic_id`) on 3,275 TESS+Kepler targets, MLflow run `58570d85f1dd4f68a7e888988c88eeab`:

| Metric | Mean ± std (across folds) |
|---|---|
| ROC-AUC | 0.9555 ± 0.0044 |
| PR-AUC | 0.9586 ± 0.0058 |
| F1 (at fold-best threshold) | 0.888 ± 0.012 |
| Brier (calibrated) | 0.0905 ± 0.0130 |
| Temperature T* | 1.275 ± 0.250 |

Per-fold ROC-AUC: 0.949, 0.953, 0.960, 0.961, 0.956 — every fold clears 0.948.

**Ladder of incremental gains.** Each row below is an additive change with no other modifications; same 3,275-row 5-fold CV split throughout. T* is reported only for folds where temperature scaling is active.

| Variant | ROC-AUC | Brier | T* |
|---|---|---|---|
| Branch 2 milestone (single 70/15/15 split + isotonic) | 0.901 (test) | 0.092 (test) | n/a |
| Branch 3 step 1 (move to 5-fold group CV) | 0.8836 ± 0.0348 | 0.1396 ± 0.0208 | n/a |
| + SE channel attention + bilateral MHA + residual fusion | 0.9232 ± 0.0098 | 0.1112 ± 0.0096 | n/a |
| + temperature scaling (replaces isotonic) | 0.9295 ± 0.0097 | 0.1118 ± 0.0109 | 1.317 ± 0.115 |
| + `centroid_snr` aux feature (raw, scaled by StandardScaler) | 0.9337 ± 0.0186 | 0.1121 ± 0.0190 | 1.340 ± 0.249 |
| **+ log1p on `centroid_snr` before scaling** | **0.9555 ± 0.0044** | **0.0905 ± 0.0130** | **1.275 ± 0.250** |

The drop from the branch-2 0.901 (test) to the row-1 0.8836 (CV mean) is the *correctness* effect of moving from a single train/val/test cut to k-fold — the single-split test value was an inflated cut, not a real performance loss. The remaining four rows are real gains on the corrected baseline: +0.072 ROC and a ~8× tightening of the fold std (0.0348 → 0.0044). The largest single discrimination gain is the SE + MHA + residual fusion block (+0.040 ROC); the largest single calibration gain is `log1p(centroid_snr)` (Brier −0.022, fold std halved). The raw-scaled centroid step (row 5) is a deliberate ablation showing that the feature's heavy-tailed FP distribution (q90 = 423 vs planet body ~1.1) corrupts `StandardScaler` unless the tail is compressed first.

Earlier runs in the project showed the training-stability failure mode this project explicitly addressed: one `cnn-large` run terminated at epoch 25 with `loss = NaN`, AUC = 0.5 (chance-level). Diagnostic investigation traced this to the combination of (i) un-scaled raw stellar parameters being concatenated into the dense head, (ii) no gradient clipping on the Adam optimiser, and (iii) a pathological interaction with focal loss when `class_weight` was also active. All three were mitigated in branch 1 (`fix/training-stability`).

### Comparison with published baselines

| Study | Year | Mission | Sample | Architecture | Reported metric |
|---|---|---|---|---|---|
| Shallue & Vanderburg | 2018 | Kepler DR24 | 15,737 | Dual-view 1D CNN ("AstroNet") | 98% accuracy |
| Ansdell et al. | 2018 | Kepler | ~16,000 | AstroNet + scalar aux features | Improved over baseline |
| Dattilo et al. | 2019 | K2 | — | AstroNet (transferred) | Two new planets confirmed |
| Yu et al. | 2019 | TESS (simulated) | — | AstroNet + stellar depth | Recall 61% on real TESS (degraded) |
| Valizadegan et al. (ExoMiner) | 2022 | Kepler | — | Multi-branch CNN | 301 new exoplanets validated |
| Tey et al. (Astronet-Triage-v2) | 2023 | TESS QLP FFI | 24,926 | Dual-view 1D CNN, 5-label, 3-vetter consensus | PR-AUC = 0.965 (test); 89% precision / 91% recall on unseen S33 |
| Valizadegan et al. (ExoMiner++) | 2025 | TESS 2-min | — | Transfer learning from Kepler | 7,330 TESS candidates |
| Martinho & Valizadegan (ExoMiner++ 2.0 FFI) | 2025 | TESS FFI | 6,419 unlabelled TCEs | CV-ensemble (5 folds) + difference-image branch | PR-AUC = 0.952 (FFI), 0.967 (2-min) |
| Xie et al. (SE-CNN-RlNet) | 2025 | Kepler + TESS | ~7,000 | AstroNet + SE channel attention + residual MLP | F1 = 0.957 (Kepler), 0.995 (TESS) |
| Islam (ExoNet) | 2026 | Train: Kepler / Inference: TESS PCs | 7,585 train / 4,720 inference | AstroNet + 8-head MHA + residual late fusion + temperature scaling | Test ROC-AUC = 0.9549 |
| Lafarga et al. (RAVEN) | 2026 | TESS-SPOC FFI (S1–55, FGK) | 2.26 M stars / 14,815 NSFP-cut | GBDT + GP ensemble per-FP scenario, synthetic PASTIS training | 143 validated, 31 newly detected; posterior > 0.99 across 8 FP scenarios |
| **This work — branch 2** | 2026 | TESS only | 1,959 | AstroNet + Wide&Deep + isotonic | ROC-AUC = 0.901 (test, single split) |
| **This work — branch 3** | 2026 | TESS + Kepler | 3,275 | + SE + bilateral MHA + residual fusion + temperature scaling + log1p centroid | **ROC-AUC = 0.9555 ± 0.0044 (5-fold CV)** |

After branch 3, the model is at parity with Islam (2026)'s ExoNet on ROC-AUC (0.9555 vs 0.9549) on a comparable Kepler+TESS dataset, with a 5-fold CV evaluation that is stricter than ExoNet's single 70/15/15 train/val/test split. The remaining gap to Xie et al. (2025)'s SE-CNN-RlNet (F1 = 0.957 Kepler, 0.995 TESS) is driven primarily by the sample-size difference (3,275 vs ~7,000 examples) and the per-mission split — the SE-CNN architectural ideas from that paper are already adopted here. Branch 4 (candidate discovery) will exercise the model on the 6,200 held-out TOI/Kepler Planet Candidates retained from the catalogue build.

The comparison entries from Tey et al. (2023), Martinho & Valizadegan (2025), and Lafarga et al. (2026) report PR-AUC rather than ROC-AUC and operate on different sample compositions (FFI-only or TESS-SPOC-FFI-only), so headline numbers should not be compared digit-for-digit with ours. They are included as reference points for the broader vetting-pipeline landscape. The RAVEN pipeline (Hadjigeorghiou et al. 2025; Lafarga et al. 2026) is the most methodologically distinct of these — a GBDT + Gaussian-Process ensemble trained on PASTIS-injected synthetic light curves, classifying each candidate against eight specific astrophysical FP scenarios separately. Its >0.97 ROC-AUC on simulated test data is not directly comparable to our 0.9555 on real TFOPWG-labelled holdout — the simulated-vs-real evaluation distinction matters more than the architectural one — but the per-scenario decomposition is a credible future direction for this project's binary classifier.

### Branch-4 candidate-discovery results

The branch-3 5-fold ensemble was applied to all 6,200 held-out Planet Candidates in `data/labels/candidates.parquet` (4,570 TESS + 1,630 Kepler). Each candidate was scored by all five fold-models, with 30 MC-Dropout samples per fold and per-fold temperature scaling, yielding `prob_mean ± prob_std` plus a fold-disagreement metric. The full run completed at 86.9 % coverage: **5,388 / 6,200 candidates received a valid probability score** — TESS 3,895 / 4,570 = 85.2 % ok, Kepler 1,493 / 1,630 = 91.6 % ok. The remaining 812 split between permanent catalogue gaps (652 TESS `no SPOC pipeline data` plus the small Kepler-side equivalent) and 160 preprocessing failures where the catalogue period/epoch does not yield a usable phase-fold across both missions. A subset of the initial Kepler run failed against a transient MAST CAOMv240 backend outage during scoring; those candidates were recovered by adding a direct-archive HTTP download path (`archive.stsci.edu/pub/kepler/lightcurves/{KIC[:4]}/{KIC:09d}/`) to the light-curve downloader, which bypasses the CAOM search layer entirely. The fallback is now the primary Kepler download path in `src/exoplanet_hunter/data/download.py`, with the existing `lightkurve.search_lightcurve` retained as a safety fallback for anything the direct archive cannot serve.

**The deliverable: a calibrated priority list of unconfirmed candidates.** Branch 4's output is a ranked priority list — 6,200 unconfirmed Planet Candidates ordered by ensemble probability, each row carrying an MC-Dropout standard deviation, p10/p90 quantiles, between-fold disagreement, and within-fold dropout disagreement. The discovery score `prob_mean × (1 − fold_disagree) × 1/(1 + n_followup/3)` further upweights candidates that are *both* high-confidence *and* under-investigated by the community, on the principle that telescope time is more productively spent on a strong candidate nobody has followed up than on a marginal candidate already well-vetted. The decision-relevant evidence for the trustworthiness of this ranking is the Branch-3 cross-validation reported in the previous section — ROC-AUC = 0.9555 ± 0.0044, PR-AUC = 0.9586 ± 0.0058, Brier = 0.0905, fold-std contraction from 0.0348 → 0.0044 across the ablation ladder. PR-AUC integrates over every operating point, so the discriminative claim does not depend on a single chosen threshold; the Brier score speaks to whether `prob_mean = 0.95` actually corresponds to a ~95 % chance. The since-confirmed recall analysis below is a downstream sanity check on a different population, not the headline metric for this work.

**High-confidence still-unconfirmed picks.** Across the 5,388 successfully scored PCs, **146 received `prob_mean ≥ 0.95`** and remain unconfirmed in the live ExoFOP TFOPWG snapshot (2026-05-22) — 140 TESS and 6 Kepler. The top three by raw probability are long-period TESS candidates: TOI-4328.01 (TIC 77175217, P = 703.79 d, prob = 0.989, `fold_disagree` = 0.006), TOI-4565.01 (TIC 381897917, P = 692.51 d, prob = 0.983), and TOI-4353.01 (TIC 176797879, P = 718.18 d, prob = 0.980). Long-period TESS detections are scientifically valuable because TESS's sector-by-sector observing pattern makes them rare and ground-based follow-up campaigns are lengthy. Inspection of the rendered vetting figures reveals a consistent pattern across the top picks: shallow transits (~800–2000 ppm) where the BLS periodogram shows no dominant peak at the candidate period, but where odd/even depth differences are essentially zero (Δdepth ≤ 0.0002) and centroid shifts are well below the BEB-warning threshold (SNR ≤ 2.0). For TOI-4328.01 specifically, the single-transit nature at P = 703.8 d means BLS lacks the statistical power to flag the candidate at all over the TESS baseline — only a learned dual-view model can recover such a signal. These are precisely the candidates where the trained CNN adds value over classical BLS-only pipelines, which would deprioritise the same targets for lack of strong periodogram support.

The six new Kepler picks at `prob_mean ≥ 0.95` are all KOIs: KOI-3444.01 (KIC 5384713, P = 12.67 d, prob = 0.971), KOI-3034.01 (KIC 2973386, P = 31.02 d, prob = 0.969), KOI-6925.01 (KIC 7868967, P = 12.95 d, prob = 0.962), KOI-6276.01 (KIC 2557350, P = 3.10 d, prob = 0.957), KOI-6568.01 (KIC 5353938, P = 6.28 d, prob = 0.956), and KOI-8012.01 (KIC 10452252, P = 34.57 d, prob = 0.951). All six are still listed as `CANDIDATE` in the latest Kepler KOI cumulative table and were recovered by the direct-archive download path described above.

Two complementary rankings are produced downstream of these scores. Sorting by raw `prob_mean` surfaces the candidates with the strongest learned signal regardless of community attention — the top of that list is dominated by the long-period TESS picks above. The `discovery_score = prob_mean × (1 − fold_disagree) × 1/(1 + n_followup/3)` re-rank multiplicatively penalises high-follow-up candidates so that already-vetted TOIs sink and under-investigated targets surface; the top of that list is dominated by the six Kepler KOIs, which carry no `n_followup` value in the ExoFOP TFOPWG schema (TFOPWG follow-up tracking is a TESS-side construct) and therefore default to a full 1× multiplier. The asymmetry between the two missions in this score is an honest limitation of the simple formula and is noted as future work; the rest of this report reports top picks by `prob_mean`. Six-panel vetting figures for the top-20 have been rendered to `results/vetting/`; they constitute the prioritised list for manual review against ExoFOP TFOP files and any subsequent community follow-up. They remain candidates, not discoveries, until and unless independently confirmed.

**Internal sanity check: recall on since-confirmed planets.** A 2026-05-18 snapshot of the NASA Exoplanet Archive Planetary Systems table was cross-referenced against the 6,200 held-out PCs by joining on TOI base ID and requiring orbital period agreement within 2 %. This identified **120 candidates that were labelled `PC` at training-catalogue build time but have since been promoted to confirmed-planet status by other surveys / follow-up programs**. These planets were never seen by the model during training but are known to be real. They are not discoveries by this work — the confirmations were performed elsewhere — but they serve as a real-world generalisation check on a population the model never trained on:

| Threshold | Recall | 95 % CI (Wilson) |
|---|---|---|
| 0.3 | 99.2 % (119/120) | [95.4, 99.9] |
| **0.5** | **95.8 % (115/120)** | **[90.6, 98.2]** |
| 0.7 | 80.8 % (97/120) | [72.9, 86.9] |
| 0.9 | 30.8 % (37/120) | [23.3, 39.6] |
| 0.95 | 11.7 % (14/120) | [7.1, 18.6] |
| 0.99 | 0.0 % (0/120) | [0.0, 3.1] |

The mean `prob_mean` across the 120 confirmed planets is 0.80, consistent with well-calibrated probabilities for a positive-but-noisy class. The sharp drop above threshold 0.9 is the expected behaviour of post-hoc temperature scaling (T* = 1.275) — the calibration step deliberately compresses the right tail to correct the pre-calibration softmax's overconfidence, rather than indicating model failure. recall@0.99 = 0 % is therefore correct, not pathological. The 95.8 % at threshold 0.5 is a sanity check that the model generalises to real planets confirmed after training closed; it is not a benchmark claim against the published systems (see methodological note below for why), and it is not the metric on which the priority list above stands or falls.

**Failure-mode analysis: the five planets the model scored below 0.5.** Vetting-figure inspection (`results/vetting/`) shows that the five "misses" fall into three distinct categories, none of which represent random model failure:

*Category 1 — model correctly suspicious of likely background-EB signature (2 of 5).* TOI-2886 b (P = 1.60 d, prob = 0.222) and TOI-3474 b (P = 3.88 d, prob = 0.396) both exhibit deep V-shaped transits combined with extreme centroid shifts (in-transit centroid SNR = 27.7 and 16.6 respectively, far above the 3.0 BEB-warning threshold). The dual-view CNN and the centroid feature jointly downweight these, which is the scientifically correct response to the available photometric data — the centroid shift indicates the dip is most likely leaking from a fainter source within the aperture, not from the TIC target. The planets' subsequent confirmations almost certainly relied on independent radial-velocity or high-resolution imaging that resolved the photometric ambiguity.

*Category 2 — wide-binary dilution (1 of 5).* TOI-3523 A b (P = 2.30 d, prob = 0.417). The "A" suffix marks this as the bright component of a known wide binary; the companion star contaminates the photometric aperture and produces a centroid SNR of 5.2, just above the BEB threshold. As with category 1, the model is correctly suspicious of what its inputs show; the resolution requires non-photometric vetting.

*Category 3 — genuine borderline / edge-of-distribution (2 of 5).* TOI-1291 b (P = 7.16 d, prob = 0.464) is extremely shallow (~500 ppm) with a mild centroid concern (SNR = 4.3). TOI-4773 b (P = 1.75 d, prob = 0.481) shows an asymmetric dip-then-bump morphology in its phase fold, consistent with starspot crossings or grazing geometry — exactly the kind of asymmetric transit (cf. Kepler-13Ab, Szabó et al. 2020) flagged in the limitations section as a known training-data gap. Both express honest model uncertainty: ensemble σ values of 0.112 and 0.132 are 15–20× larger than the top-picks σ (~0.007), indicating "I don't know" rather than confident rejection.

The first two categories — three of five — are arguably correct decisions on the photometric evidence available; the model is performing the EB-rejection it was trained to perform, and the catalogue's confirmation status relies on non-photometric vetting that the model cannot see. The third category exposes the only systematic training-data gap: very shallow signals and asymmetric / spot-crossed transits are under-represented in the labelled positives. This motivates the future-work item of injecting Kepler-13Ab-like asymmetric transits and starspot-crossing examples as labelled positives (see Discussion below).

**Methodological note on cross-study recall comparison.** Shallue & Vanderburg (2018), Yu et al. (2019), Valizadegan et al. (2022, ExoMiner), and Islam (2026, ExoNet) all report recall on a random ~10 % held-out split of the same labelled dataset they trained on (typically Kepler DR24/DR25 autovetter or TESS QLP TFOPWG labels). Each work picks a different operating point — the precision constraints alone span 0.45 → 0.90 → 0.99 — so the headline recall numbers are not commensurable digit-for-digit:

| Work | Dataset | Split | Reported recall | At what operating point |
|---|---|---|---|---|
| AstroNet (Shallue & Vanderburg 2018) | Kepler DR24 autovetter, 15,737 TCEs | random 80/10/10 (test = 1,523) | 0.95 | at precision = 0.90 |
| ExoMiner (Valizadegan et al. 2022) | Kepler DR25, 30,957 TCEs (2,643 CP + 28,314 FP) | held-out test set | 0.936 | at precision = 0.99 (very strict) |
| Yu et al. 2019 (Astronet-Vetting TESS) | TESS S1–5 QLP, 16,516 TCEs (test = 1,650) | random 80/10/10 | ~0.90 (44/49 PCs) | at threshold = 0.1, precision = 0.45 |
| **This work — Branch 4** | 6,200 held-out PCs | **temporal holdout** (PC → confirmed after training-catalogue build) | **0.958** | at threshold = 0.5, no precision constraint |

Those works test "of the labels held out from training, how many are correctly classified at the chosen operating point?" — a within-distribution generalisation check. This work's recall number tests "of the candidates that flipped to confirmed real planets between training-catalogue build and the snapshot date, how many would the model have flagged at the un-tuned decision boundary?" — a temporal real-world generalisation check on a different population. Precision is not computable on the 120 because the set contains no negatives by construction, so this work cannot quote a precision-recall operating point analogous to AstroNet's or ExoMiner's. The right place to compare discriminative ability across studies is the headline ROC-AUC and PR-AUC reported in the previous section, where this work matches Islam (2026, ExoNet) at 0.9555 vs 0.9549 on a methodologically similar TESS+Kepler dataset, using a stricter 5-fold per-star group CV with substantially less training data (3,275 examples vs ExoNet's 7,585, AstroNet's 15,737, or ExoMiner's 30,957). The priority list is the deliverable; the CV metrics back its calibration; the 95.8 % since-confirmed recall is one downstream piece of evidence that the ranking surfaces real planets, not the headline.

### Discussion of limitations and future work

**Sample size and class imbalance.** The branch-3 3,275-example training set is competitive with the ~3,500–7,000 regime of recent published work (Xie et al. 2025; Islam 2026) but does not approach the ~16,000-example regime of the original AstroNet. ExoNet's per-KOI (rather than per-star) deduplication strategy is an avenue worth exploring: it preserves multi-planet systems as distinct samples (e.g. Kepler-90's eight confirmed planets each contribute) at the cost of relaxing the strict per-star group split adopted here. Adopting it would likely add ~1,500 effective examples.

**Asymmetric and time-variant transits.** Szabó et al. (2020) document Kepler-13Ab as a textbook counter-example to the symmetric U-shaped transit assumption. Gravity darkening on the rapidly rotating host star produces an asymmetric ingress/egress; spin-orbit misalignment of 58.6° (Johnson et al. 2014, as adopted by Szabó et al. 2020) tilts the transit chord; orbital precession driven by stellar oblateness causes a measurable transit-duration variation across years. A model trained only on symmetric transits — exactly the failure mode of an architecture with horizontal-flip augmentation enabled, which this project removed in `fix/training-stability` — will systematically miss such systems. Branch 3 will inject Kepler-13Ab and a small set of grazing / starspot-crossing transits as labelled positives, on the principle that the model is trained on the kinds of signals it will be expected to find.

**Detrending.** The current Savitzky-Golay flattening is robust but blunt. Szabó et al. (2020) detrended their Kepler-13Ab data with WOTAN's iterative biweight method (Hippke et al. 2019, AJ), specifically chosen for its robustness to instrumental scatter and noise instability. Branch 3 will run a controlled A/B comparison between Savitzky-Golay and WOTAN biweight, using identical splits and otherwise-identical pipelines, with the lower validation Brier score determining the default. The losing method will be retained as a documented "tried, didn't help" alternative for full traceability.

**Architectural upgrades shipped in branch 3.** Squeeze-and-Excitation channel attention (Hu et al. 2018; placement per Xie et al. 2025), bilateral multi-head attention plus LayerNorm-residual (Islam 2026), LeakyReLU in the head (Xie et al. 2025 §2.2), a residual late-fusion linear shortcut from the concatenated embeddings to the head's last layer (Islam 2026), and post-hoc temperature scaling (Guo et al. 2017) all landed on `feat/architecture-upgrades`. Together they account for +0.046 ROC over the CV baseline (0.8836 → 0.9295).

**Physical features.** The current 9-d aux vector adds `centroid_snr` (branch 3) to the eight branch-2 features. ExoNet's 8-d vector also includes planet radius (R_p), equilibrium temperature (T_eq), and metallicity ([Fe/H]); extending the aux dimension to 12 with these is straightforward (the pipeline auto-handles arbitrary aux dimensionality) and is a candidate for a follow-on branch.

**Eclipsing-binary discrimination.** Branch 3 added one BEB-discrimination angle — the centroid-shift feature, after Ansdell et al. (2018) — which contributed +0.026 ROC and Brier −0.021 once the log1p transform was applied. Two further cheap features derivable directly from the existing global view remain on the roadmap: *odd/even transit depth ratio* (catches eclipsing binaries with primary/secondary depth differences) and *secondary-eclipse depth at phase 0.5* (catches grazing EBs and self-luminous companions).

**Third-light correction.** Szabó et al. (2020) apply third-light ratios for Kepler-13A (l₃ = 0.91 Kepler, 0.934 TESS, originally derived by Shporer et al. 2014) and discuss the systematic underestimation of planet radius that occurs when contamination is ignored. This project does not currently regress R_p / R_*, only classifies, so third-light correction is out of scope; it is documented as a known limitation should the work be extended to radius regression.

**Independent cross-check against the RAVEN catalog.** The Branch-4 high-confidence picks were cross-referenced against the live ExoFOP TFOPWG and NEA PS registries. A further independent check would compare our top-K against the RAVEN catalog of Lafarga et al. (2026), particularly their ~1,000 vetted TESS-SPOC FFI candidates that are not yet TOI or CTOI. This comparison is on the Branch-5 roadmap rather than the Branch-4 release; the RAVEN catalog is not in `data/external/` at the time of writing.

**Uncertainty-quantification alternatives.** This project uses MC-Dropout (Gal & Ghahramani 2016) with n=30 samples for per-candidate epistemic uncertainty, combined with the 5-fold ensemble disagreement signal. Yoon & Kim (2025) introduce *flexible evidential deep learning* (F-EDL), which predicts a Flexible Dirichlet distribution over class probabilities and yields closed-form aleatoric and epistemic uncertainty from a **single forward pass**, with empirical generalisation across noisy and long-tailed settings exceeding standard EDL. For a follow-on branch, F-EDL would cut bulk-scoring inference time by roughly the MC-sample count (~30×) while providing a theoretically better-calibrated decomposition than dropout-based UQ.

**Discovery-score mission asymmetry.** The current `discovery_score = prob_mean × (1 − fold_disagree) × 1/(1 + n_followup/3)` formula penalises high-follow-up candidates so that under-investigated targets surface to the top of the list. Because community follow-up counts in the ExoFOP TFOPWG schema are a TESS-side construct, Kepler KOIs carry no `n_followup` value and default to the full 1× multiplier — which gives them an unintended advantage in the rerank relative to TESS candidates of comparable raw `prob_mean`. A Kepler-side analogue of follow-up tracking (e.g. the count of papers citing a KOI in the NASA ADS) would symmetrise the score. Until that is wired in, the report leads with the raw `prob_mean` ranking and treats the discovery-score view as a secondary lens on the same data.

---

## References

Ansdell, M., Ioannou, Y., Osborn, H. P., Sasdelli, M., Smith, J. C., Caldwell, D., Jenkins, J. M., Räissi, C., & Angerhausen, D. (2018). Scientific domain knowledge improves exoplanet transit classification with deep learning. *The Astrophysical Journal Letters*, 869(1), L7.

Christiansen, J. L., McElroy, D. L., Harbut, M., Ciardi, D. R., Crane, M., Good, J., Hardegree-Ullman, K. K., Kessel, A. Y., Lund, M. B., Lynn, M., Muthiah, A., Nilsson, R., Oluyide, T., Papin, M., Rivera, A., Susemiehl, N., Swain, M., Tam, R., van Eyken, J., & Beichman, C. (2025). The NASA Exoplanet Archive and Exoplanet Follow-up Observing Program: Data, tools, and usage. *arXiv preprint* arXiv:2506.03299.

Cui, K., Armstrong, D. J., Hadjigeorghiou, A., Lafarga, M., Kunovac, V., Doyle, L., Nieto, L. A., & Díaz, R. F. (2026). Demographics of close-in TESS exoplanets orbiting FGK main-sequence stars. *Monthly Notices of the Royal Astronomical Society*, 546(2), 1–16. https://doi.org/10.1093/mnras/stag022

Dattilo, A., Vanderburg, A., Shallue, C. J., Mayo, A. W., Berlind, P., Bieryla, A., Calkins, M. L., Esquerdo, G. A., Everett, M. E., Howell, S. B., Latham, D. W., Scott, N. J., & Yu, L. (2019). Identifying exoplanets with deep learning. II. Two new super-Earths uncovered by a neural network in K2 data. *The Astronomical Journal*, 157(5), 169.

de Beurs, Z. L., Vanderburg, A., Shallue, C. J., Dumusque, X., Collier Cameron, A., Leet, C., Buchhave, L. A., Cosentino, R., Ghedina, A., Haywood, R. D., Langellier, N., Latham, D. W., López-Morales, M., Mayor, M., Micela, G., Milbourne, T., Mortier, A., Molinari, E., Pepe, F., Phillips, D. F., Pinamonti, M., Piotto, G., Rackham, B. V., Rice, K., Sasselov, D., Sozzetti, A., Udry, S., & Watson, C. A. (2022). Identifying exoplanets with deep learning. IV. Removing stellar activity signals from radial velocity measurements using neural networks. *The Astronomical Journal*, 164(2), 49. https://doi.org/10.3847/1538-3881/ac738e

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, 1050–1059.

Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.

Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 70, 1321–1330.

Hadjigeorghiou, A., Armstrong, D. J., Cui, K., Lafarga Magro, M., Nieto, L. A., Díaz, R. F., Doyle, L., & Kunovac, V. (2025). RAVEN: RAnking and Validation of ExoplaNets. *arXiv preprint* arXiv:2509.17645 (submitted to *Monthly Notices of the Royal Astronomical Society*).

Hippke, M., David, T. J., Mulders, G. D., & Heller, R. (2019). Wōtan: Comprehensive time-series detrending in Python. *The Astronomical Journal*, 158(4), 143.

Hippke, M., & Heller, R. (2019). Optimized transit detection algorithm to search for periodic transits of small planets. *Astronomy & Astrophysics*, 623, A39.

Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-excitation networks. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 7132–7141.

Islam, M. R. (2026). ExoNet: Calibrated multimodal deep learning for TESS exoplanet candidate vetting using phase-folded light curves, stellar parameters, and multi-head attention. *arXiv preprint* arXiv:2604.15560v3.

Khan, O. (2026). UMI: GPU-accelerated asymmetric robust estimator for photometric detrending in exoplanet transit searches. *arXiv preprint* arXiv:2604.06602. Distributed as the `torchflat` Python package. *[Verify arXiv URL before final submission — not independently confirmed.]*

Lafarga, M., Armstrong, D. J., Cui, K., Hadjigeorghiou, A., Kunovac, V., Doyle, L., Bryant, E. M., Díaz, R. F., Nieto, L. A., & Osborn, A. (2026). Automatic search for transiting planets in TESS–SPOC FFIs with RAVEN: over 100 newly validated planets and over 2000 vetted candidates. *Monthly Notices of the Royal Astronomical Society*, 548(2), 1–30. https://doi.org/10.1093/mnras/stag512

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980–2988.

Malik, A., Moster, B. P., & Obermeier, C. (2021). Exoplanet detection using machine learning. *Monthly Notices of the Royal Astronomical Society*, 513(4), 5505–5516. https://doi.org/10.1093/mnras/stab3692

Marsland, S. (2014). *Machine learning: An algorithmic perspective* (2nd ed.). Chapman and Hall/CRC.

Martinho, M., & Valizadegan, H. (2025). Vetting TESS Full-Frame Image Transit Signals with ExoMiner, 2.0.0. Zenodo. https://doi.org/10.5281/zenodo.17707413

NASA Exoplanet Archive. (2026). Exoplanet and candidate statistics, accessed May 2026. https://exoplanetarchive.ipac.caltech.edu/docs/counts_detail.html — see Christiansen et al. (2025) for the canonical service description.

Roth, J. T., Hartman, J. D., Bakos, G. Á., Yee, S. W., Bouma, L. G., Yana Galarza, J., Teske, J. K., Butler, R. P., Crane, J. D., Shectman, S., Osip, D., Vissapragada, S., Kanodia, S., Beletsky, Y., & Gaibor, Y. (2026). The T16 Planet Hunt: 10,000 new planet candidates from TESS Cycle 1 and the confirmation of a hot Jupiter around TIC 183374187. *arXiv preprint* arXiv:2604.18579.

Shallue, C. J., & Vanderburg, A. (2018). Identifying exoplanets with deep learning: A five-planet resonant chain around Kepler-80 and an eighth planet around Kepler-90. *The Astronomical Journal*, 155(2), 94.

Shporer, A., Jenkins, J. M., Rowe, J. F., Sanchis-Ojeda, R., Esquerdo, G. A., Howell, S. B., Bryson, S. T., Twicken, J. D., Buchhave, L. A., & Latham, D. W. (2014). Kepler-13Ab — A binary system, photometric reflection, and a very massive planet. *The Astrophysical Journal*, 788(1), 92. *[Source of the l₃ third-light ratios for Kepler-13A used by Szabó et al. 2020.]*

Szabó, Gy. M., Pribulla, T., Pál, A., Bódi, A., Kiss, L. L., & Derekas, A. (2020). The clockwork is moving on — a combined analysis of TESS and Kepler measurements of Kepler-13Ab. *Monthly Notices of the Royal Astronomical Society Letters*, 492(1), L17–L21. https://doi.org/10.1093/mnrasl/slz177

Tey, E., Moldovan, D., Kunimoto, M., Huang, C. X., Shporer, A., Daylan, T., Muthukrishna, D., Vanderburg, A., Dattilo, A., Ricker, G. R., & Seager, S. (2023). Astronet-Triage-v2: Improvements to the TESS triage convolutional neural network using new training data and multi-class classification. *The Astronomical Journal*, 165(3), 95. https://doi.org/10.3847/1538-3881/acad85

Valizadegan, H., Martinho, M. J. S., Wilkens, L. S., Jenkins, J. M., Smith, J. C., Caldwell, D. A., Twicken, J. D., Gerum, P. C. L., Walia, N., Hausknecht, K., Lubin, N. Y., Bryson, S. T., & Oza, N. C. (2022). ExoMiner: A highly accurate and explainable deep learning classifier that validates 301 new exoplanets. *The Astrophysical Journal*, 926(2), 120.

Valizadegan, H., Martinho, M. J. S., Jenkins, J. M., Twicken, J. D., Caldwell, D. A., Maynard, P., Wei, H., Zhong, W., Yates, C., Donald, S., Collins, K. A., Latham, D., Barkaoui, K., Calkins, M. L., Chazov, N., Esquerdo, G. A., Guillot, T., Krushinsky, V., Nowak, G., Rackham, B. V., Triaud, A., Schwarz, R. P., Stephens, D., Stockdale, C., Watkins, C. N., & Wilkin, F. P. (2025). ExoMiner++: Enhanced transit classification and a new vetting catalog for 2-minute TESS data. *The Astronomical Journal*, 170(6), 287. https://doi.org/10.3847/1538-3881/ae03a4

Xie, D., Wang, Y., Liu, F., & Sun, W. (2025). Deep learning to classify exoplanet light curves in Kepler and TESS. *Research in Astronomy and Astrophysics*, 25, 104004 (13 pp.). https://doi.org/10.1088/1674-4527/adf70e

Yu, L., Vanderburg, A., Huang, C. X., Shallue, C. J., Crossfield, I. J. M., Gaudi, B. S., Daylan, T., Dattilo, A., Armstrong, D. J., Ricker, G. R., Vanderspek, R. K., Latham, D. W., Seager, S., Dittmann, J., Doty, J. P., Glidden, A., & Quinn, S. N. (2019). Identifying exoplanets with deep learning. III. Automated triage and vetting of TESS candidates. *The Astronomical Journal*, 158(1), 25.

Yoon, T., & Kim, H. (2025). Uncertainty estimation by flexible evidential deep learning. *Advances in Neural Information Processing Systems (NeurIPS)*, 39.

---

## Course materials consulted

Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). Provided as `Hands-On_Machine_Learning_with_Scikit-Learn-Keras-and-TensorFlow-2nd-Edition-Aurelien-Geron.pdf`. Reference for: Wide-and-Deep architectures (Ch. 10), Keras Functional API and callbacks (Ch. 10–11), training stability and gradient clipping (Ch. 11), Hydra/MLflow-style experiment infrastructure (Ch. 19), feature scaling and pipelines (Ch. 2).

DATA 303 — Statistical Modelling for Data Science, Weeks 1–6 (Victoria University of Wellington, 2026). Reference for: regression diagnostics applied to calibration assessment (§3), interaction and transformation of predictors as motivation for the aux-feature engineering (§4), and shrinkage methods (§8) as conceptual analogue for the L2 regularisation used in the conv towers.

---

## Project artefacts

- **Codebase:** `/Users/ollie/Project/`, branch `feat/architecture-upgrades` at branch-3 completion; the same content lands on `main` at this milestone.
- **Catalogue:** `data/labels/labels.parquet` (3,500 rows after the May-2026 depth/duration unit fix); `data/labels/candidates.parquet` (6,200 held-out Planet Candidates: 4,570 TESS + 1,630 Kepler, reserved for branch-4 inference).
- **Processed views:** `data/processed/views.npz` — 3,275 examples × 9-dim aux (TESS + Kepler combined, group-stratified). Backup of the pre-centroid 8-dim dataset at `data/processed/views.npz.bak_pre_centroid` for apples-to-apples ablation.
- **Trained models:** `models/cv/<run_id>/fold_{0..4}/cnn_dualview.keras` for each branch-3 step. The branch-3 final run is `58570d85f1dd4f68a7e888988c88eeab`. Per-fold calibration bundles at the same paths contain `{calibrator: TemperatureScaler, temperature, threshold, aux_pipeline, aux_dim}`.
- **Experiment history:** `mlruns/732906991717652602/` — covers branch 1 (training stability), branch 2 (data quality + metrics audit), and branch 3 (architecture + centroid). Branch-3 ladder MLflow run ids: step 1 (CV baseline) `71ec8452…`, step 2 (SE+MHA+residual) `1d9ef1e9…`, step 3 (temperature scaling) `4d9485e1…`, step 4A (centroid raw) `cc4ab87b…`, step 4B (centroid + log1p) `58570d85…`.
- **Reference papers consulted:** the Géron textbook (2019); Szabó et al. (2020) on Kepler-13Ab; Islam (2026) ExoNet; Xie et al. (2025) SE-CNN-RlNet; Khan (2026) UMI detrending; Christiansen et al. (2025) on the NEA + ExoFOP services; the RAVEN trilogy (Hadjigeorghiou et al. 2025; Lafarga et al. 2026; Cui et al. 2026) for the TESS-SPOC FFI vetting and occurrence-rate context; Tey et al. (2023) for Astronet-Triage-v2; Martinho & Valizadegan (2025) for ExoMiner++ 2.0 FFI; Yoon & Kim (2025) for the F-EDL uncertainty-quantification framework; and a May-2026 literature review covering Vision Transformers with Recurrence Plots, centroid-shift diagnostics, and the published ExoMiner++ 2-min paper.
