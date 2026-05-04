# Detecting Exoplanet Transits in TESS and Kepler Light Curves with a Calibrated Dual-View 1D CNN

**Author:** Oliver Donaldson
**Project type:** Personal research / portfolio piece
**Background:** Data Science student, Victoria University of Wellington (DATA 305 — Machine Learning, DATA 303 — Statistical Modelling)
**Status:** *DRAFT — written 2026-05-03. Final metrics pending the in-progress combined TESS+Kepler training run. All numbers reported as "current" reflect the 21 April 2026 baseline (`mlflow run 8dce07454c`); numbers reported as "target" come from the closest published comparator.*

---

## How this project was built

This is a personal-interest project. It started with a question I wanted to answer for myself — *can a deep-learning model trained on real NASA TESS data actually identify exoplanet transits, and what would it take to build one?* — and it became the vehicle I used to apply the deep-learning material from DATA 305 to a real research problem rather than a textbook one.

I built it with Anthropic's Claude (Sonnet 4.7, via the Claude Code CLI) as a pair-programming and tutoring collaborator. The collaboration was deliberately learning-first: I supplied the vision, the references, and the judgment calls; Claude wrote and explained code grounded in those references; I executed everything on my own machine, reviewed every change, and pushed back where it didn't make sense. Listing what each side actually contributed feels more honest than a generic "AI was used" line, and is what I'd want to see in someone else's portfolio:

**What I brought:**
- The original idea, the choice of problem (exoplanet transit detection), datasets (TESS + Kepler via MAST), labels (NASA Exoplanet Archive), and target architecture family (dual-view CNN per Shallue & Vanderburg 2018).
- All reference materials: my coursework notes from DATA 305 (Marsland 2014) and DATA 303 (2026), the Géron (2019) textbook, and the research papers cited in this report (Howarth & Morello 2020 / Kepler-13Ab; Islam 2026 / ExoNet; Xie et al. 2025 / SE-CNN-RlNet).
- All design judgment calls: branch scoping, when to retrain, what to keep vs scrap, storage allocation between internal SSD and external USB, prioritising data scaling over architecture upgrades (and vice versa).
- All code execution and review — every dataset build, every training run, every git commit — on my own laptop, with my eyes on the output before moving on.
- Pushback when things didn't make sense: questioning verbose comments, questioning sample sizes before committing to a long retrain, catching dead config knobs that no longer did anything.
- Final shape of this report — what to emphasise, what to acknowledge as a limitation, how to frame it.

**What Claude drafted, with my approval:**
- The initial codebase scaffold (Hydra + MLflow + dual-view CNN + RF baseline + preprocessing modules + tests).
- Specific patches on the `fix/training-stability` branch: gradient clipping, the impute-and-scale aux pipeline, the persisted calibrator/scaler bundle, the removal of dead features, the `.gitignore` bug fix, the cross-script synchronisation of aux dimensionality.
- Explanatory writeups linking design choices to specific references (e.g. how Wide-and-Deep from Géron Ch. 10 maps onto the aux-feature path; why Howarth & Morello's findings argue against horizontal-flip augmentation; how ExoNet's per-KOI deduplication contrasts with the per-TIC group split adopted here).
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

We present an end-to-end deep-learning pipeline for detecting transiting exoplanets in NASA TESS and Kepler light curves. The system downloads stitched SPOC/Kepler light curves directly from MAST via `lightkurve`, builds the dual-view (global + local) phase-folded representation of Shallue & Vanderburg (2018), and feeds it through a 1D convolutional neural network with a Wide-and-Deep auxiliary stellar-feature path implemented in the Keras Functional API. The training pipeline is group-stratified by host star to prevent multi-planet leakage, uses gradient clipping and a fitted `SimpleImputer→StandardScaler` aux pipeline to ensure numerical stability, and produces calibrated transit probabilities via isotonic regression with an F1-optimised decision threshold. A baseline Random Forest on hand-crafted features provides a classical-ML reference. On a 1,959-example TESS-only training set, the current model achieves ROC-AUC = 0.945 / test ROC-AUC = 0.901 / F1 = 0.91. Ongoing work expands the dataset to ~3,500 TESS+Kepler examples, addresses the "training stability" failures observed in earlier runs, and incorporates findings from three recent papers — Howarth & Morello (2020) on transit asymmetry, Islam (2026)'s ExoNet, and Xie et al. (2025)'s SE-CNN-RlNet — to motivate planned architectural and feature upgrades. The full pipeline is reproducible via Hydra + MLflow, and all code is open source.

---

## Introduction

The transit method has confirmed 4,307 of the 5,787 known exoplanets to date (>74% of the catalogue). NASA's *Transiting Exoplanet Survey Satellite* (TESS), operating since April 2018, has catalogued over 7,800 candidates, of which fewer than 720 have been independently confirmed (Islam, 2026). The classification backlog — over 7,000 unreviewed candidate signals — is structurally beyond the capacity of manual expert vetting and motivates an automated, calibrated, and reproducible machine-learning vetting pipeline.

This project is a portfolio piece for DATA 305 (Marsland, 2014), grounded in the deep-learning tooling covered in Géron (2019) — Keras Functional API, callbacks, Wide-and-Deep architectures, MLflow experiment tracking — and the regression diagnostics taught in DATA 303 (which informed the calibration and uncertainty-quantification components). The technical objective is to build, from real data only (no synthetic light curves in the final training set), a pipeline that can:

1. **Classify** brightness dips in unreviewed targets as *transits* vs *false positives*;
2. **Calibrate** its predicted probabilities so they reflect true likelihoods, suitable for downstream candidate prioritisation;
3. **Generalise** beyond the symmetric textbook transit, on which most published models are trained, to handle the gravity-darkened, spin-orbit-misaligned, and time-variant transit shapes that occur in practice;
4. **Be reproducible**, with every experiment tracked, every hyperparameter version-controlled, and every dataset rebuildable from the catalogue queries.

The architectural baseline is Shallue & Vanderburg (2018)'s "AstroNet" — a dual-view 1D CNN that ingests a low-resolution global phase fold and a high-resolution local zoom of the transit window. Subsequent work (Ansdell et al., 2018; Yu et al., 2019; Dattilo et al., 2019; Valizadegan et al., 2022, 2025) has extended this baseline with stellar-context features, transfer learning across missions, and richer multi-branch encodings. Two 2025–2026 papers in particular shape the upgrade path adopted here: Xie et al. (2025) demonstrated that channel-attention (Squeeze-and-Excitation) blocks plus a residual fully-connected head substantially improve training stability and accuracy; Islam (2026) introduced trimodal late fusion with multi-head attention over the CNN feature map, achieving ROC-AUC = 0.955 on Kepler. The present work adopts these as the planned architecture for branch 2 of the project.

A separate physical-realism concern is raised by Howarth & Morello (2020), whose detailed analysis of Kepler-13Ab — a hot Jupiter orbiting a rapidly rotating, gravity-darkened star — demonstrates that real transits can be substantially asymmetric, can exhibit transit-duration variation (TDV) due to orbital precession, and require third-light correction when contaminated by a binary companion. Models trained exclusively on perfectly U-shaped transits will systematically under-detect such cases. Branch 3 of this project addresses this by including hard-example asymmetric transits in the training set and by adding the planet-radius and stellar-metallicity features that ExoNet uses to disambiguate eclipsing-binary contaminants from genuine planetary signals.

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

The output is a single compressed numpy archive (`data/processed/views.npz`) containing `global_views`, `local_views`, `labels`, `tic_ids`, and an 8-dimensional `aux_features` vector per target: `[T_eff, R_*, log g, T_mag, depth, duration, log P, SNR]`.

### Model architecture

The principal model is a dual-view 1D CNN (`src/exoplanet_hunter/models/cnn_dualview.py`) implemented in the Keras Functional API. The architecture follows Shallue & Vanderburg (2018) with extensions adopted from Ansdell et al. (2018):

- **Global tower:** 3 convolutional blocks (16, 32, 64 filters), 2 conv layers per block with kernel 5, BatchNorm, ReLU, MaxPool size 5, optional SpatialDropout. Terminates in GlobalAveragePooling1D producing a 64-d embedding.
- **Local tower:** 2 convolutional blocks (16, 32 filters), 2 conv layers per block with kernel 5, MaxPool size 3. Terminates in GlobalAveragePooling1D producing a 32-d embedding.
- **Wide path (auxiliary):** the 8-d standardised stellar/transit feature vector concatenated *directly* with the global and local embeddings (the Wide-and-Deep pattern from Géron Ch. 10).
- **Head:** two fully-connected layers (256, 128 units) with ReLU + BatchNorm + Dropout (p = 0.4), followed by a sigmoid output unit. Dropout is left enabled at inference time (`training=True`) so MC-Dropout uncertainty estimation is available downstream (Gal & Ghahramani, 2016).

A baseline Random Forest classifier on hand-crafted features (depth, duration, depth-SNR, ingress slope, secondary-eclipse depth, odd/even depth ratio) provides the classical-ML reference, with k-fold CV and SHAP feature-importance plots for interpretability.

### Training

Training runs are launched via Hydra (`scripts/train_model.py`); all hyperparameters live in composable YAML configs under `conf/`. Per-run experiment tracking, including resolved configs, learning curves, evaluation plots, and model artefacts, is logged to MLflow.

- **Split:** 70/15/15 train/val/test, *grouped by `tic_id`* with `sklearn.model_selection.GroupShuffleSplit`. Multi-planet systems and re-observed TICs are kept entirely within a single split; without this, test AUC is inflated by 2–5 points through "seen this star before" leakage.
- **Optimiser:** Adam, learning rate 5×10⁻⁴, **with `clipnorm = 1.0` to cap gradient norms**. This was added after multiple earlier runs collapsed to `loss = NaN` mid-training, traced to unscaled stellar-parameter inputs (T_eff ≈ 5,800 vs log_period ≈ 1) producing exploding gradients in the dense head.
- **Aux feature pipeline:** raw aux features are passed through a `sklearn.Pipeline([SimpleImputer(strategy="median"), StandardScaler])`, fitted on the training split only and reused (not refit) at val/test/inference. The fitted pipeline is persisted alongside the model checkpoint so `score_target.py` reproduces the exact training-time preprocessing.
- **Loss:** binary cross-entropy by default; binary focal loss (γ = 2, α = 0.75) optionally available for stronger negative-class downweighting. When focal loss is active, `class_weight` is disabled to prevent double-counting.
- **Augmentation:** small Gaussian noise (σ = 5×10⁻⁴), small phase shifts (±0.5 %), random depth scaling (±5 %), and 2 % random bin masking. *Time-flip augmentation was removed* on the basis of Howarth & Morello (2020): real transits are not symmetric, and flipping them mislabels asymmetric ingress/egress shapes.
- **Callbacks:** `EarlyStopping` on `val_auc` (patience 25, restore best), `ModelCheckpoint` on `val_auc`, `ReduceLROnPlateau` on `val_loss` (factor 0.5, patience 8). All standard from the DATA 305 / Géron Ch. 11 toolbox.
- **Calibration:** isotonic regression fitted on validation predictions, applied to test scores. Decision threshold selected by sweeping ∈ [0.05, 0.95] on the validation set and choosing the value that maximises F1.

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

### Current performance

The most recent stable training run (`mlflow run 8dce07454c`, 21 April 2026, `cnn-large` configuration on 1,959 examples) achieved:

| Metric | Train | Validation | Test |
|---|---|---|---|
| ROC-AUC | 0.945 | 0.887 | 0.901 |
| Accuracy | 0.883 | 0.871 | — |
| Precision | 0.971 | 0.922 | 0.918 |
| Recall | 0.879 | 0.915 | 0.911 |
| F1 | — | — | 0.915 |
| Brier (uncalibrated) | — | 0.109 | — |
| Brier (calibrated) | — | — | 0.092 |

Earlier runs in the same session showed the training-stability failure mode this project explicitly addressed: one `cnn-large` run terminated at epoch 25 with `loss = NaN`, AUC = 0.5 (chance-level). Diagnostic investigation traced this to the combination of (i) un-scaled raw stellar parameters being concatenated into the dense head, (ii) no gradient clipping on the Adam optimiser, and (iii) a pathological interaction with focal loss when `class_weight` was also active. All three are now mitigated; the `fix/training-stability` branch implements the changes summarised in the methodology.

### In-progress: combined TESS+Kepler training

A combined-mission build is currently running (~3,500 examples target — 1,000 TESS + 2,500 KOI). When complete, this run will provide the first data point at a sample size comparable to published work and will isolate the effect of the stability fixes from any sample-size-driven gain. *Numbers to be inserted once the training run completes.*

### Comparison with published baselines

| Study | Year | Mission | Sample | Architecture | Reported metric |
|---|---|---|---|---|---|
| Shallue & Vanderburg | 2018 | Kepler DR24 | 15,737 | Dual-view 1D CNN ("AstroNet") | 98% accuracy |
| Ansdell et al. | 2018 | Kepler | ~16,000 | AstroNet + scalar aux features | Improved over baseline |
| Dattilo et al. | 2019 | K2 | — | AstroNet (transferred) | Two new planets confirmed |
| Yu et al. | 2019 | TESS (simulated) | — | AstroNet + stellar depth | Recall 61% on real TESS (degraded) |
| Valizadegan et al. (ExoMiner) | 2022 | Kepler | — | Multi-branch CNN | 301 new exoplanets validated |
| Valizadegan et al. (ExoMiner++) | 2025 | TESS 2-min | — | Transfer learning from Kepler | 7,330 TESS candidates |
| Xie et al. (SE-CNN-RlNet) | 2025 | Kepler + TESS | ~7,000 | AstroNet + SE channel attention + residual MLP | F1 = 0.957 (Kepler), 0.995 (TESS) |
| Islam (ExoNet) | 2026 | Kepler + TESS | 7,585 | AstroNet + Multi-Head Attention + residual late fusion + temperature scaling | ROC-AUC = 0.955 |
| **This work — current** | 2026 | TESS only | 1,959 | AstroNet + Wide&Deep | ROC-AUC = 0.901 (test) |
| **This work — projected** | 2026 | TESS + Kepler | ~3,500 | as above + branch 2 architecture upgrades | TBD |

The current model is competitive with, but not yet at, the SOTA reported by ExoNet and SE-CNN-RlNet. The two principal sample-size gaps (1,959 vs ~7,500) and architectural gaps (no attention, no residual head) explain most of the deficit. Branches 2 and 3 of this project address both.

### Discussion of limitations and future work

**Sample size and class imbalance.** The current 1,959-example training set is small by the standards of published work (5–8× smaller than ExoNet, 8× smaller than Shallue & Vanderburg). The combined TESS+Kepler build now in progress addresses this for the dual-mission case but does not approach the ~16,000-example regime of the original AstroNet. ExoNet's per-KOI (rather than per-star) deduplication strategy is an avenue worth exploring in branch 2: it preserves multi-planet systems as distinct samples (e.g. Kepler-90's eight confirmed planets each contribute) at the cost of relaxing the strict per-star group split adopted here.

**Asymmetric and time-variant transits.** Howarth & Morello (2020) document Kepler-13Ab as a textbook counter-example to the symmetric U-shaped transit assumption. Gravity darkening on the rapidly rotating host star produces an asymmetric ingress/egress; spin-orbit misalignment tilts the transit chord; orbital precession driven by stellar oblateness causes a measurable transit-duration variation across years. A model trained only on symmetric transits — exactly the failure mode of an architecture with horizontal-flip augmentation enabled, which this project removed in `fix/training-stability` — will systematically miss such systems. Branch 3 will inject Kepler-13Ab and a small set of grazing / starspot-crossing transits as labelled positives, on the principle that the model is trained on the kinds of signals it will be expected to find.

**Detrending.** The current Savitzky-Golay flattening is robust but blunt. Howarth & Morello (2020) detrended their data with WOTAN's biweight method, specifically chosen for its robustness to instrumental scatter and noise instability. Branch 3 will run a controlled A/B comparison between Savitzky-Golay and WOTAN biweight, using identical splits and otherwise-identical pipelines, with the lower validation Brier score determining the default. The losing method will be retained as a documented "tried, didn't help" alternative for full traceability.

**Architectural upgrades.** Two cheap, high-leverage upgrades from Xie et al. (2025) — substituting LeakyReLU for ReLU, and inserting Squeeze-and-Excitation channel-attention blocks into each conv tower — are scheduled for branch 2. Islam (2026)'s Multi-Head Attention layer over the global feature map and residual late-fusion head are also on the roadmap. Calibration will be migrated from isotonic regression to temperature scaling, which produces a single learnable scalar that preserves prediction rankings.

**Physical features.** The current 8-d aux vector includes T_eff, R_*, log g, T_mag, transit depth, duration, log period, and SNR. ExoNet's 8-d vector also includes planet radius (R_p), equilibrium temperature (T_eq), and metallicity ([Fe/H]). Branch 3 will extend the aux dimension to 11 with these additions; the existing pipeline auto-handles arbitrary aux dimensionality.

**Eclipsing-binary discrimination.** Two cheap features derivable directly from the existing global view — *odd/even transit depth ratio* and *secondary-eclipse depth at phase 0.5* — are strong discriminators of the most common false-positive class. Both will be added in branch 3.

**Third-light correction.** Howarth & Morello (2020) provide third-light ratios (l₃ = 0.91 Kepler, 0.93 TESS) for Kepler-13A and discuss the systematic underestimation of planet radius that occurs when contamination is ignored. This project does not currently regress R_p / R_*, only classifies, so third-light correction is out of scope; it is documented as a known limitation should the work be extended to radius regression.

**Inference and discovery.** Branch 4 (planned) will score the 6,203 held-out TOI Planet Candidates in `data/labels/candidates.parquet`, build a six-panel vetting figure (phase-folded global, local, odd/even, BLS periodogram, centroid drift, model probability + MC-Dropout uncertainty), and submit the top high-confidence candidates for manual review against ExoFOP / TFOP records. This is the discovery pathway: any candidates surviving manual review against existing dispositions would be eligible for community follow-up.

---

## References

Ansdell, M., Ioannou, Y., Osborn, H. P., Sasdelli, M., Smith, J. C., Caldwell, D., Jenkins, J. M., Räissi, C., & Angerhausen, D. (2018). Scientific domain knowledge improves exoplanet transit classification with deep learning. *The Astrophysical Journal Letters*, 869(1), L7.

Dattilo, A., Vanderburg, A., Shallue, C. J., Mayo, A. W., Berlind, P., Bieryla, A., Calkins, M. L., Esquerdo, G. A., Everett, M. E., Howell, S. B., Latham, D. W., Scott, N. J., & Yu, L. (2019). Identifying exoplanets with deep learning. II. Two new super-Earths uncovered by a neural network in K2 data. *The Astronomical Journal*, 157(5), 169.

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. *Proceedings of the 33rd International Conference on Machine Learning (ICML)*, 1050–1059.

Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.

Hippke, M., & Heller, R. (2019). Optimized transit detection algorithm to search for periodic transits of small planets. *Astronomy & Astrophysics*, 623, A39.

Howarth, I. D., & Morello, G. (2020). Kepler-13Ab — gravity darkening, transit timing, and the changing transit duration. *Monthly Notices of the Royal Astronomical Society Letters*, 492(1), L17–L21.
*[The "Kepler-13Ab" paper provided to us as `mnrasl_492_1_l17.pdf`. Verify exact author/title before final submission.]*

Islam, M. R. (2026). ExoNet: Calibrated multimodal deep learning for TESS exoplanet candidate vetting using phase-folded light curves, stellar parameters, and multi-head attention. *arXiv preprint* arXiv:2604.15560v3.

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2980–2988.

Marsland, S. (2014). *Machine learning: An algorithmic perspective* (2nd ed.). Chapman and Hall/CRC.

Shallue, C. J., & Vanderburg, A. (2018). Identifying exoplanets with deep learning: A five-planet resonant chain around Kepler-80 and an eighth planet around Kepler-90. *The Astronomical Journal*, 155(2), 94.

Valizadegan, H., Martinho, M. J. S., Wilkens, L. S., Jenkins, J. M., Smith, J. C., Caldwell, D. A., Twicken, J. D., Gerum, P. C. L., Walia, N., Hausknecht, K., Lubin, N. Y., Bryson, S. T., & Oza, N. C. (2022). ExoMiner: A highly accurate and explainable deep learning classifier that validates 301 new exoplanets. *The Astrophysical Journal*, 926(2), 120.

Valizadegan, H., et al. (2025). ExoMiner++: Transfer-learning-based exoplanet vetting for TESS. *[ApJ in press / arXiv pending — confirm citation.]*

Xie, D., Wang, Y., Liu, F., & Sun, W. (2025). Deep learning to classify exoplanet light curves in Kepler and TESS. *Research in Astronomy and Astrophysics*, 25, 104004 (13 pp.). https://doi.org/10.1088/1674-4527/adf70e

Yu, L., Vanderburg, A., Huang, C. X., Shallue, C. J., Crossfield, I. J. M., Gaudi, B. S., Daylan, T., Dattilo, A., Armstrong, D. J., Ricker, G. R., Vanderspek, R. K., Latham, D. W., Seager, S., Dittmann, J., Doty, J. P., Glidden, A., & Quinn, S. N. (2019). Identifying exoplanets with deep learning. III. Automated triage and vetting of TESS candidates. *The Astronomical Journal*, 158(1), 25.

---

## Course materials consulted

Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). Provided as `Hands-On_Machine_Learning_with_Scikit-Learn-Keras-and-TensorFlow-2nd-Edition-Aurelien-Geron.pdf`. Reference for: Wide-and-Deep architectures (Ch. 10), Keras Functional API and callbacks (Ch. 10–11), training stability and gradient clipping (Ch. 11), Hydra/MLflow-style experiment infrastructure (Ch. 19), feature scaling and pipelines (Ch. 2).

DATA 303 — Statistical Modelling for Data Science, Weeks 1–6 (Victoria University of Wellington, 2026). Reference for: regression diagnostics applied to calibration assessment (§3), interaction and transformation of predictors as motivation for the aux-feature engineering (§4), and shrinkage methods (§8) as conceptual analogue for the L2 regularisation used in the conv towers.

---

## Project artefacts

- **Codebase:** `/Users/ollie/Project/`, branch `fix/training-stability` at time of writing.
- **Catalogue:** `data/labels/labels.parquet` (4,000 rows), `data/labels/candidates.parquet` (6,203 held-out planet candidates).
- **Processed views:** `data/processed/views.npz` (currently 1,959 examples; combined-mission rebuild in progress).
- **Trained model:** `models/cnn_dualview.keras` (1.5 MB), `models/cnn_calibrator.joblib` (calibrator + threshold + aux pipeline bundle).
- **Experiment history:** `mlruns/732906991717652602/` — 14 runs across stability-fix iterations.
- **Reference papers consulted:** `mnrasl_492_1_l17.pdf` (Howarth & Morello 2020), `2604.15560v3.pdf` (ExoNet), `Deep Learning to Classify Exoplanet Light Curves in Kepler and TESS.pdf` (Xie et al. 2025), and the Géron textbook.
