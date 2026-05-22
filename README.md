# Exoplanet Hunter

A calibrated deep-learning pipeline that **vets unconfirmed exoplanet transit
candidates** from NASA TESS and Kepler light curves. Trained on 3,275
human-vetted candidates, achieves headline parity with published vetting
models (Islam 2026 / ExoNet) using ~½ the training data, and produces a
ranked priority list of 6,200 unconfirmed candidates with uncertainty bands.

| Metric | Value |
|---|---|
| **ROC-AUC** (5-fold group CV) | **0.9555 ± 0.0044** |
| **PR-AUC** (5-fold group CV) | 0.9586 ± 0.0058 |
| **F1** (at fold-optimal τ\*) | 0.888 ± 0.012 |
| **Brier** (calibrated) | 0.0905 |
| **Training examples** | 3,275 (TESS + Kepler) |
| **Discovery pool scored** | **5,388 / 6,200 = 86.9 %** |
| **High-confidence picks** (`prob_mean ≥ 0.95`) | **146** (140 TESS, 6 Kepler) |

> **Honest framing.** This is a *vetting* pipeline, not a discovery one. It
> takes already-flagged candidates (TOI-PC, KOI-CANDIDATE) and produces a
> calibrated probability that the signal is a real transit. Independent
> follow-up is still required for confirmation. See
> [docs/research_report_draft.md](docs/research_report_draft.md#methodological-note-on-cross-study-recall-comparison)
> for the careful comparison with published baselines.

## Headline top pick — TOI-4328.01

![TOI-4328.01 vetting figure](docs/figures/toi-4328-01_tic_77175217.png)

A P = 703.79 d, ~800 ppm long-period TESS candidate (TIC 77175217) where
the BLS periodogram lacks the statistical power to flag the transit but the
dual-view CNN scores it at `prob_mean = 0.989, fold_disagree = 0.006`. The
single-transit nature over the TESS baseline makes long-period detections
exactly the regime where a learned model adds value over classical
periodogram-only pipelines.

## What I built

I wanted to answer a personal-interest question: *can a deep-learning model
trained on real NASA light curves actually identify exoplanet transits,
and what would it take to build one end-to-end?* The result is this repo —
a reproducible pipeline that ingests Kepler + TESS data from MAST,
preprocesses it with the canonical phase-fold-and-bin scheme of Shallue &
Vanderburg (2018), and feeds the views through a 1-D CNN extended with
Squeeze-and-Excitation channel attention (Hu et al. 2018; Xie et al. 2025),
bilateral multi-head attention with residual late-fusion (Islam 2026), and
a Wide & Deep auxiliary path carrying 9 stellar/transit/centroid features.

Training uses 5-fold `StratifiedGroupKFold` grouped on `tic_id` so
multi-planet systems and re-observed stars are kept together (without this,
test AUC was inflated by 2–5 points through "seen-this-star-before"
leakage). Calibration is post-hoc temperature scaling (Guo et al. 2017),
uncertainty is MC-Dropout (Gal & Ghahramani 2016) plus 5-fold ensemble
disagreement.

Applied to the 6,200 held-out unconfirmed Planet Candidates from the NEA
TOI / KOI catalogues, the model produced a ranked priority list — 146 of
those candidates received `prob_mean ≥ 0.95` and are not yet confirmed in
the live ExoFOP TFOPWG snapshot. Of 120 candidates that were `PC` at
training-catalogue build but have since been confirmed elsewhere, **115
(95.8 %) score above the standard 0.5 threshold** — an internal
generalisation check, not a benchmark claim.

## Top-3 unconfirmed picks (by `prob_mean`)

| Candidate | TIC / KIC | Period (d) | `prob_mean` | `fold_disagree` | Vetting figure |
|---|---|---|---|---|---|
| TOI-4328.01 | TIC 77175217 | 703.79 | **0.989** | 0.006 | [docs/figures/toi-4328-01_tic_77175217.png](docs/figures/toi-4328-01_tic_77175217.png) |
| TOI-4565.01 | TIC 381897917 | 692.51 | 0.983 | 0.008 | [docs/figures/toi-4565-01_tic_381897917.png](docs/figures/toi-4565-01_tic_381897917.png) |
| TOI-4353.01 | TIC 176797879 | 718.18 | 0.980 | 0.009 | [docs/figures/toi-4353-01_tic_176797879.png](docs/figures/toi-4353-01_tic_176797879.png) |

Full top-20 figures in [`results/vetting/`](results/vetting). All three
top picks are long-period TESS detections — the regime where TESS's
sector-by-sector observing pattern makes confirmation hard and follow-up
campaigns are lengthy, so prioritisation has real value.

## How it works

Pipeline (`src/exoplanet_hunter/`):

```
catalog ──► downloader ──► preprocess ──► dual-view CNN ──► calibrated probability
(NEA TAP)   (lightkurve   (clean +       (SE + MHA +       (temperature-
            +             Savitzky-      Wide & Deep +     scaled, MC-Dropout
            archive.stsci Golay flatten  residual fusion)  uncertainty)
            direct)       + phase fold
                          + bin)
```

Architecture (full detail in [`docs/methodology.md`](docs/methodology.md)):

- **Global view** (2,001 bins covering full phase) and **local view** (201
  bins covering ±2 transit durations around phase 0).
- **Global tower:** 3 conv blocks (16/32/64 filters), SE block after each.
  **Local tower:** 2 conv blocks (16/32 filters). 8-head Multi-Head
  Attention + residual + LayerNorm at the end of each tower before
  GlobalAveragePooling.
- **Wide & Deep auxiliary path:** 9-dim vector `[Teff, R*, log g, Tmag,
  depth, duration, log P, SNR, centroid_snr]` concatenated with the pooled
  tower embeddings, bypassing the conv layers.
- **Residual fusion head:** FC(256) → FC(128) → FC(1) with LeakyReLU +
  BatchNorm + Dropout(0.4), wrapped in a linear shortcut from the
  concatenated embeddings.

## Reproduce

```bash
# 1. Create the conda env (Python 3.11)
make env && conda activate exoplanet-hunter
make hooks                           # ruff + nbstripout + mypy

# 2. Build the labelled catalogue + processed views
python scripts/build_dataset.py      # NEA TAP queries + MAST downloads + preprocess

# 3. Train the 5-fold dual-view CNN ensemble
python scripts/train_model.py model=cnn_dualview

# 4. Score a known target end-to-end (single TIC sanity check)
python scripts/score_target.py tic_id=150428135    # TOI-700

# 5. Bulk-score the 6,200-candidate discovery pool (Branch 4)
python scripts/score_candidates.py

# 6. Generate the priority shortlist + render vetting figures
python scripts/discovery_shortlist.py
python scripts/render_vetting.py top_k=20
```

Experiment tracking is MLflow (`mlruns/`); the branch-3 final run is
`58570d85f1dd4f68a7e888988c88eeab`.

## Repo layout

```
src/exoplanet_hunter/   importable package
├── data/               catalogue, downloader (direct-archive + lightkurve), stellar lookups
├── preprocess/         clean, flatten, fold, build dual views
├── features/           hand-crafted features + centroid extraction
├── models/             dual-view CNN, RF baseline, focal loss, MC dropout
├── training/           Hydra/MLflow training entry
├── eval/               metrics, calibration, six-panel vetting figure
├── search/             BLS / TLS period search (Branch-5 candidate)
└── utils/              logging, paths, seeds

conf/                   Hydra configs (model, data, train, preprocess)
scripts/                CLI entry points
docs/methodology.md     full architecture + design write-up
docs/research_report_draft.md  results, ablation ladder, comparisons, limitations
results/vetting/        six-panel vetting figures for the top-20 picks
notebooks/              exploration notebooks
tests/                  pytest — synthetic-transit fixtures
```

## Tooling

| Tool | Purpose |
|---|---|
| **Hydra** | Composable YAML configs (`model=foo data=bar`) |
| **MLflow** | Experiment tracking — every hyperparameter, metric, plot, artifact |
| **lightkurve** | MAST querying + Kepler/TESS FITS handling |
| **astroquery** | TIC v8 / Gaia DR3 stellar-parameter lookups |
| **Ruff + mypy + pytest** | Pre-commit linting, type checking, fixture tests |

## Data sources

All free, no auth required:

- **MAST archive** — TESS SPOC + Kepler stitched light curves via
  `lightkurve` and direct HTTP from `archive.stsci.edu/pub/kepler/`.
- **NASA Exoplanet Archive** — confirmed planets, TOI dispositions, KOI
  cumulative table via the public TAP service.
- **ExoFOP-TESS** — current TFOPWG dispositions + follow-up counts.
- **TIC v8 / Gaia DR3** — stellar parameters via `astroquery`.

## See also

- [**docs/research_report_draft.md**](docs/research_report_draft.md) —
  full results, ablation ladder, comparisons with published baselines,
  Branch-4 discovery results, limitations & future work.
- [**docs/methodology.md**](docs/methodology.md) — architecture detail,
  preprocessing choices, training protocol, design rationale.

## Selected references

- Shallue, C. & Vanderburg, A. (2018). *Identifying Exoplanets with Deep
  Learning.* **AJ** 155, 94. *(AstroNet baseline)*
- Islam, M. R. (2026). *ExoNet: Calibrated Multimodal Deep Learning for
  TESS Exoplanet Candidate Vetting.* **arXiv:2604.15560.** *(closest
  methodological comparison)*
- Xie, D., et al. (2025). *Deep learning to classify exoplanet light curves
  in Kepler and TESS.* **RAA** 25, 104004. *(SE-CNN architecture)*
- Hu, J., Shen, L., & Sun, G. (2018). *Squeeze-and-Excitation Networks.*
  **CVPR.**
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation.*
  **ICML.**
- Guo, C., et al. (2017). *On Calibration of Modern Neural Networks.*
  **ICML.** *(temperature scaling)*
- Marsland, S. (2014). *Machine Learning: An Algorithmic Perspective*, 2nd
  ed. *(DATA 305 reference text)*

Full bibliography in
[docs/research_report_draft.md](docs/research_report_draft.md#references).

## License

MIT.
