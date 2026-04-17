# Exoplanet Hunter

Deep-learning pipeline for detecting exoplanet transits in NASA TESS light curves.

> Built around the dual-view 1D CNN architecture from
> Shallue & Vanderburg (2018), trained on real labelled TESS data,
> instrumented with MLflow + Hydra + Optuna, and shipped with Docker.

## What it does

1. **Downloads** confirmed-planet, false-positive, and quiet-star light curves
   directly from MAST via `lightkurve` — no synthetic data.
2. **Preprocesses** them with the canonical pipeline: clean → flatten →
   phase-fold → extract global + local views.
3. **Trains** two models on the same labelled set:
   - A **Random Forest** baseline on hand-crafted features (the natural
     classical-ML comparison from DATA 305 Week 2).
   - A **dual-view 1D CNN** (Keras Functional API) — the deep-learning headline
     model, with an optional Wide & Deep extension that fuses stellar
     parameters (Teff, R*, log g) into the head.
4. **Scores** unreviewed targets with MC-Dropout uncertainty quantification —
   each prediction comes with a calibrated standard deviation.
5. **Tracks** every experiment in MLflow and lets you sweep hyperparameters
   with Optuna.

## Quickstart

```bash
# 1. Create the conda env (uses Python 3.11)
make env
conda activate exoplanet-hunter

# 2. Install hooks (ruff + nbstripout + mypy)
make hooks

# 3. Smoke-test the pipeline on 30 targets (~10 min)
python scripts/build_dataset.py data=small
python scripts/train_model.py    model=random_forest data=small
python scripts/train_model.py    model=cnn_dualview  data=small

# 4. View experiments
make mlflow      # opens at http://localhost:5000

# 5. Score a known confirmed planet
python scripts/score_target.py tic_id=150428135   # TOI-700
```

## Project layout

```
src/exoplanet_hunter/   # importable package
├── data/               # catalog + downloader + stellar lookups
├── preprocess/         # clean, flatten, fold, build views
├── features/           # hand-crafted scalar features (RF baseline)
├── models/             # RF, dual-view CNN, focal loss, MC dropout
├── training/           # Hydra/MLflow training entry + Optuna tuner
├── eval/               # metrics, calibration, candidate vetting
├── search/             # BLS + TLS period search
├── viz/                # Plotly Dash dashboard (scaffolded)
└── utils/              # logging, paths, seeds

conf/                   # Hydra configs (model, data, train, preprocess)
notebooks/              # exploration + model-walkthrough notebooks
scripts/                # CLI entry points (Hydra-driven)
tests/                  # pytest — synthetic-transit fixtures
docs/methodology.md     # portfolio writeup
```

## Tooling

| Tool       | What for                                                         |
|------------|------------------------------------------------------------------|
| **Hydra**  | Composable YAML configs; `model=foo data=bar` from the CLI.      |
| **MLflow** | Track every run — hyperparams, metrics, plots, model artifacts.  |
| **Optuna** | Bayesian hyperparameter search with median pruning.              |
| **Docker** | Reproducible build (CPU); MLflow + Jupyter via `docker compose`. |
| **Ruff**   | Lint + format on commit (pre-commit hook).                       |
| **mypy**   | Strict types on our code, lenient on third-party libs.           |
| **pytest** | Synthetic-transit fixtures so tests don't touch MAST.            |

## Data sources

All free, no auth required:

- **MAST archive** — TESS SPOC light curves via `lightkurve`.
- **NASA Exoplanet Archive** — confirmed planets + TOI dispositions, queried
  via the public TAP service.
- **TIC v8 / Gaia DR3** — stellar parameters via `astroquery`.

## Where it goes from here

Scaffolded files marked `TODO(Oliver)` are the next learning chunks:

- `eval/vetting.py` — six-panel vetting figure for triaging candidates.
- `notebooks/05_candidate_search.ipynb` — discovery loop on TOI-PC + random TICs.
- `viz/dashboard.py` — interactive Plotly Dash app for browsing predictions.
- `models/cnn_dualview_torch.py` — PyTorch port for framework comparison.

See `docs/methodology.md` for the full write-up.

## References

- Shallue, C. & Vanderburg, A. (2018). *Identifying Exoplanets with Deep Learning*. AJ 155, 94.
- Hippke, M. & Heller, R. (2019). *Optimised Transit Detection — TLS*. A&A 623, A39.
- Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation*. ICML.
- Lin, T-Y. et al. (2017). *Focal Loss for Dense Object Detection*. ICCV.
- Marsland, S. (2014). *Machine Learning: An Algorithmic Perspective*, 2nd ed.

## License

MIT.
