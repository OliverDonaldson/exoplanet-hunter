# --- Exoplanet Hunter — common workflows --------------------------------------
# Run `make help` for a list. Most targets assume the conda env is active:
#   conda activate exoplanet-hunter

SHELL := /bin/bash
PY    := python
SCRIPT_DIR := scripts

.DEFAULT_GOAL := help

.PHONY: help env install hooks clean lint format type test test-network \
        data data-small train train-rf train-cnn tune mlflow jupyter \
        pdf docs lock

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} \
	     /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# --- Environment --------------------------------------------------------------

env:  ## Create the conda env from environment.yml
	conda env create -f environment.yml

install:  ## Install the package in editable mode
	pip install -e .

hooks:  ## Install pre-commit hooks
	pre-commit install

lock:  ## Freeze the active conda env to requirements-lock.txt for exact-pin reproducibility
	pip freeze --exclude-editable > requirements-lock.txt
	@echo "Wrote requirements-lock.txt"

# --- Code quality -------------------------------------------------------------

lint:  ## Ruff lint (no fixes)
	ruff check .

format:  ## Ruff format + auto-fix
	ruff check --fix .
	ruff format .

type:  ## mypy type check
	mypy

test:  ## Run fast tests (no network)
	pytest -m "not network" -q

test-network:  ## Run network tests (hits MAST/NASA archives)
	pytest -m network -q

# --- Data + training ---------------------------------------------------------

data:  ## Build the full labelled dataset (long; downloads many TESS sectors)
	$(PY) $(SCRIPT_DIR)/build_dataset.py data=default

data-small:  ## Tiny dataset for fast iteration / smoke testing
	$(PY) $(SCRIPT_DIR)/build_dataset.py data=small

train-rf:  ## Train Random Forest baseline (small dataset)
	$(PY) $(SCRIPT_DIR)/train_model.py model=random_forest data=small

train-cnn:  ## Train dual-view CNN (small dataset)
	$(PY) $(SCRIPT_DIR)/train_model.py model=cnn_dualview data=small

train: train-rf train-cnn  ## Train RF then CNN baselines

tune:  ## Optuna HP search for the dual-view CNN
	$(PY) $(SCRIPT_DIR)/train_model.py -m model=cnn_dualview train=tune

# --- Services -----------------------------------------------------------------

mlflow:  ## Start a local MLflow tracking UI at http://localhost:5050
	mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5050

jupyter:  ## Start JupyterLab
	jupyter lab --no-browser

# --- Docs ---------------------------------------------------------------------
# `make pdf` regenerates docs/Research_Report.pdf from docs/Research_Report.md.
# Requires: pandoc + xelatex (e.g. brew install pandoc; brew install --cask
# mactex-no-gui) plus the DejaVu fonts (brew install --cask font-dejavu).
# To swap fonts: `make pdf MAINFONT="TeX Gyre Termes" MONOFONT="DejaVu Sans Mono"`.

MAINFONT ?= DejaVu Serif
MONOFONT ?= DejaVu Sans Mono
PANDOC_OPTS := --pdf-engine=xelatex --toc --toc-depth=2 \
               --include-in-header=preamble.tex \
               -V geometry:margin=0.85in -V fontsize=10pt \
               -V mainfont="$(MAINFONT)" -V monofont="$(MONOFONT)" \
               -V colorlinks=true -V linkcolor=RoyalBlue \
               -V urlcolor=RoyalBlue -V toccolor=black

pdf:  ## Rebuild docs/Research_Report.pdf from Research_Report.md
	@command -v pandoc >/dev/null || { echo "pandoc not found — brew install pandoc"; exit 1; }
	cd docs && pandoc Research_Report.md -o Research_Report.pdf $(PANDOC_OPTS)
	@echo "Wrote docs/Research_Report.pdf"

docs: pdf  ## Alias for `make pdf`

# --- Misc ---------------------------------------------------------------------

clean:  ## Remove caches and build artifacts (keeps data/models)
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
