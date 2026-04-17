# --- Exoplanet Hunter — common workflows --------------------------------------
# Run `make help` for a list. Most targets assume the conda env is active:
#   conda activate exoplanet-hunter

SHELL := /bin/bash
PY    := python
SCRIPT_DIR := scripts

.DEFAULT_GOAL := help

.PHONY: help env install hooks clean lint format type test test-network \
        data data-small train train-rf train-cnn tune mlflow jupyter \
        docker-build docker-up docker-down

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

mlflow:  ## Start a local MLflow tracking UI at :5000
	mlflow ui --backend-store-uri ./mlruns --host 0.0.0.0 --port 5000

jupyter:  ## Start JupyterLab
	jupyter lab --no-browser

# --- Docker -------------------------------------------------------------------

docker-build:  ## Build the Docker image
	docker compose build

docker-up:  ## Start mlflow + jupyter services
	docker compose up

docker-down:  ## Stop services
	docker compose down

# --- Misc ---------------------------------------------------------------------

clean:  ## Remove caches and build artifacts (keeps data/models)
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
