# --- Exoplanet Hunter — reproducible build (CPU; no GPU on this image) -------
# For Apple Silicon GPU acceleration, run natively with `tensorflow-metal`
# instead of inside this container — Docker on macOS can't expose the GPU.
#
# Build:  docker compose build
# Run:    docker compose up

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    LIGHTKURVE_CACHE_DIR=/data/.lightkurve-cache

# System deps for FITS, plotting, and scientific Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        graphviz \
        libgl1 \
        libglib2.0-0 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python deps first (cached layer).
COPY pyproject.toml README.md ./
COPY src/ src/

# Mirror of environment.yml's pip section, for non-conda builds.
# Keep this in sync with environment.yml.
RUN pip install --upgrade pip wheel \
 && pip install \
        "lightkurve>=2.5" \
        "astropy>=6.1" \
        "astroquery>=0.4.7" \
        "transitleastsquares>=1.0.31" \
        "numpy>=1.26,<2.1" \
        "scipy>=1.13" \
        "pandas>=2.2" \
        "pyarrow>=16" \
        "matplotlib>=3.9" \
        "plotly>=5.22" \
        "dash>=2.17" \
        "scikit-learn>=1.5" \
        "shap>=0.46" \
        "joblib>=1.4" \
        "tensorflow>=2.16,<2.18" \
        "mlflow>=2.14" \
        "hydra-core>=1.3" \
        "omegaconf>=2.3" \
        "optuna>=3.6" \
        "optuna-integration[mlflow,tfkeras]>=3.6" \
        "pytest>=8" \
        "ruff>=0.5" \
        "mypy>=1.10" \
        "jupyterlab>=4.2" \
        "tqdm>=4.66" \
        "rich>=13.7" \
 && pip install -e .

# Bring in the rest of the project (configs, scripts, notebooks, tests, docs).
COPY . .

# Default ports: MLflow (5000), Jupyter (8888), Dash (8050).
EXPOSE 5000 8888 8050

# Default command runs JupyterLab; docker-compose overrides for MLflow.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password="]
