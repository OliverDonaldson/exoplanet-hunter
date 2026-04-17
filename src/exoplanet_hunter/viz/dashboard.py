"""Interactive Plotly Dash dashboard — scaffolded.

Planned features:
  * Search a TIC ID, fetch its light curve, run inference, display:
      - raw + flattened light curve
      - phase-folded global view + local view
      - model probability with MC-Dropout uncertainty band
      - feature contributions (SHAP for RF, Integrated Gradients for CNN)
  * Gallery view of top-scoring candidates from the latest model.

Run with:
    python -m exoplanet_hunter.viz.dashboard

TODO(Oliver): build out after the training pipeline is producing a saved model.
"""

from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "dashboard scaffolded — implement after a CNN model is saved to models/cnn_dualview.keras"
    )


if __name__ == "__main__":
    main()
