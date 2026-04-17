"""Train a model — thin Hydra wrapper around `exoplanet_hunter.training.train`.

Kept as a script so `make train-rf` etc. work without the `-m` flag.
"""

from __future__ import annotations

from exoplanet_hunter.training.train import main

if __name__ == "__main__":
    main()
