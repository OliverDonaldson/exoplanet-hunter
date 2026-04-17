"""Reproducibility — seed every PRNG that affects training."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow.

    Call this at the very top of every entry-point script. Doesn't make TF
    fully deterministic on GPU, but eliminates the easy sources of variance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass
