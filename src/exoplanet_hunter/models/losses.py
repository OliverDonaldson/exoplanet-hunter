"""Custom losses for class-imbalanced training.

`binary_focal_loss` down-weights easy negatives so the model focuses on the
hard cases. With a confirmed:false:quiet ratio of roughly 1:1:5 and SPOC
producing many obvious non-planets, the gradient signal from the rare planet
class would otherwise be drowned out.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import backend as K


def binary_focal_loss(gamma: float = 2.0, alpha: float = 0.75) -> tf.types.experimental.Callable:
    """Focal loss (Lin et al. 2017) for binary classification.

        L = -alpha       * (1 - p)^gamma * log(p)        if y == 1
            -(1 - alpha) * p^gamma       * log(1 - p)    if y == 0

    α is the positive-class weight: with α > 0.5 the rare positive class
    gets a larger gradient contribution. For exoplanet detection positives
    are rare (~1:6), so the sensible default is α ≈ 0.75, *not* the 0.25
    from the original object-detection paper (which assumed rare negatives).

    Never combine this with a second `class_weight` on model.fit — the
    caller must choose one or the other, else the minority class gets
    double-weighted.
    """

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        eps = K.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), eps, 1.0 - eps)
        pt_pos = y_pred
        pt_neg = 1.0 - y_pred
        loss_pos = -alpha       * tf.pow(1.0 - pt_pos, gamma) * tf.math.log(pt_pos)
        loss_neg = -(1 - alpha) * tf.pow(pt_neg,       gamma) * tf.math.log(1.0 - pt_neg)
        return tf.reduce_mean(y_true * loss_pos + (1.0 - y_true) * loss_neg)

    loss.__name__ = f"focal_g{gamma:g}_a{alpha:g}"
    return loss
