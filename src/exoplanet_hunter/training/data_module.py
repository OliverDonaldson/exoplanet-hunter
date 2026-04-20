"""Loading + slicing the processed view dataset for training.

The processed dataset is a single .npz file written by the build pipeline:

    data/processed/views.npz
        global_views   (N, 2001) float32
        local_views    (N, 201)  float32
        labels         (N,)      int8     {0, 1}
        tic_ids        (N,)      int64
        aux_features   (N, A)    float32  (may be all NaN if not used)

This module provides:

  * `load_views`             — read the .npz and return a dict.
  * `train_val_test_split`   — stratified split.
  * `LightcurveDataset`      — wraps a split into a `tf.data.Dataset` with
                               on-the-fly augmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit


@dataclass
class ViewArrays:
    global_views: np.ndarray
    local_views: np.ndarray
    labels: np.ndarray
    tic_ids: np.ndarray
    aux_features: np.ndarray | None


def load_views(path: Path) -> ViewArrays:
    """Read a processed views .npz built by `build_dataset.py`."""
    with np.load(path) as f:
        aux = f["aux_features"] if "aux_features" in f.files else None
        return ViewArrays(
            global_views=f["global_views"].astype(np.float32),
            local_views=f["local_views"].astype(np.float32),
            labels=f["labels"].astype(np.int8),
            tic_ids=f["tic_ids"].astype(np.int64),
            aux_features=aux.astype(np.float32) if aux is not None else None,
        )


def train_val_test_split(
    views: ViewArrays,
    train: float = 0.7,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
) -> tuple[ViewArrays, ViewArrays, ViewArrays]:
    """Stratified group split — no TIC ID appears in more than one fold.

    Multi-planet systems share a TIC ID. Without grouping, the model
    gets a "seen that star before" shortcut and test AUC is inflated
    by 2-5 pts. We use `GroupShuffleSplit` with `groups=tic_ids` so
    every sample from a given star stays in the same split.
    """
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1")

    idx = np.arange(len(views.labels))
    groups = views.tic_ids

    # First split: train vs (val+test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(val + test), random_state=seed)
    train_idx, rest_idx = next(gss1.split(idx, views.labels, groups))

    # Second split: val vs test (within the rest group)
    rest_groups = groups[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test / (val + test), random_state=seed)
    val_rel, test_rel = next(gss2.split(rest_idx, views.labels[rest_idx], rest_groups))
    val_idx = rest_idx[val_rel]
    test_idx = rest_idx[test_rel]

    return _slice(views, train_idx), _slice(views, val_idx), _slice(views, test_idx)


def _slice(v: ViewArrays, idx: np.ndarray) -> ViewArrays:
    return ViewArrays(
        global_views=v.global_views[idx],
        local_views=v.local_views[idx],
        labels=v.labels[idx],
        tic_ids=v.tic_ids[idx],
        aux_features=None if v.aux_features is None else v.aux_features[idx],
    )


class LightcurveDataset:
    """Build `tf.data.Dataset` pipelines from a `ViewArrays`."""

    def __init__(
        self,
        views: ViewArrays,
        *,
        batch_size: int = 64,
        shuffle: bool = True,
        augment: bool = False,
        time_shift_frac: float = 0.005,
        noise_std: float = 1e-4,
        flip_prob: float = 0.5,
        scale_range: float = 0.05,
        mask_prob: float = 0.05,
        use_aux: bool = False,
        seed: int | None = None,
    ) -> None:
        self.v = views
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.time_shift_frac = time_shift_frac
        self.noise_std = noise_std
        self.flip_prob = flip_prob
        self.scale_range = scale_range
        self.mask_prob = mask_prob
        self.use_aux = use_aux and views.aux_features is not None
        # If the caller doesn't pass a seed we still want a deterministic
        # shuffle for the default case (test fixtures, CI). A `None` here
        # would mean "use TF global state", which is non-reproducible.
        self.seed = 42 if seed is None else int(seed)

    # ------------------------------------------------------------ augmentation

    @staticmethod
    def _shift_1d(x: tf.Tensor, max_frac: float) -> tf.Tensor:
        n = tf.shape(x)[-1]
        shift = tf.cast(
            tf.random.uniform([], minval=-max_frac, maxval=max_frac) * tf.cast(n, tf.float32),
            tf.int32,
        )
        return tf.roll(x, shift=shift, axis=-1)

    def _augment(
        self, g: tf.Tensor, l: tf.Tensor, *, aux: tf.Tensor | None = None
    ) -> tuple[tf.Tensor, tf.Tensor] | tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        # Coherent phase shift — the same random offset for both views,
        # because they represent the same physical star at the same time.
        shift_frac = tf.random.uniform(
            [], minval=-self.time_shift_frac, maxval=self.time_shift_frac
        )
        g_n = tf.shape(g)[-1]
        l_n = tf.shape(l)[-1]
        g = tf.roll(g, shift=tf.cast(shift_frac * tf.cast(g_n, tf.float32), tf.int32), axis=-1)
        l = tf.roll(l, shift=tf.cast(shift_frac * tf.cast(l_n, tf.float32), tf.int32), axis=-1)
        # Noise (independent per view — sensor noise is uncorrelated)
        g = g + tf.random.normal(tf.shape(g), stddev=self.noise_std)
        l = l + tf.random.normal(tf.shape(l), stddev=self.noise_std)
        # Random scaling — simulate different transit depths / stellar variability
        if self.scale_range > 0:
            scale = tf.random.uniform(
                [],
                minval=1.0 - self.scale_range,
                maxval=1.0 + self.scale_range,
            )
            g = g * scale
            l = l * scale
        # Random masking — simulate missing cadences / data gaps
        if self.mask_prob > 0:
            g_mask = tf.cast(
                tf.random.uniform(tf.shape(g)) > self.mask_prob,
                tf.float32,
            )
            l_mask = tf.cast(
                tf.random.uniform(tf.shape(l)) > self.mask_prob,
                tf.float32,
            )
            g = g * g_mask
            l = l * l_mask
        # Flip (transit is symmetric in time) — same decision for both views
        do_flip = tf.random.uniform([]) < self.flip_prob
        g = tf.cond(do_flip, lambda: tf.reverse(g, axis=[-1]), lambda: g)
        l = tf.cond(do_flip, lambda: tf.reverse(l, axis=[-1]), lambda: l)
        return (g, l, aux) if aux is not None else (g, l)

    # ----------------------------------------------------------------- public

    def to_tf_dataset(self) -> tf.data.Dataset:
        """Materialise as a `tf.data.Dataset` of (inputs_dict, label)."""
        g = self.v.global_views[..., None]  # add channel axis
        l = self.v.local_views[..., None]
        y = self.v.labels.astype(np.float32)

        if self.use_aux:
            aux = self.v.aux_features
            ds = tf.data.Dataset.from_tensor_slices(((g, l, aux), y))
        else:
            ds = tf.data.Dataset.from_tensor_slices(((g, l), y))

        if self.shuffle:
            ds = ds.shuffle(buffer_size=min(1024, len(y)), seed=self.seed)

        if self.augment:

            def _aug(inputs, label):  # type: ignore[no-untyped-def]
                if self.use_aux:
                    g_, l_, a_ = inputs
                    g_, l_, a_ = self._augment(g_, l_, aux=a_)  # type: ignore[misc]
                    return (g_, l_, a_), label
                g_, l_ = inputs
                g_, l_ = self._augment(g_, l_)  # type: ignore[misc]
                return (g_, l_), label

            ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)

        # Convert tuple-inputs to dict-inputs for the Functional model.
        if self.use_aux:
            ds = ds.map(
                lambda inputs, y_: (
                    {"global_view": inputs[0], "local_view": inputs[1], "aux_features": inputs[2]},
                    y_,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            ds = ds.map(
                lambda inputs, y_: (
                    {"global_view": inputs[0], "local_view": inputs[1]},
                    y_,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
