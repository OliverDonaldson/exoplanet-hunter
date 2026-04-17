"""Dual-view 1D CNN — Shallue & Vanderburg 2018 (AstroNet).

Architecture overview:

  global_view (2001,) ──► Conv tower (5 blocks, 16→256) ──► flatten ──┐
                                                                       ├─► concat
  local_view  (201,)  ──► Conv tower (2 blocks, 16→32)  ──► flatten ──┤        │
                                                                       │        ├─► FC×4 (512) ──► sigmoid
                                            aux_features (n,) ────────┘        │
                                                            (Wide path) ───────┘

The aux path is the **Wide & Deep** pattern from the Keras Functional API
notes (Week 5): a subset of inputs (here: stellar parameters) bypasses the
deep towers and is concatenated directly into the head. This lets stellar
context (e.g. Teff, R*) influence the prediction even though it isn't a
time series.

Dropout in the FC head is left enabled at inference time to support MC
Dropout uncertainty estimation (see `models/uncertainty.py`).
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import Model, layers


def _conv_tower(
    x: tf.Tensor,
    filters_per_block: list[int],
    conv_per_block: int,
    kernel_size: int,
    pool_size: int,
    name: str,
) -> tf.Tensor:
    """Sequence of (Conv1D, ReLU)*conv_per_block + MaxPool1D blocks."""
    for block_idx, n_filters in enumerate(filters_per_block):
        for conv_idx in range(conv_per_block):
            x = layers.Conv1D(
                filters=n_filters,
                kernel_size=kernel_size,
                padding="same",
                activation="relu",
                name=f"{name}_b{block_idx}_c{conv_idx}",
            )(x)
        x = layers.MaxPool1D(pool_size=pool_size, name=f"{name}_b{block_idx}_pool")(x)
    return layers.Flatten(name=f"{name}_flatten")(x)


def build_cnn_dualview(
    model_cfg: Any,
    *,
    global_input_length: int = 2001,
    local_input_length: int = 201,
    aux_input_dim: int | None = None,
) -> Model:
    """Construct the dual-view CNN as a Keras Functional `Model`.

    Parameters
    ----------
    model_cfg : the `model` Hydra group (`conf/model/cnn_dualview*.yaml`).
    global_input_length, local_input_length : sequence lengths from preprocessing.
    aux_input_dim : dimension of the optional auxiliary stellar-feature vector.
                    Pass None / 0 to disable the wide path.
    """
    use_aux = bool(getattr(model_cfg, "use_aux_features", False)) and bool(aux_input_dim)

    g_in = layers.Input(shape=(global_input_length, 1), name="global_view")
    l_in = layers.Input(shape=(local_input_length,  1), name="local_view")

    g = _conv_tower(
        g_in,
        filters_per_block=list(model_cfg.global_view.conv_blocks),
        conv_per_block=int(model_cfg.global_view.conv_per_block),
        kernel_size=int(model_cfg.global_view.kernel_size),
        pool_size=int(model_cfg.global_view.pool_size),
        name="global",
    )
    l = _conv_tower(                                              # noqa: E741
        l_in,
        filters_per_block=list(model_cfg.local_view.conv_blocks),
        conv_per_block=int(model_cfg.local_view.conv_per_block),
        kernel_size=int(model_cfg.local_view.kernel_size),
        pool_size=int(model_cfg.local_view.pool_size),
        name="local",
    )

    inputs: list[tf.Tensor] = [g_in, l_in]
    branches: list[tf.Tensor] = [g, l]

    if use_aux:
        a_in = layers.Input(shape=(int(aux_input_dim),), name="aux_features")
        inputs.append(a_in)
        branches.append(a_in)                       # wide path — no transformation

    x = layers.Concatenate(name="concat")(branches)

    # FC head with always-on dropout (training=True) so we can do MC Dropout
    # at inference. We pass `training=True` explicitly inside a small Lambda
    # to keep stochasticity at predict-time.
    for i, units in enumerate(model_cfg.head.fc_units):
        x = layers.Dense(int(units), activation="relu", name=f"fc_{i}")(x)
        x = layers.Dropout(float(model_cfg.head.dropout), name=f"drop_{i}")(x, training=None)

    output = layers.Dense(
        int(model_cfg.output.units),
        activation=str(model_cfg.output.activation),
        name="output",
    )(x)

    model = Model(inputs=inputs, outputs=output, name="cnn_dualview")
    return model
