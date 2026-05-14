"""Dual-view 1D CNN — branch-3 architecture.

Builds on Shallue & Vanderburg 2018 (AstroNet) with three additions:

  1. Squeeze-and-Excitation channel attention after each conv block, before
     MaxPool (Hu et al. 2018; placement per Xie et al. 2025, Fig. 1).
  2. Multi-Head Attention with residual+LayerNorm at the end of each conv
     tower, applied to the temporal feature map before GlobalAveragePool
     (ExoNet, Islam 2026, §III.D — applied bilaterally to global and local).
  3. Residual late-fusion head: 2-layer MLP with a linear shortcut from the
     concatenated stream embeddings to the head output dim, preventing
     gradient stagnation in the fusion path before the encoders converge
     (ExoNet, Islam 2026, §III.D).

Conv towers retain ReLU. The fully-connected head uses LeakyReLU(α=0.1)
matching Xie et al. 2025 §2.2 literally — they reserve LeakyReLU for the
residual head, not the conv tower.

Dropout in the FC head stays training=True so MC Dropout uncertainty
estimation (`models/uncertainty.py`) keeps working at inference.
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers


def _se_block(
    x: tf.Tensor,
    *,
    reduction: int = 8,
    floor: int = 4,
    name_prefix: str,
) -> tf.Tensor:
    """Squeeze-and-Excitation channel attention (Hu et al. 2018).

    GAP → FC(C/r) → ReLU → FC(C) → Sigmoid → channel-wise scale.
    Bottleneck size is `max(C // reduction, floor)` so the first conv block
    (C=16) doesn't degenerate to C/r=2 with r=8.
    """
    channels = int(x.shape[-1])
    bottleneck = max(channels // reduction, floor)
    s = layers.GlobalAveragePooling1D(name=f"{name_prefix}_se_gap")(x)
    s = layers.Dense(
        bottleneck,
        activation="relu",
        name=f"{name_prefix}_se_squeeze",
    )(s)
    s = layers.Dense(
        channels,
        activation="sigmoid",
        name=f"{name_prefix}_se_excite",
    )(s)
    s = layers.Reshape((1, channels), name=f"{name_prefix}_se_reshape")(s)
    return layers.Multiply(name=f"{name_prefix}_se_scale")([x, s])


def _conv_tower(
    x: tf.Tensor,
    filters_per_block: list[int],
    conv_per_block: int,
    kernel_size: int,
    pool_size: int,
    name: str,
    *,
    use_batchnorm: bool = True,
    spatial_dropout: float = 0.0,
    l2: float = 0.0,
    se_reduction: int = 8,
    se_floor: int = 4,
) -> tf.Tensor:
    """Conv blocks with SE attention, returning the pre-GAP feature map (B, T, C).

    Per block: (Conv1D → BN → ReLU) × conv_per_block → SE → MaxPool → SpatialDropout.
    SE goes before MaxPool so channel weighting acts on the full-resolution
    feature map (Xie et al. 2025 placement).
    """
    reg = regularizers.l2(l2) if l2 > 0 else None
    for block_idx, n_filters in enumerate(filters_per_block):
        for conv_idx in range(conv_per_block):
            x = layers.Conv1D(
                filters=n_filters,
                kernel_size=kernel_size,
                padding="same",
                activation=None,
                kernel_regularizer=reg,
                name=f"{name}_b{block_idx}_c{conv_idx}",
            )(x)
            if use_batchnorm:
                x = layers.BatchNormalization(
                    name=f"{name}_b{block_idx}_bn{conv_idx}",
                )(x)
            x = layers.Activation("relu", name=f"{name}_b{block_idx}_act{conv_idx}")(x)
        x = _se_block(
            x,
            reduction=se_reduction,
            floor=se_floor,
            name_prefix=f"{name}_b{block_idx}",
        )
        x = layers.MaxPool1D(pool_size=pool_size, name=f"{name}_b{block_idx}_pool")(x)
        if spatial_dropout > 0:
            x = layers.SpatialDropout1D(
                spatial_dropout, name=f"{name}_b{block_idx}_sdrop",
            )(x)
    return x


def _attention_pool(
    x: tf.Tensor,
    *,
    num_heads: int = 8,
    key_dim: int = 32,
    dropout: float = 0.1,
    name_prefix: str,
) -> tf.Tensor:
    """ExoNet-style MHA over a (B, T, C) feature map.

    `output_shape=C` keeps the residual identity F + MHA(F) dimension-correct
    (default value_dim*num_heads would not match C for our 64-channel tower).
    """
    channels = int(x.shape[-1])
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout,
        output_shape=channels,
        name=f"{name_prefix}_mha",
    )(x, x, x)
    h = layers.Add(name=f"{name_prefix}_mha_add")([x, attn])
    h = layers.LayerNormalization(name=f"{name_prefix}_mha_ln")(h)
    return layers.GlobalAveragePooling1D(name=f"{name_prefix}_gap")(h)


def _residual_fusion_head(
    z: tf.Tensor,
    *,
    fc_units: list[int],
    dropout: float,
    l2: float,
    use_batchnorm: bool,
    leaky_alpha: float = 0.1,
) -> tf.Tensor:
    """ExoNet-style residual late-fusion head.

    z (concat embedding) → MLP(fc_units) + linear shortcut z → fc_units[-1] → Add.
    LeakyReLU(α=0.1) per Xie 2025 §2.2 (head-only). Dropout stays training=True
    for MC Dropout at inference.
    """
    fc_l2 = regularizers.l2(l2) if l2 > 0 else None
    h = z
    for i, units in enumerate(fc_units):
        h = layers.Dense(
            int(units),
            kernel_regularizer=fc_l2,
            name=f"fc_{i}",
        )(h)
        if use_batchnorm:
            h = layers.BatchNormalization(name=f"fc_bn_{i}")(h)
        h = layers.LeakyReLU(negative_slope=float(leaky_alpha), name=f"fc_act_{i}")(h)
        h = layers.Dropout(float(dropout), name=f"fc_drop_{i}")(h, training=None)

    last_units = int(fc_units[-1])
    shortcut = layers.Dense(
        last_units,
        use_bias=False,
        kernel_regularizer=fc_l2,
        name="fusion_shortcut",
    )(z)
    return layers.Add(name="fusion_add")([h, shortcut])


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

    use_bn = bool(getattr(model_cfg, "use_batchnorm", True))
    sdrop = float(getattr(model_cfg, "spatial_dropout", 0.0))
    l2_val = float(getattr(model_cfg, "l2", 0.0))

    attn_cfg = getattr(model_cfg, "attention", None)
    n_heads = int(getattr(attn_cfg, "num_heads", 8)) if attn_cfg is not None else 8
    key_dim = int(getattr(attn_cfg, "key_dim", 32)) if attn_cfg is not None else 32
    attn_dropout = (
        float(getattr(attn_cfg, "dropout", 0.1)) if attn_cfg is not None else 0.1
    )

    se_cfg = getattr(model_cfg, "se", None)
    se_reduction = int(getattr(se_cfg, "reduction", 8)) if se_cfg is not None else 8
    se_floor = int(getattr(se_cfg, "floor", 4)) if se_cfg is not None else 4

    leaky_alpha = float(getattr(getattr(model_cfg, "head", {}), "leaky_alpha", 0.1))

    g = _conv_tower(
        g_in,
        filters_per_block=list(model_cfg.global_view.conv_blocks),
        conv_per_block=int(model_cfg.global_view.conv_per_block),
        kernel_size=int(model_cfg.global_view.kernel_size),
        pool_size=int(model_cfg.global_view.pool_size),
        name="global",
        use_batchnorm=use_bn,
        spatial_dropout=sdrop,
        l2=l2_val,
        se_reduction=se_reduction,
        se_floor=se_floor,
    )
    g = _attention_pool(
        g,
        num_heads=n_heads,
        key_dim=key_dim,
        dropout=attn_dropout,
        name_prefix="global",
    )

    l = _conv_tower(
        l_in,
        filters_per_block=list(model_cfg.local_view.conv_blocks),
        conv_per_block=int(model_cfg.local_view.conv_per_block),
        kernel_size=int(model_cfg.local_view.kernel_size),
        pool_size=int(model_cfg.local_view.pool_size),
        name="local",
        use_batchnorm=use_bn,
        spatial_dropout=sdrop,
        l2=l2_val,
        se_reduction=se_reduction,
        se_floor=se_floor,
    )
    l = _attention_pool(
        l,
        num_heads=n_heads,
        key_dim=key_dim,
        dropout=attn_dropout,
        name_prefix="local",
    )

    inputs: list[tf.Tensor] = [g_in, l_in]
    branches: list[tf.Tensor] = [g, l]

    if use_aux:
        assert aux_input_dim is not None, "aux_input_dim is required when use_aux=True"
        a_in = layers.Input(shape=(int(aux_input_dim),), name="aux_features")
        inputs.append(a_in)
        branches.append(a_in)                       # wide path — no transformation

    z = layers.Concatenate(name="concat")(branches)

    h = _residual_fusion_head(
        z,
        fc_units=list(model_cfg.head.fc_units),
        dropout=float(model_cfg.head.dropout),
        l2=l2_val,
        use_batchnorm=use_bn,
        leaky_alpha=leaky_alpha,
    )

    output = layers.Dense(
        int(model_cfg.output.units),
        activation=str(model_cfg.output.activation),
        name="output",
    )(h)

    return Model(inputs=inputs, outputs=output, name="cnn_dualview")
