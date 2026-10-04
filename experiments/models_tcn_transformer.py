"""
models_tcn_transformer.py — the Multi-Task TCN-Transformer architecture described in the paper
=========================================================================
Step 2 of paper 035S_IC3MT2026 describes it as:
  "cascades a Temporal Convolutional Network (TCN) with a Transformer
    encoder to capture both localized spatial-temporal dynamics and
    global kinematic dependencies"

★ Current status (important, to avoid misunderstanding):
  All formal results in exp_personalization currently use `loso_train_and_save.build_model`,
  which is Conv1D×2 + GlobalAveragePooling + Dense, **with no Transformer**.
  The TCN-Transformer described in the paper only exists in the old code `src/TrainingModel/MIA.py`,
  and that old code has two problems that this file fixes:
    1. no static-feature (age/height/weight/sex) branch — incompatible with the current pipeline
    2. the TCN is just two flat Conv1D layers with no residual connections, so it is not really a TCN

This file provides two comparable architectures whose interface is identical to build_model (same two inputs,
same sigmoid output, same loss), so they can be dropped straight into the existing multi-seed comparison.

  build_tcn_transformer : residual TCN stack + Transformer encoder
  build_baseline        : delegates to the current Conv1D architecture (control)

⚠️ The parameter counts differ a lot (TCN-T is several times the baseline); at the scale of only 10 subjects and
   ~34k training windows, more capacity is not necessarily better — which is exactly why it must be measured.
"""

import os
import sys

import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, Conv1D,          # noqa: E402
                                     BatchNormalization, LayerNormalization,
                                     GlobalAveragePooling1D, Concatenate,
                                     MultiHeadAttention, Add, Activation)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from loso_train_and_save import build_model as _baseline_build, WINDOW_SIZE  # noqa: E402


def _tcn_block(x, filters, dilation, dropout, name):
    """Standard TCN residual block: two causal dilated convolutions + a residual connection.

    The residual is the key difference between a TCN and "a few stacked Conv1D layers" — without it, going deeper degrades.
    """
    prev = x
    for i in (1, 2):
        x = Conv1D(filters, 3, padding='causal', dilation_rate=dilation,
                   name=f'{name}_conv{i}')(x)
        x = BatchNormalization(name=f'{name}_bn{i}')(x)
        x = Activation('relu', name=f'{name}_relu{i}')(x)
        x = Dropout(dropout, name=f'{name}_drop{i}')(x)
    # when channel counts differ, align them with a 1x1 convolution so they can be added
    if prev.shape[-1] != filters:
        prev = Conv1D(filters, 1, padding='same', name=f'{name}_proj')(prev)
    return Add(name=f'{name}_add')([prev, x])


def _transformer_encoder(x, num_heads, key_dim, ff_dim, dropout, name):
    """Pre-LN Transformer encoder block (easier to train than Post-LN, especially on small data)."""
    h = LayerNormalization(epsilon=1e-6, name=f'{name}_ln1')(x)
    h = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout,
                           name=f'{name}_mha')(h, h)
    x = Add(name=f'{name}_add1')([x, h])
    h = LayerNormalization(epsilon=1e-6, name=f'{name}_ln2')(x)
    h = Dense(ff_dim, activation='relu', name=f'{name}_ff1')(h)
    h = Dropout(dropout, name=f'{name}_ffdrop')(h)
    h = Dense(x.shape[-1], name=f'{name}_ff2')(h)
    return Add(name=f'{name}_add2')([x, h])


def build_tcn_transformer(n_ts_feat, n_static,
                          tcn_filters=64, dilations=(1, 2, 4),
                          num_heads=2, ff_dim=128, dropout=0.1,
                          lr=1e-3):
    """Residual TCN stack → Transformer encoder → fuse with static features → %MVC.

    Layer names keep the 'feature*' prefix so phase2's "frozen feature layers" logic
    (layer.trainable = 'feature' not in layer.name) works unchanged.
    """
    input_ts = Input(shape=(WINDOW_SIZE, n_ts_feat), name='ts_input')
    input_static = Input(shape=(n_static,), name='static_input')

    x = input_ts
    for d in dilations:
        x = _tcn_block(x, tcn_filters, d, dropout, name=f'feature_tcn_d{d}')
    x = _transformer_encoder(x, num_heads, tcn_filters // num_heads, ff_dim,
                             dropout, name='feature_attn')
    ts_feat = GlobalAveragePooling1D(name='feature_pool')(x)

    fused = Concatenate(name='fusion_concat')([ts_feat, input_static])
    fused = Dense(64, activation='relu', name='fusion_dense')(fused)
    fused = Dropout(0.2, name='fusion_dropout')(fused)
    emg = Dense(32, activation='relu', name='emg_dense')(fused)
    out = Dense(2, activation='sigmoid', name='out_emg')(emg)

    model = tf.keras.Model(inputs=[input_ts, input_static], outputs=[out])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss={'out_emg': 'mse'}, loss_weights={'out_emg': 3.0},
                  metrics={'out_emg': 'mae'})
    return model


def build_baseline(n_ts_feat, n_static):
    """The current architecture (Conv1D×2 + GAP), as the control."""
    return _baseline_build(n_ts_feat, n_static)


ARCHS = {'baseline': build_baseline, 'tcn_transformer': build_tcn_transformer}


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    for name, fn in ARCHS.items():
        m = fn(16, 4)
        n = m.count_params()
        print(f"{name:<18} params {n:>9,}")
