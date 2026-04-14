from __future__ import annotations

import numpy as np

import mlx.core as mx

from chai_mlx.nn.layers.attention import MSAPairWeightedAveraging


def test_msa_pair_weighted_averaging_masks_msa_values() -> None:
    module = MSAPairWeightedAveraging(msa_dim=4, pair_dim=3, num_heads=1, value_dim=2)

    pair = mx.zeros((1, 2, 2, 3), dtype=mx.float32)
    token_pair_mask = mx.ones((1, 2, 2), dtype=mx.float32)
    msa_mask = mx.array([[[1.0, 1.0], [0.0, 0.0]]], dtype=mx.float32)

    msa_base = mx.array(
        [
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            ]
        ],
        dtype=mx.float32,
    )
    msa_variant = mx.array(
        [
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
                [[101.0, 202.0, 303.0, 404.0], [505.0, 606.0, 707.0, 808.0]],
            ]
        ],
        dtype=mx.float32,
    )

    out_base = module(
        msa_base,
        pair,
        token_pair_mask=token_pair_mask,
        msa_mask=msa_mask,
    )
    out_variant = module(
        msa_variant,
        pair,
        token_pair_mask=token_pair_mask,
        msa_mask=msa_mask,
    )

    np.testing.assert_allclose(
        np.array(out_base),
        np.array(out_variant),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.array(out_base[:, 1]),
        np.zeros((1, 2, 4), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
