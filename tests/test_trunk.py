from __future__ import annotations

import numpy as np

import mlx.core as mx

from chai_mlx.config import ChaiConfig
from chai_mlx.model.trunk import MSAModule
from chai_mlx.nn.layers.common import Transition


class _LinearStub:
    def __init__(self, delta: mx.array) -> None:
        self.delta = delta

    def __call__(self, single: mx.array) -> mx.array:
        return self.delta


class _OuterProductRecorder:
    def __init__(self) -> None:
        self.last_msa: mx.array | None = None

    def __call__(self, msa: mx.array, msa_mask: mx.array | None = None) -> mx.array:
        self.last_msa = msa
        return mx.zeros((msa.shape[0], msa.shape[2], msa.shape[2], 256), dtype=mx.float32)


class _ZeroPairTransition:
    def __call__(self, pair: mx.array) -> mx.array:
        return mx.zeros_like(pair)


class _IdentityTriangle:
    def __call__(self, pair: mx.array, pair_mask: mx.array | None = None) -> mx.array:
        return pair


class _ChunkRecorder:
    def __init__(self) -> None:
        self.calls: list[np.ndarray] = []

    def __call__(self, x: mx.array) -> mx.array:
        self.calls.append(np.array(x, copy=False))
        return x


class _ConcatSelf:
    def __init__(self) -> None:
        self.weight = mx.zeros((8, 4), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.concatenate([x, x], axis=-1)


def test_msa_module_broadcasts_single_bias_to_all_msa_rows() -> None:
    module = MSAModule(ChaiConfig())

    delta = mx.array(
        [
            [
                [1.0] * 64,
                [2.0] * 64,
            ]
        ],
        dtype=mx.float32,
    )
    recorder = _OuterProductRecorder()
    module.linear_s2m = _LinearStub(delta)
    module.outer_product_mean = [recorder]
    module.msa_transition = []
    module.msa_pair_weighted_averaging = []
    module.pair_transition = [_ZeroPairTransition()]
    module.triangular_multiplication = [_IdentityTriangle()]
    module.triangular_attention = [_IdentityTriangle()]

    single = mx.zeros((1, 2, 384), dtype=mx.float32)
    pair = mx.zeros((1, 2, 2, 256), dtype=mx.float32)
    msa_input = mx.array(
        [
            [
                [[0.0] * 64, [10.0] * 64],
                [[100.0] * 64, [110.0] * 64],
                [[200.0] * 64, [210.0] * 64],
            ]
        ],
        dtype=mx.float32,
    )

    module(single, pair, msa_input)

    assert recorder.last_msa is not None
    expected = msa_input + delta[:, None, :, :]
    np.testing.assert_allclose(
        np.array(recorder.last_msa, copy=False),
        np.array(expected, copy=False),
        rtol=1e-6,
        atol=1e-6,
    )


def test_transition_chunks_along_token_axis_when_budget_exceeded() -> None:
    transition = Transition(dim=4, expansion=1)
    transition.chunk_budget = 64

    norm = _ChunkRecorder()
    down = _ChunkRecorder()
    transition.norm = norm
    transition.up = _ConcatSelf()
    transition.down = down

    x = mx.arange(1 * 3 * 6 * 4, dtype=mx.float32).reshape(1, 3, 6, 4)
    out = transition(x)

    assert len(norm.calls) == 2
    assert len(down.calls) == 2
    assert norm.calls[0].shape == (1, 3, 3, 4)
    assert norm.calls[1].shape == (1, 3, 3, 4)
    np.testing.assert_allclose(norm.calls[0], np.array(x[:, :, :3, :], copy=False))
    np.testing.assert_allclose(norm.calls[1], np.array(x[:, :, 3:, :], copy=False))

    x_np = np.array(x, copy=False)
    expected = np.concatenate(
        [
            (x_np[:, :, :3, :] / (1.0 + np.exp(-x_np[:, :, :3, :]))) * x_np[:, :, :3, :],
            (x_np[:, :, 3:, :] / (1.0 + np.exp(-x_np[:, :, 3:, :]))) * x_np[:, :, 3:, :],
        ],
        axis=-2,
    )
    np.testing.assert_allclose(np.array(out, copy=False), expected, rtol=1e-6, atol=1e-6)
