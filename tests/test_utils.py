from __future__ import annotations

import numpy as np

import mlx.core as mx

from chai_mlx.utils import (
    chunk_last,
    edm_gammas,
    edm_sigmas,
    make_additive_mask,
    masked_mean,
    stable_softmax,
)


def test_make_additive_mask_converts_float_masks_to_bias() -> None:
    mask = mx.array([[1.0, 0.0, 1.0]], dtype=mx.float32)

    actual = np.array(make_additive_mask(mask))

    np.testing.assert_allclose(actual, np.array([[0.0, -10000.0, 0.0]], dtype=np.float32))


def test_chunk_last_splits_evenly() -> None:
    x = mx.arange(12, dtype=mx.float32).reshape(2, 6)

    left, right = chunk_last(x, 2)

    np.testing.assert_array_equal(np.array(left), np.array([[0, 1, 2], [6, 7, 8]]))
    np.testing.assert_array_equal(np.array(right), np.array([[3, 4, 5], [9, 10, 11]]))


def test_masked_mean_respects_mask_and_eps() -> None:
    x = mx.array([[[1.0, 10.0], [3.0, 30.0]]], dtype=mx.float32)
    mask = mx.array([[1.0, 0.0]], dtype=mx.float32)

    actual = np.array(masked_mean(x, mask, axis=1))

    np.testing.assert_allclose(actual, np.array([[1.0, 10.0]], dtype=np.float32))


def test_stable_softmax_matches_numpy_reference() -> None:
    logits = mx.array([[1000.0, 1001.0, 1002.0]], dtype=mx.float32)

    actual = np.array(stable_softmax(logits, axis=-1))
    shifted = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    shifted = shifted - shifted.max(axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_edm_schedule_matches_reference_formulas() -> None:
    num_steps = 8
    sigma_data = 16.0
    s_min, s_max, p = 4e-4, 80.0, 7.0
    s_churn, s_tmin, s_tmax = 80.0, 4e-4, 80.0

    sigmas = np.array(edm_sigmas(num_steps, sigma_data, s_min, s_max, p))
    gammas = np.array(edm_gammas(mx.array(sigmas), s_churn, s_tmin, s_tmax))

    t_ref = np.linspace(0.0, 1.0, 2 * num_steps + 1, dtype=np.float64)[1::2]
    ref_sigmas = sigma_data * (
        t_ref * s_min ** (1.0 / p) + (1.0 - t_ref) * s_max ** (1.0 / p)
    ) ** p
    ref_sigmas = ref_sigmas.astype(np.float32)
    ref_gamma = min(s_churn / num_steps, np.sqrt(2.0) - 1.0)
    expected_gammas = np.where(
        (ref_sigmas >= s_tmin) & (ref_sigmas <= s_tmax),
        ref_gamma,
        0.0,
    ).astype(np.float32)

    np.testing.assert_allclose(sigmas, ref_sigmas, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gammas, expected_gammas, rtol=1e-6, atol=1e-6)
