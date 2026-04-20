"""Unit tests for :func:`scripts.cuda_parity._compare`.

Covers the four regressions historically seen in the parity harness:

1. All-zero reference tensor (e.g. empty MSA): comparison should not
   report ``rel = inf`` unless MLX also produced non-zero noise.
2. Non-finite values (NaN / Inf) in either side: should be a hard fail
   with a readable label, not a silent ``inf`` that pollutes the CSV.
3. Shape mismatch: should be a hard fail with a readable label.
4. Masked comparison: only non-pad entries should drive the verdict,
   and a mask that selects zero entries should degrade gracefully to a
   trivially-passing row (the "structural null" case).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from cuda_parity import _compare  # type: ignore[import-not-found]


def test_all_zero_reference_with_all_zero_mlx_passes() -> None:
    cuda = np.zeros((4, 4), dtype=np.float32)
    mlx = np.zeros((4, 4), dtype=np.float32)
    c = _compare("zero_match", cuda, mlx, tol=1e-3)
    assert c.passed
    assert c.max_abs == 0.0
    assert c.ref_range == 0.0
    assert c.rel == 0.0


def test_all_zero_reference_with_noisy_mlx_fails_with_inf_rel() -> None:
    cuda = np.zeros((4, 4), dtype=np.float32)
    mlx = np.full((4, 4), 0.01, dtype=np.float32)
    c = _compare("noise_in_void", cuda, mlx, tol=1e-3)
    assert not c.passed
    assert c.ref_range == 0.0
    assert c.rel == float("inf")
    assert c.max_abs == pytest.approx(0.01)


def test_nan_in_mlx_is_reported_with_label() -> None:
    cuda = np.ones((4, 4), dtype=np.float32)
    mlx = np.ones((4, 4), dtype=np.float32)
    mlx[0, 0] = np.nan
    c = _compare("nan_leak", cuda, mlx, tol=1e-3)
    assert not c.passed
    assert c.max_abs == float("inf")
    assert "non-finite" in c.name


def test_inf_in_cuda_is_reported_with_label() -> None:
    cuda = np.ones((4, 4), dtype=np.float32)
    cuda[1, 1] = np.inf
    mlx = np.ones((4, 4), dtype=np.float32)
    c = _compare("inf_leak", cuda, mlx, tol=1e-3)
    assert not c.passed
    assert "non-finite" in c.name


def test_shape_mismatch_is_hard_fail_with_readable_label() -> None:
    cuda = np.zeros((4, 4), dtype=np.float32)
    mlx = np.zeros((4, 5), dtype=np.float32)
    c = _compare("shape_mismatch", cuda, mlx, tol=1e-3)
    assert not c.passed
    assert c.max_abs == float("inf")
    assert "shape mismatch" in c.name


def test_mask_with_zero_selection_is_trivial_pass() -> None:
    """When the CUDA mask is entirely False (e.g. no MSA rows), the
    comparison is structurally vacuous. We report it as a trivial
    pass rather than letting pre-mask numerical noise drive the
    verdict either way.
    """
    cuda = np.random.RandomState(0).randn(4, 4).astype(np.float32)
    mlx = np.random.RandomState(1).randn(4, 4).astype(np.float32)
    mask = np.zeros((4,), dtype=bool)
    c = _compare("empty_mask", cuda, mlx, tol=1e-6, mask=mask)
    assert c.passed
    assert c.max_abs == 0.0
    assert "mask selects 0" in c.name
    assert c.ref_range > 0.0


def test_mask_with_partial_selection_only_compares_selected_entries() -> None:
    """Differences in masked-out entries must not influence the row."""
    cuda = np.zeros((4, 4), dtype=np.float32)
    mlx = np.zeros((4, 4), dtype=np.float32)
    cuda[0, 0] = 1.0
    mlx[0, 0] = 1.0
    # Masked-out (row 1) differs wildly but should not be seen:
    mlx[1, 0] = 999.0
    mask = np.array([True, False, True, True], dtype=bool)
    c = _compare("partial_mask", cuda, mlx, tol=1e-6, mask=mask)
    assert c.passed
    assert c.max_abs == 0.0


def test_mask_broadcasts_over_trailing_feature_dim() -> None:
    """A 1D mask over the leading axis should broadcast over the
    (N, C) reference, matching how chai-lab's ``msa_mask`` (shape
    ``(batch, depth, tokens)``) broadcasts over a ``(batch, depth,
    tokens, feat)`` embedding.
    """
    cuda = np.zeros((4, 3), dtype=np.float32)
    mlx = np.zeros((4, 3), dtype=np.float32)
    cuda[0] = [1.0, 2.0, 3.0]
    mlx[0] = [1.0, 2.0, 3.0]
    mlx[3] = [99.0, 99.0, 99.0]  # masked out
    mask = np.array([True, True, True, False], dtype=bool)
    c = _compare("broadcast_mask", cuda, mlx, tol=1e-6, mask=mask)
    assert c.passed
    assert c.max_abs == 0.0


def test_within_tolerance_passes_at_nonzero_reference() -> None:
    cuda = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
    mlx = cuda + 0.001
    c = _compare("close_match", cuda, mlx, tol=1e-2)
    assert c.passed
    assert c.max_abs == pytest.approx(0.001, abs=1e-4)
    assert c.ref_range == 4.0
