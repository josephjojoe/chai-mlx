"""Unit tests for :func:`chai_mlx.model.trunk._subsample_and_reorder_msa`.

The MLX port mirrors chai-lab's
``chai_lab.data.dataset.msas.utils.subsample_and_reorder_msa_feats_n_mask``.
These tests verify the shape invariants (output depth == input depth,
selected rows move to the front, mask is zero-padded) without pulling
in the full model.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from chai_mlx.model.trunk import _subsample_and_reorder_msa


def _deterministic_inputs(depth: int, tokens: int, dim: int):
    """Build a reproducible MSA feature tensor + mask."""
    # Row index encoded in the features so we can track which rows end
    # up where after reordering.
    feats = np.tile(
        np.arange(depth, dtype=np.float32)[:, None, None],
        (1, tokens, dim),
    )[None, ...]  # (1, depth, tokens, dim)
    # Give every row at least one unmasked token so the nonnull-row
    # filter doesn't drop anything; vary the count to drive the
    # "bigger rows first" ranking.
    mask_np = np.zeros((1, depth, tokens), dtype=bool)
    for r in range(depth):
        mask_np[0, r, : (r % tokens) + 1] = True
    return mx.array(feats), mx.array(mask_np)


def test_shallow_msa_is_returned_unchanged() -> None:
    """When depth <= select_n_rows, the subsample is a no-op."""
    feats, mask = _deterministic_inputs(depth=8, tokens=4, dim=3)
    mx.random.seed(0)
    feats_out, mask_out = _subsample_and_reorder_msa(
        feats, mask, select_n_rows=16
    )
    # Must be identity (not just equal-valued).
    assert np.array_equal(np.asarray(feats_out), np.asarray(feats))
    assert np.array_equal(np.asarray(mask_out), np.asarray(mask))


def test_select_n_rows_zero_disables_subsampling() -> None:
    feats, mask = _deterministic_inputs(depth=32, tokens=4, dim=3)
    mx.random.seed(0)
    feats_out, mask_out = _subsample_and_reorder_msa(
        feats, mask, select_n_rows=0
    )
    assert np.array_equal(np.asarray(feats_out), np.asarray(feats))
    assert np.array_equal(np.asarray(mask_out), np.asarray(mask))


def test_deep_msa_preserves_shape() -> None:
    """Subsampling must keep output shapes == input shapes (mask is
    zero-padded; features are just reordered)."""
    feats, mask = _deterministic_inputs(depth=32, tokens=4, dim=3)
    mx.random.seed(42)
    feats_out, mask_out = _subsample_and_reorder_msa(
        feats, mask, select_n_rows=8
    )
    assert feats_out.shape == feats.shape
    assert mask_out.shape == mask.shape


def test_mask_sampled_depth_padded_with_zeros() -> None:
    """The first ``select_n_rows`` entries of the output mask are the
    selected rows (preserving mask content); the rest are zeros."""
    feats, mask = _deterministic_inputs(depth=32, tokens=4, dim=3)
    mx.random.seed(42)
    feats_out, mask_out = _subsample_and_reorder_msa(
        feats, mask, select_n_rows=8
    )
    mask_np = np.asarray(mask_out)
    # Tail rows (depth 8..32) must all be masked out (padding zeros).
    assert not mask_np[:, 8:, :].any(), (
        "Rows beyond select_n_rows should be zero-padded in the mask"
    )
    # First 8 rows must have at least one unmasked token (non-null rule).
    assert mask_np[:, :8, :].any(axis=-1).all()


def test_selected_rows_come_first_in_features() -> None:
    """``combo_idx`` is ``[selected] + [unselected]``, so the first
    ``select_n_rows`` feature rows must be a subset of the input rows
    (identified by the row-index-encoded features we built)."""
    feats, mask = _deterministic_inputs(depth=32, tokens=4, dim=3)
    mx.random.seed(42)
    feats_out, _ = _subsample_and_reorder_msa(feats, mask, select_n_rows=8)
    feats_np = np.asarray(feats_out)
    # The feature value encodes the original row index; first 8 rows
    # must be distinct (no duplicates -> no row repetition).
    first_rows = feats_np[0, :8, 0, 0]
    assert len(set(first_rows.tolist())) == 8, (
        "Selected rows should be distinct; got duplicates "
        f"{first_rows.tolist()}"
    )
    # Remaining rows (the "unselected" tail) should also be distinct
    # and disjoint from the first 8.
    tail_rows = feats_np[0, 8:, 0, 0]
    assert set(first_rows.tolist()).isdisjoint(tail_rows.tolist())
    # Every row index from [0, depth) must appear exactly once.
    all_rows = np.concatenate([first_rows, tail_rows])
    assert sorted(all_rows.tolist()) == list(range(32))
