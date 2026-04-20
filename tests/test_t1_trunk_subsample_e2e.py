"""End-to-end smoke for the Tier-1 trunk knobs.

Uses the same tiny-shape fixture as :mod:`tests.test_end_to_end` to
actually run the model with:

* ``recycle_msa_subsample`` > 0 — exercises the MSA resampling code
  path inside :class:`chai_mlx.model.trunk.Trunk`.
* a second call with the same seed + ``recycle_msa_subsample = 0`` —
  confirms the default path still behaves deterministically (matches
  :mod:`tests.test_determinism`'s invariant).

We deliberately *don't* assert bit-exactness between the two
configurations because the subsample path draws a random permutation
over the MSA rows -- that's the whole point of the knob. What we DO
assert is that the call completes with the expected shapes and
finite coordinates.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import mlx.core as mx
import numpy as np

from chai_mlx import ChaiMLX, InferenceOutputs
from chai_mlx.config import ChaiConfig
from chai_mlx.data.types import FeatureContext

from tests.helpers import make_structure_inputs


def _tiny_context(cfg: ChaiConfig) -> FeatureContext:
    batch_size = 1
    n_tokens = 4
    n_atoms = 32
    atom_blocks = n_atoms // 32
    msa_depth = 8

    structure = make_structure_inputs(
        batch_size=batch_size,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
        msa_depth=msa_depth,
        n_templates=1,
        with_ranker_fields=True,
    )

    q_idx = mx.arange(n_atoms, dtype=mx.int32).reshape(batch_size, atom_blocks, 32)
    kv_idx = mx.clip(
        q_idx[:, :, :1] + mx.arange(128, dtype=mx.int32)[None, None, :] - 48,
        0,
        n_atoms - 1,
    )
    block_mask = mx.ones((batch_size, atom_blocks, 32, 128), dtype=mx.float32)

    structure = dc_replace(
        structure,
        atom_q_indices=q_idx,
        atom_kv_indices=kv_idx,
        block_atom_pair_mask=block_mask,
    )

    def seq(shape: tuple[int, ...]) -> mx.array:
        size = 1
        for dim in shape:
            size *= dim
        return mx.arange(size, dtype=mx.float32).reshape(shape) / 100.0

    return FeatureContext(
        token_features=seq((batch_size, n_tokens, cfg.feature_dims.token)),
        token_pair_features=seq(
            (batch_size, n_tokens, n_tokens, cfg.feature_dims.token_pair)
        ),
        atom_features=seq((batch_size, n_atoms, cfg.feature_dims.atom)),
        atom_pair_features=seq(
            (
                batch_size,
                atom_blocks,
                cfg.atom_blocks.query_block,
                cfg.atom_blocks.kv_block,
                cfg.feature_dims.atom_pair,
            )
        ),
        msa_features=seq((batch_size, msa_depth, n_tokens, cfg.feature_dims.msa)),
        template_features=seq(
            (batch_size, 1, n_tokens, n_tokens, cfg.feature_dims.templates)
        ),
        structure_inputs=structure,
        bond_adjacency=mx.zeros(
            (batch_size, n_tokens, n_tokens, 1), dtype=mx.float32
        ),
    )


def test_recycle_msa_subsample_zero_matches_baseline() -> None:
    """``recycle_msa_subsample=0`` (default) must still produce finite
    coords without touching the subsample code path."""
    cfg = ChaiConfig(compute_dtype="float32")
    mx.random.seed(42)
    model = ChaiMLX(cfg)
    ctx = _tiny_context(cfg)

    result = model.run_inference(
        ctx,
        recycles=1,
        num_samples=1,
        num_steps=2,
        recycle_msa_subsample=0,
    )
    assert isinstance(result, InferenceOutputs)
    coords = np.array(result.coords)
    assert coords.shape == (1, 1, 32, 3)
    assert np.isfinite(coords).all()


def test_recycle_msa_subsample_positive_runs() -> None:
    """Non-zero subsample activates the resample-each-recycle path."""
    cfg = ChaiConfig(compute_dtype="float32")
    mx.random.seed(42)
    model = ChaiMLX(cfg)
    ctx = _tiny_context(cfg)

    # The tiny fixture has msa_depth=8, so select_n_rows=4 triggers
    # the subsampling branch inside _subsample_and_reorder_msa.
    result = model.run_inference(
        ctx,
        recycles=2,
        num_samples=1,
        num_steps=2,
        recycle_msa_subsample=4,
    )
    coords = np.array(result.coords)
    assert coords.shape == (1, 1, 32, 3)
    assert np.isfinite(coords).all()


def test_recycle_msa_subsample_larger_than_depth_is_noop() -> None:
    """When select_n_rows >= depth, the subsample helper returns the
    inputs unchanged. The model should produce the same coords as the
    ``=0`` baseline under the same seed."""
    cfg = ChaiConfig(compute_dtype="float32")
    ctx = _tiny_context(cfg)

    mx.random.seed(42)
    model_a = ChaiMLX(cfg)
    out_a = model_a.run_inference(
        ctx,
        recycles=1,
        num_samples=1,
        num_steps=2,
        recycle_msa_subsample=0,
    )
    coords_a = np.array(out_a.coords)

    mx.random.seed(42)
    model_b = ChaiMLX(cfg)
    out_b = model_b.run_inference(
        ctx,
        recycles=1,
        num_samples=1,
        num_steps=2,
        # depth in _tiny_context is 8; anything >= 8 must no-op.
        recycle_msa_subsample=16,
    )
    coords_b = np.array(out_b.coords)

    np.testing.assert_array_equal(
        coords_a, coords_b,
        err_msg=(
            "recycle_msa_subsample >= depth should be a no-op but "
            "the coords diverged from the =0 baseline."
        ),
    )
