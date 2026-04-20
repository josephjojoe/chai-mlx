"""Run the same inputs through ``run_inference`` twice and assert
the coords are bit-identical.

A silent loss of same-machine determinism (for example from a backend or
driver change introducing nondeterministic reductions) would be easy to
miss in normal use. This test keeps a tight deterministic guardrail on a
small no-weights configuration. It uses ``float32`` so bf16 rounding is
not a confounder.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import mlx.core as mx
import numpy as np
import pytest

from chai_mlx import ChaiMLX
from chai_mlx.config import ChaiConfig
from chai_mlx.data.types import FeatureContext

from tests.helpers import make_structure_inputs


def _tiny_context(cfg: ChaiConfig) -> FeatureContext:
    batch_size = 1
    n_tokens = 4
    n_atoms = 32
    atom_blocks = n_atoms // 32

    structure = make_structure_inputs(
        batch_size=batch_size,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
        msa_depth=2,
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
        msa_features=seq((batch_size, 2, n_tokens, cfg.feature_dims.msa)),
        template_features=seq(
            (batch_size, 1, n_tokens, n_tokens, cfg.feature_dims.templates)
        ),
        structure_inputs=structure,
        bond_adjacency=mx.zeros(
            (batch_size, n_tokens, n_tokens, 1), dtype=mx.float32
        ),
    )


def test_run_inference_is_bit_exact_on_same_seed() -> None:
    """Same seed, same inputs, same model → bit-identical coords.

    If this test ever starts failing, it means MLX / Metal has
    introduced a non-deterministic reduction on our hot path. That is
    worth investigating before shipping a release.
    """
    cfg = ChaiConfig(compute_dtype="float32")

    mx.random.seed(42)
    model1 = ChaiMLX(cfg)
    ctx1 = _tiny_context(cfg)
    out1 = model1.run_inference(ctx1, recycles=1, num_samples=1, num_steps=2)
    coords1 = np.array(out1.coords)
    mx.eval(out1.coords)

    mx.random.seed(42)
    model2 = ChaiMLX(cfg)
    ctx2 = _tiny_context(cfg)
    out2 = model2.run_inference(ctx2, recycles=1, num_samples=1, num_steps=2)
    coords2 = np.array(out2.coords)
    mx.eval(out2.coords)

    if not np.array_equal(coords1, coords2):
        max_abs = float(np.max(np.abs(coords1 - coords2)))
        n_diff = int(np.sum(coords1 != coords2))
        pytest.fail(
            "ChaiMLX.run_inference is not bit-exact under the same seed.\n"
            f"  max abs diff: {max_abs}\n"
            f"  diffs: {n_diff}/{coords1.size}\n"
            "  this usually means MLX/Metal has introduced a non-"
            "deterministic reduction on our hot path."
        )
