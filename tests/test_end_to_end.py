"""End-to-end smoke of :meth:`ChaiMLX.run_inference` on a tiny shape.

This is intentionally the smallest possible exercise of the full pipeline
(embedding + trunk + diffusion + confidence + ranking) with random weights
and tiny tensors. It protects the public ``run_inference`` entry point
against the kind of silent breakage that broke ``examples/basic_inference.py``
when the ranker was ported.

Numerics are meaningless (weights are randomly initialised) — we only
assert that the call completes and returns the expected dataclass shapes.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import mlx.core as mx

from chai_mlx import ChaiMLX, InferenceOutputs
from chai_mlx.config import ChaiConfig
from chai_mlx.data.types import FeatureContext

from tests.helpers import make_structure_inputs


def _tiny_context(cfg: ChaiConfig) -> FeatureContext:
    """Build a FeatureContext wired to sizes that satisfy atom-attention.

    ``n_atoms`` must be a multiple of 32 (the atom-attention query block)
    and we need a realistic enough ``atom_pair_features`` shape for the
    encoder to run. The rest of the shape is kept small so the test is fast.
    """
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


def test_run_inference_tiny_smoke() -> None:
    cfg = ChaiConfig(compute_dtype="float32")
    model = ChaiMLX(cfg)
    ctx = _tiny_context(cfg)

    result = model.run_inference(ctx, recycles=1, num_samples=1, num_steps=2)

    assert isinstance(result, InferenceOutputs)
    assert result.coords.shape == (1, 1, 32, 3)
    assert result.confidence.pae_logits.shape[:2] == (1, 1)
    assert result.confidence.pde_logits.shape[:2] == (1, 1)
    assert result.confidence.plddt_logits.shape[:2] == (1, 1)
    assert result.ranking.aggregate_score.shape == (1, 1)

    mx.eval(
        result.coords,
        result.confidence.pae_logits,
        result.confidence.pde_logits,
        result.confidence.plddt_logits,
        result.ranking.aggregate_score,
    )
