"""Minimal end-to-end smoke example using random dummy inputs.

Runs ``ChaiMLX`` across embedding, trunk, one diffusion step, confidence, and
ranking on random inputs and prints the resulting shapes. No weights, no
featurizer; useful only as a sanity check that the pipeline is wired up.

For a real FASTA-backed inference run, see ``examples/fasta_smoke.py`` or
``scripts/inference.py``.
"""

from __future__ import annotations

import mlx.core as mx

from chai_mlx import ChaiMLX, featurize


def build_dummy_inputs(
    batch_size: int = 1,
    n_tokens: int = 256,
    msa_depth: int = 32,
    n_templates: int = 4,
) -> dict:
    """Build a FeatureContext-shaped dict of random inputs.

    All tensors are random where the shape is non-trivial, and mask-like
    fields are set to ones so the model exercises every code path. Index
    fields used by the ranker (``token_centre_atom_index``,
    ``token_residue_index``, etc.) are populated with reasonable defaults
    so ``rank_outputs`` runs cleanly; the numeric values are not meaningful.

    ``structure_inputs`` must include ``msa_mask`` and
    ``template_input_masks``: the trunk reads both via ``StructureInputs``
    (see ``chai_mlx/model/trunk.py`` — the MSA-subsample path needs
    ``msa_mask`` and the template embedder needs ``template_input_masks``).
    Omitting them leaves the fields at their ``Optional`` default of
    ``None`` and the first ``.astype(...)`` call inside the trunk fails
    with ``AttributeError: 'NoneType' object has no attribute 'astype'``.
    """
    n_atoms = 23 * n_tokens
    num_blocks = n_atoms // 32
    token_exists = mx.ones((batch_size, n_tokens), dtype=mx.bool_)
    atom_exists = mx.ones((batch_size, n_atoms), dtype=mx.bool_)
    atom_token_index = mx.arange(n_atoms)[None, :] // 23
    atom_within_token_index = mx.arange(n_atoms)[None, :] % 37
    token_ref_atom_index = mx.arange(n_tokens)[None, :] * 23
    token_centre_atom_index = mx.arange(n_tokens)[None, :] * 23
    q_idx = mx.arange(n_atoms).reshape(1, num_blocks, 32)
    kv_idx = mx.clip(
        q_idx[:, :, :1] + mx.arange(128)[None, None, :] - 48,
        0,
        n_atoms - 1,
    )
    block_mask = mx.ones((batch_size, num_blocks, 32, 128), dtype=mx.bool_)
    msa_mask = mx.ones((batch_size, msa_depth, n_tokens), dtype=mx.float32)
    template_input_masks = mx.ones(
        (batch_size, n_templates, n_tokens, n_tokens), dtype=mx.float32
    )

    return {
        "token_features": mx.random.normal((batch_size, n_tokens, 2638)),
        "token_pair_features": mx.random.normal((batch_size, n_tokens, n_tokens, 163)),
        "atom_features": mx.random.normal((batch_size, n_atoms, 395)),
        "atom_pair_features": mx.random.normal((batch_size, num_blocks, 32, 128, 14)),
        "msa_features": mx.random.normal((batch_size, msa_depth, n_tokens, 42)),
        "template_features": mx.random.normal((batch_size, n_templates, n_tokens, n_tokens, 76)),
        "bond_adjacency": mx.zeros((batch_size, n_tokens, n_tokens, 1)),
        "structure_inputs": {
            "atom_exists_mask": atom_exists,
            "token_exists_mask": token_exists,
            "token_pair_mask": token_exists[:, :, None] & token_exists[:, None, :],
            "atom_token_index": atom_token_index,
            "atom_within_token_index": atom_within_token_index,
            "token_reference_atom_index": token_ref_atom_index,
            "token_centre_atom_index": token_centre_atom_index,
            "token_asym_id": mx.ones((batch_size, n_tokens), dtype=mx.int32),
            "token_entity_id": mx.ones((batch_size, n_tokens), dtype=mx.int32),
            "token_chain_id": mx.ones((batch_size, n_tokens), dtype=mx.int32),
            "token_is_polymer": mx.ones((batch_size, n_tokens), dtype=mx.bool_),
            "token_residue_index": mx.arange(n_tokens, dtype=mx.int32)[None, :],
            "token_entity_type": mx.zeros((batch_size, n_tokens), dtype=mx.int32),
            "token_backbone_frame_mask": mx.ones((batch_size, n_tokens), dtype=mx.bool_),
            "token_backbone_frame_index": mx.zeros((batch_size, n_tokens, 3), dtype=mx.int32),
            "bond_adjacency": mx.zeros((batch_size, n_tokens, n_tokens, 1)),
            "atom_q_indices": q_idx,
            "atom_kv_indices": kv_idx,
            "block_atom_pair_mask": block_mask,
            "msa_mask": msa_mask,
            "template_input_masks": template_input_masks,
        },
    }


def main() -> None:
    model = ChaiMLX()
    ctx = featurize(build_dummy_inputs())
    emb = model.embed_inputs(ctx)
    trunk_out = model.trunk(emb, recycles=1)
    cache = model.prepare_diffusion_cache(trunk_out)
    coords = model.init_noise(batch_size=1, num_samples=1, structure=emb.structure_inputs)
    sigma_curr, sigma_next, gamma = next(model.schedule(num_steps=8))
    coords = model.diffusion_step(cache, coords, sigma_curr, sigma_next, gamma)
    conf = model.confidence(trunk_out, coords)
    rank = model.rank_outputs(conf, coords, emb.structure_inputs)
    print(coords.shape, conf.pae_logits.shape, rank.aggregate_score.shape)


if __name__ == "__main__":
    main()
