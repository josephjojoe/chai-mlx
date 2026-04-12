"""Collation: padding, block index construction, and assembly into
``FeatureContext`` + ``StructureInputs``.

Ported from ``chai_lab/data/collate/`` and ``chai_lab/model/utils.py``.
"""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx
import numpy as np

from ..config import Chai1Config
from ..types import FeatureContext, StructureInputs
from .embeddings import EmbeddingContext
from .feature_generators import AllFeatures, generate_all_features
from .msa import MSAContext
from .structure import AllAtomStructureContext, EntityType
from .templates import TemplateContext

AVAILABLE_MODEL_SIZES = [256, 384, 512, 768, 1024, 1536, 2048]
ATOM_MULTIPLIER = 23
MAX_MSA_DEPTH = 16_384
MAX_NUM_TEMPLATES = 4


def get_pad_sizes(
    n_tokens: int,
    n_atoms: int,
    supported_sizes: Sequence[int] = AVAILABLE_MODEL_SIZES,
) -> tuple[int, int]:
    """Compute padded token and atom counts."""
    for size in sorted(supported_sizes):
        if size >= n_tokens:
            padded_tokens = size
            padded_atoms = ATOM_MULTIPLIER * padded_tokens
            if padded_atoms >= n_atoms:
                assert padded_atoms % 32 == 0, f"Padded atoms {padded_atoms} not divisible by 32"
                return padded_tokens, padded_atoms
    raise ValueError(
        f"n_tokens={n_tokens} exceeds max supported size {max(supported_sizes)}"
    )


# ── Block index construction ─────────────────────────────────────────

def get_qkv_indices_for_blocks(
    n_atoms: int,
    num_query_atoms: int = 32,
    num_key_atoms: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build blocked atom-pair Q/KV indices and validity mask.

    Returns:
        q_indices: [n_blocks, num_query_atoms] int — query atom indices
        kv_indices: [n_blocks, num_key_atoms] int — key/value atom indices
        kv_mask: [n_blocks, num_query_atoms, num_key_atoms] bool — validity mask
    """
    assert n_atoms % num_query_atoms == 0, f"{n_atoms} not divisible by {num_query_atoms}"
    n_blocks = n_atoms // num_query_atoms

    q_indices = np.arange(n_atoms, dtype=np.int64).reshape(n_blocks, num_query_atoms)

    kv_half = num_key_atoms // 2
    kv_indices = np.zeros((n_blocks, num_key_atoms), dtype=np.int64)
    kv_mask = np.zeros((n_blocks, num_query_atoms, num_key_atoms), dtype=bool)

    for b in range(n_blocks):
        center = b * num_query_atoms + num_query_atoms // 2
        start = center - kv_half
        end = start + num_key_atoms

        for k in range(num_key_atoms):
            idx = start + k
            if 0 <= idx < n_atoms:
                kv_indices[b, k] = idx
                kv_mask[b, :, k] = True
            else:
                kv_indices[b, k] = idx % n_atoms

    return q_indices, kv_indices, kv_mask


def get_block_atom_pair_mask(
    atom_ref_mask: np.ndarray,
    q_indices: np.ndarray,
    kv_indices: np.ndarray,
    kv_mask: np.ndarray,
) -> np.ndarray:
    """Combine ref-mask with structural validity for block atom pairs.

    Returns: [n_blocks, num_query_atoms, num_key_atoms] bool
    """
    q_valid = atom_ref_mask[q_indices].astype(bool)   # [nb, q]
    kv_valid = atom_ref_mask[kv_indices].astype(bool)  # [nb, kv]
    pair_valid = q_valid[:, :, None] & kv_valid[:, None, :]  # [nb, q, kv]
    return pair_valid & kv_mask


# ── Collation ────────────────────────────────────────────────────────

def collate(
    structure: AllAtomStructureContext,
    *,
    msa_ctx: MSAContext | None = None,
    template_ctx: TemplateContext | None = None,
    embedding_ctx: EmbeddingContext | None = None,
    cfg: Chai1Config | None = None,
) -> tuple[FeatureContext, AllAtomStructureContext]:
    """Pad structure/contexts, build block indices, generate features, and
    assemble into ``FeatureContext``.
    """
    cfg = cfg or Chai1Config()
    n_tokens_raw = structure.num_tokens
    n_atoms_raw = structure.num_atoms

    # Determine padded sizes
    n_tokens, n_atoms = get_pad_sizes(
        n_tokens_raw, n_atoms_raw, cfg.supported_token_sizes
    )

    # Pad structure
    padded = structure.pad(n_tokens, n_atoms)

    # Pad external contexts
    msa = (msa_ctx or MSAContext.empty(n_tokens)).pad(n_tokens, MAX_MSA_DEPTH)
    tmpl = (template_ctx or TemplateContext.empty(n_tokens)).pad(n_tokens, MAX_NUM_TEMPLATES)
    emb = (embedding_ctx or EmbeddingContext.empty(n_tokens)).pad(n_tokens)

    # Build block indices
    q_idx, kv_idx, kv_mask = get_qkv_indices_for_blocks(
        n_atoms,
        num_query_atoms=cfg.atom_blocks.query_block,
        num_key_atoms=cfg.atom_blocks.kv_block,
    )
    block_pair_mask = get_block_atom_pair_mask(
        padded.atom_ref_mask.astype(np.float32), q_idx, kv_idx, kv_mask
    )

    # Generate all features
    features = generate_all_features(
        padded,
        esm_embeddings=emb.esm_embeddings,
        msa_tokens=msa.tokens,
        msa_deletion_matrix=msa.deletion_matrix,
        msa_mask=msa.mask,
        msa_species=msa.species,
        template_restype=tmpl.template_restype,
        template_distances=tmpl.template_distances,
        template_unit_vector=tmpl.template_unit_vector,
        template_mask=tmpl.template_mask,
        q_indices=q_idx,
        kv_indices=kv_idx,
        block_mask=block_pair_mask,
    )

    # Build token_pair_mask
    tok_mask = padded.token_exists_mask.astype(np.float32)
    token_pair_mask_np = tok_mask[:, None] * tok_mask[None, :]

    # Determine polymer mask
    polymer_types = {EntityType.PROTEIN.value, EntityType.RNA.value, EntityType.DNA.value}
    is_polymer = np.array(
        [int(t in polymer_types) for t in padded.token_entity_type], dtype=np.float32
    ).reshape(-1)

    # Convert to MLX arrays
    def _mx(arr: np.ndarray) -> mx.array:
        return mx.array(arr)

    structure_inputs = StructureInputs(
        atom_exists_mask=_mx(padded.atom_exists_mask.astype(np.float32)),
        token_exists_mask=_mx(padded.token_exists_mask.astype(np.float32)),
        token_pair_mask=_mx(token_pair_mask_np),
        atom_token_index=_mx(padded.atom_token_index),
        atom_within_token_index=_mx(padded.atom_within_token_index),
        token_reference_atom_index=_mx(padded.token_ref_atom_index),
        token_asym_id=_mx(padded.token_asym_id),
        token_entity_id=_mx(padded.token_entity_id),
        token_chain_id=_mx(padded.token_asym_id),
        token_is_polymer=_mx(is_polymer),
        atom_ref_positions=_mx(padded.atom_ref_pos),
        atom_ref_space_uid=_mx(padded.atom_ref_space_uid),
        atom_q_indices=_mx(q_idx),
        atom_kv_indices=_mx(kv_idx),
        block_atom_pair_mask=_mx(block_pair_mask.astype(np.float32)),
    )

    # Add batch dimension
    token_feat = _mx(features.token[None])
    pair_feat = _mx(features.token_pair[None])
    atom_feat = _mx(features.atom[None])
    atom_pair_feat = _mx(features.atom_pair[None]) if features.atom_pair.size > 0 else _mx(
        np.zeros((1, q_idx.shape[0], cfg.atom_blocks.query_block, cfg.atom_blocks.kv_block, 14), dtype=np.float32)
    )
    msa_feat = _mx(features.msa[None]) if features.msa.size > 0 else _mx(
        np.zeros((1, 0, n_tokens, 42), dtype=np.float32)
    )
    tmpl_feat = _mx(features.template[None]) if features.template.size > 0 else _mx(
        np.zeros((1, MAX_NUM_TEMPLATES, n_tokens, n_tokens, 76), dtype=np.float32)
    )

    feature_ctx = FeatureContext(
        token_features=token_feat,
        token_pair_features=pair_feat,
        atom_features=atom_feat,
        atom_pair_features=atom_pair_feat,
        msa_features=msa_feat,
        template_features=tmpl_feat,
        structure_inputs=structure_inputs,
    )

    return feature_ctx, padded
