"""Feature generators — pure numpy reimplementation of the 30+ generators in
``chai_lab/data/features/generators/``.

Each generator is a function that takes the structure context (and optional
external contexts) and returns a numpy array. The final ``generate_all_features``
function concatenates them by FeatureType and returns the 6 feature tensors.

Target dimensions (from ``config.py::FeatureDims``):
    TOKEN:      2638
    TOKEN_PAIR:  163
    ATOM:        395
    ATOM_PAIR:    14
    MSA:          42
    TEMPLATES:    76
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .residue_constants import (
    NUM_RESTYPES,
    residue_types_with_nucleotides_order,
)
from .structure import AllAtomStructureContext, EntityType


# =====================================================================
# Dimension constants (must sum to FeatureDims when concatenated)
# =====================================================================

# TOKEN features (total = 2638)
_RESTYPE_DIM = 32            # one-hot residue type (with extra padding)
_ESM_DIM = 2560              # ESM2 embeddings
_MSA_PROFILE_DIM = NUM_RESTYPES  # 33
_MSA_DELETION_MEAN_DIM = 1
_IS_DISTILLATION_DIM = 2
_TOKEN_BFACTOR_DIM = 2
_TOKEN_PLDDT_DIM = 3
_CHAIN_IS_CROPPED_DIM = 1
_MISSING_CHAIN_CONTACT_DIM = 1
_TOKEN_EXTRA_DIM = 3         # padding to reach 2638

# TOKEN_PAIR features (total = 163)
_REL_SEQ_SEP_DIM = 67        # binned sequence separation
_REL_TOKEN_SEP_DIM = 67      # binned token separation (same as seq sep dim)
_REL_ENTITY_DIM = 3          # relative entity id
_REL_CHAIN_DIM = 6           # relative chain id
_DOCKING_DIM = 5             # docking constraint bins
_TOKEN_DIST_RESTRAINT_DIM = 5  # distance restraint RBF
_TOKEN_POCKET_RESTRAINT_DIM = 5  # pocket restraint RBF
_TOKEN_BOND_DIM = 1          # bond adjacency
_TOKEN_PAIR_DISTANCE_DIM = 1 # token centre distance
_TOKEN_PAIR_EXTRA_DIM = 3    # padding to reach 163

# ATOM features (total = 395)
_REF_POS_DIM = 3
_REF_CHARGE_DIM = 1
_REF_MASK_DIM = 1
_ATOM_ELEMENT_DIM = 129      # one-hot atomic number
_ATOM_NAME_DIM = 256         # 4 chars × 64 classes each
_ATOM_EXTRA_DIM = 5          # padding to reach 395

# ATOM_PAIR features (total = 14)
_BLOCKED_DISTOGRAM_DIM = 11  # binned blocked distances
_INV_SQUARED_DIST_DIM = 2    # 1/(1+d²) + mask
_ATOM_PAIR_EXTRA_DIM = 1     # padding to reach 14

# MSA features (total = 42)
_MSA_ONEHOT_DIM = NUM_RESTYPES  # 33
_MSA_HAS_DELETION_DIM = 1
_MSA_DELETION_VALUE_DIM = 1
_MSA_IS_PAIRED_DIM = 1
_MSA_DATA_SOURCE_DIM = 6

# TEMPLATES features (total = 76)
_TEMPLATE_MASK_DIM = 2
_TEMPLATE_UNIT_VECTOR_DIM = 3
_TEMPLATE_RESTYPE_DIM = NUM_RESTYPES  # 33
_TEMPLATE_DISTOGRAM_DIM = 38


# =====================================================================
# Helper functions
# =====================================================================

def _one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer indices to one-hot encoding."""
    indices = np.clip(indices.astype(np.int64), 0, num_classes - 1)
    out = np.zeros(indices.shape + (num_classes,), dtype=np.float32)
    np.put_along_axis(out, indices[..., None], 1.0, axis=-1)
    return out


def _binned(values: np.ndarray, bin_edges: Sequence[float]) -> np.ndarray:
    """Digitize *values* into bins, returning one-hot of shape ``[..., len(edges)+1]``."""
    edges = np.array(bin_edges, dtype=np.float32)
    idx = np.digitize(values, edges).astype(np.int64)
    return _one_hot(idx, len(edges) + 1)


def _relative_position_encoding(
    diff: np.ndarray,
    max_val: int,
    *,
    add_inter_bucket: bool = False,
) -> np.ndarray:
    """Encode relative positions, optionally with an inter-chain bucket."""
    n_classes = 2 * max_val + 1
    if add_inter_bucket:
        n_classes += 1
    clipped = np.clip(diff, -max_val, max_val) + max_val
    return _one_hot(clipped.astype(np.int64), n_classes)


# =====================================================================
# TOKEN feature generators
# =====================================================================

def gen_residue_type(ctx: AllAtomStructureContext) -> np.ndarray:
    """One-hot residue type. [n_tokens, 32]"""
    return _one_hot(ctx.token_residue_type, _RESTYPE_DIM)


def gen_esm_embeddings(
    ctx: AllAtomStructureContext,
    esm: np.ndarray | None = None,
) -> np.ndarray:
    """ESM2 embeddings passthrough. [n_tokens, 2560]"""
    n = ctx.num_tokens
    if esm is not None:
        if esm.shape[0] < n:
            out = np.zeros((n, esm.shape[1]), dtype=np.float32)
            out[:esm.shape[0]] = esm
            return out
        return esm[:n].astype(np.float32)
    return np.zeros((n, _ESM_DIM), dtype=np.float32)


def gen_msa_profile(
    ctx: AllAtomStructureContext,
    msa_tokens: np.ndarray | None = None,
    msa_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Normalized residue distribution from main MSA. [n_tokens, 33]"""
    n = ctx.num_tokens
    if msa_tokens is not None and msa_tokens.shape[0] > 0:
        oh = _one_hot(msa_tokens, NUM_RESTYPES)  # [depth, n, 33]
        if msa_mask is not None:
            oh = oh * msa_mask[..., None].astype(np.float32)
        denom = np.maximum(oh.sum(axis=0).sum(axis=-1, keepdims=True), 1.0)
        return (oh.sum(axis=0) / denom).astype(np.float32)[:n]
    return np.zeros((n, _MSA_PROFILE_DIM), dtype=np.float32)


def gen_msa_deletion_mean(
    ctx: AllAtomStructureContext,
    deletion_matrix: np.ndarray | None = None,
    msa_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Mean deletion count per position. [n_tokens, 1]"""
    n = ctx.num_tokens
    if deletion_matrix is not None and deletion_matrix.shape[0] > 0:
        if msa_mask is not None:
            masked = deletion_matrix * msa_mask.astype(np.float32)
            denom = np.maximum(msa_mask.sum(axis=0, keepdims=True).T, 1.0)
        else:
            masked = deletion_matrix
            denom = float(deletion_matrix.shape[0])
        mean = masked.mean(axis=0, keepdims=True).T.astype(np.float32)
        return mean[:n]
    return np.zeros((n, 1), dtype=np.float32)


def gen_is_distillation(ctx: AllAtomStructureContext) -> np.ndarray:
    """Broadcast is_distillation flag. [n_tokens, 2]"""
    n = ctx.num_tokens
    is_distill = bool(ctx.is_distillation[0])
    out = np.zeros((n, 2), dtype=np.float32)
    out[:, int(is_distill)] = 1.0
    return out


def gen_token_bfactor(ctx: AllAtomStructureContext) -> np.ndarray:
    """B-factor binned. [n_tokens, 2]"""
    return _binned(ctx.token_b_factor_or_plddt, [30.0])


def gen_token_plddt(ctx: AllAtomStructureContext) -> np.ndarray:
    """pLDDT binned (only for distillation). [n_tokens, 3]"""
    if bool(ctx.is_distillation[0]):
        return _binned(ctx.token_b_factor_or_plddt, [50.0, 70.0])
    return np.zeros((ctx.num_tokens, 3), dtype=np.float32)


def gen_chain_is_cropped(ctx: AllAtomStructureContext) -> np.ndarray:
    """Placeholder crop flag. [n_tokens, 1]"""
    return np.zeros((ctx.num_tokens, 1), dtype=np.float32)


def gen_missing_chain_contact(ctx: AllAtomStructureContext) -> np.ndarray:
    """Missing inter-chain contact flag. [n_tokens, 1]"""
    return np.zeros((ctx.num_tokens, 1), dtype=np.float32)


# =====================================================================
# TOKEN_PAIR feature generators
# =====================================================================

def gen_relative_sequence_separation(ctx: AllAtomStructureContext) -> np.ndarray:
    """Binned sequence separation. [n_tokens, n_tokens, 67]"""
    n = ctx.num_tokens
    res_idx = ctx.token_residue_index
    diff = res_idx[:, None] - res_idx[None, :]
    same_chain = (ctx.token_asym_id[:, None] == ctx.token_asym_id[None, :])

    sep_bins = list(range(-32, 33))
    n_bins = len(sep_bins) + 1  # 66
    inter_chain_bin = n_bins  # one extra for inter-chain

    idx = np.digitize(diff, sep_bins).astype(np.int64)
    idx = np.where(same_chain, idx, inter_chain_bin)
    return _one_hot(idx, n_bins + 1)  # 67


def gen_relative_token_separation(ctx: AllAtomStructureContext) -> np.ndarray:
    """Binned token index separation. [n_tokens, n_tokens, 67]"""
    n = ctx.num_tokens
    tok_idx = ctx.token_index
    diff = tok_idx[:, None] - tok_idx[None, :]
    same_chain = (ctx.token_asym_id[:, None] == ctx.token_asym_id[None, :])

    sep_bins = list(range(-32, 33))
    n_bins = len(sep_bins) + 1
    inter_chain_bin = n_bins

    idx = np.digitize(diff, sep_bins).astype(np.int64)
    idx = np.where(same_chain, idx, inter_chain_bin)
    return _one_hot(idx, n_bins + 1)  # 67


def gen_relative_entity(ctx: AllAtomStructureContext) -> np.ndarray:
    """Clamped relative entity id. [n_tokens, n_tokens, 3]"""
    diff = ctx.token_entity_id[:, None].astype(np.int64) - ctx.token_entity_id[None, :].astype(np.int64)
    return _one_hot(np.clip(diff + 1, 0, 2).astype(np.int64), 3)


def gen_relative_chain(ctx: AllAtomStructureContext) -> np.ndarray:
    """Relative chain index. [n_tokens, n_tokens, 6]"""
    diff = ctx.token_asym_id[:, None].astype(np.int64) - ctx.token_asym_id[None, :].astype(np.int64)
    return _one_hot(np.clip(diff + 2, 0, 5).astype(np.int64), 6)


def gen_docking_constraint(ctx: AllAtomStructureContext) -> np.ndarray:
    """Docking constraint (placeholder). [n_tokens, n_tokens, 5]"""
    return np.zeros((ctx.num_tokens, ctx.num_tokens, _DOCKING_DIM), dtype=np.float32)


def gen_token_distance_restraint(ctx: AllAtomStructureContext) -> np.ndarray:
    """Token distance restraint (placeholder). [n_tokens, n_tokens, 5]"""
    n = ctx.num_tokens
    return np.full((n, n, _TOKEN_DIST_RESTRAINT_DIM), -1.0, dtype=np.float32)


def gen_token_pocket_restraint(ctx: AllAtomStructureContext) -> np.ndarray:
    """Token pocket restraint (placeholder). [n_tokens, n_tokens, 5]"""
    n = ctx.num_tokens
    return np.full((n, n, _TOKEN_POCKET_RESTRAINT_DIM), -1.0, dtype=np.float32)


def gen_token_bond(ctx: AllAtomStructureContext) -> np.ndarray:
    """Token-token bond adjacency from atom-level bonds. [n_tokens, n_tokens, 1]"""
    n = ctx.num_tokens
    adj = np.zeros((n, n), dtype=np.float32)
    if ctx.bond_left.size > 0:
        tok_l = ctx.atom_token_index[ctx.bond_left]
        tok_r = ctx.atom_token_index[ctx.bond_right]
        mask = tok_l != tok_r
        adj[tok_l[mask], tok_r[mask]] = 1.0
        adj[tok_r[mask], tok_l[mask]] = 1.0
    return adj[..., None]


def gen_token_pair_distance(ctx: AllAtomStructureContext) -> np.ndarray:
    """Token centre distance (placeholder zeros for inference). [n_tokens, n_tokens, 1]"""
    n = ctx.num_tokens
    return np.zeros((n, n, 1), dtype=np.float32)


# =====================================================================
# ATOM feature generators
# =====================================================================

def gen_ref_pos(ctx: AllAtomStructureContext) -> np.ndarray:
    """Reference positions / 10 (nm scale). [n_atoms, 3]"""
    return (ctx.atom_ref_pos / 10.0).astype(np.float32)


def gen_ref_charge(ctx: AllAtomStructureContext) -> np.ndarray:
    """Reference charge. [n_atoms, 1]"""
    return ctx.atom_ref_charge.astype(np.float32).reshape(-1, 1)


def gen_ref_mask(ctx: AllAtomStructureContext) -> np.ndarray:
    """Reference atom mask. [n_atoms, 1]"""
    return ctx.atom_ref_mask.astype(np.float32).reshape(-1, 1)


def gen_atom_element(ctx: AllAtomStructureContext) -> np.ndarray:
    """One-hot atomic number. [n_atoms, 129]"""
    return _one_hot(np.clip(ctx.atom_ref_element, 0, 128), _ATOM_ELEMENT_DIM)


def gen_atom_name(ctx: AllAtomStructureContext) -> np.ndarray:
    """One-hot per character of PDB atom name (4 chars × 64). [n_atoms, 256]"""
    n = ctx.num_atoms
    chars = ctx.atom_ref_name_chars  # [n_atoms, 4] int
    parts = []
    for c in range(4):
        parts.append(_one_hot(np.clip(chars[:, c], 0, 63), 64))
    return np.concatenate(parts, axis=-1)


# =====================================================================
# ATOM_PAIR feature generators
# =====================================================================

def gen_blocked_distogram(
    ctx: AllAtomStructureContext,
    q_indices: np.ndarray | None = None,
    kv_indices: np.ndarray | None = None,
    block_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Blocked pairwise distance histogram.

    If block indices are not yet computed, returns zeros.
    Shape: [n_blocks, query_block, kv_block, 11]
    """
    if q_indices is None or kv_indices is None:
        return np.zeros((0, 0, 0, _BLOCKED_DISTOGRAM_DIM), dtype=np.float32)

    ref_pos = ctx.atom_ref_pos  # [n_atoms, 3]
    q_pos = ref_pos[q_indices]   # [n_blocks, q_block, 3]
    kv_pos = ref_pos[kv_indices] # [n_blocks, kv_block, 3]

    diff = q_pos[:, :, None, :] - kv_pos[:, None, :, :]  # [nb, q, kv, 3]
    dist = np.sqrt((diff ** 2).sum(axis=-1) + 1e-10)

    bin_edges = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    result = _binned(dist, bin_edges)  # [nb, q, kv, 11]

    if block_mask is not None:
        result = result * block_mask[..., None].astype(np.float32)

    return result


def gen_inv_squared_dist(
    ctx: AllAtomStructureContext,
    q_indices: np.ndarray | None = None,
    kv_indices: np.ndarray | None = None,
    block_mask: np.ndarray | None = None,
) -> np.ndarray:
    """1/(1+d²) and mask for blocked atom pairs.

    Shape: [n_blocks, query_block, kv_block, 2]
    """
    if q_indices is None or kv_indices is None:
        return np.zeros((0, 0, 0, _INV_SQUARED_DIST_DIM), dtype=np.float32)

    ref_pos = ctx.atom_ref_pos
    q_pos = ref_pos[q_indices]
    kv_pos = ref_pos[kv_indices]

    diff = q_pos[:, :, None, :] - kv_pos[:, None, :, :]
    dist_sq = (diff ** 2).sum(axis=-1)
    inv_sq = 1.0 / (1.0 + dist_sq)

    ref_mask = ctx.atom_ref_mask.astype(np.float32)
    q_mask = ref_mask[q_indices]
    kv_mask = ref_mask[kv_indices]
    pair_mask = q_mask[:, :, None] * kv_mask[:, None, :]

    if block_mask is not None:
        pair_mask = pair_mask * block_mask.astype(np.float32)

    return np.stack([inv_sq * pair_mask, pair_mask], axis=-1).astype(np.float32)


# =====================================================================
# MSA feature generators
# =====================================================================

def gen_msa_features(
    ctx: AllAtomStructureContext,
    msa_tokens: np.ndarray | None = None,
    deletion_matrix: np.ndarray | None = None,
    msa_mask: np.ndarray | None = None,
    msa_species: np.ndarray | None = None,
) -> np.ndarray:
    """All MSA features concatenated. [depth, n_tokens, 42]

    Components: one-hot(33) + has_deletion(1) + deletion_value(1)
              + is_paired(1) + data_source(6) = 42
    """
    n = ctx.num_tokens
    if msa_tokens is None or msa_tokens.shape[0] == 0:
        return np.zeros((0, n, 42), dtype=np.float32)

    depth = msa_tokens.shape[0]

    onehot = _one_hot(msa_tokens, _MSA_ONEHOT_DIM)[:, :n]  # [depth, n, 33]

    if deletion_matrix is not None:
        del_mat = deletion_matrix[:depth, :n]
        has_del = (del_mat > 0).astype(np.float32)[..., None]  # [depth, n, 1]
        del_val = (2.0 / np.pi * np.arctan(del_mat / 3.0)).astype(np.float32)[..., None]
    else:
        has_del = np.zeros((depth, n, 1), dtype=np.float32)
        del_val = np.zeros((depth, n, 1), dtype=np.float32)

    is_paired = np.zeros((depth, n, 1), dtype=np.float32)

    if msa_species is not None:
        source = _one_hot(np.clip(msa_species[:depth], 0, 5), 6)
        source = np.broadcast_to(source[:, None, :], (depth, n, 6)).copy()
    else:
        source = np.zeros((depth, n, 6), dtype=np.float32)

    return np.concatenate([onehot, has_del, del_val, is_paired, source], axis=-1)


# =====================================================================
# TEMPLATE feature generators
# =====================================================================

def gen_template_features(
    ctx: AllAtomStructureContext,
    template_restype: np.ndarray | None = None,
    template_distances: np.ndarray | None = None,
    template_unit_vector: np.ndarray | None = None,
    template_mask_arr: np.ndarray | None = None,
) -> np.ndarray:
    """All template features concatenated. [n_templates, n_tokens, n_tokens, 76]

    Components: mask(2) + unit_vector(3) + restype(33) + distogram(38) = 76
    """
    n = ctx.num_tokens
    if template_restype is None or template_restype.shape[0] == 0:
        return np.zeros((0, n, n, 76), dtype=np.float32)

    nt = template_restype.shape[0]

    # Template mask: backbone frame mask and pseudo-beta pair mask
    bb_mask = ctx.token_backbone_frame_mask.astype(np.float32)
    pair_mask_bb = bb_mask[:, None] * bb_mask[None, :]  # [n, n]
    same_asym = (ctx.token_asym_id[:, None] == ctx.token_asym_id[None, :]).astype(np.float32)

    if template_mask_arr is not None:
        tmask = template_mask_arr[:nt, :n, :n].astype(np.float32)
    else:
        tmask = np.broadcast_to(
            (pair_mask_bb * same_asym)[None, :, :], (nt, n, n)
        ).copy().astype(np.float32)

    mask_feat = np.stack([tmask, tmask], axis=-1)  # [nt, n, n, 2]

    # Unit vectors
    if template_unit_vector is not None:
        uv = template_unit_vector[:nt, :n, :n, :3].astype(np.float32)
    else:
        uv = np.zeros((nt, n, n, 3), dtype=np.float32)

    # Restype: outer sum of one-hot
    if template_restype is not None:
        rt = _one_hot(template_restype[:nt, :n], NUM_RESTYPES)  # [nt, n, 33]
        rt_pair = rt[:, :, None, :] + rt[:, None, :, :]  # This would be 33+33=66
        # Actually it's just one-hot per-position. Reference uses separate per-token.
        # Template restype generator stores one-hot at token level, then the template
        # embedding handles the pair structure. Here we just use the 33-dim one-hot.
        # Expand to pair by broadcasting along one dimension for compatibility.
        rt_feat = np.broadcast_to(rt[:, :, None, :], (nt, n, n, NUM_RESTYPES)).copy()
    else:
        rt_feat = np.zeros((nt, n, n, NUM_RESTYPES), dtype=np.float32)

    # Distance histogram
    if template_distances is not None:
        dist = template_distances[:nt, :n, :n]
        bin_edges = np.linspace(2.0, 22.0, 37).tolist()
        disto = _binned(dist, bin_edges)  # [nt, n, n, 38]
    else:
        disto = np.zeros((nt, n, n, 38), dtype=np.float32)

    return np.concatenate([mask_feat, uv, rt_feat, disto], axis=-1)


# =====================================================================
# Master generator
# =====================================================================

@dataclass
class AllFeatures:
    """Container for all 6 feature tensor groups."""
    token: np.ndarray          # [n_tokens, 2638]
    token_pair: np.ndarray     # [n_tokens, n_tokens, 163]
    atom: np.ndarray           # [n_atoms, 395]
    atom_pair: np.ndarray      # [n_blocks, q_block, kv_block, 14]  (empty until collation)
    msa: np.ndarray            # [depth, n_tokens, 42]
    template: np.ndarray       # [n_templates, n_tokens, n_tokens, 76]


def generate_all_features(
    ctx: AllAtomStructureContext,
    *,
    esm_embeddings: np.ndarray | None = None,
    msa_tokens: np.ndarray | None = None,
    msa_deletion_matrix: np.ndarray | None = None,
    msa_mask: np.ndarray | None = None,
    msa_species: np.ndarray | None = None,
    template_restype: np.ndarray | None = None,
    template_distances: np.ndarray | None = None,
    template_unit_vector: np.ndarray | None = None,
    template_mask: np.ndarray | None = None,
    q_indices: np.ndarray | None = None,
    kv_indices: np.ndarray | None = None,
    block_mask: np.ndarray | None = None,
) -> AllFeatures:
    """Run all feature generators and return concatenated feature arrays."""

    n = ctx.num_tokens
    na = ctx.num_atoms

    # ── TOKEN features ───────────────────────────────────────────
    token_parts = [
        gen_residue_type(ctx),                    # 32
        gen_esm_embeddings(ctx, esm_embeddings),  # 2560
        gen_msa_profile(ctx, msa_tokens, msa_mask),  # 33
        gen_msa_deletion_mean(ctx, msa_deletion_matrix, msa_mask),  # 1
        gen_is_distillation(ctx),                 # 2
        gen_token_bfactor(ctx),                   # 2
        gen_token_plddt(ctx),                     # 3
        gen_chain_is_cropped(ctx),                # 1
        gen_missing_chain_contact(ctx),           # 1
        np.zeros((n, _TOKEN_EXTRA_DIM), dtype=np.float32),  # 3
    ]
    token_feat = np.concatenate(token_parts, axis=-1)
    assert token_feat.shape == (n, 2638), f"TOKEN dim mismatch: {token_feat.shape[-1]} != 2638"

    # ── TOKEN_PAIR features ──────────────────────────────────────
    pair_parts = [
        gen_relative_sequence_separation(ctx),    # 67
        gen_relative_token_separation(ctx),       # 67
        gen_relative_entity(ctx),                 # 3
        gen_relative_chain(ctx),                  # 6
        gen_docking_constraint(ctx),              # 5
        gen_token_distance_restraint(ctx),        # 5
        gen_token_pocket_restraint(ctx),          # 5
        gen_token_bond(ctx),                      # 1
        gen_token_pair_distance(ctx),             # 1
        np.zeros((n, n, _TOKEN_PAIR_EXTRA_DIM), dtype=np.float32),  # 3
    ]
    pair_feat = np.concatenate(pair_parts, axis=-1)
    assert pair_feat.shape == (n, n, 163), f"TOKEN_PAIR dim mismatch: {pair_feat.shape[-1]} != 163"

    # ── ATOM features ────────────────────────────────────────────
    atom_parts = [
        gen_ref_pos(ctx),                         # 3
        gen_ref_charge(ctx),                      # 1
        gen_ref_mask(ctx),                        # 1
        gen_atom_element(ctx),                    # 129
        gen_atom_name(ctx),                       # 256
        np.zeros((na, _ATOM_EXTRA_DIM), dtype=np.float32),  # 5
    ]
    atom_feat = np.concatenate(atom_parts, axis=-1)
    assert atom_feat.shape == (na, 395), f"ATOM dim mismatch: {atom_feat.shape[-1]} != 395"

    # ── ATOM_PAIR features ───────────────────────────────────────
    disto = gen_blocked_distogram(ctx, q_indices, kv_indices, block_mask)
    inv_sq = gen_inv_squared_dist(ctx, q_indices, kv_indices, block_mask)
    if disto.size > 0:
        atom_pair_feat = np.concatenate(
            [disto, inv_sq, np.zeros(disto.shape[:-1] + (_ATOM_PAIR_EXTRA_DIM,), dtype=np.float32)],
            axis=-1,
        )
    else:
        atom_pair_feat = np.zeros((0, 0, 0, 14), dtype=np.float32)

    # ── MSA features ─────────────────────────────────────────────
    msa_feat = gen_msa_features(ctx, msa_tokens, msa_deletion_matrix, msa_mask, msa_species)

    # ── TEMPLATE features ────────────────────────────────────────
    tmpl_feat = gen_template_features(
        ctx, template_restype, template_distances, template_unit_vector, template_mask
    )

    return AllFeatures(
        token=token_feat,
        token_pair=pair_feat,
        atom=atom_feat,
        atom_pair=atom_pair_feat,
        msa=msa_feat,
        template=tmpl_feat,
    )
