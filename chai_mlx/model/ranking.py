"""Sample ranking in MLX.

Faithful port of :mod:`chai_lab.ranking.rank` (+ ``ranking/{ptm,clashes,
plddt,frames,utils}``).

The reference formulas are:

* ``expectation(logits, weights) = Σ_b softmax(logits)_b · weights_b``
* ``tm_d0(n_tokens) = 1.24·(max(n_tokens, 19) - 15)^(1/3) - 1.8``
* ``bin_weights_b = 1 / (1 + (bin_centers_b / d0)^2)``
* Per-pair expected TM = ``expectation(pae_logits_ij, bin_weights)``
* pTM = ``max_i Σ_j qk_weights_ij · expected_pair_tm_ij`` over valid pairs
  (pairs where the *query* token has a valid backbone/single-atom frame
  and both tokens exist / satisfy the key mask), normalised by the number
  of key tokens with ``clamp_min(1)``.

The frame-validity mask is the union of ``token_backbone_frame_mask`` and
the single-atom "nearest two same-residue atoms" frames from
``chai_lab/ranking/frames.py``.

Clashes follow ``chai_lab/ranking/clashes.py``: we 0-index ``asym_id``,
build an ``[n_chains, a]`` one-hot membership matrix, aggregate pairwise
clash indicators into a dense ``[n_chains, n_chains]`` matrix (with the
same symmetric-halving convention), and apply the ``max_clashes=100`` /
``max_clash_ratio=0.5`` polymer-only "has_inter_chain_clashes" check.

pLDDT follows ``chai_lab/ranking/plddt.py``: bin centers over ``[0, 1]``,
complex score = masked mean over atoms, per-chain score via gathered
chain masks, per-atom = per-bin expectation.

All ops are vectorised so a single call can handle ``[S, N, N, bins]``
inputs without Python-side loops over samples.  The only Python loops are
over unique chains (bounded by the number of chains, typically O(1..10)).

Output aggregate score matches chai-lab:
``0.2·ptm + 0.8·iptm - 100·has_inter_chain_clashes``.
"""

from __future__ import annotations

import math

import mlx.core as mx

from chai_mlx.config import ChaiConfig
from chai_mlx.data.types import ConfidenceOutputs, RankingOutputs, StructureInputs
from chai_mlx.utils import (
    expectation_from_logits,
    gather_tokens_to_atoms,
    masked_mean,
    pairwise_distance,
    stable_softmax,
)


# ---------------------------------------------------------------------------
# Low-level helpers (faithful ports of chai_lab.ranking.utils)
# ---------------------------------------------------------------------------


def _bin_centers(min_bin: float, max_bin: float, no_bins: int) -> mx.array:
    """Centers of ``no_bins`` equal-width bins across ``[min_bin, max_bin]``.

    Mirrors ``chai1._bin_centers``: ``linspace(min, max, 2·no_bins + 1)[1::2]``.
    """
    full = mx.linspace(float(min_bin), float(max_bin), 2 * int(no_bins) + 1)
    return full[1::2]


def _expectation(logits: mx.array, weights: mx.array) -> mx.array:
    """``Σ_b softmax(logits)_b · weights_b`` — drops the trailing bin dim."""
    probs = stable_softmax(logits, axis=-1).astype(weights.dtype)
    return mx.sum(probs * weights, axis=-1)


def _unique_sorted_asyms(
    asym_id: mx.array, mask: mx.array | None
) -> list[int]:
    """Sorted unique integer asym ids, restricted to tokens where mask is true.

    Equivalent to ``torch.unique(asym_id[mask])`` but returned as a Python list
    of ints so the chain loops below can index MLX tensors by scalar chain id.
    """
    if mask is not None:
        mask_bool = mask.astype(mx.bool_)
        vals = mx.where(mask_bool, asym_id, mx.array(-1, dtype=asym_id.dtype))
        items = set(int(v) for v in vals.reshape(-1).tolist() if int(v) != -1)
    else:
        items = set(int(v) for v in asym_id.reshape(-1).tolist())
    return sorted(items)


def _chain_masks(
    asym_id: mx.array,
    token_exists: mx.array,
    asyms: list[int],
) -> mx.array:
    """Return ``[..., c, n]`` bool mask: ``chain_i`` AND ``token_exists``.

    Matches ``get_chain_masks_and_asyms`` from chai_lab/ranking/utils.py.
    """
    if not asyms:
        shape = (*asym_id.shape[:-1], 0, asym_id.shape[-1])
        return mx.zeros(shape, dtype=mx.bool_)
    stacked = mx.stack(
        [asym_id == mx.array(c, dtype=asym_id.dtype) for c in asyms],
        axis=-2,
    )
    return stacked & token_exists.astype(mx.bool_)[..., None, :]


# ---------------------------------------------------------------------------
# Valid frames (port of chai_lab.ranking.frames)
# ---------------------------------------------------------------------------


def _abc_is_colinear(a: mx.array, b: mx.array, c: mx.array) -> mx.array:
    """Return True where triplet (a,b,c) is too colinear (<25° or >155°)."""
    eps = mx.array(1e-8, dtype=a.dtype)
    w1 = a - b
    w1_norm = mx.maximum(mx.sqrt(mx.sum(w1 * w1, axis=-1, keepdims=True)), eps)
    w1 = w1 / w1_norm
    w2 = c - b
    w2_norm = mx.maximum(mx.sqrt(mx.sum(w2 * w2, axis=-1, keepdims=True)), eps)
    w2 = w2 / w2_norm
    cos_sim = mx.clip(mx.sum(w1 * w2, axis=-1), -1.0, 1.0)
    angle = mx.arccos(cos_sim)
    low = math.radians(25.0)
    high = math.radians(155.0)
    nan_or_bad = mx.isnan(angle) | (angle < low) | (angle > high)
    return nan_or_bad


def _gather_per_batch(values: mx.array, indices: mx.array) -> mx.array:
    """``values[batch, indices]`` with broadcasting on leading axes."""
    batch = mx.arange(values.shape[0]).reshape((-1,) + (1,) * (indices.ndim - 1))
    return values[batch, indices]


def _single_atom_frames_mask(
    coords: mx.array,
    structure: StructureInputs,
) -> mx.array:
    """Per-token validity of single-atom (topk-2 nearest) frames.

    Returns a ``[B, N]`` bool mask identifying tokens for which a coherent
    single-atom frame (a_i, b_i, c_i) can be defined.  The rules follow
    ``chai_lab.ranking.frames.get_single_atom_frames``:

    1. Each token's "centre atom" is ``b_i``; we find the two nearest
       centre atoms in the same residue/chain to serve as ``a_i`` / ``c_i``.
    2. The triplet must be non-colinear, defined (centre masks true),
       within the same residue/chain, and the token must be single-atom
       (i.e. no duplicates in ``atom_token_index``), and must not already
       have a backbone frame.

    This only reproduces the *mask* (not the abc indices), which is what
    ``rank.rank()`` consumes via ``all_frames_mask``.
    """
    token_exists = structure.token_exists_mask.astype(mx.bool_)
    atom_exists = structure.atom_exists_mask.astype(mx.bool_)
    centre_idx = structure.token_centre_atom_index
    asym_id = structure.token_asym_id
    res_idx = structure.token_residue_index
    bb_mask = structure.token_backbone_frame_mask.astype(mx.bool_)
    atom_token_index = structure.atom_token_index

    if (
        centre_idx is None
        or res_idx is None
        or bb_mask is None
    ):
        raise ValueError(
            "single_atom_frames requires token_centre_atom_index, "
            "token_residue_index and token_backbone_frame_mask on StructureInputs"
        )

    B, N = token_exists.shape
    # Centre coords + validity (coords gathered along the atoms axis).
    centre_coords = _gather_per_batch(coords, centre_idx)
    centre_ok = _gather_per_batch(atom_exists, centre_idx) & token_exists

    # Squared pairwise distances, infinity-filled where pair is invalid /
    # cross-residue / cross-chain / self-diagonal.
    diffs = centre_coords[:, :, None, :] - centre_coords[:, None, :, :]
    dists_sq = mx.sum(diffs * diffs, axis=-1)
    asym_match = asym_id[:, :, None] == asym_id[:, None, :]
    res_match = res_idx[:, :, None] == res_idx[:, None, :]
    both_ok = centre_ok[:, :, None] & centre_ok[:, None, :]
    eye = mx.eye(N, dtype=mx.bool_)[None, :, :]
    keep = both_ok & asym_match & res_match & ~eye
    inf_val = mx.array(float("inf"), dtype=dists_sq.dtype)
    dists_sq = mx.where(keep, dists_sq, inf_val)

    # Indices of the two smallest distances per row.  ``argsort`` on the
    # last axis and take the first two entries (argtopk/smallest-k).
    order = mx.argsort(dists_sq, axis=-1)
    a = order[..., 0].astype(mx.int32)
    c = order[..., 1].astype(mx.int32)
    b_tok = mx.broadcast_to(mx.arange(N, dtype=mx.int32)[None, :], (B, N))

    # If the row was entirely invalid, the min distance is still +inf;
    # the a/c we pulled out are meaningless.  Track this so we can veto.
    batch = mx.arange(B)[:, None]
    dists_a = dists_sq[batch, b_tok, a]
    dists_c = dists_sq[batch, b_tok, c]
    row_has_two = (dists_a < inf_val) & (dists_c < inf_val)

    # Same-residue / same-chain checks (redundant given the inf mask above
    # but cheap and mirrors the reference's explicit checks).
    a_res = _gather_per_batch(res_idx, a)
    b_res = _gather_per_batch(res_idx, b_tok)
    c_res = _gather_per_batch(res_idx, c)
    same_residue = (a_res == b_res) & (b_res == c_res)
    a_asym = _gather_per_batch(asym_id, a)
    b_asym = _gather_per_batch(asym_id, b_tok)
    c_asym = _gather_per_batch(asym_id, c)
    same_chain = (a_asym == b_asym) & (b_asym == c_asym)

    # Collinearity check on the corresponding centre positions.
    a_pos = _gather_per_batch(centre_coords, a)
    b_pos = _gather_per_batch(centre_coords, b_tok)
    c_pos = _gather_per_batch(centre_coords, c)
    colinear = _abc_is_colinear(a_pos, b_pos, c_pos)

    # All three centre atoms must exist.
    abc_coords_mask = (
        _gather_per_batch(centre_ok, a)
        & _gather_per_batch(centre_ok, b_tok)
        & _gather_per_batch(centre_ok, c)
    )

    # Single-atom-token check: the token index must occur exactly once in
    # ``atom_token_index`` (i.e. the residue has been tokenised as a single
    # atom).  ``torch.unique(..., return_counts=True)`` is not available in
    # MLX, so we compute ``segment_count`` with a one-hot indicator.
    atok = atom_token_index
    atok_mask = atom_exists
    # ``counts[b, n] = Σ_a 1[atok[b,a] == n] · atok_mask[b,a]`` via one-hot.
    atok_onehot = mx.eye(N, dtype=mx.int32)[atok]
    if atok_mask is not None:
        atok_onehot = atok_onehot * atok_mask.astype(mx.int32)[..., None]
    counts = mx.sum(atok_onehot, axis=-2)  # [B, N]
    is_single_atom = counts == mx.array(1, dtype=counts.dtype)

    mask = (
        is_single_atom
        & ~bb_mask
        & same_residue
        & same_chain
        & ~colinear
        & abc_coords_mask
        & token_exists
        & row_has_two
    )
    return mask


def _all_frames_mask(
    coords: mx.array,
    structure: StructureInputs,
) -> mx.array:
    """Union of ``token_backbone_frame_mask`` and single-atom frames.

    Matches ``get_frames_and_mask(...)[1]`` in chai_lab/ranking/frames.py.
    """
    single = _single_atom_frames_mask(coords, structure)
    bb = structure.token_backbone_frame_mask.astype(mx.bool_)
    return single | bb


# ---------------------------------------------------------------------------
# pTM / ipTM (port of chai_lab.ranking.ptm)
# ---------------------------------------------------------------------------


def _tm_d0(n_key_tokens: mx.array) -> mx.array:
    n = mx.maximum(n_key_tokens, mx.array(19.0, dtype=n_key_tokens.dtype))
    return 1.24 * (n - 15.0) ** (1.0 / 3.0) - 1.8


def _compute_ptm(
    logits: mx.array,
    query_res_mask: mx.array,
    query_has_frame_mask: mx.array,
    key_res_mask: mx.array,
    bin_centers: mx.array,
) -> mx.array:
    """Reference-faithful pTM score.

    Shapes::

        logits:              [..., N, N, bins]
        query_res_mask:      [..., N]
        query_has_frame_mask:[..., N]
        key_res_mask:        [..., N]
        bin_centers:         [bins]

    Returns ``[...]``.
    """
    key_bool = key_res_mask.astype(mx.bool_)
    num_key = mx.sum(key_bool.astype(logits.dtype), axis=-1)  # [...]
    d0 = _tm_d0(num_key)[..., None]  # [..., 1]
    bin_weights = 1.0 / (1.0 + (bin_centers.astype(d0.dtype) / d0) ** 2)  # [..., bins]
    bin_weights = bin_weights[..., None, None, :]  # [..., 1, 1, bins]

    expected_pair_tm = _expectation(logits, bin_weights)  # [..., N, N]

    valid_pairs = (
        query_has_frame_mask.astype(mx.bool_)[..., :, None]
        & query_res_mask.astype(mx.bool_)[..., :, None]
        & key_bool[..., None, :]
    )
    num_key_denom = mx.maximum(num_key, mx.array(1.0, dtype=num_key.dtype))[..., None, None]
    qk_weights = valid_pairs.astype(expected_pair_tm.dtype) / num_key_denom
    per_query = mx.sum(qk_weights * expected_pair_tm, axis=-1)  # [..., N]
    return mx.max(per_query, axis=-1)  # [...]


def _complex_ptm(
    pae_logits: mx.array,
    token_exists: mx.array,
    valid_frames: mx.array,
    bin_centers: mx.array,
) -> mx.array:
    return _compute_ptm(
        logits=pae_logits,
        query_res_mask=token_exists,
        query_has_frame_mask=valid_frames,
        key_res_mask=token_exists,
        bin_centers=bin_centers,
    )


def _interface_ptm(
    pae_logits: mx.array,
    token_exists: mx.array,
    valid_frames: mx.array,
    bin_centers: mx.array,
    asym_id: mx.array,
    asyms: list[int],
) -> tuple[mx.array, mx.array]:
    """Return (iptm scalar per batch, per_chain_ptm ``[..., c]``).

    ``iptm`` is the max over chains of "pTM restricted to rows in chain c,
    cols outside chain c".  We unroll the outer chain loop in Python since
    the number of chains is small; the inner pTM computation is fully
    vectorised across the sample/batch axes.
    """
    if not asyms:
        b_shape = pae_logits.shape[:-3]
        return (
            mx.zeros(b_shape, dtype=mx.float32),
            mx.zeros((*b_shape, 0), dtype=mx.float32),
        )
    chain_masks = _chain_masks(asym_id, token_exists, asyms)  # [..., c, n]
    token_exists_bool = token_exists.astype(mx.bool_)
    per_chain_ptm_list: list[mx.array] = []
    for ci in range(len(asyms)):
        cm = chain_masks[..., ci, :]
        key = (~cm) & token_exists_bool
        per_chain_ptm_list.append(
            _compute_ptm(
                logits=pae_logits,
                query_res_mask=cm,
                query_has_frame_mask=valid_frames,
                key_res_mask=key,
                bin_centers=bin_centers,
            )
        )
    per_chain_ptm = mx.stack(per_chain_ptm_list, axis=-1)  # [..., c]
    iptm = mx.max(per_chain_ptm, axis=-1)
    return iptm, per_chain_ptm


def _per_chain_ptm(
    pae_logits: mx.array,
    token_exists: mx.array,
    valid_frames: mx.array,
    bin_centers: mx.array,
    asym_id: mx.array,
    asyms: list[int],
) -> mx.array:
    """Per-chain complex pTM (chain vs itself).  Shape ``[..., c]``."""
    if not asyms:
        return mx.zeros((*pae_logits.shape[:-3], 0), dtype=mx.float32)
    chain_masks = _chain_masks(asym_id, token_exists, asyms)
    out: list[mx.array] = []
    for ci in range(len(asyms)):
        cm = chain_masks[..., ci, :]
        out.append(
            _compute_ptm(
                logits=pae_logits,
                query_res_mask=cm,
                query_has_frame_mask=valid_frames,
                key_res_mask=cm,
                bin_centers=bin_centers,
            )
        )
    return mx.stack(out, axis=-1)


def _per_chain_pair_iptm(
    pae_logits: mx.array,
    token_exists: mx.array,
    valid_frames: mx.array,
    bin_centers: mx.array,
    asym_id: mx.array,
    asyms: list[int],
) -> mx.array:
    """``[..., c, c]`` matrix of (query_chain, key_chain) pTM scores."""
    c = len(asyms)
    if c == 0:
        return mx.zeros((*pae_logits.shape[:-3], 0, 0), dtype=mx.float32)
    chain_masks = _chain_masks(asym_id, token_exists, asyms)
    rows: list[mx.array] = []
    for qi in range(c):
        q_mask = chain_masks[..., qi, :]
        cols: list[mx.array] = []
        for kj in range(c):
            k_mask = chain_masks[..., kj, :]
            cols.append(
                _compute_ptm(
                    logits=pae_logits,
                    query_res_mask=q_mask,
                    query_has_frame_mask=valid_frames,
                    key_res_mask=k_mask,
                    bin_centers=bin_centers,
                )
            )
        rows.append(mx.stack(cols, axis=-1))
    return mx.stack(rows, axis=-2)


# ---------------------------------------------------------------------------
# Clashes (port of chai_lab.ranking.clashes)
# ---------------------------------------------------------------------------


def _clash_scores(
    coords: mx.array,
    atom_mask: mx.array,
    atom_asym_id: mx.array,
    atom_entity_type: mx.array,
    polymer_types: tuple[int, ...],
    *,
    clash_threshold: float = 1.1,
    max_clashes: int = 100,
    max_clash_ratio: float = 0.5,
) -> dict[str, mx.array]:
    """Dense per-chain-pair clash matrix and scalar flags.

    ``coords``     ``[B, A, 3]``
    ``atom_mask``  ``[B, A]`` bool
    ``atom_asym_id`` ``[B, A]`` integer 1..C (shifted to 0..C-1 internally)
    ``atom_entity_type`` ``[B, A]``

    Returns dict with ``total_clashes``, ``total_inter_chain_clashes``,
    ``chain_chain_clashes`` (``[B, C, C]``), and ``has_inter_chain_clashes``.
    """
    atom_mask_bool = atom_mask.astype(mx.bool_)
    B, A = atom_mask_bool.shape
    asym0 = atom_asym_id.astype(mx.int32) - mx.array(1, dtype=mx.int32)
    n_chains = int(mx.max(asym0).item()) + 1 if int(mx.max(asym0).item()) >= 0 else 0
    if n_chains == 0:
        return {
            "total_clashes": mx.zeros((B,), dtype=mx.int32),
            "total_inter_chain_clashes": mx.zeros((B,), dtype=mx.int32),
            "chain_chain_clashes": mx.zeros((B, 0, 0), dtype=mx.int32),
            "has_inter_chain_clashes": mx.zeros((B,), dtype=mx.bool_),
        }

    dists = pairwise_distance(coords)  # [B, A, A]
    valid = atom_mask_bool[:, :, None] & atom_mask_bool[:, None, :]
    not_self = ~mx.eye(A, dtype=mx.bool_)[None, :, :]
    clash_ij_bool = valid & not_self & (dists < clash_threshold)
    # Use float32 accumulators because MLX einsum/matmul require floats;
    # round to int32 after the aggregation since counts are exact integers.
    clash_ij = clash_ij_bool.astype(mx.float32)

    # Membership matrix ``M[b, c, a] = (asym0[b, a] == c) & atom_mask[b, a]``.
    member_bool = mx.stack(
        [(asym0 == c) & atom_mask_bool for c in range(n_chains)],
        axis=-2,
    )  # [B, C, A] bool
    member = member_bool.astype(mx.float32)

    # Aggregate clashes: ``M @ clash_ij @ M^T`` counts pairs.
    atom_to_chain = mx.einsum("bca,baj->bcj", member, clash_ij)  # [B, C, A]
    chain_chain = mx.einsum("bcj,bdj->bcd", atom_to_chain, member)  # [B, C, C]
    chain_chain = mx.round(chain_chain).astype(mx.int32)

    total_clashes = mx.sum(chain_chain, axis=(-1, -2)) // mx.array(2, dtype=chain_chain.dtype)
    diag = mx.eye(n_chains, dtype=mx.int32)[None, :, :]
    off_diag = mx.array(1, dtype=mx.int32) - diag
    chain_chain_adj = chain_chain // (mx.array(1, dtype=chain_chain.dtype) + diag)
    inter_chain_chain = chain_chain_adj * off_diag
    total_inter_chain = mx.sum(inter_chain_chain, axis=(-1, -2)) // mx.array(2, dtype=chain_chain.dtype)

    atoms_per_chain = mx.sum(member_bool.astype(mx.int32), axis=-1)  # [B, C]
    per_row_denom = mx.maximum(atoms_per_chain[:, :, None], mx.array(1, dtype=atoms_per_chain.dtype))
    per_col_denom = mx.maximum(atoms_per_chain[:, None, :], mx.array(1, dtype=atoms_per_chain.dtype))

    has_many = inter_chain_chain >= mx.array(max_clashes, dtype=inter_chain_chain.dtype)
    ratio_row = inter_chain_chain.astype(mx.float32) / per_row_denom.astype(mx.float32)
    ratio_col = inter_chain_chain.astype(mx.float32) / per_col_denom.astype(mx.float32)
    has_ratio = (ratio_row >= max_clash_ratio) | (ratio_col >= max_clash_ratio)
    has_clashes = has_many | has_ratio

    # Only count polymer pair chains.  A chain is "polymer" if any atom in
    # that chain is flagged as a polymer entity type.
    polymer_vec = mx.zeros(atom_entity_type.shape, dtype=mx.bool_)
    for v in polymer_types:
        polymer_vec = polymer_vec | (atom_entity_type == mx.array(v, dtype=atom_entity_type.dtype))
    chain_polymer_any = mx.stack(
        [mx.any(polymer_vec & (asym0 == c) & atom_mask_bool, axis=-1) for c in range(n_chains)],
        axis=-1,
    )  # [B, C]
    polymer_pair = chain_polymer_any[:, :, None] & chain_polymer_any[:, None, :]
    has_inter_chain = mx.any(
        (has_clashes & polymer_pair).reshape(B, -1), axis=-1
    )

    return {
        "total_clashes": total_clashes.astype(mx.int32),
        "total_inter_chain_clashes": total_inter_chain.astype(mx.int32),
        "chain_chain_clashes": chain_chain_adj.astype(mx.int32),
        "has_inter_chain_clashes": has_inter_chain,
    }


# ---------------------------------------------------------------------------
# pLDDT (port of chai_lab.ranking.plddt)
# ---------------------------------------------------------------------------


def _plddt_scores(
    plddt_logits: mx.array,
    atom_mask: mx.array,
    atom_asym_id: mx.array,
    asyms: list[int],
) -> tuple[mx.array, mx.array, mx.array]:
    """Return ``(complex_plddt, per_chain_plddt, per_atom_plddt)``.

    Bin centers cover ``[0, 1]`` with ``plddt_logits.shape[-1]`` equal-width
    bins (matching ``_bin_centers(0, 1, n_bins)``).
    """
    n_bins = int(plddt_logits.shape[-1])
    centers = _bin_centers(0.0, 1.0, n_bins)
    per_atom = _expectation(plddt_logits, centers)  # [..., A]
    mask_bool = atom_mask.astype(mx.bool_)
    complex_plddt = masked_mean(per_atom, mask_bool, axis=-1)
    per_chain_list: list[mx.array] = []
    for c in asyms:
        cmask = (atom_asym_id == mx.array(c, dtype=atom_asym_id.dtype)) & mask_bool
        per_chain_list.append(masked_mean(per_atom, cmask, axis=-1))
    per_chain = mx.stack(per_chain_list, axis=-1) if per_chain_list else mx.zeros(
        (*per_atom.shape[:-1], 0), dtype=per_atom.dtype
    )
    return complex_plddt, per_chain, per_atom


# ---------------------------------------------------------------------------
# Ranker entrypoint
# ---------------------------------------------------------------------------


# chai-lab treats PROTEIN / RNA / DNA / POLYMER_HYBRID as polymers for the
# has_inter_chain_clashes check.  Keep the enum values in sync with
# ``chai_lab.data.parsing.structure.entity_type.EntityType``.
_POLYMER_ENTITY_TYPES: tuple[int, ...] = (0, 1, 2, 4)


def _ranking_for_sample(
    pae_logits: mx.array,
    pde_logits: mx.array,
    plddt_logits: mx.array,
    coords: mx.array,
    structure: StructureInputs,
    pae_bin_centers: mx.array,
) -> RankingOutputs:
    token_exists = structure.token_exists_mask.astype(mx.bool_)
    atom_mask = structure.atom_exists_mask.astype(mx.bool_)
    asym_id = structure.token_asym_id
    atom_token_index = structure.atom_token_index

    # Broadcast asym / entity_type / frame_mask to the per-atom axis for
    # the clash + plddt chain breakdowns.
    atom_asym = gather_tokens_to_atoms(asym_id[:, :, None], atom_token_index)[..., 0]
    if structure.token_entity_type is not None:
        atom_entity_type = gather_tokens_to_atoms(
            structure.token_entity_type[:, :, None], atom_token_index
        )[..., 0]
    else:
        atom_entity_type = mx.zeros(atom_mask.shape, dtype=mx.int32)

    valid_frames = _all_frames_mask(coords, structure)

    ptm = _complex_ptm(pae_logits, token_exists, valid_frames, pae_bin_centers)
    asyms = _unique_sorted_asyms(asym_id, token_exists)
    iptm, per_chain_ptm_interface = _interface_ptm(
        pae_logits, token_exists, valid_frames, pae_bin_centers, asym_id, asyms
    )
    per_chain_ptm = _per_chain_ptm(
        pae_logits, token_exists, valid_frames, pae_bin_centers, asym_id, asyms
    )
    per_chain_pair_iptm = _per_chain_pair_iptm(
        pae_logits, token_exists, valid_frames, pae_bin_centers, asym_id, asyms
    )

    clash = _clash_scores(
        coords=coords,
        atom_mask=atom_mask,
        atom_asym_id=atom_asym,
        atom_entity_type=atom_entity_type,
        polymer_types=_POLYMER_ENTITY_TYPES,
    )
    has_clashes = clash["has_inter_chain_clashes"]

    complex_plddt, per_chain_plddt, per_atom_plddt = _plddt_scores(
        plddt_logits, atom_mask, atom_asym, asyms
    )

    aggregate = (
        0.2 * ptm.astype(mx.float32)
        + 0.8 * iptm.astype(mx.float32)
        - 100.0 * has_clashes.astype(mx.float32)
    )

    # Expectation-based summaries decoded directly from the logits. They
    # are returned alongside the ranking outputs, but the aggregate score
    # itself is driven by pTM, ipTM, and clashes.
    pae_expectation = expectation_from_logits(pae_logits, max_value=32.0)
    pde_expectation = expectation_from_logits(pde_logits, max_value=32.0)
    plddt_expectation = expectation_from_logits(plddt_logits, max_value=1.0)

    asym_ids_arr = mx.array(asyms, dtype=mx.int32) if asyms else mx.zeros((0,), dtype=mx.int32)

    return RankingOutputs(
        plddt=plddt_expectation,
        pae=pae_expectation,
        pde=pde_expectation,
        ptm=ptm,
        iptm=iptm,
        has_inter_chain_clashes=has_clashes.astype(mx.float32),
        aggregate_score=aggregate,
        per_chain_ptm=per_chain_ptm,
        per_chain_pair_iptm=per_chain_pair_iptm,
        complex_plddt=complex_plddt,
        per_chain_plddt=per_chain_plddt,
        per_atom_plddt=per_atom_plddt,
        chain_chain_clashes=clash["chain_chain_clashes"],
        total_clashes=clash["total_clashes"],
        total_inter_chain_clashes=clash["total_inter_chain_clashes"],
        asym_ids=asym_ids_arr,
    )


def _stack_rankings(outputs: list[RankingOutputs]) -> RankingOutputs:
    """Stack per-sample ``RankingOutputs`` along a new axis=1."""
    def _stack_opt(attr: str) -> mx.array | None:
        vals = [getattr(o, attr) for o in outputs]
        if any(v is None for v in vals):
            return None
        return mx.stack(vals, axis=1)

    return RankingOutputs(
        plddt=mx.stack([o.plddt for o in outputs], axis=1),
        pae=mx.stack([o.pae for o in outputs], axis=1),
        pde=mx.stack([o.pde for o in outputs], axis=1),
        ptm=mx.stack([o.ptm for o in outputs], axis=1),
        iptm=mx.stack([o.iptm for o in outputs], axis=1),
        has_inter_chain_clashes=mx.stack(
            [o.has_inter_chain_clashes for o in outputs], axis=1
        ),
        aggregate_score=mx.stack([o.aggregate_score for o in outputs], axis=1),
        per_chain_ptm=_stack_opt("per_chain_ptm"),
        per_chain_pair_iptm=_stack_opt("per_chain_pair_iptm"),
        complex_plddt=_stack_opt("complex_plddt"),
        per_chain_plddt=_stack_opt("per_chain_plddt"),
        per_atom_plddt=_stack_opt("per_atom_plddt"),
        chain_chain_clashes=_stack_opt("chain_chain_clashes"),
        total_clashes=_stack_opt("total_clashes"),
        total_inter_chain_clashes=_stack_opt("total_inter_chain_clashes"),
        asym_ids=outputs[0].asym_ids,
    )


class Ranker:
    """Sample ranker mirroring :func:`chai_lab.ranking.rank.rank`.

    Inputs are the logits from the confidence head plus the
    :class:`StructureInputs` that carry ``token_asym_id``,
    ``token_entity_type``, ``token_backbone_frame_mask``,
    ``token_centre_atom_index``, ``token_residue_index``, and the
    atom-level masks / indices.

    ``__call__`` handles both ``coords.ndim == 3`` (``[B, A, 3]``) and
    ``coords.ndim == 4`` (``[B, S, A, 3]``) by ranking each sample
    independently and stacking along axis 1.  Confidence logits are
    expected to have a matching sample axis when ``ndim == 4``.
    """

    def __init__(self, cfg: ChaiConfig) -> None:
        self.cfg = cfg
        self._pae_bin_centers = _bin_centers(0.0, 32.0, int(cfg.confidence.pair_bins))

    def _rank_single(
        self,
        conf: ConfidenceOutputs,
        coords: mx.array,
        structure: StructureInputs,
    ) -> RankingOutputs:
        return _ranking_for_sample(
            pae_logits=conf.pae_logits,
            pde_logits=conf.pde_logits,
            plddt_logits=conf.plddt_logits,
            coords=coords,
            structure=structure,
            pae_bin_centers=self._pae_bin_centers,
        )

    def __call__(
        self,
        conf: ConfidenceOutputs,
        coords: mx.array,
        structure: StructureInputs,
    ) -> RankingOutputs:
        if coords.ndim == 3:
            return self._rank_single(conf, coords, structure)
        outputs = [
            self._rank_single(
                ConfidenceOutputs(
                    pae_logits=conf.pae_logits[:, i],
                    pde_logits=conf.pde_logits[:, i],
                    plddt_logits=conf.plddt_logits[:, i],
                ),
                coords[:, i],
                structure,
            )
            for i in range(coords.shape[1])
        ]
        return _stack_rankings(outputs)
