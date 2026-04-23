from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import mlx.core as mx
except Exception:  # pragma: no cover - syntax-only environments
    mx = Any  # type: ignore[assignment]


Array = Any


@dataclass
class StructureInputs:
    atom_exists_mask: Array
    token_exists_mask: Array
    token_pair_mask: Array
    atom_token_index: Array
    atom_within_token_index: Array
    token_reference_atom_index: Array
    token_asym_id: Array
    token_entity_id: Array
    token_chain_id: Array
    token_is_polymer: Array
    token_centre_atom_index: Array | None = None
    atom_ref_positions: Array | None = None
    atom_ref_space_uid: Array | None = None
    atom_coords: Array | None = None
    bond_adjacency: Array | None = None
    atom_q_indices: Array | None = None
    atom_kv_indices: Array | None = None
    block_atom_pair_mask: Array | None = None
    reference_coords: Array | None = None
    msa_mask: Array | None = None
    template_input_masks: Array | None = None
    token_residue_index: Array | None = None
    token_entity_type: Array | None = None
    token_backbone_frame_mask: Array | None = None
    token_backbone_frame_index: Array | None = None


RawFeatures = dict[str, Array]
"""Per-feature-name raw tensors (indices/floats) before encoding.

When present on a ``FeatureContext``, ``FeatureEmbedding`` uses the
memory-efficient path that encodes, concatenates, and projects each
feature group without materialising the full wide encoded tensor
across all groups simultaneously.
"""


@dataclass
class FeatureContext:
    token_features: Array
    token_pair_features: Array
    atom_features: Array
    atom_pair_features: Array
    msa_features: Array
    template_features: Array
    structure_inputs: StructureInputs
    bond_adjacency: Array | None = None
    raw_features: RawFeatures | None = None


@dataclass
class InputBundle:
    """User-facing frontend bundle.

    This is intentionally permissive: in practice callers often already have
    precomputed features or partially tokenized structure inputs.
    """

    features: FeatureContext | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingOutputs:
    token_single_input: Array
    token_pair_input: Array
    token_pair_structure_input: Array
    atom_single_input: Array
    atom_single_structure_input: Array
    atom_pair_input: Array
    atom_pair_structure_input: Array
    msa_input: Array
    template_input: Array
    single_initial: Array
    single_structure: Array
    pair_initial: Array
    pair_structure: Array
    structure_inputs: StructureInputs


@dataclass
class TrunkOutputs:
    single_initial: Array
    single_trunk: Array
    single_structure: Array
    pair_initial: Array
    pair_trunk: Array
    pair_structure: Array
    atom_single_structure_input: Array
    atom_pair_structure_input: Array
    msa_input: Array
    template_input: Array
    structure_inputs: StructureInputs


@dataclass
class DiffusionCache:
    s_static: Array
    z_cond: Array
    pair_biases: tuple[Array, ...]
    blocked_pair_base: Array
    atom_cond: Array
    atom_single_cond: Array
    trunk_outputs: TrunkOutputs
    structure_inputs: StructureInputs


@dataclass
class ConfidenceOutputs:
    pae_logits: Array
    pde_logits: Array
    plddt_logits: Array
    token_single: Array | None = None
    token_pair: Array | None = None
    structure_inputs: StructureInputs | None = None


@dataclass
class RankingOutputs:
    """Ranking outputs matching chai-lab's ``SampleRanking``.

    Scalar scores (``ptm``, ``iptm``, ``aggregate_score``,
    ``has_inter_chain_clashes``, ``total_clashes``,
    ``total_inter_chain_clashes``) are computed from the full PAE logit
    distribution via per-bin TM weighting (not from ``E[PAE]``), matching
    :func:`chai_lab.ranking.rank.rank` exactly up to MLX numerical noise.

    Fields:

    - ``plddt``, ``pae``, ``pde``: expectation-based summaries decoded
      directly from the logits.
    - ``ptm``, ``iptm``: complex pTM / interface pTM (chai-lab semantics).
    - ``per_chain_ptm``: pTM for each chain in the complex, shape ``[..., c]``.
    - ``per_chain_pair_iptm``: per-chain-pair ipTM, shape ``[..., c, c]``.
    - ``per_atom_plddt``, ``per_chain_plddt``, ``complex_plddt``: plddt
      scores (complex, per-chain, per-atom) computed from logits.
    - ``chain_chain_clashes``: integer ``[..., c, c]`` matrix of inter-chain
      clashes (off-diagonal); diagonal entries are intra-chain clashes.
    - ``total_clashes``, ``total_inter_chain_clashes``: summed counts.
    - ``has_inter_chain_clashes``: boolean flag using the reference's
      ``max_clashes=100`` / ``max_clash_ratio=0.5`` / polymer-only policy.
    - ``asym_ids``: sorted unique asym ids used to index the per-chain arrays.
    - ``aggregate_score``: ``0.2*ptm + 0.8*iptm - 100*has_inter_chain_clashes``.
    """

    plddt: Array
    pae: Array
    pde: Array
    ptm: Array
    iptm: Array
    has_inter_chain_clashes: Array
    aggregate_score: Array
    per_chain_ptm: Array | None = None
    per_chain_pair_iptm: Array | None = None
    complex_plddt: Array | None = None
    per_chain_plddt: Array | None = None
    per_atom_plddt: Array | None = None
    chain_chain_clashes: Array | None = None
    total_clashes: Array | None = None
    total_inter_chain_clashes: Array | None = None
    asym_ids: Array | None = None
