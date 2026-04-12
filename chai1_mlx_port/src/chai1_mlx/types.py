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
    plddt: Array
    pae: Array
    pde: Array
    ptm: Array
    iptm: Array
    has_inter_chain_clashes: Array
    aggregate_score: Array
