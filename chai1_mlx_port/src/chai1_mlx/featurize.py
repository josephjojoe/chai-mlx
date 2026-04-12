"""Frontend featurization adapters.

``featurize()`` accepts precomputed tensors (the fast path for callers who
already have encoded features).

``featurize_fasta()`` delegates to **chai-lab's** featurization pipeline for
correctness, then converts the batch dict into a ``FeatureContext`` that the
MLX model consumes.  This avoids reimplementing the 30+ feature generators
and their encoding quirks.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .types import FeatureContext, InputBundle, StructureInputs

_REQUIRED_KEYS = {
    "token_features",
    "token_pair_features",
    "atom_features",
    "atom_pair_features",
    "msa_features",
    "template_features",
    "structure_inputs",
}


def _coerce_structure_inputs(obj: Any) -> StructureInputs:
    if isinstance(obj, StructureInputs):
        return obj
    if is_dataclass(obj):
        return StructureInputs(**asdict(obj))
    if isinstance(obj, dict):
        return StructureInputs(**obj)
    raise TypeError("structure_inputs must be a StructureInputs instance or dict")


def featurize(inputs: FeatureContext | InputBundle | dict[str, Any]) -> FeatureContext:
    """Return a frontend-independent FeatureContext.

    Accepts precomputed ``FeatureContext``, ``InputBundle``, or raw dict of
    tensors.  For FASTA-based featurization, use ``featurize_fasta()`` instead.
    """

    if isinstance(inputs, FeatureContext):
        return inputs

    if isinstance(inputs, InputBundle):
        if inputs.features is not None:
            return inputs.features
        inputs = inputs.raw

    if not isinstance(inputs, dict):
        raise TypeError(
            "featurize() expects a FeatureContext, InputBundle, or dict of precomputed tensors"
        )

    missing = sorted(_REQUIRED_KEYS - set(inputs))
    if missing:
        raise ValueError(
            "This MLX port expects precomputed encoded feature tensors. Missing keys: "
            + ", ".join(missing)
        )

    payload = dict(inputs)
    payload["structure_inputs"] = _coerce_structure_inputs(payload["structure_inputs"])
    return FeatureContext(**payload)


# ---------------------------------------------------------------------------
# FASTA-based featurization via chai-lab
# ---------------------------------------------------------------------------

def featurize_fasta(
    fasta_file: str | Path,
    *,
    output_dir: str | Path | None = None,
    msa_directory: Path | None = None,
    constraint_path: Path | None = None,
    use_esm_embeddings: bool = True,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    use_templates_server: bool = False,
    templates_path: Path | None = None,
) -> FeatureContext:
    """Full FASTA-to-FeatureContext entry point using chai-lab's pipeline.

    This calls the upstream chai-lab featurization (parsing, tokenization,
    MSA/template/ESM loading, feature generation, and collation) and then
    converts the resulting batch dict into the ``FeatureContext`` that the
    MLX model consumes.

    Requires ``chai-lab`` and ``torch`` to be installed.
    """
    import tempfile

    import torch

    fasta_file = Path(fasta_file)
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="chai_mlx_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    from chai_lab.chai1 import (
        Collate,
        TokenBondRestraint,
        feature_factory,
        make_all_atom_feature_context,
    )

    feature_context = make_all_atom_feature_context(
        fasta_file,
        output_dir=output_dir,
        use_esm_embeddings=use_esm_embeddings,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_directory=msa_directory,
        constraint_path=constraint_path,
        use_templates_server=use_templates_server,
        templates_path=templates_path,
    )

    collator = Collate(
        feature_factory=feature_factory,
        num_key_atoms=128,
        num_query_atoms=32,
    )
    batch = collator([feature_context])

    bond_ft = TokenBondRestraint().generate(batch=batch).data

    return _batch_to_feature_context(batch, bond_ft)


# ---------------------------------------------------------------------------
# One-hot widths from the TorchScript feature_embedding.pt IR.
# These are the exact second argument to torch.one_hot() in forward_256.
# For can_mask=True features this is num_classes + 1 (except MSADataSource
# whose num_classes already includes the mask class).
# ---------------------------------------------------------------------------
_ONE_HOT_WIDTH: dict[str, int] = {
    "AtomNameOneHot": 65,
    "AtomRefElement": 130,
    "BlockedAtomPairDistogram": 12,
    "DockingConstraintGenerator": 6,
    "IsDistillation": 2,
    "MSADataSource": 6,
    "MSAOneHot": 33,
    "RelativeChain": 6,
    "RelativeEntity": 3,
    "RelativeSequenceSeparation": 67,
    "RelativeTokenSeparation": 67,
    "ResidueType": 33,
    "TemplateDistogram": 39,
    "TokenBFactor": 3,
    "TokenPLDDT": 4,
}

_RBF_FEATURES = frozenset({"TokenDistanceRestraint", "TokenPairPocketRestraint"})
_OUTERSUM_FEATURES = frozenset({"TemplateResType"})


# ---------------------------------------------------------------------------
# Batch-dict → FeatureContext conversion
# ---------------------------------------------------------------------------

def _batch_to_feature_context(
    batch: dict[str, Any],
    bond_features: Any,
) -> FeatureContext:
    """Convert a chai-lab batch dict into a ``FeatureContext``.

    The encoding applied here matches the TorchScript ``feature_embedding.pt``
    internal encoding for all features that do NOT require learned parameters.
    Features with learned parameters (TemplateResType embedding, RBF restraint
    radii) are left as raw data in auxiliary fields and encoded at runtime
    by ``FeatureEmbedding``.
    """
    import torch
    import torch.nn.functional as F

    import mlx.core as mx

    from chai_lab.data.features.feature_type import FeatureType
    from chai_lab.data.features.generators.base import EncodingType
    from chai_lab.chai1 import feature_generators

    features = batch["features"]
    inputs = batch["inputs"]

    # -- helpers --------------------------------------------------------

    def _encode_one_hot(
        name: str, feat: torch.Tensor, gen: Any,
    ) -> torch.Tensor:
        width = _ONE_HOT_WIDTH[name]
        idx = feat.long()
        if idx.shape[-1] == 1:
            idx = idx.squeeze(-1)
        result = F.one_hot(idx.clamp(0, width - 1), width).float()
        if result.ndim > feat.ndim:
            result = result.reshape(*result.shape[: feat.ndim - 1], -1)
        return result

    def _encode_identity(feat: torch.Tensor) -> torch.Tensor:
        out = feat.float()
        if out.ndim >= 2 and out.shape[-1] != 1:
            return out
        if out.ndim < 3:
            while out.ndim < 3:
                out = out.unsqueeze(-1)
        return out

    def _encode_feature(
        name: str, gen: Any, feat: torch.Tensor,
    ) -> torch.Tensor:
        if gen.encoding_ty == EncodingType.ONE_HOT:
            return _encode_one_hot(name, feat, gen)
        if gen.encoding_ty in (EncodingType.IDENTITY, EncodingType.ESM):
            return _encode_identity(feat)
        if gen.encoding_ty == EncodingType.RBF:
            return _encode_identity(feat)
        return _encode_identity(feat)

    # -- group generators by FeatureType, sorted alphabetically ---------
    # The TorchScript concatenates features in alphabetical order within
    # each type group (verified against feature_embedding_forward256.py).

    groups: dict[FeatureType, list[tuple[str, Any]]] = {}
    for name, gen in feature_generators.items():
        groups.setdefault(gen.ty, []).append((name, gen))
    for ft in groups:
        groups[ft].sort(key=lambda x: x[0])

    # -- stash raw data for features encoded by FeatureEmbedding --------

    template_restype_raw = None
    distance_restraint_raw = None
    pocket_restraint_raw = None

    def _concat_for_type(
        ft: FeatureType, target_dim: int,
    ) -> torch.Tensor:
        nonlocal template_restype_raw, distance_restraint_raw, pocket_restraint_raw

        parts: list[torch.Tensor] = []
        for name, gen in groups.get(ft, []):
            if name in _OUTERSUM_FEATURES:
                template_restype_raw = features[name].squeeze(-1).long()
                placeholder = torch.zeros(
                    *features[name].shape[:-1],
                    features[name].shape[-2],
                    32,
                    dtype=torch.float32,
                )
                parts.append(placeholder)
                continue
            if name in _RBF_FEATURES:
                raw = features[name].float()
                if raw.shape[-1] == 1:
                    pass
                if name == "TokenDistanceRestraint":
                    distance_restraint_raw = raw
                else:
                    pocket_restraint_raw = raw
                placeholder = torch.zeros(
                    *raw.shape[:-1], 7, dtype=torch.float32,
                )
                parts.append(placeholder)
                continue
            encoded = _encode_feature(name, gen, features[name])
            parts.append(encoded)
        if not parts:
            raise RuntimeError(f"No features found for {ft}")
        cat = torch.cat(parts, dim=-1)
        if cat.shape[-1] != target_dim:
            raise RuntimeError(
                f"{ft.value} feature dim {cat.shape[-1]} != target {target_dim}"
            )
        return cat

    token_feat = _concat_for_type(FeatureType.TOKEN, 2638)
    pair_feat = _concat_for_type(FeatureType.TOKEN_PAIR, 163)
    atom_feat = _concat_for_type(FeatureType.ATOM, 395)
    atom_pair_feat = _concat_for_type(FeatureType.ATOM_PAIR, 14)
    msa_feat = _concat_for_type(FeatureType.MSA, 42)
    tmpl_feat = _concat_for_type(FeatureType.TEMPLATES, 76)

    # -- convert to MLX -------------------------------------------------

    def _mx(t: torch.Tensor) -> mx.array:
        return mx.array(t.detach().cpu().numpy())

    # -- build StructureInputs ------------------------------------------

    token_exists = inputs["token_exists_mask"]
    atom_exists = inputs["atom_exists_mask"]
    token_pair_mask = torch.einsum("bi,bj->bij", token_exists.float(), token_exists.float())

    from chai_lab.data.parsing.structure.entity_type import EntityType as ChaiEntityType

    polymer_values = {
        ChaiEntityType.PROTEIN.value,
        ChaiEntityType.RNA.value,
        ChaiEntityType.DNA.value,
    }
    entity_type = inputs["token_entity_type"].long()
    is_polymer = torch.zeros_like(entity_type, dtype=torch.float32)
    for v in polymer_values:
        is_polymer[entity_type == v] = 1.0

    structure = StructureInputs(
        atom_exists_mask=_mx(atom_exists.float()),
        token_exists_mask=_mx(token_exists.float()),
        token_pair_mask=_mx(token_pair_mask),
        atom_token_index=_mx(inputs["atom_token_index"].long()),
        atom_within_token_index=_mx(inputs["atom_within_token_index"].long()),
        token_reference_atom_index=_mx(inputs["token_ref_atom_index"].long()),
        token_asym_id=_mx(inputs["token_asym_id"].long()),
        token_entity_id=_mx(inputs["token_entity_id"].long()),
        token_chain_id=_mx(inputs["token_asym_id"].long()),
        token_is_polymer=_mx(is_polymer),
        atom_ref_positions=_mx(inputs["atom_ref_pos"].float()),
        atom_ref_space_uid=_mx(inputs["atom_ref_space_uid"].long()),
        atom_q_indices=_mx(inputs["block_atom_pair_q_idces"]),
        atom_kv_indices=_mx(inputs["block_atom_pair_kv_idces"]),
        block_atom_pair_mask=_mx(inputs["block_atom_pair_mask"].float()),
    )

    return FeatureContext(
        token_features=_mx(token_feat),
        token_pair_features=_mx(pair_feat),
        atom_features=_mx(atom_feat),
        atom_pair_features=_mx(atom_pair_feat),
        msa_features=_mx(msa_feat),
        template_features=_mx(tmpl_feat),
        structure_inputs=structure,
        bond_adjacency=_mx(bond_features),
        template_restype_indices=(
            _mx(template_restype_raw) if template_restype_raw is not None else None
        ),
        distance_restraint_data=(
            _mx(distance_restraint_raw) if distance_restraint_raw is not None else None
        ),
        pocket_restraint_data=(
            _mx(pocket_restraint_raw) if pocket_restraint_raw is not None else None
        ),
    )
