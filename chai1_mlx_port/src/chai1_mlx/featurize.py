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
# Batch-dict → FeatureContext conversion
# ---------------------------------------------------------------------------

def _batch_to_feature_context(
    batch: dict[str, Any],
    bond_features: Any,
) -> FeatureContext:
    """Convert a chai-lab batch dict into a ``FeatureContext``.

    The batch is produced by ``Collate(feature_factory, ...)(contexts)`` and
    contains:

    - ``batch["inputs"]``: padded raw tensors (masks, indices, coords, …)
    - ``batch["features"]``: per-generator feature tensors (raw, pre-encoding)

    The TorchScript ``feature_embedding.pt`` module encodes and concatenates
    these internally.  We replicate that encoding here so the MLX model's
    ``nn.Linear``-based ``FeatureEmbedding`` receives the same input.

    **Important**: the per-feature encoding below must match the TorchScript
    model's internal encoding.  Verify via parity tests against the reference
    ``feature_embedding.pt`` output.
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

    def _encode_one_hot(feat: torch.Tensor, num_classes: int) -> torch.Tensor:
        idx = feat.squeeze(-1).long().clamp(0, num_classes - 1)
        return F.one_hot(idx, num_classes).float()

    def _encode_outersum(feat: torch.Tensor, num_classes: int) -> torch.Tensor:
        idx = feat.squeeze(-1).long().clamp(0, num_classes - 1)
        oh = F.one_hot(idx, num_classes).float()
        return oh[..., :, None, :] + oh[..., None, :, :]

    def _encode_identity(feat: torch.Tensor) -> torch.Tensor:
        return feat.float()

    def _encode_feature(
        gen: Any, feat: torch.Tensor,
    ) -> torch.Tensor:
        if gen.encoding_ty == EncodingType.ONE_HOT:
            return _encode_one_hot(feat, gen.num_classes)
        if gen.encoding_ty == EncodingType.OUTERSUM:
            return _encode_outersum(feat, gen.num_classes)
        return _encode_identity(feat)

    # -- group generators by FeatureType --------------------------------

    groups: dict[FeatureType, list[tuple[str, Any]]] = {}
    for name, gen in feature_generators.items():
        groups.setdefault(gen.ty, []).append((name, gen))

    def _concat_for_type(
        ft: FeatureType, target_dim: int,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        for name, gen in groups.get(ft, []):
            encoded = _encode_feature(gen, features[name])
            parts.append(encoded)
        if not parts:
            raise RuntimeError(f"No features found for {ft}")
        cat = torch.cat(parts, dim=-1)
        deficit = target_dim - cat.shape[-1]
        if deficit > 0:
            pad_shape = cat.shape[:-1] + (deficit,)
            cat = torch.cat([cat, torch.zeros(pad_shape, dtype=cat.dtype)], dim=-1)
        elif deficit < 0:
            raise RuntimeError(
                f"{ft.value} feature dim {cat.shape[-1]} exceeds target {target_dim}"
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
        return mx.array(t.cpu().numpy())

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
    )
