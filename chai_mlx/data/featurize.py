"""Frontend featurization adapters.

``featurize()`` accepts precomputed tensors (the fast path for callers who
already have encoded features).

``featurize_fasta()`` delegates to **chai-lab's** featurization pipeline for
correctness, then converts the batch dict into a ``FeatureContext`` that the
MLX model consumes.  This avoids reimplementing the 30+ feature generators
and their encoding quirks.

When ``featurize_fasta()`` is used, raw per-feature tensors are stored as
``FeatureContext.raw_features`` so that ``FeatureEmbedding`` can encode,
concatenate, and project each feature group independently — avoiding the
multi-GB materialisation of a single wide encoded tensor that the
pre-computed path requires.
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

    Requires ``chai-lab`` and ``torch`` to be installed.  Returns a
    ``FeatureContext`` with ``raw_features`` populated — the heavy-duty
    encoding + projection is deferred to ``FeatureEmbedding.__call__``.
    """
    import tempfile

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

    Rather than encoding and concatenating into wide dense tensors on the
    CPU, we simply convert each raw feature tensor to MLX and store them
    in ``raw_features``.  The ``FeatureEmbedding`` will encode + project
    one group at a time, limiting peak memory to the largest single group
    rather than the sum of all groups.

    The wide ``*_features`` fields are set to zero-sized placeholders since
    they are unused when ``raw_features`` is present.
    """
    import torch

    import mlx.core as mx

    from chai_lab.chai1 import feature_generators

    features = batch["features"]
    inputs = batch["inputs"]

    def _mx(t: torch.Tensor) -> mx.array:
        return mx.array(t.detach().cpu().numpy())

    raw: dict[str, mx.array] = {}
    for name in feature_generators:
        raw[name] = _mx(features[name])

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

    B = token_exists.shape[0]

    structure = StructureInputs(
        atom_exists_mask=_mx(atom_exists.float()),
        token_exists_mask=_mx(token_exists.float()),
        token_pair_mask=_mx(token_pair_mask),
        atom_token_index=_mx(inputs["atom_token_index"].long()),
        atom_within_token_index=_mx(inputs["atom_within_token_index"].long()),
        token_reference_atom_index=_mx(inputs["token_ref_atom_index"].long()),
        token_centre_atom_index=_mx(inputs["token_centre_atom_index"].long()),
        token_asym_id=_mx(inputs["token_asym_id"].long()),
        token_entity_id=_mx(inputs["token_entity_id"].long()),
        token_chain_id=_mx(inputs["token_asym_id"].long()),
        token_is_polymer=_mx(is_polymer),
        atom_ref_positions=_mx(inputs["atom_ref_pos"].float()),
        atom_ref_space_uid=_mx(inputs["atom_ref_space_uid"].long()),
        atom_q_indices=_mx(inputs["block_atom_pair_q_idces"].unsqueeze(0).expand(B, -1, -1)
                           if inputs["block_atom_pair_q_idces"].dim() == 2
                           else inputs["block_atom_pair_q_idces"]),
        atom_kv_indices=_mx(inputs["block_atom_pair_kv_idces"].unsqueeze(0).expand(B, -1, -1)
                            if inputs["block_atom_pair_kv_idces"].dim() == 2
                            else inputs["block_atom_pair_kv_idces"]),
        block_atom_pair_mask=_mx(inputs["block_atom_pair_mask"].float()),
        msa_mask=_mx(inputs["msa_mask"]),
        template_input_masks=_mx(
            torch.einsum(
                "btn,btm->btnm",
                inputs["template_mask"].float(),
                inputs["template_mask"].float(),
            )
        ),
    )

    N = token_exists.shape[1]
    empty = mx.zeros((B, 0))

    return FeatureContext(
        token_features=empty,
        token_pair_features=empty,
        atom_features=empty,
        atom_pair_features=empty,
        msa_features=empty,
        template_features=empty,
        structure_inputs=structure,
        bond_adjacency=_mx(bond_features),
        raw_features=raw,
    )
