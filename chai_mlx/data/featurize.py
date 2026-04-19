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

The ``esm_backend`` knob on :func:`featurize_fasta` selects how ESM-2
embeddings are obtained:

* ``"off"`` (default): no ESM embeddings; the feature is zero-filled.
  Matches every existing harness call site.
* ``"chai"``: pass ``use_esm_embeddings=True`` through to chai-lab, which
  loads its traced CUDA fp16 checkpoint.  Only sensible on a machine that
  has torch + CUDA.
* ``"mlx"``: pre-compute embeddings with ``esm-mlx`` on Apple silicon
  and inject them into the chai-lab featurization pipeline.  Requires
  the ``[esm]`` extra.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

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
    use_esm_embeddings: bool | None = None,
    esm_backend: Literal["off", "chai", "mlx", "mlx_cache"] = "off",
    esm_cache_dir: str | Path | None = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    use_templates_server: bool = False,
    templates_path: Path | None = None,
) -> FeatureContext:
    """Full FASTA-to-FeatureContext entry point using chai-lab's pipeline.

    Requires ``chai-lab`` and ``torch`` to be installed.  Returns a
    ``FeatureContext`` with ``raw_features`` populated — the heavy-duty
    encoding + projection is deferred to ``FeatureEmbedding.__call__``.

    Parameters
    ----------
    esm_backend:
        * ``"off"`` (default) — zero-fill ESM embeddings. All existing
          harness scripts use this.
        * ``"chai"`` — delegate to chai-lab's traced CUDA fp16 checkpoint.
          Only sensible on a CUDA host.
        * ``"mlx"`` — compute embeddings with ``esm-mlx`` in-process.
          Loads the full 3B model; not advisable alongside chai-mlx
          inference on a 16 GB Mac.
        * ``"mlx_cache"`` — load embeddings from the directory pointed to
          by ``esm_cache_dir`` (as produced by
          :mod:`scripts.precompute_esm_mlx`). Zero extra RAM cost at
          inference time.

    esm_cache_dir:
        Directory of pre-computed ``<sha1>.npy`` files. Required when
        ``esm_backend="mlx_cache"``; ignored otherwise.

    use_esm_embeddings:
        **Deprecated.** Retained for backward compatibility with callers
        that pass this boolean.  ``True`` is equivalent to
        ``esm_backend="chai"``; ``False`` is equivalent to the default
        ``esm_backend="off"``.  Passing both raises ``ValueError``.
    """
    import tempfile

    # Patch chai-lab's RDKit timeout on macOS before its modules are
    # imported inside ``make_all_atom_feature_context``.
    from chai_mlx.data._rdkit_timeout_patch import apply_rdkit_timeout_patch

    apply_rdkit_timeout_patch()

    fasta_file = Path(fasta_file)
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="chai_mlx_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    if use_esm_embeddings is not None:
        if esm_backend != "off":
            raise ValueError(
                "Pass only one of use_esm_embeddings= or esm_backend= "
                "(use_esm_embeddings is deprecated)"
            )
        esm_backend = "chai" if use_esm_embeddings else "off"

    if esm_backend not in ("off", "chai", "mlx", "mlx_cache"):
        raise ValueError(
            f"esm_backend must be 'off', 'chai', 'mlx', or 'mlx_cache'; got {esm_backend!r}"
        )
    if esm_backend == "mlx_cache" and esm_cache_dir is None:
        raise ValueError("esm_backend='mlx_cache' requires esm_cache_dir=")

    from chai_lab.chai1 import (
        Collate,
        TokenBondRestraint,
        feature_factory,
        make_all_atom_feature_context,
    )

    feature_context = make_all_atom_feature_context(
        fasta_file,
        output_dir=output_dir,
        # Use the FASTA entity names as chain IDs so constraint CSVs can
        # reference chains by the same label the user wrote in the FASTA
        # header.  Without this, chai-lab auto-assigns A/B/C... and
        # constraint lookups silently return zero matches.
        entity_name_as_subchain=True,
        use_esm_embeddings=(esm_backend == "chai"),
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_directory=msa_directory,
        constraint_path=constraint_path,
        use_templates_server=use_templates_server,
        templates_path=templates_path,
    )

    if esm_backend == "mlx":
        from chai_mlx.data.esm_mlx_adapter import build_embedding_context

        feature_context.embedding_context = build_embedding_context(
            feature_context.chains
        )
    elif esm_backend == "mlx_cache":
        from chai_mlx.data.esm_mlx_adapter import build_embedding_context_from_cache

        feature_context.embedding_context = build_embedding_context_from_cache(
            feature_context.chains, cache_dir=esm_cache_dir
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
        token_residue_index=_mx(inputs["token_residue_index"].long()),
        token_entity_type=_mx(inputs["token_entity_type"].long()),
        token_backbone_frame_mask=_mx(inputs["token_backbone_frame_mask"]),
        token_backbone_frame_index=_mx(inputs["token_backbone_frame_index"].long()),
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
