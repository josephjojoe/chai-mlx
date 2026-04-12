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


def featurize_fasta(
    fasta_file: str | Path,
    *,
    msa_directory: Path | None = None,
    templates_path: Path | None = None,
    esm_embeddings: Path | None = None,
    constraint_path: Path | None = None,
) -> FeatureContext:
    """Full FASTA-to-FeatureContext entry point.

    This is the standalone featurization pipeline that makes ``chai-mlx``
    self-contained.  It parses a FASTA file, tokenizes chains, loads optional
    precomputed MSA / template / ESM data, runs all feature generators, and
    returns a ready-to-use ``FeatureContext``.

    Parameters
    ----------
    fasta_file
        Path to a multi-record FASTA file with ``entity|...|name=...`` headers.
    msa_directory
        Optional directory containing per-chain ``.a3m`` files (from ColabFold
        or any MSA tool).  File names should match chain index (``0.a3m``,
        ``1.a3m``, ...) or chain name.
    templates_path
        Optional path to a precomputed ``.npz`` template file.
    esm_embeddings
        Optional path to precomputed ESM embeddings (``.npy`` / ``.npz`` /
        ``.safetensors``).
    constraint_path
        Reserved for future restraint support (currently unused).
    """
    from .data.collate import collate
    from .data.embeddings import EmbeddingContext
    from .data.msa import MSAContext
    from .data.parsing import load_chains_from_fasta
    from .data.templates import TemplateContext

    fasta_file = Path(fasta_file)
    chain_inputs, structure = load_chains_from_fasta(fasta_file)
    n_tokens = structure.num_tokens

    # Load MSA if available
    msa_ctx: MSAContext | None = None
    if msa_directory is not None:
        msa_dir = Path(msa_directory)
        a3m_files = sorted(msa_dir.glob("*.a3m"))
        if a3m_files:
            msa_ctx = MSAContext.load_from_file(a3m_files[0], n_tokens)

    # Load templates if available
    tmpl_ctx: TemplateContext | None = None
    if templates_path is not None:
        tmpl_ctx = TemplateContext.load_from_file(templates_path, n_tokens)

    # Load ESM embeddings if available
    emb_ctx: EmbeddingContext | None = None
    if esm_embeddings is not None:
        emb_ctx = EmbeddingContext.load_from_file(esm_embeddings, n_tokens)

    feature_ctx, _ = collate(
        structure,
        msa_ctx=msa_ctx,
        template_ctx=tmpl_ctx,
        embedding_ctx=emb_ctx,
    )

    return feature_ctx
