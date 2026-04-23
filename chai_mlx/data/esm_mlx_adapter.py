"""MLX-side ESM-2 embedding adapter for chai-lab's featurization pipeline.

This module is the bridge that lets ``featurize_fasta(esm_backend="mlx")``
produce Chai-1-compatible ESM-2 embeddings entirely on Apple silicon,
without touching the CUDA-only ``traced_sdpa_esm2_t36_3B_UR50D_fp16.pt``
checkpoint that chai-lab normally depends on.

The mapping to chai-lab's own
``chai_lab.data.dataset.embeddings.esm.get_esm_embedding_context`` is
deliberately one-to-one:

* Tokenization: esm-mlx's ``Tokenizer`` shares the official ESM-2 alphabet
  with chai-lab's ``DumbTokenizer`` (indices are bit-identical), so feeding
  the same ``<cls>{seq}<eos>`` token stream through either implementation
  yields activations in the same order.
* Per-chain dispatch: proteins are embedded once per unique sequence, then
  re-exploded via ``chain.structure_context.token_residue_index``.  Non-
  protein chains are padded with zeros of the same dim (2560).
* Output dtype: chai-lab casts to fp32 on CPU; we mirror that so downstream
  chai-lab code (which does ``embedding_context.pad(...)`` with fp32 zeros)
  is typesafe.

All ``esm_mlx`` imports are lazy: the core ``chai_mlx`` package never
loads MLX ESM weights unless a caller explicitly opts in via
``esm_backend="mlx"``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chai_lab.data.dataset.embeddings.embedding_context import EmbeddingContext
    from chai_lab.data.dataset.structure.chain import Chain


_ESM_MLX_MODEL_NAME = "esm2_t36_3B_UR50D"
_ESM_EMBED_DIM = 2560


def _sha1(sequence: str) -> str:
    return hashlib.sha1(sequence.encode()).hexdigest()[:16]


def build_embedding_context_from_cache(
    chains: "list[Chain]",
    cache_dir: str | Path,
) -> "EmbeddingContext":
    """Build an ``EmbeddingContext`` from a pre-computed ESM-MLX cache.

    Pairs with ``chai-mlx-precompute-esm``: every protein sequence
    is looked up by ``sha1(sequence)[:16]`` and loaded from
    ``<cache-dir>/<sha1>.npy``.  Non-protein chains are zero-filled to
    match chai-lab's convention.

    This function has no dependency on ``esm-mlx`` or ``mlx`` -- it
    exists specifically so we can free the 6 GB ESM-2 3B weights before
    ``ChaiMLX.from_pretrained`` loads the ~1.2 GB chai-1 weights on a
    16 GB Mac.
    """
    import numpy as np
    import torch

    from chai_lab.data.dataset.embeddings.embedding_context import EmbeddingContext
    from chai_lab.data.parsing.structure.entity_type import EntityType

    cache_dir = Path(cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"ESM-MLX cache directory not found: {cache_dir}. "
            "Run chai-mlx-precompute-esm first."
        )

    seq_to_context: dict[str, EmbeddingContext] = {}
    for chain in chains:
        if chain.entity_data.entity_type != EntityType.PROTEIN:
            continue
        seq = chain.entity_data.sequence
        if seq in seq_to_context:
            continue
        sha = _sha1(seq)
        npy_path = cache_dir / f"{sha}.npy"
        if not npy_path.is_file():
            raise FileNotFoundError(
                f"ESM-MLX cache miss: sequence of length {len(seq)} "
                f"(sha {sha}) not found at {npy_path}. "
                "Run chai-mlx-precompute-esm for the FASTA you plan to fold."
            )
        emb_np = np.load(npy_path).astype(np.float32, copy=False)
        if emb_np.shape != (len(seq), _ESM_EMBED_DIM):
            raise ValueError(
                f"Cached embedding shape {emb_np.shape} does not match sequence length {len(seq)}"
            )
        seq_to_context[seq] = EmbeddingContext(
            esm_embeddings=torch.from_numpy(emb_np)
        )

    chain_embs: list[EmbeddingContext] = []
    for chain in chains:
        if chain.entity_data.entity_type == EntityType.PROTEIN:
            chain_embs.append(seq_to_context[chain.entity_data.sequence])
        else:
            chain_embs.append(
                EmbeddingContext.empty(
                    n_tokens=chain.structure_context.num_tokens,
                    d_emb=_ESM_EMBED_DIM,
                )
            )

    exploded = [
        emb.esm_embeddings[chain.structure_context.token_residue_index, :]
        for emb, chain in zip(chain_embs, chains, strict=True)
    ]
    merged = torch.cat(exploded, dim=0)
    return EmbeddingContext(esm_embeddings=merged)


def build_embedding_context(
    chains: "list[Chain]",
    *,
    model_name: str = _ESM_MLX_MODEL_NAME,
) -> "EmbeddingContext":
    """Build a chai-lab ``EmbeddingContext`` from chains using esm-mlx.

    Mirrors ``chai_lab.data.dataset.embeddings.esm.get_esm_embedding_context``
    step-for-step: per-unique-protein-sequence ESM-2 forward, zero
    embeddings for non-polymer chains, then
    ``token_residue_index``-driven explosion back to the per-token layout.

    The returned ``EmbeddingContext`` is ready to be assigned directly to
    ``AllAtomFeatureContext.embedding_context`` before ``Collate`` runs;
    the downstream featurizer does not need to know or care that the
    embeddings originated on MLX rather than CUDA.
    """
    import numpy as np
    import torch
    import mlx.core as mx

    from chai_lab.data.dataset.embeddings.embedding_context import EmbeddingContext
    from chai_lab.data.parsing.structure.entity_type import EntityType

    try:
        from esm_mlx import ESM2, Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "esm_backend='mlx' requires the esm_mlx package. Install with:\n"
            "    pip install 'chai-mlx[esm]'\n"
            "    pip install -e '.[esm]'\n"
            "(or pre-compute embeddings with chai-mlx-precompute-esm "
            "and pass esm_backend='mlx_cache' + esm_cache_dir= instead)."
        ) from exc

    protein_sequences: set[str] = {
        chain.entity_data.sequence
        for chain in chains
        if chain.entity_data.entity_type == EntityType.PROTEIN
    }

    seq_to_context: dict[str, EmbeddingContext] = {}
    if protein_sequences:
        model = ESM2.from_pretrained(model_name)
        tokenizer = Tokenizer()
        for seq in protein_sequences:
            tokens = tokenizer.encode(seq)
            out = model(tokens, repr_layers=[model.num_layers])
            last_hidden = out["representations"][model.num_layers]
            trimmed = last_hidden[0, 1:-1]
            mx.eval(trimmed)
            emb_np = np.asarray(trimmed).astype(np.float32, copy=False)
            assert emb_np.shape == (len(seq), _ESM_EMBED_DIM), (
                f"Unexpected ESM-MLX output shape {emb_np.shape} for sequence "
                f"of length {len(seq)} (expected ({len(seq)}, {_ESM_EMBED_DIM}))"
            )
            seq_to_context[seq] = EmbeddingContext(
                esm_embeddings=torch.from_numpy(emb_np)
            )

    chain_embs: list[EmbeddingContext] = []
    for chain in chains:
        if chain.entity_data.entity_type == EntityType.PROTEIN:
            chain_embs.append(seq_to_context[chain.entity_data.sequence])
        else:
            chain_embs.append(
                EmbeddingContext.empty(
                    n_tokens=chain.structure_context.num_tokens,
                    d_emb=_ESM_EMBED_DIM,
                )
            )

    exploded = [
        emb.esm_embeddings[chain.structure_context.token_residue_index, :]
        for emb, chain in zip(chain_embs, chains, strict=True)
    ]
    merged = torch.cat(exploded, dim=0)
    return EmbeddingContext(esm_embeddings=merged)
