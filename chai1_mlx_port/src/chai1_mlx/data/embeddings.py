"""ESM embedding context — glue for precomputed embeddings.

Provides ``EmbeddingContext.load_from_file()`` for ``.npy`` / ``.npz`` /
``.safetensors`` files and ``EmbeddingContext.empty()`` when no embeddings
are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ESM_DIM = 2560


@dataclass
class EmbeddingContext:
    esm_embeddings: np.ndarray  # [n_tokens, esm_dim] float32

    @property
    def n_tokens(self) -> int:
        return self.esm_embeddings.shape[0]

    @property
    def dim(self) -> int:
        return self.esm_embeddings.shape[1]

    @classmethod
    def empty(cls, n_tokens: int, dim: int = ESM_DIM) -> "EmbeddingContext":
        return cls(esm_embeddings=np.zeros((n_tokens, dim), dtype=np.float32))

    @classmethod
    def load_from_file(cls, path: str | Path, n_tokens: int | None = None) -> "EmbeddingContext":
        """Load precomputed ESM embeddings.

        Supported formats:
          - ``.npy``: raw array of shape ``[n_tokens, dim]``
          - ``.npz``: expects key ``esm_embeddings``
          - ``.safetensors``: expects key ``esm_embeddings``
        """
        path = Path(path)
        if path.suffix == ".npy":
            arr = np.load(path)
        elif path.suffix == ".npz":
            data = np.load(path)
            arr = data.get("esm_embeddings", data.get("embeddings", next(iter(data.values()))))
        elif path.suffix == ".safetensors":
            try:
                from safetensors.numpy import load_file
                data = load_file(str(path))
                arr = data.get("esm_embeddings", data.get("embeddings", next(iter(data.values()))))
            except ImportError:
                raise ImportError("safetensors required to load .safetensors embedding files")
        else:
            raise ValueError(f"Unsupported embedding format: {path.suffix}")

        arr = arr.astype(np.float32)
        if n_tokens is not None and arr.shape[0] != n_tokens:
            padded = np.zeros((n_tokens, arr.shape[1]), dtype=np.float32)
            n = min(arr.shape[0], n_tokens)
            padded[:n] = arr[:n]
            arr = padded

        return cls(esm_embeddings=arr)

    def pad(self, n_tokens: int) -> "EmbeddingContext":
        """Pad or truncate to *n_tokens*."""
        cur = self.n_tokens
        if cur == n_tokens:
            return self
        out = np.zeros((n_tokens, self.dim), dtype=np.float32)
        n = min(cur, n_tokens)
        out[:n] = self.esm_embeddings[:n]
        return EmbeddingContext(esm_embeddings=out)
