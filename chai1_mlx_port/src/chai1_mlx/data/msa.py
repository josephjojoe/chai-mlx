"""MSA context — glue for reading precomputed .a3m alignments.

Provides ``MSAContext.load_from_file()`` for ColabFold-style ``.a3m`` files,
and ``MSAContext.empty()`` when no MSA is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .residue_constants import residue_types_with_nucleotides_order

MAX_MSA_DEPTH = 16_384


@dataclass
class MSAContext:
    tokens: np.ndarray          # [depth, n_tokens] int
    deletion_matrix: np.ndarray # [depth, n_tokens] float32
    mask: np.ndarray            # [depth, n_tokens] bool
    species: np.ndarray         # [depth] int  (data source id per row)

    @property
    def depth(self) -> int:
        return self.tokens.shape[0]

    @property
    def n_tokens(self) -> int:
        return self.tokens.shape[1]

    @classmethod
    def empty(cls, n_tokens: int) -> "MSAContext":
        return cls(
            tokens=np.zeros((0, n_tokens), dtype=np.int64),
            deletion_matrix=np.zeros((0, n_tokens), dtype=np.float32),
            mask=np.zeros((0, n_tokens), dtype=bool),
            species=np.zeros(0, dtype=np.int64),
        )

    @classmethod
    def load_from_file(cls, path: str | Path, n_tokens: int) -> "MSAContext":
        """Parse an ``.a3m`` alignment file into an ``MSAContext``.

        The first sequence in the a3m is the query. Lowercase letters are
        insertions (deleted from alignment columns). Each row is mapped to
        the residue vocabulary and aligned to *n_tokens* columns (truncated
        or padded).
        """
        path = Path(path)
        if path.suffix not in (".a3m", ".afa"):
            raise ValueError(f"Expected .a3m file, got {path.suffix}")

        sequences: list[str] = []
        current: list[str] = []
        with open(path) as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if current:
                        sequences.append("".join(current))
                        current = []
                elif line:
                    current.append(line)
        if current:
            sequences.append("".join(current))

        if not sequences:
            return cls.empty(n_tokens)

        depth = min(len(sequences), MAX_MSA_DEPTH)
        tokens = np.zeros((depth, n_tokens), dtype=np.int64)
        deletions = np.zeros((depth, n_tokens), dtype=np.float32)
        mask = np.zeros((depth, n_tokens), dtype=bool)

        gap_idx = residue_types_with_nucleotides_order.get("-", 31)

        for row_i in range(depth):
            seq = sequences[row_i]
            col = 0
            del_count = 0
            for ch in seq:
                if ch.islower():
                    del_count += 1
                    continue
                if col >= n_tokens:
                    break
                if ch == "-":
                    tokens[row_i, col] = gap_idx
                else:
                    tokens[row_i, col] = residue_types_with_nucleotides_order.get(
                        ch.upper(), 20
                    )
                deletions[row_i, col] = del_count
                del_count = 0
                mask[row_i, col] = True
                col += 1

        return cls(
            tokens=tokens,
            deletion_matrix=deletions,
            mask=mask,
            species=np.zeros(depth, dtype=np.int64),
        )

    def pad(self, n_tokens: int, max_depth: int = MAX_MSA_DEPTH) -> "MSAContext":
        """Pad or truncate to *n_tokens* columns and *max_depth* rows."""
        d = min(self.depth, max_depth)
        cur_t = self.n_tokens

        def _pad2d(arr: np.ndarray, target_rows: int, target_cols: int) -> np.ndarray:
            out = np.zeros((target_rows, target_cols), dtype=arr.dtype)
            r = min(arr.shape[0], target_rows)
            c = min(arr.shape[1], target_cols)
            out[:r, :c] = arr[:r, :c]
            return out

        return MSAContext(
            tokens=_pad2d(self.tokens, d, n_tokens),
            deletion_matrix=_pad2d(self.deletion_matrix, d, n_tokens),
            mask=_pad2d(self.mask, d, n_tokens),
            species=self.species[:d],
        )
