"""Template context — glue for reading precomputed template hits.

Provides ``TemplateContext.load_from_file()`` for ``.m8`` hit files
and ``TemplateContext.empty()`` when no templates are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .residue_constants import NUM_RESTYPES, residue_types_with_nucleotides_order

MAX_NUM_TEMPLATES = 4


@dataclass
class TemplateContext:
    template_restype: np.ndarray    # [n_templates, n_tokens] int
    template_distances: np.ndarray  # [n_templates, n_tokens, n_tokens] float32
    template_unit_vector: np.ndarray  # [n_templates, n_tokens, n_tokens, 3] float32
    template_mask: np.ndarray       # [n_templates, n_tokens, n_tokens] bool

    @property
    def n_templates(self) -> int:
        return self.template_restype.shape[0]

    @property
    def n_tokens(self) -> int:
        return self.template_restype.shape[1]

    @classmethod
    def empty(cls, n_tokens: int) -> "TemplateContext":
        return cls(
            template_restype=np.zeros((0, n_tokens), dtype=np.int64),
            template_distances=np.zeros((0, n_tokens, n_tokens), dtype=np.float32),
            template_unit_vector=np.zeros((0, n_tokens, n_tokens, 3), dtype=np.float32),
            template_mask=np.zeros((0, n_tokens, n_tokens), dtype=bool),
        )

    @classmethod
    def load_from_file(cls, path: str | Path, n_tokens: int) -> "TemplateContext":
        """Load precomputed template data from an ``.npz`` file.

        Expected keys: ``template_restype``, ``template_distances``,
        ``template_unit_vector``, ``template_mask``.

        For ``.m8`` tabular hit files, a dedicated parser would align hits
        to the query sequence. This simplified version loads pre-aligned data.
        """
        path = Path(path)
        if path.suffix == ".npz":
            data = np.load(path)
            return cls(
                template_restype=data["template_restype"][:MAX_NUM_TEMPLATES],
                template_distances=data["template_distances"][:MAX_NUM_TEMPLATES],
                template_unit_vector=data["template_unit_vector"][:MAX_NUM_TEMPLATES],
                template_mask=data["template_mask"][:MAX_NUM_TEMPLATES],
            )
        return cls.empty(n_tokens)

    def pad(self, n_tokens: int, max_templates: int = MAX_NUM_TEMPLATES) -> "TemplateContext":
        """Pad to *n_tokens* and *max_templates*."""
        t = min(self.n_templates, max_templates)
        cur_n = self.n_tokens

        def _pad_t(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
            out = np.zeros(shape, dtype=arr.dtype)
            slices = tuple(slice(0, min(s, a)) for s, a in zip(arr.shape, shape))
            out[slices] = arr[slices]
            return out

        return TemplateContext(
            template_restype=_pad_t(self.template_restype[:t], (max_templates, n_tokens)),
            template_distances=_pad_t(self.template_distances[:t], (max_templates, n_tokens, n_tokens)),
            template_unit_vector=_pad_t(self.template_unit_vector[:t], (max_templates, n_tokens, n_tokens, 3)),
            template_mask=_pad_t(self.template_mask[:t], (max_templates, n_tokens, n_tokens)),
        )
