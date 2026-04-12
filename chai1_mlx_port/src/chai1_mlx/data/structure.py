"""AllAtomStructureContext — pure-numpy port of the reference dataclass.

Fields mirror ``chai_lab.data.dataset.structure.all_atom_structure_context``
but use numpy arrays instead of PyTorch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
from typing import ClassVar, Sequence

import numpy as np


class EntityType(IntEnum):
    PROTEIN = 0
    RNA = 1
    DNA = 2
    LIGAND = 3
    POLYMER_HYBRID = 4
    WATER = 5
    UNKNOWN = 6
    MANUAL_GLYCAN = 7


@dataclass
class AllAtomStructureContext:
    # ── token-level ──────────────────────────────────────────────
    token_residue_type: np.ndarray       # [n_tokens]       int
    token_residue_index: np.ndarray      # [n_tokens]       int
    token_index: np.ndarray              # [n_tokens]       int
    token_centre_atom_index: np.ndarray  # [n_tokens]       int
    token_ref_atom_index: np.ndarray     # [n_tokens]       int
    token_exists_mask: np.ndarray        # [n_tokens]       bool
    token_backbone_frame_mask: np.ndarray  # [n_tokens]     bool
    token_backbone_frame_index: np.ndarray # [n_tokens, 3]  int
    token_asym_id: np.ndarray            # [n_tokens]       int
    token_entity_id: np.ndarray          # [n_tokens]       int
    token_sym_id: np.ndarray             # [n_tokens]       int
    token_entity_type: np.ndarray        # [n_tokens]       int
    token_residue_name: np.ndarray       # [n_tokens, 8]    uint8
    token_b_factor_or_plddt: np.ndarray  # [n_tokens]       float32

    # ── atom-level ───────────────────────────────────────────────
    atom_token_index: np.ndarray         # [n_atoms]        int
    atom_within_token_index: np.ndarray  # [n_atoms]        int
    atom_ref_pos: np.ndarray             # [n_atoms, 3]     float32
    atom_ref_mask: np.ndarray            # [n_atoms]        bool
    atom_ref_element: np.ndarray         # [n_atoms]        int
    atom_ref_charge: np.ndarray          # [n_atoms]        int
    atom_ref_name: list[str]
    atom_ref_name_chars: np.ndarray      # [n_atoms, 4]     int
    atom_ref_space_uid: np.ndarray       # [n_atoms]        int
    atom_is_not_padding_mask: np.ndarray # [n_atoms]        bool

    # ── supervision / GT ─────────────────────────────────────────
    atom_gt_coords: np.ndarray           # [n_atoms, 3]     float32
    atom_exists_mask: np.ndarray         # [n_atoms]        bool

    # ── structure-level metadata ─────────────────────────────────
    pdb_id: np.ndarray                   # [n_tokens, 32]   uint8
    source_pdb_chain_id: np.ndarray      # [n_tokens, 4]    uint8
    subchain_id: np.ndarray              # [n_tokens, 4]    uint8
    resolution: np.ndarray               # [1]              float32
    is_distillation: np.ndarray          # [1]              bool

    # ── symmetries & bonds ───────────────────────────────────────
    symmetries: np.ndarray               # [n_atoms, n_sym] int
    bond_left: np.ndarray                # [n_bonds]        int
    bond_right: np.ndarray               # [n_bonds]        int

    @property
    def num_tokens(self) -> int:
        return len(self.token_residue_type)

    @property
    def num_atoms(self) -> int:
        return len(self.atom_token_index)

    # ── Pad ──────────────────────────────────────────────────────

    def pad(self, n_tokens: int, n_atoms: int) -> "AllAtomStructureContext":
        """Return a new context padded to exactly *n_tokens* / *n_atoms*."""
        nt = self.num_tokens
        na = self.num_atoms
        if nt > n_tokens or na > n_atoms:
            raise ValueError(
                f"Cannot pad {nt} tokens / {na} atoms to "
                f"{n_tokens} / {n_atoms}"
            )
        dt = n_tokens - nt
        da = n_atoms - na

        def _p1(arr: np.ndarray, pad_n: int, val=0) -> np.ndarray:
            if pad_n == 0:
                return arr
            pad_width = [(0, pad_n)] + [(0, 0)] * (arr.ndim - 1)
            return np.pad(arr, pad_width, constant_values=val)

        return AllAtomStructureContext(
            token_residue_type=_p1(self.token_residue_type, dt),
            token_residue_index=_p1(self.token_residue_index, dt),
            token_index=_p1(self.token_index, dt),
            token_centre_atom_index=_p1(self.token_centre_atom_index, dt),
            token_ref_atom_index=_p1(self.token_ref_atom_index, dt),
            token_exists_mask=_p1(self.token_exists_mask, dt),
            token_backbone_frame_mask=_p1(self.token_backbone_frame_mask, dt),
            token_backbone_frame_index=_p1(self.token_backbone_frame_index, dt),
            token_asym_id=_p1(self.token_asym_id, dt),
            token_entity_id=_p1(self.token_entity_id, dt),
            token_sym_id=_p1(self.token_sym_id, dt),
            token_entity_type=_p1(self.token_entity_type, dt),
            token_residue_name=_p1(self.token_residue_name, dt),
            token_b_factor_or_plddt=_p1(self.token_b_factor_or_plddt, dt),
            atom_token_index=_p1(self.atom_token_index, da),
            atom_within_token_index=_p1(self.atom_within_token_index, da),
            atom_ref_pos=_p1(self.atom_ref_pos, da),
            atom_ref_mask=_p1(self.atom_ref_mask, da),
            atom_ref_element=_p1(self.atom_ref_element, da),
            atom_ref_charge=_p1(self.atom_ref_charge, da),
            atom_ref_name=self.atom_ref_name + [""] * da,
            atom_ref_name_chars=_p1(self.atom_ref_name_chars, da),
            atom_ref_space_uid=_p1(self.atom_ref_space_uid, da, val=-1),
            atom_is_not_padding_mask=_p1(self.atom_is_not_padding_mask, da),
            atom_gt_coords=_p1(self.atom_gt_coords, da),
            atom_exists_mask=_p1(self.atom_exists_mask, da),
            pdb_id=_p1(self.pdb_id, dt),
            source_pdb_chain_id=_p1(self.source_pdb_chain_id, dt),
            subchain_id=_p1(self.subchain_id, dt),
            resolution=self.resolution,
            is_distillation=self.is_distillation,
            symmetries=_p1(self.symmetries, da, val=-1),
            bond_left=self.bond_left,
            bond_right=self.bond_right,
        )

    # ── Merge ────────────────────────────────────────────────────

    @classmethod
    def merge(cls, contexts: Sequence["AllAtomStructureContext"]) -> "AllAtomStructureContext":
        """Concatenate multiple contexts, re-indexing atoms and tokens."""
        if len(contexts) == 1:
            return contexts[0]

        tok_offsets: list[int] = []
        atom_offsets: list[int] = []
        cum_tok, cum_atom = 0, 0
        for c in contexts:
            tok_offsets.append(cum_tok)
            atom_offsets.append(cum_atom)
            cum_tok += c.num_tokens
            cum_atom += c.num_atoms

        def _cat(arrays: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(arrays, axis=0) if arrays else np.array([], dtype=arrays[0].dtype)

        token_index = np.arange(cum_tok, dtype=np.int64)
        atom_token_index = _cat([
            c.atom_token_index + tok_offsets[i] for i, c in enumerate(contexts)
        ])
        token_centre_atom_index = _cat([
            c.token_centre_atom_index + atom_offsets[i] for i, c in enumerate(contexts)
        ])
        token_ref_atom_index = _cat([
            c.token_ref_atom_index + atom_offsets[i] for i, c in enumerate(contexts)
        ])
        token_backbone_frame_index = _cat([
            c.token_backbone_frame_index + atom_offsets[i] for i, c in enumerate(contexts)
        ])

        # Reindex atom_ref_space_uid per context
        space_uid_parts = []
        space_offset = 0
        for c in contexts:
            uid = c.atom_ref_space_uid.copy()
            valid = uid >= 0
            if valid.any():
                uid[valid] += space_offset
                space_offset = uid[valid].max() + 1
            space_uid_parts.append(uid)

        # Symmetries — pad to max width, concat
        max_sym = max(c.symmetries.shape[1] for c in contexts) if contexts else 0
        sym_parts = []
        for c in contexts:
            s = c.symmetries
            if s.shape[1] < max_sym:
                pad = np.full((s.shape[0], max_sym - s.shape[1]), -1, dtype=s.dtype)
                s = np.concatenate([s, pad], axis=1)
            sym_parts.append(s)

        # Bonds
        bond_l = _cat([c.bond_left + atom_offsets[i] for i, c in enumerate(contexts)])
        bond_r = _cat([c.bond_right + atom_offsets[i] for i, c in enumerate(contexts)])

        resolution = np.maximum.reduce([c.resolution for c in contexts])
        is_distill = np.logical_or.reduce([c.is_distillation for c in contexts])

        return cls(
            token_residue_type=_cat([c.token_residue_type for c in contexts]),
            token_residue_index=_cat([c.token_residue_index for c in contexts]),
            token_index=token_index,
            token_centre_atom_index=token_centre_atom_index,
            token_ref_atom_index=token_ref_atom_index,
            token_exists_mask=_cat([c.token_exists_mask for c in contexts]),
            token_backbone_frame_mask=_cat([c.token_backbone_frame_mask for c in contexts]),
            token_backbone_frame_index=token_backbone_frame_index,
            token_asym_id=_cat([c.token_asym_id for c in contexts]),
            token_entity_id=_cat([c.token_entity_id for c in contexts]),
            token_sym_id=_cat([c.token_sym_id for c in contexts]),
            token_entity_type=_cat([c.token_entity_type for c in contexts]),
            token_residue_name=_cat([c.token_residue_name for c in contexts]),
            token_b_factor_or_plddt=_cat([c.token_b_factor_or_plddt for c in contexts]),
            atom_token_index=atom_token_index,
            atom_within_token_index=_cat([c.atom_within_token_index for c in contexts]),
            atom_ref_pos=_cat([c.atom_ref_pos for c in contexts]),
            atom_ref_mask=_cat([c.atom_ref_mask for c in contexts]),
            atom_ref_element=_cat([c.atom_ref_element for c in contexts]),
            atom_ref_charge=_cat([c.atom_ref_charge for c in contexts]),
            atom_ref_name=sum((c.atom_ref_name for c in contexts), []),
            atom_ref_name_chars=_cat([c.atom_ref_name_chars for c in contexts]),
            atom_ref_space_uid=_cat(space_uid_parts),
            atom_is_not_padding_mask=_cat([c.atom_is_not_padding_mask for c in contexts]),
            atom_gt_coords=_cat([c.atom_gt_coords for c in contexts]),
            atom_exists_mask=_cat([c.atom_exists_mask for c in contexts]),
            pdb_id=_cat([c.pdb_id for c in contexts]),
            source_pdb_chain_id=_cat([c.source_pdb_chain_id for c in contexts]),
            subchain_id=_cat([c.subchain_id for c in contexts]),
            resolution=resolution,
            is_distillation=is_distill,
            symmetries=_cat(sym_parts),
            bond_left=bond_l,
            bond_right=bond_r,
        )

    # ── Empty context factory ────────────────────────────────────

    @classmethod
    def empty(cls) -> "AllAtomStructureContext":
        z1 = np.zeros(0, dtype=np.int64)
        z1f = np.zeros(0, dtype=np.float32)
        z1b = np.zeros(0, dtype=bool)
        z23 = np.zeros((0, 3), dtype=np.float32)
        z2i3 = np.zeros((0, 3), dtype=np.int64)
        z2u8 = np.zeros((0, 8), dtype=np.uint8)
        z2u4 = np.zeros((0, 4), dtype=np.uint8)
        z2u32 = np.zeros((0, 32), dtype=np.uint8)
        z2i4 = np.zeros((0, 4), dtype=np.int64)
        z2s1 = np.zeros((0, 1), dtype=np.int64)
        return cls(
            token_residue_type=z1, token_residue_index=z1, token_index=z1,
            token_centre_atom_index=z1, token_ref_atom_index=z1,
            token_exists_mask=z1b, token_backbone_frame_mask=z1b,
            token_backbone_frame_index=z2i3,
            token_asym_id=z1, token_entity_id=z1, token_sym_id=z1,
            token_entity_type=z1, token_residue_name=z2u8,
            token_b_factor_or_plddt=z1f,
            atom_token_index=z1, atom_within_token_index=z1,
            atom_ref_pos=z23, atom_ref_mask=z1b, atom_ref_element=z1,
            atom_ref_charge=z1, atom_ref_name=[], atom_ref_name_chars=z2i4,
            atom_ref_space_uid=z1, atom_is_not_padding_mask=z1b,
            atom_gt_coords=z23, atom_exists_mask=z1b,
            pdb_id=z2u32, source_pdb_chain_id=z2u4, subchain_id=z2u4,
            resolution=np.zeros(1, dtype=np.float32),
            is_distillation=np.zeros(1, dtype=bool),
            symmetries=z2s1,
            bond_left=z1, bond_right=z1,
        )
