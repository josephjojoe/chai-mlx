"""FASTA parsing and entity type detection for inference inputs.

Ported from ``chai_lab/data/dataset/inference_dataset.py`` and
``chai_lab/data/parsing/fasta.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .residue_constants import (
    residue_types_with_nucleotides_order,
    restype_1to3_with_x,
    standard_residue_pdb_codes,
)
from .structure import AllAtomStructureContext, EntityType


# ── FASTA parsing ────────────────────────────────────────────────────

@dataclass
class FastaEntry:
    description: str
    sequence: str


def read_fasta(path: str | Path) -> list[FastaEntry]:
    """Read a multi-record FASTA file (no BioPython dependency)."""
    entries: list[FastaEntry] = []
    desc: str | None = None
    seq_lines: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if desc is not None:
                    entries.append(FastaEntry(desc, "".join(seq_lines)))
                desc = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line)
    if desc is not None:
        entries.append(FastaEntry(desc, "".join(seq_lines)))
    return entries


# ── Entity type detection ────────────────────────────────────────────

_ENTITY_RE = re.compile(r"^(\w+)\|")


def _parse_entity_type(description: str) -> EntityType:
    """Extract entity type from FASTA description line."""
    m = _ENTITY_RE.match(description)
    if m is None:
        return EntityType.PROTEIN
    label = m.group(1).lower()
    return {
        "protein": EntityType.PROTEIN,
        "ligand": EntityType.LIGAND,
        "rna": EntityType.RNA,
        "dna": EntityType.DNA,
        "glycan": EntityType.MANUAL_GLYCAN,
    }.get(label, EntityType.PROTEIN)


# ── Input representation ─────────────────────────────────────────────

@dataclass
class ChainInput:
    """A single chain/entity from a FASTA file."""
    sequence: str
    entity_type: EntityType
    entity_name: str = ""


def read_inputs(fasta_path: str | Path) -> list[ChainInput]:
    """Parse a FASTA file into a list of chain inputs.

    The FASTA description may use the ``entity|...|name=...`` convention
    from the reference ``chai-lab`` inference pipeline.
    """
    entries = read_fasta(fasta_path)
    inputs: list[ChainInput] = []
    for e in entries:
        etype = _parse_entity_type(e.description)
        name_match = re.search(r"name=([^|]+)", e.description)
        name = name_match.group(1).strip() if name_match else ""
        inputs.append(ChainInput(sequence=e.sequence, entity_type=etype, entity_name=name))
    return inputs


# ── Tokenization helpers ─────────────────────────────────────────────

def _str_to_uint8(s: str, width: int) -> np.ndarray:
    """Encode *s* into a fixed-width uint8 array (truncate or pad)."""
    arr = np.zeros(width, dtype=np.uint8)
    for i, ch in enumerate(s[:width]):
        arr[i] = ord(ch)
    return arr


def _atom_name_chars(name: str) -> np.ndarray:
    """Encode a 4-char PDB atom name as int array."""
    arr = np.zeros(4, dtype=np.int64)
    for i, ch in enumerate(name[:4]):
        arr[i] = ord(ch) - 32  # space-based offset
    return arr


def tokenize_protein_chain(
    sequence: str,
    *,
    chain_id: int = 0,
    entity_id: int = 0,
    sym_id: int = 0,
) -> AllAtomStructureContext:
    """Tokenize a single protein chain from its amino acid sequence.

    This is a simplified tokenizer that assigns one token per residue with
    standard atom37 layout. It does NOT compute reference conformer geometry
    from RDKit/CCD -- instead it produces zero reference positions (to be
    populated from weights or external data).
    """
    from .residue_constants import atom_order, residue_atoms, restype_1to3_with_x

    n_tokens = len(sequence)
    max_atoms_per_residue = len(atom_order)  # 37

    all_atom_names: list[str] = []
    atom_token_idx: list[int] = []
    atom_within_token_idx: list[int] = []
    atom_ref_element: list[int] = []
    token_centre: list[int] = []
    token_ref: list[int] = []
    token_residue_type: list[int] = []
    backbone_frame_indices: list[list[int]] = []
    backbone_frame_mask: list[bool] = []

    atom_offset = 0
    for tok_i, aa in enumerate(sequence):
        aa_upper = aa.upper()
        resname_3 = restype_1to3_with_x.get(aa_upper, "UNK")
        restype_idx = residue_types_with_nucleotides_order.get(aa_upper, 20)
        token_residue_type.append(restype_idx)

        known_atoms = residue_atoms.get(resname_3, [])
        n_atoms_this = max(len(known_atoms), 1)

        for a_i, aname in enumerate(known_atoms):
            all_atom_names.append(aname)
            atom_token_idx.append(tok_i)
            within_idx = atom_order.get(aname, a_i)
            atom_within_token_idx.append(within_idx)
            elem = _element_from_name(aname)
            atom_ref_element.append(elem)

        ca_idx = atom_offset + (atom_order.get("CA", 1) if "CA" in known_atoms else 0)
        n_idx = atom_offset + (atom_order.get("N", 0) if "N" in known_atoms else 0)
        c_idx = atom_offset + (atom_order.get("C", 2) if "C" in known_atoms else 0)
        token_centre.append(ca_idx)
        token_ref.append(ca_idx)

        has_backbone = all(a in known_atoms for a in ("N", "CA", "C"))
        backbone_frame_mask.append(has_backbone)
        if has_backbone:
            actual_n = atom_offset + [i for i, a in enumerate(known_atoms) if a == "N"][0]
            actual_ca = atom_offset + [i for i, a in enumerate(known_atoms) if a == "CA"][0]
            actual_c = atom_offset + [i for i, a in enumerate(known_atoms) if a == "C"][0]
            backbone_frame_indices.append([actual_n, actual_ca, actual_c])
        else:
            backbone_frame_indices.append([ca_idx, ca_idx, ca_idx])

        atom_offset += len(known_atoms)

    n_atoms = len(all_atom_names)

    pdb_id_arr = np.tile(_str_to_uint8("PRED", 32), (n_tokens, 1))
    chain_str = chr(ord("A") + chain_id % 26)
    chain_id_arr = np.tile(_str_to_uint8(chain_str, 4), (n_tokens, 1))
    subchain_arr = np.tile(_str_to_uint8(chain_str, 4), (n_tokens, 1))

    residue_names = np.zeros((n_tokens, 8), dtype=np.uint8)
    for i, aa in enumerate(sequence):
        rn = restype_1to3_with_x.get(aa.upper(), "UNK")
        residue_names[i] = _str_to_uint8(rn, 8)

    return AllAtomStructureContext(
        token_residue_type=np.array(token_residue_type, dtype=np.int64),
        token_residue_index=np.arange(n_tokens, dtype=np.int64),
        token_index=np.arange(n_tokens, dtype=np.int64),
        token_centre_atom_index=np.array(token_centre, dtype=np.int64),
        token_ref_atom_index=np.array(token_ref, dtype=np.int64),
        token_exists_mask=np.ones(n_tokens, dtype=bool),
        token_backbone_frame_mask=np.array(backbone_frame_mask, dtype=bool),
        token_backbone_frame_index=np.array(backbone_frame_indices, dtype=np.int64),
        token_asym_id=np.full(n_tokens, chain_id, dtype=np.int64),
        token_entity_id=np.full(n_tokens, entity_id, dtype=np.int64),
        token_sym_id=np.full(n_tokens, sym_id, dtype=np.int64),
        token_entity_type=np.full(n_tokens, EntityType.PROTEIN.value, dtype=np.int64),
        token_residue_name=residue_names,
        token_b_factor_or_plddt=np.zeros(n_tokens, dtype=np.float32),
        atom_token_index=np.array(atom_token_idx, dtype=np.int64),
        atom_within_token_index=np.array(atom_within_token_idx, dtype=np.int64),
        atom_ref_pos=np.zeros((n_atoms, 3), dtype=np.float32),
        atom_ref_mask=np.ones(n_atoms, dtype=bool),
        atom_ref_element=np.array(atom_ref_element, dtype=np.int64),
        atom_ref_charge=np.zeros(n_atoms, dtype=np.int64),
        atom_ref_name=all_atom_names,
        atom_ref_name_chars=np.array([_atom_name_chars(n) for n in all_atom_names], dtype=np.int64).reshape(-1, 4) if all_atom_names else np.zeros((0, 4), dtype=np.int64),
        atom_ref_space_uid=np.arange(n_atoms, dtype=np.int64),
        atom_is_not_padding_mask=np.ones(n_atoms, dtype=bool),
        atom_gt_coords=np.zeros((n_atoms, 3), dtype=np.float32),
        atom_exists_mask=np.ones(n_atoms, dtype=bool),
        pdb_id=pdb_id_arr,
        source_pdb_chain_id=chain_id_arr,
        subchain_id=subchain_arr,
        resolution=np.array([0.0], dtype=np.float32),
        is_distillation=np.array([False], dtype=bool),
        symmetries=np.arange(n_atoms, dtype=np.int64).reshape(-1, 1),
        bond_left=np.zeros(0, dtype=np.int64),
        bond_right=np.zeros(0, dtype=np.int64),
    )


def _element_from_name(atom_name: str) -> int:
    """Rough atomic number from PDB atom name."""
    name = atom_name.strip().upper()
    first = name[0] if name else "C"
    return {"C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "H": 1, "F": 9, "B": 5}.get(first, 6)


def tokenize_nucleic_acid_chain(
    sequence: str,
    *,
    entity_type: EntityType,
    chain_id: int = 0,
    entity_id: int = 0,
    sym_id: int = 0,
) -> AllAtomStructureContext:
    """Tokenize an RNA or DNA chain (one token per nucleotide)."""
    from .residue_constants import nucleic_acid_atoms

    prefix = "R" if entity_type == EntityType.RNA else "D"
    n_tokens = len(sequence)
    all_atom_names: list[str] = []
    atom_token_idx: list[int] = []
    atom_within_token_idx: list[int] = []
    atom_ref_element: list[int] = []
    token_centre: list[int] = []
    token_ref: list[int] = []
    token_residue_type: list[int] = []

    atom_offset = 0
    for tok_i, base in enumerate(sequence):
        key = f"{prefix}{base.upper()}" if f"{prefix}{base.upper()}" in nucleic_acid_atoms else f"{prefix}X"
        restype_idx = residue_types_with_nucleotides_order.get(key, 20)
        token_residue_type.append(restype_idx)

        atom_slots = nucleic_acid_atoms[key]
        real_atoms = [(i, a) for i, a in enumerate(atom_slots) if a is not None]

        for within_i, (_, aname) in enumerate(real_atoms):
            all_atom_names.append(aname)
            atom_token_idx.append(tok_i)
            atom_within_token_idx.append(within_i)
            atom_ref_element.append(_element_from_name(aname))

        c1_idx = atom_offset
        for j, (_, aname) in enumerate(real_atoms):
            if aname == "C1'":
                c1_idx = atom_offset + j
                break
        token_centre.append(c1_idx)
        token_ref.append(c1_idx)
        atom_offset += len(real_atoms)

    n_atoms = len(all_atom_names)
    pdb_id_arr = np.tile(_str_to_uint8("PRED", 32), (n_tokens, 1))
    chain_str = chr(ord("A") + chain_id % 26)

    residue_names = np.zeros((n_tokens, 8), dtype=np.uint8)
    for i, base in enumerate(sequence):
        key = f"{prefix}{base.upper()}"
        residue_names[i] = _str_to_uint8(key, 8)

    return AllAtomStructureContext(
        token_residue_type=np.array(token_residue_type, dtype=np.int64),
        token_residue_index=np.arange(n_tokens, dtype=np.int64),
        token_index=np.arange(n_tokens, dtype=np.int64),
        token_centre_atom_index=np.array(token_centre, dtype=np.int64),
        token_ref_atom_index=np.array(token_ref, dtype=np.int64),
        token_exists_mask=np.ones(n_tokens, dtype=bool),
        token_backbone_frame_mask=np.zeros(n_tokens, dtype=bool),
        token_backbone_frame_index=np.zeros((n_tokens, 3), dtype=np.int64),
        token_asym_id=np.full(n_tokens, chain_id, dtype=np.int64),
        token_entity_id=np.full(n_tokens, entity_id, dtype=np.int64),
        token_sym_id=np.full(n_tokens, sym_id, dtype=np.int64),
        token_entity_type=np.full(n_tokens, entity_type.value, dtype=np.int64),
        token_residue_name=residue_names,
        token_b_factor_or_plddt=np.zeros(n_tokens, dtype=np.float32),
        atom_token_index=np.array(atom_token_idx, dtype=np.int64),
        atom_within_token_index=np.array(atom_within_token_idx, dtype=np.int64),
        atom_ref_pos=np.zeros((n_atoms, 3), dtype=np.float32),
        atom_ref_mask=np.ones(n_atoms, dtype=bool),
        atom_ref_element=np.array(atom_ref_element, dtype=np.int64),
        atom_ref_charge=np.zeros(n_atoms, dtype=np.int64),
        atom_ref_name=all_atom_names,
        atom_ref_name_chars=np.array([_atom_name_chars(n) for n in all_atom_names], dtype=np.int64).reshape(-1, 4) if all_atom_names else np.zeros((0, 4), dtype=np.int64),
        atom_ref_space_uid=np.arange(n_atoms, dtype=np.int64),
        atom_is_not_padding_mask=np.ones(n_atoms, dtype=bool),
        atom_gt_coords=np.zeros((n_atoms, 3), dtype=np.float32),
        atom_exists_mask=np.ones(n_atoms, dtype=bool),
        pdb_id=pdb_id_arr,
        source_pdb_chain_id=np.tile(_str_to_uint8(chain_str, 4), (n_tokens, 1)),
        subchain_id=np.tile(_str_to_uint8(chain_str, 4), (n_tokens, 1)),
        resolution=np.array([0.0], dtype=np.float32),
        is_distillation=np.array([False], dtype=bool),
        symmetries=np.arange(n_atoms, dtype=np.int64).reshape(-1, 1),
        bond_left=np.zeros(0, dtype=np.int64),
        bond_right=np.zeros(0, dtype=np.int64),
    )


def tokenize_ligand_smiles(
    smiles: str,
    *,
    chain_id: int = 0,
    entity_id: int = 0,
    sym_id: int = 0,
) -> AllAtomStructureContext:
    """Tokenize a ligand from SMILES (one token per heavy atom).

    This simplified tokenizer does NOT call RDKit for conformer generation.
    It counts heavy atoms from the SMILES string and creates placeholder tokens.
    For full accuracy, use RDKit externally and supply coordinates.
    """
    n_heavy = _count_heavy_atoms_smiles(smiles)
    if n_heavy == 0:
        n_heavy = 1

    n_tokens = n_heavy
    n_atoms = n_heavy

    return AllAtomStructureContext(
        token_residue_type=np.full(n_tokens, residue_types_with_nucleotides_order.get(":", 32), dtype=np.int64),
        token_residue_index=np.zeros(n_tokens, dtype=np.int64),
        token_index=np.arange(n_tokens, dtype=np.int64),
        token_centre_atom_index=np.arange(n_tokens, dtype=np.int64),
        token_ref_atom_index=np.arange(n_tokens, dtype=np.int64),
        token_exists_mask=np.ones(n_tokens, dtype=bool),
        token_backbone_frame_mask=np.zeros(n_tokens, dtype=bool),
        token_backbone_frame_index=np.zeros((n_tokens, 3), dtype=np.int64),
        token_asym_id=np.full(n_tokens, chain_id, dtype=np.int64),
        token_entity_id=np.full(n_tokens, entity_id, dtype=np.int64),
        token_sym_id=np.full(n_tokens, sym_id, dtype=np.int64),
        token_entity_type=np.full(n_tokens, EntityType.LIGAND.value, dtype=np.int64),
        token_residue_name=np.tile(_str_to_uint8("LIG", 8), (n_tokens, 1)),
        token_b_factor_or_plddt=np.zeros(n_tokens, dtype=np.float32),
        atom_token_index=np.arange(n_atoms, dtype=np.int64),
        atom_within_token_index=np.zeros(n_atoms, dtype=np.int64),
        atom_ref_pos=np.zeros((n_atoms, 3), dtype=np.float32),
        atom_ref_mask=np.ones(n_atoms, dtype=bool),
        atom_ref_element=np.full(n_atoms, 6, dtype=np.int64),  # default to carbon
        atom_ref_charge=np.zeros(n_atoms, dtype=np.int64),
        atom_ref_name=[f"C{i}" for i in range(n_atoms)],
        atom_ref_name_chars=np.zeros((n_atoms, 4), dtype=np.int64),
        atom_ref_space_uid=np.full(n_atoms, 0, dtype=np.int64),
        atom_is_not_padding_mask=np.ones(n_atoms, dtype=bool),
        atom_gt_coords=np.zeros((n_atoms, 3), dtype=np.float32),
        atom_exists_mask=np.ones(n_atoms, dtype=bool),
        pdb_id=np.tile(_str_to_uint8("PRED", 32), (n_tokens, 1)),
        source_pdb_chain_id=np.tile(_str_to_uint8(chr(ord("A") + chain_id % 26), 4), (n_tokens, 1)),
        subchain_id=np.tile(_str_to_uint8(chr(ord("A") + chain_id % 26), 4), (n_tokens, 1)),
        resolution=np.array([0.0], dtype=np.float32),
        is_distillation=np.array([False], dtype=bool),
        symmetries=np.arange(n_atoms, dtype=np.int64).reshape(-1, 1),
        bond_left=np.zeros(0, dtype=np.int64),
        bond_right=np.zeros(0, dtype=np.int64),
    )


def _count_heavy_atoms_smiles(smiles: str) -> int:
    """Rough heavy-atom count from SMILES (not fully accurate without RDKit)."""
    clean = re.sub(r"\[.*?\]", "X", smiles)
    clean = re.sub(r"[^A-Za-z]", "", clean)
    clean = clean.replace("H", "")
    return max(len(clean), 1)


def tokenize_chain(
    inp: ChainInput,
    *,
    chain_id: int = 0,
    entity_id: int = 0,
    sym_id: int = 0,
) -> AllAtomStructureContext:
    """Tokenize a chain input into a structure context."""
    if inp.entity_type == EntityType.PROTEIN:
        return tokenize_protein_chain(
            inp.sequence, chain_id=chain_id, entity_id=entity_id, sym_id=sym_id
        )
    elif inp.entity_type in (EntityType.RNA, EntityType.DNA):
        return tokenize_nucleic_acid_chain(
            inp.sequence,
            entity_type=inp.entity_type,
            chain_id=chain_id,
            entity_id=entity_id,
            sym_id=sym_id,
        )
    elif inp.entity_type == EntityType.LIGAND:
        return tokenize_ligand_smiles(
            inp.sequence, chain_id=chain_id, entity_id=entity_id, sym_id=sym_id
        )
    else:
        return tokenize_protein_chain(
            inp.sequence, chain_id=chain_id, entity_id=entity_id, sym_id=sym_id
        )


def load_chains_from_fasta(
    fasta_path: str | Path,
) -> tuple[list[ChainInput], AllAtomStructureContext]:
    """Parse a FASTA file, tokenize all chains, and merge into a single context."""
    inputs = read_inputs(fasta_path)
    contexts: list[AllAtomStructureContext] = []
    for i, inp in enumerate(inputs):
        ctx = tokenize_chain(inp, chain_id=i, entity_id=i, sym_id=0)
        contexts.append(ctx)

    if not contexts:
        return inputs, AllAtomStructureContext.empty()

    merged = AllAtomStructureContext.merge(contexts) if len(contexts) > 1 else contexts[0]
    return inputs, merged
