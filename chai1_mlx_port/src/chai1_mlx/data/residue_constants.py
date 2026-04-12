"""Residue vocabulary, atom tables, and lookup maps.

Ported verbatim from ``chai_lab/data/residue_constants.py`` (Apache-2.0,
original copyright AlQuraishi Laboratory / DeepMind Technologies / Chai
Discovery).
"""

from __future__ import annotations

from enum import Enum

# ── Heavy-atom lists per amino acid (PDB naming) ─────────────────────
residue_atoms: dict[str, list[str]] = {
    "ALA": ["C", "CA", "CB", "N", "O"],
    "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
    "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
    "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
    "CYS": ["C", "CA", "CB", "N", "O", "SG"],
    "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
    "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
    "GLY": ["C", "CA", "N", "O"],
    "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
    "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
    "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
    "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
    "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
    "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
    "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
    "SER": ["C", "CA", "CB", "N", "O", "OG"],
    "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
    "TRP": [
        "C", "CA", "CB", "CG", "CD1", "CD2", "CE2", "CE3",
        "CZ2", "CZ3", "CH2", "N", "NE1", "O",
    ],
    "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
    "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
}

# ── Nucleic-acid all-atom slots (RosettaFold-AA convention) ──────────
nucleic_acid_atoms: dict[str, tuple[str | None, ...]] = {
    "DA": ("O4'","C1'","C2'","OP1","P","OP2","O5'","C5'","C4'","C3'","O3'",
           "N9","C4","N3","C2","N1","C6","C5","N7","C8","N6",None,None,
           "H5''","H5'","H4'","H3'","H2''","H2'","H1'","H2","H61","H62","H8",None,None),
    "DC": ("O4'","C1'","C2'","OP1","P","OP2","O5'","C5'","C4'","C3'","O3'",
           "N1","C2","O2","N3","C4","N4","C5","C6",None,None,None,None,
           "H5''","H5'","H4'","H3'","H2''","H2'","H1'","H42","H41","H5","H6",None,None),
    "DG": ("O4'","C1'","C2'","OP1","P","OP2","O5'","C5'","C4'","C3'","O3'",
           "N9","C4","N3","C2","N1","C6","C5","N7","C8","N2","O6",None,
           "H5''","H5'","H4'","H3'","H2''","H2'","H1'","H1","H22","H21","H8",None,None),
    "DT": ("O4'","C1'","C2'","OP1","P","OP2","O5'","C5'","C4'","C3'","O3'",
           "N1","C2","O2","N3","C4","O4","C5","C7","C6",None,None,None,
           "H5''","H5'","H4'","H3'","H2''","H2'","H1'","H3","H71","H72","H73","H6",None),
    "DX": ("O4'","C1'","C2'","OP1","P","OP2","O5'","C5'","C4'","C3'","O3'",
           "O2'","N1","C2","N3","C4","C5","C6","N6","N7","C8","N9",None,
           "H5'","H5''","H4'","H3'","H2'","HO2'","H1'","H2","H61","H62","H8",None,None),
    "RA": ("O4'","C1'","C2'","OP1","P","OP2","O5'","C5'","C4'","C3'","O3'",
           "O2'","N1","C2","N3","C4","C5","C6","N6","N7","C8","N9",None,
           "H5'","H5''","H4'","H3'","H2'","HO2'","H1'","H2","H61","H62","H8",None,None),
    "RC": ("O4'","C1'","C2'","OP1","P","OP2","O5'","C5'","C4'","C3'","O3'",
           "O2'","N1","C2","O2","N3","C4","N4","C5","C6",None,None,None,
           "H5'","H5''","H4'","H3'","H2'","HO2'","H1'","H42","H41","H5","H6",None,None),
    "RG": ("O4'","C1'","C2'","OP1","P","OP2","O5'","C5'","C4'","C3'","O3'",
           "O2'","N1","C2","N2","N3","C4","C5","C6","O6","N7","C8","N9",
           "H5'","H5''","H4'","H3'","H2'","HO2'","H1'","H1","H22","H21","H8",None,None),
    "RU": ("O4'","C1'","C2'","OP1","P","OP2","O5'","C5'","C4'","C3'","O3'",
           "O2'","N1","C2","O2","N3","C4","O4","C5","C6",None,None,None,
           "H5'","H5''","H4'","H3'","H2'","HO2'","H1'","H3","H5","H6",None,None,None),
    "RX": ("O4'","C1'","C2'","OP1","P","OP2","O5'","C5'","C4'","C3'","O3'",
           "O2'",None,None,None,None,None,None,None,None,None,None,None,
           None,"H5'","H5''","H4'","H3'","H2'","HO2'","H1'",None,None,None,None,None,None),
}

# ── Fixed-size atom37 index table ────────────────────────────────────
atom_indices = Enum(
    "atom_indices",
    [
        "N","CA","C","CB","O","CG","CG1","CG2","OG","OG1","SG",
        "CD","CD1","CD2","ND1","ND2","OD1","OD2","SD",
        "CE","CE1","CE2","CE3","NE","NE1","NE2","OE1","OE2",
        "CH2","NH1","NH2","OH","CZ","CZ2","CZ3","NZ","OXT",
    ],
    start=0,
)
atom_types = [a.name for a in atom_indices]
atom_order: dict[str, int] = {name: i for i, name in enumerate(atom_types)}

# ── Residue vocabulary ───────────────────────────────────────────────
restypes = [
    "A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V",
]

residue_types_with_nucleotides: list[str] = (
    restypes
    + ["X"]
    + ["RA","RC","RG","RU","RX"]
    + ["DA","DC","DG","DT","DX"]
    + ["-"]  # gap
    + [":"]  # non-existent / mask
)

residue_types_with_nucleotides_order: dict[str, int] = {
    rt: i for i, rt in enumerate(residue_types_with_nucleotides)
}

NUM_RESTYPES = len(residue_types_with_nucleotides)  # 33

standard_residue_pdb_codes: set[str] = {
    "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
    "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","UNK",
    "A","G","C","U","DA","DG","DC","DT",
}

new_ligand_residue_name = "LIG"

restype_1to3: dict[str, str] = {
    "A":"ALA","R":"ARG","N":"ASN","D":"ASP","C":"CYS","Q":"GLN","E":"GLU",
    "G":"GLY","H":"HIS","I":"ILE","L":"LEU","K":"LYS","M":"MET","F":"PHE",
    "P":"PRO","S":"SER","T":"THR","W":"TRP","Y":"TYR","V":"VAL",
}

restype_1to3_with_x: dict[str, str] = {**restype_1to3, "X": "UNK"}
restype_3to1: dict[str, str] = {v: k for k, v in restype_1to3.items()}
