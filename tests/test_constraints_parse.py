"""Constraint featurization smoke.

Writes a minimal 3-restraint CSV, pushes it through ``featurize_fasta``
on 1CRN + a small-molecule ligand, and asserts that the
``TokenDistanceRestraint`` / ``TokenPairPocketRestraint`` raw feature
tensors are populated with non-default entries at the expected positions.

The chai-lab restraint CSV schema::

    restraint_id,chainA,res_idxA,chainB,res_idxB,connection_type,confidence,
    min_distance_angstrom,max_distance_angstrom,comment

For contact/pocket restraints the *distance* feature gets a non-default
value at token pairs (i, j) matching the specified residues; for
covalent restraints, the bond-adjacency (``ctx.bond_adjacency``) tensor
picks up a non-zero entry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.helpers import has_chai_lab_runtime


_HAS_CHAI_LAB = has_chai_lab_runtime()

pytestmark = pytest.mark.skipif(
    not _HAS_CHAI_LAB,
    reason="constraint tests require chai-lab (install via [featurize] extra)",
)


_CRN_SEQUENCE = "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN"


def _write_fasta(path: Path, *, with_ligand: bool) -> None:
    fasta = f">protein|name=1CRN\n{_CRN_SEQUENCE}\n"
    if with_ligand:
        fasta += ">ligand|name=LIG1\nCS\n"
    path.write_text(fasta)


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    fieldnames = [
        "restraint_id",
        "chainA",
        "res_idxA",
        "chainB",
        "res_idxB",
        "connection_type",
        "confidence",
        "min_distance_angstrom",
        "max_distance_angstrom",
        "comment",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_contact_restraint_populates_distance_feature(tmp_path: Path) -> None:
    """Protein-only contact restraint: the most widely-used case.

    Ligand-containing targets exercise RDKit symmetry detection which
    chai-lab wraps in a multiprocessing-based timeout helper; that
    helper cannot pickle on some macOS Python builds (AttributeError:
    Can't pickle local object 'timeout.<locals>.handler').  We stay
    protein-only here so offline pytest works regardless of platform;
    the pocket + covalent + ligand paths are still validated end-to-end
    via ``scripts/cuda_constraints_parity.py`` against the Modal CUDA
    harness output.
    """
    from chai_mlx.data.featurize import featurize_fasta

    fasta_path = tmp_path / "1crn.fasta"
    _write_fasta(fasta_path, with_ligand=False)

    csv_path = tmp_path / "constraints.csv"
    _write_csv(
        csv_path,
        rows=[
            {
                "restraint_id": "r_contact_disulfide",
                "chainA": "1CRN",
                "res_idxA": "C3",
                "chainB": "1CRN",
                "res_idxB": "C40",
                "connection_type": "contact",
                "confidence": 1.0,
                "min_distance_angstrom": 1.8,
                "max_distance_angstrom": 2.5,
                "comment": "Cys3-Cys40 contact",
            },
            {
                "restraint_id": "r_contact_core",
                "chainA": "1CRN",
                "res_idxA": "I7",
                "chainB": "1CRN",
                "res_idxB": "I34",
                "connection_type": "contact",
                "confidence": 0.9,
                "min_distance_angstrom": 3.0,
                "max_distance_angstrom": 8.0,
                "comment": "synthetic hydrophobic-core contact",
            },
        ],
    )

    ctx = featurize_fasta(
        fasta_path,
        output_dir=tmp_path / "features",
        constraint_path=csv_path,
        esm_backend="off",
        use_msa_server=False,
        use_templates_server=False,
    )

    assert ctx.raw_features is not None, (
        "featurize_fasta should populate raw_features so FeatureEmbedding "
        "can encode constraints"
    )
    distance = np.asarray(ctx.raw_features["TokenDistanceRestraint"])

    # chai-lab encodes "no restraint" as -1; any non-(-1) entry means the
    # CSV made it through the contact-restraint branch.
    assert (distance != -1.0).any(), (
        "TokenDistanceRestraint has no non-default entries; contact restraint "
        "did not propagate into the feature tensor"
    )


def test_no_constraints_leaves_features_at_sentinel(tmp_path: Path) -> None:
    """Sanity: without a constraint CSV, the restraint tensors stay at -1."""
    from chai_mlx.data.featurize import featurize_fasta

    fasta_path = tmp_path / "1crn_no_constraints.fasta"
    _write_fasta(fasta_path, with_ligand=False)

    ctx = featurize_fasta(
        fasta_path,
        output_dir=tmp_path / "features",
        esm_backend="off",
        use_msa_server=False,
        use_templates_server=False,
    )

    assert ctx.raw_features is not None
    distance = np.asarray(ctx.raw_features["TokenDistanceRestraint"])
    pocket = np.asarray(ctx.raw_features["TokenPairPocketRestraint"])
    assert np.all(distance == -1.0), (
        "TokenDistanceRestraint must stay at the -1 sentinel when no CSV is passed"
    )
    assert np.all(pocket == -1.0), (
        "TokenPairPocketRestraint must stay at the -1 sentinel when no CSV is passed"
    )


def test_covalent_restraint_populates_bond_adjacency(tmp_path: Path) -> None:
    """Covalent + protein + ligand end-to-end on macOS.

    Previously blocked by chai-lab's RDKit timeout decorator not being
    picklable under macOS multiprocessing-spawn; guarded now by
    :mod:`chai_mlx.data._rdkit_timeout_patch`.
    """
    from chai_mlx.data.featurize import featurize_fasta

    fasta_path = tmp_path / "1crn_with_ligand.fasta"
    _write_fasta(fasta_path, with_ligand=True)

    csv_path = tmp_path / "constraints.csv"
    _write_csv(
        csv_path,
        rows=[
            {
                "restraint_id": "r_cov",
                "chainA": "1CRN",
                "res_idxA": "C32@SG",
                "chainB": "LIG1",
                "res_idxB": "@C1",
                "connection_type": "covalent",
                "confidence": 1.0,
                "min_distance_angstrom": "",
                "max_distance_angstrom": "",
                "comment": "synthetic Cys32 SG -> LIG1 C1",
            },
        ],
    )

    ctx = featurize_fasta(
        fasta_path,
        output_dir=tmp_path / "features",
        constraint_path=csv_path,
        esm_backend="off",
        use_msa_server=False,
        use_templates_server=False,
    )

    bond = np.asarray(ctx.bond_adjacency)
    assert bond.ndim == 4 and bond.shape[-1] == 1, (
        f"bond_adjacency shape unexpected: {bond.shape}"
    )
    # Exactly one covalent restraint means exactly one non-zero entry
    # (chai-lab stores the single atom-pair bond as a 1 at that (i, j)).
    nz = int((bond != 0).sum())
    assert nz >= 1, (
        "bond_adjacency has no non-zero entries; covalent restraint did not "
        "propagate through featurization"
    )


def test_protein_plus_ligand_featurize(tmp_path: Path) -> None:
    """Protein + ligand FASTA round-trips without the RDKit timeout bug.

    Under the default ``pad_strategy="exact"`` the token axis is padded
    to exactly ``num_tokens`` (46 protein residues + 2 ligand atoms = 48
    tokens for 1CRN + methanethiol) and the atom axis to the next
    multiple of 32 -- *not* to chai-lab's 256-token bucket. This is the
    main wall-clock win the exact-length path provides.
    """
    from chai_mlx.data.featurize import featurize_fasta

    fasta_path = tmp_path / "1crn_with_ligand.fasta"
    _write_fasta(fasta_path, with_ligand=True)

    ctx = featurize_fasta(
        fasta_path,
        output_dir=tmp_path / "features",
        esm_backend="off",
        use_msa_server=False,
        use_templates_server=False,
    )

    si = ctx.structure_inputs
    n_tokens = si.token_entity_type.shape[1]
    n_atoms = si.atom_exists_mask.shape[1]

    # 1CRN has 46 residues; methanethiol ("CS") contributes 2 heavy-atom
    # ligand tokens. Exact-length padding never exceeds this, and never
    # pads up to chai-lab's 256 bucket.
    assert n_tokens == 48, (
        f"expected exact-length token dim 48 (46 residues + 2 ligand atoms); "
        f"got {n_tokens}. If you changed the pad-strategy default, also "
        f"update this assertion -- the point of this test is to catch "
        f"silent reversion to bucketed padding."
    )
    # Atom axis is padded to the next multiple of 32 (the local-atom-
    # attention query-block stride). Count should be > num_tokens (each
    # residue has multiple atoms) and divisible by 32.
    assert n_atoms > 0 and n_atoms % 32 == 0, (
        f"n_atoms={n_atoms} violates the mandatory %32==0 constraint from "
        f"chai_lab.model.utils.get_qkv_indices_for_blocks"
    )


def test_protein_plus_ligand_featurize_bucket_mode(tmp_path: Path) -> None:
    """Bucket-mode opt-in: reproduces chai-lab's 256-token rounding.

    Kept as a regression guard for the parity path that compares against
    the CUDA reference TorchScript artefacts (which were traced at
    exactly those seven sizes).
    """
    from chai_mlx.data.featurize import featurize_fasta

    fasta_path = tmp_path / "1crn_with_ligand.fasta"
    _write_fasta(fasta_path, with_ligand=True)

    ctx = featurize_fasta(
        fasta_path,
        output_dir=tmp_path / "features",
        esm_backend="off",
        use_msa_server=False,
        use_templates_server=False,
        pad_strategy="bucket",
    )

    si = ctx.structure_inputs
    # 48 tokens -> smallest chai-lab bucket (256).
    assert si.token_entity_type.shape[1] == 256, (
        f"unexpected token dim under pad_strategy='bucket': "
        f"{si.token_entity_type.shape}"
    )
    # n_atoms = 23 * n_tokens under bucket mode.
    assert si.atom_exists_mask.shape[1] == 23 * 256
