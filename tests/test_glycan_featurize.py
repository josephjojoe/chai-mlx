"""Glycan featurization smoke.

Runs ``featurize_fasta`` on a small FASTA that contains a glycan record
and asserts the resulting ``structure_inputs`` carry
``EntityType.MANUAL_GLYCAN`` for the glycan tokens. No covalent bond is
attached (the covalent-bond path is validated elsewhere via
``test_constraints_parse.py``); this test isolates the glycan parser +
featurization plumbing.

Retires the "glycan: plumbed but not exercised" row from
HANDOFF.md §8.1.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


_HAS_CHAI_LAB = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("chai_lab") is not None
)

pytestmark = pytest.mark.skipif(
    not _HAS_CHAI_LAB,
    reason="featurization tests require chai-lab (install via [featurize] extra)",
)


def test_glycan_only_featurization(tmp_path: Path) -> None:
    from chai_lab.data.parsing.structure.entity_type import EntityType

    from chai_mlx.data.featurize import featurize_fasta

    fasta_path = tmp_path / "glycan.fasta"
    # Small single-sugar glycan. Chai-lab's grammar is
    # ``NAME`` or ``NAME(link NAME)``; NAG is a standard sugar (N-acetyl-
    # glucosamine) that chai-lab's sugar table supports.
    fasta_path.write_text(">glycan|name=GLY1\nNAG\n")

    ctx = featurize_fasta(
        fasta_path,
        output_dir=tmp_path / "features",
        esm_backend="off",
    )
    si = ctx.structure_inputs

    entity_type = np.asarray(si.token_entity_type).ravel()
    token_exists = np.asarray(si.token_exists_mask).astype(bool).ravel()

    active_types = set(entity_type[token_exists].tolist())
    assert active_types == {EntityType.MANUAL_GLYCAN.value}, (
        f"expected every live token to be MANUAL_GLYCAN, got {active_types}"
    )


def test_protein_plus_glycan_featurization(tmp_path: Path) -> None:
    """Mixed protein + glycan without covalent bonds. Makes sure the
    featurizer produces a sensible structure context with two entity
    types active and both chains assigned distinct asym IDs.
    """
    from chai_lab.data.parsing.structure.entity_type import EntityType

    from chai_mlx.data.featurize import featurize_fasta

    fasta_path = tmp_path / "protein_glycan.fasta"
    fasta_path.write_text(
        ">protein|name=P1\nMKFLILFNILVSTLSFSSAQA\n"
        ">glycan|name=GLY1\nNAG(4-1 NAG)\n"
    )

    ctx = featurize_fasta(
        fasta_path,
        output_dir=tmp_path / "features",
        esm_backend="off",
    )
    si = ctx.structure_inputs

    entity_type = np.asarray(si.token_entity_type).ravel()
    token_exists = np.asarray(si.token_exists_mask).astype(bool).ravel()
    asym_id = np.asarray(si.token_asym_id).ravel()

    active_types = set(entity_type[token_exists].tolist())
    assert EntityType.PROTEIN.value in active_types
    assert EntityType.MANUAL_GLYCAN.value in active_types
    assert len(set(asym_id[token_exists].tolist())) >= 2, (
        "protein + glycan should produce at least two asym IDs"
    )


def test_glycan_with_chain_linkage_featurization(tmp_path: Path) -> None:
    """Two-sugar glycan using the (n-n NAME) linkage grammar. This is
    the worked example from chai-lab/examples/covalent_bonds/README.md;
    a featurize-only test keeps the cost low but verifies the parser
    path end-to-end.
    """
    from chai_lab.data.parsing.structure.entity_type import EntityType

    from chai_mlx.data.featurize import featurize_fasta

    fasta_path = tmp_path / "glycan2.fasta"
    fasta_path.write_text(">glycan|name=G2\nNAG(4-1 NAG)\n")

    ctx = featurize_fasta(
        fasta_path,
        output_dir=tmp_path / "features",
        esm_backend="off",
    )
    si = ctx.structure_inputs

    entity_type = np.asarray(si.token_entity_type).ravel()
    token_exists = np.asarray(si.token_exists_mask).astype(bool).ravel()
    active_types = set(entity_type[token_exists].tolist())
    assert active_types == {EntityType.MANUAL_GLYCAN.value}
