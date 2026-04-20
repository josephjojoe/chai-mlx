"""RNA featurization smoke.

Runs ``featurize_fasta`` on an RNA-only FASTA and asserts the resulting
``structure_inputs`` carry ``EntityType.RNA`` and that the
``token_is_polymer`` mask treats RNA as a polymer (matching chai-1's
convention: PROTEIN, RNA, DNA all share the is_polymer umbrella).

Pure featurization, no model weights, so this stays cheap for CI.
Retires the "RNA: plumbed but not exercised" row from HANDOFF.md §8.1.
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


def test_rna_only_featurization_entity_type(tmp_path: Path) -> None:
    from chai_lab.data.parsing.structure.entity_type import EntityType

    from chai_mlx.data.featurize import featurize_fasta

    # A short self-complementary RNA hairpin-ish stretch keeps the test
    # cheap. We just want to verify the RNA featurization code-path
    # produces plausible structural metadata; the end-to-end numerical
    # sweep uses the slightly longer 2KOC hairpin in DEFAULT_TARGETS.
    fasta_path = tmp_path / "rna.fasta"
    fasta_path.write_text(
        ">rna|name=RNA1\nGCGCAAAAGCGC\n"
    )

    ctx = featurize_fasta(
        fasta_path,
        output_dir=tmp_path / "features",
        esm_backend="off",
        use_msa_server=False,
        use_templates_server=False,
    )
    si = ctx.structure_inputs

    entity_type = np.asarray(si.token_entity_type).ravel()
    is_polymer = np.asarray(si.token_is_polymer).ravel()
    token_exists = np.asarray(si.token_exists_mask).astype(bool).ravel()
    asym_id = np.asarray(si.token_asym_id).ravel()

    active_types = set(entity_type[token_exists].tolist())
    assert active_types == {EntityType.RNA.value}, (
        f"expected every live token to be RNA, got entity_type values {active_types}"
    )
    assert np.all(is_polymer[token_exists] == 1.0), (
        "RNA tokens must be flagged as polymers"
    )
    # One chain -> exactly one live asym id.
    assert len(set(asym_id[token_exists].tolist())) == 1


def test_rna_multimer_featurization(tmp_path: Path) -> None:
    """Two RNA strands (duplex-like input). Verifies the multi-chain
    path for RNA still produces distinct asym IDs -- matches what the
    DNA duplex test covers for ``EntityType.DNA``.
    """
    from chai_lab.data.parsing.structure.entity_type import EntityType

    from chai_mlx.data.featurize import featurize_fasta

    fasta_path = tmp_path / "rna_duplex.fasta"
    fasta_path.write_text(
        ">rna|name=RNA1\nAAGCGCUU\n"
        ">rna|name=RNA2\nAAGCGCUU\n"
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
    assert active_types == {EntityType.RNA.value}
    assert len(set(asym_id[token_exists].tolist())) >= 2, (
        "two RNA chains should yield two distinct asym IDs"
    )
