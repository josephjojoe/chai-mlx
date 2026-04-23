"""Nucleic-acid featurization smoke.

Runs ``featurize_fasta`` on a tiny synthetic DNA duplex and asserts the
resulting ``structure_inputs`` carry ``EntityType.DNA`` and that the
``token_is_polymer`` mask treats DNA as a polymer (as Chai-1 does: it
merges PROTEIN, RNA, DNA under the is_polymer umbrella).

This is pure featurization -- no model weights, no diffusion, no MLX
forward pass -- so it is safe to run on CI without GPUs or weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.helpers import has_chai_lab_runtime


_HAS_CHAI_LAB = has_chai_lab_runtime()

pytestmark = pytest.mark.skipif(
    not _HAS_CHAI_LAB,
    reason="featurization tests require chai-lab (install via [featurize] extra)",
)


def test_dna_only_featurization_entity_type(tmp_path: Path) -> None:
    from chai_lab.data.parsing.structure.entity_type import EntityType

    from chai_mlx.data.featurize import featurize_fasta

    # Short self-complementary duplex; two chains, 12 bp each.
    fasta_path = tmp_path / "dna.fasta"
    # Entity names capped at 4 chars for chai-lab's fixed-length subchain
    # ID packing (required whenever entity_name_as_subchain=True, which
    # featurize_fasta now forwards unconditionally).
    fasta_path.write_text(
        ">dna|name=DNA1\nCGCGAATTCGCG\n"
        ">dna|name=DNA2\nCGCGAATTCGCG\n"
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
    assert active_types == {EntityType.DNA.value}, (
        f"expected every live token to be DNA, got entity_type values {active_types}"
    )
    assert np.all(is_polymer[token_exists] == 1.0), (
        "DNA tokens must be flagged as polymers (is_polymer mask covers DNA+RNA+PROTEIN)"
    )
    assert len(set(asym_id[token_exists].tolist())) >= 2, (
        "two DNA chains should yield two distinct asym IDs"
    )
