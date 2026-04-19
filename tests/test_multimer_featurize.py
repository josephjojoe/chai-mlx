"""Multi-chain protein featurization smoke.

Runs ``featurize_fasta`` on a synthetic two-protein FASTA and asserts
the returned context carries two distinct chain IDs tagged as PROTEIN.
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


def test_two_protein_chain_featurization(tmp_path: Path) -> None:
    from chai_lab.data.parsing.structure.entity_type import EntityType

    from chai_mlx.data.featurize import featurize_fasta

    # Two short, synthetic protein chains; shapes + bookkeeping only.
    seq_a = "MKTAYIAKQRQISFVKSHFS"
    seq_b = "GVQVETISPGDGRTFPKRGQ"
    fasta_path = tmp_path / "dimer.fasta"
    # Entity names capped at 4 chars (chai-lab packs subchain IDs into a
    # fixed-length tensor whenever entity_name_as_subchain=True).
    fasta_path.write_text(
        f">protein|name=CHNA\n{seq_a}\n"
        f">protein|name=CHNB\n{seq_b}\n"
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
    token_exists = np.asarray(si.token_exists_mask).astype(bool).ravel()
    asym_id = np.asarray(si.token_asym_id).ravel()

    active_types = set(entity_type[token_exists].tolist())
    assert active_types == {EntityType.PROTEIN.value}, (
        f"expected every live token to be PROTEIN, got {active_types}"
    )

    live_asyms = sorted(set(asym_id[token_exists].tolist()))
    assert len(live_asyms) == 2, (
        f"expected exactly two chains, got asym IDs {live_asyms}"
    )

    live_mask = token_exists.astype(int)
    assert live_mask.sum() == len(seq_a) + len(seq_b), (
        "token count should match the sum of the two chain lengths"
    )
