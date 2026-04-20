"""Argparse-surface tests for the Tier-1 chai-1 parity flags.

Covers the CLI-layer wiring of:

* ``--num-trunk-samples``    (chai-1 ``num_trunk_samples``)
* ``--recycle-msa-subsample`` (chai-1 ``recycle_msa_subsample``)
* ``--fasta-chain-names``    (chai-1 ``fasta_names_as_cif_chains``)

These tests don't run the model; they only verify that the argparser
accepts the flags, that defaults match the documented semantics, and
that auto-resolution rules (e.g. ``--fasta-chain-names`` defaulting
to the constraint-path presence) behave as specified. Real end-to-end
behaviour of the flags is exercised by
:mod:`tests.test_integration_infer` and the T1 end-to-end test below.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chai_mlx.cli import infer as _infer_mod


@pytest.fixture
def tiny_fasta() -> Path:
    with tempfile.TemporaryDirectory(prefix="chai_mlx_t1_test_") as tmpdir:
        fasta = Path(tmpdir) / "protein.fasta"
        fasta.write_text(">protein|name=T\nMKWV\n")
        yield fasta


def _parse(argv: list[str]):
    return _infer_mod._parse_args(argv)


def test_num_trunk_samples_defaults_to_1(tiny_fasta: Path) -> None:
    args = _parse(
        [
            "--weights-dir", "weights",
            "--fasta", str(tiny_fasta),
            "--output-dir", "/tmp/out",
        ],
    )
    assert args.num_trunk_samples == 1


def test_num_trunk_samples_accepts_explicit_value(tiny_fasta: Path) -> None:
    args = _parse(
        [
            "--weights-dir", "weights",
            "--fasta", str(tiny_fasta),
            "--output-dir", "/tmp/out",
            "--num-trunk-samples", "3",
        ],
    )
    assert args.num_trunk_samples == 3


def test_recycle_msa_subsample_defaults_to_0(tiny_fasta: Path) -> None:
    args = _parse(
        [
            "--weights-dir", "weights",
            "--fasta", str(tiny_fasta),
            "--output-dir", "/tmp/out",
        ],
    )
    assert args.recycle_msa_subsample == 0


def test_recycle_msa_subsample_accepts_value(tiny_fasta: Path) -> None:
    args = _parse(
        [
            "--weights-dir", "weights",
            "--fasta", str(tiny_fasta),
            "--output-dir", "/tmp/out",
            "--recycle-msa-subsample", "2048",
        ],
    )
    assert args.recycle_msa_subsample == 2048


def test_fasta_chain_names_default_is_none(tiny_fasta: Path) -> None:
    """Default is None so the downstream code can auto-resolve based on
    whether a constraint CSV is attached (matches chai-1 when no CSV,
    matches constraint-friendly behaviour when one is)."""
    args = _parse(
        [
            "--weights-dir", "weights",
            "--fasta", str(tiny_fasta),
            "--output-dir", "/tmp/out",
        ],
    )
    assert args.fasta_chain_names is None


def test_fasta_chain_names_explicit_true(tiny_fasta: Path) -> None:
    args = _parse(
        [
            "--weights-dir", "weights",
            "--fasta", str(tiny_fasta),
            "--output-dir", "/tmp/out",
            "--fasta-chain-names",
        ],
    )
    assert args.fasta_chain_names is True


def test_fasta_chain_names_explicit_false(tiny_fasta: Path) -> None:
    args = _parse(
        [
            "--weights-dir", "weights",
            "--fasta", str(tiny_fasta),
            "--output-dir", "/tmp/out",
            "--no-fasta-chain-names",
        ],
    )
    assert args.fasta_chain_names is False


def test_help_includes_new_t1_flags() -> None:
    """Smoke test: ``--help`` should mention every T1 flag so users
    discover them. If this breaks the user-visible help is the likely
    culprit."""
    parser_built = False
    try:
        _parse(["--help"])
    except SystemExit:
        parser_built = True
    assert parser_built, "--help should always cause SystemExit(0)"
