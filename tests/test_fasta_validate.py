"""Tests for :mod:`chai_mlx.data.fasta`.

Covers the light-weight header parser and the set of user-facing
validation errors it surfaces (4-char name limit, duplicate names,
unknown kinds, empty sequences, malformed headers).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chai_mlx.data.fasta import (
    FastaRecord,
    MAX_ENTITY_NAME_LENGTH,
    find_fasta_issues,
    parse_fasta_records,
    validate_fasta_or_raise,
)


def _write(tmpdir: Path, text: str) -> Path:
    p = tmpdir / "input.fasta"
    p.write_text(text)
    return p


def test_parser_reads_single_protein():
    with tempfile.TemporaryDirectory() as td:
        fasta = _write(
            Path(td),
            ">protein|name=T1\nMKWVTFISLLFLFSSAYS\n",
        )
        records = parse_fasta_records(fasta)
        assert len(records) == 1
        r = records[0]
        assert r.kind == "protein"
        assert r.name == "T1"
        assert r.sequence == "MKWVTFISLLFLFSSAYS"
        assert r.line_number == 1


def test_parser_handles_multi_record_and_blank_lines():
    with tempfile.TemporaryDirectory() as td:
        fasta = _write(
            Path(td),
            ">protein|name=A\nMKWV\n\n>ligand|name=L1\nCS\n\n>dna|name=D1\nACGT\n",
        )
        records = parse_fasta_records(fasta)
        assert [r.kind for r in records] == ["protein", "ligand", "dna"]
        assert [r.name for r in records] == ["A", "L1", "D1"]
        assert [r.sequence for r in records] == ["MKWV", "CS", "ACGT"]


def test_find_fasta_issues_rejects_long_name():
    # FKBP is fine (4 chars); FKBPA is 5 chars and must be flagged.
    recs = [
        FastaRecord(kind="protein", name="FKBPA", sequence="MKWV", line_number=1),
    ]
    issues = find_fasta_issues(recs)
    assert len(issues) == 1
    assert "FKBPA" in issues[0].message
    assert f"cannot exceed {MAX_ENTITY_NAME_LENGTH}" in issues[0].message


def test_find_fasta_issues_rejects_duplicate_name():
    recs = [
        FastaRecord(kind="protein", name="T", sequence="MKWV", line_number=1),
        FastaRecord(kind="protein", name="T", sequence="MKWV", line_number=3),
    ]
    issues = find_fasta_issues(recs)
    assert len(issues) == 1
    assert "used more than once" in issues[0].message
    assert issues[0].line_number == 3


def test_find_fasta_issues_rejects_unknown_kind():
    recs = [
        FastaRecord(kind="blerg", name="X", sequence="AAA", line_number=1),
    ]
    issues = find_fasta_issues(recs)
    assert len(issues) == 1
    assert "blerg" in issues[0].message


def test_find_fasta_issues_rejects_empty_sequence():
    recs = [
        FastaRecord(kind="protein", name="X", sequence="", line_number=1),
    ]
    issues = find_fasta_issues(recs)
    assert len(issues) == 1
    assert "no sequence" in issues[0].message


def test_find_fasta_issues_rejects_missing_name():
    # Header was '>protein' without '|name=...'
    recs = [
        FastaRecord(kind="protein", name="", sequence="AAA", line_number=1),
    ]
    issues = find_fasta_issues(recs)
    assert len(issues) == 1
    assert "missing" in issues[0].message


def test_validate_passes_on_good_fasta():
    with tempfile.TemporaryDirectory() as td:
        fasta = _write(
            Path(td),
            ">protein|name=T1\nMKWV\n>ligand|name=L1\nCS\n",
        )
        records = validate_fasta_or_raise(fasta)
        assert len(records) == 2


def test_validate_raises_on_long_name_lists_offender():
    with tempfile.TemporaryDirectory() as td:
        fasta = _write(
            Path(td),
            ">protein|name=MYPROTEIN\nMKWV\n",
        )
        with pytest.raises(SystemExit) as exc:
            validate_fasta_or_raise(fasta)
        msg = str(exc.value)
        assert "MYPROTEIN" in msg
        assert "line 1" in msg


def test_validate_raises_on_empty_file():
    with tempfile.TemporaryDirectory() as td:
        fasta = _write(Path(td), "")
        with pytest.raises(SystemExit) as exc:
            validate_fasta_or_raise(fasta)
        assert "contains no FASTA records" in str(exc.value)


def test_validate_rejects_bad_ligand_smiles_when_rdkit_present():
    try:
        import rdkit  # noqa: F401
    except ImportError:
        pytest.skip("rdkit not installed")
    with tempfile.TemporaryDirectory() as td:
        fasta = _write(
            Path(td),
            ">protein|name=P1\nMKWV\n>ligand|name=L1\n][not_smiles\n",
        )
        with pytest.raises(SystemExit) as exc:
            validate_fasta_or_raise(fasta)
        msg = str(exc.value)
        assert "L1" in msg
        assert "RDKit could not parse" in msg


def test_validate_rejects_bad_glycan_when_chailab_present():
    try:
        from chai_lab.data.parsing.glycans import _glycan_string_to_sugars_and_bonds  # noqa: F401
    except ImportError:
        pytest.skip("chai_lab not installed")
    with tempfile.TemporaryDirectory() as td:
        fasta = _write(
            Path(td),
            ">protein|name=P1\nMKWV\n>glycan|name=G1\nZZZ(unparseable)\n",
        )
        with pytest.raises(SystemExit) as exc:
            validate_fasta_or_raise(fasta)
        msg = str(exc.value)
        assert "G1" in msg
        assert "glycan parser" in msg


def test_validate_rejects_modified_residue_token_by_default(tmp_path):
    """Inline ``[FOO]`` / ``(FOO)`` tokens surface a loud pointer at
    HANDOFF §8.1; that's chai-lab's real PTM syntax (see
    ``chai_lab.data.parsing.input_validation.constituents_of_modified_fasta``).
    """
    fasta = tmp_path / "ptm.fasta"
    fasta.write_text(
        ">protein|name=P1\nAPNGL[HIP]TRP\n"
    )
    with pytest.raises(SystemExit) as exc:
        validate_fasta_or_raise(fasta)
    msg = str(exc.value)
    assert "modified-residue" in msg.lower()
    assert "CHAI_MLX_ALLOW_MODIFIED_RESIDUES" in msg


def test_validate_allows_modified_residue_token_via_env(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
):
    """The escape hatch lets power users opt in at their own risk."""
    fasta = tmp_path / "ptm.fasta"
    fasta.write_text(
        ">protein|name=P1\nAPNGL[HIP]TRP\n"
    )
    monkeypatch.setenv("CHAI_MLX_ALLOW_MODIFIED_RESIDUES", "1")
    # With the env var set, validation should pass (the downstream
    # featurizer may still reject the sequence, but that's chai-lab's
    # problem, not ours).
    records = validate_fasta_or_raise(fasta)
    assert len(records) == 1
    assert records[0].kind == "protein"


def test_validate_accepts_good_smiles_and_glycan_when_deps_present():
    with tempfile.TemporaryDirectory() as td:
        fasta = _write(
            Path(td),
            ">protein|name=P1\nMKWV\n"
            ">ligand|name=L1\nCS\n"
            ">glycan|name=G1\nNAG(4-1 NAG)\n",
        )
        # Only runs clean if both rdkit and chai_lab are available; in
        # the standard featurize test env, both are installed.
        try:
            records = validate_fasta_or_raise(fasta)
        except SystemExit as exc:  # pragma: no cover - env without deps
            msg = str(exc)
            if "RDKit" in msg or "glycan" in msg:
                pytest.skip(f"deps missing for preflight: {msg}")
            raise
        kinds = [r.kind for r in records]
        assert kinds == ["protein", "ligand", "glycan"]
