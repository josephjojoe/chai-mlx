"""Argparse-level smoke tests for the ``chai-mlx-infer`` CLI.

These tests exercise the CLI surface without actually running inference:
they invoke :func:`chai_mlx.cli.infer._parse_args` with crafted argv
lists, and assert that mutually-exclusive / required-combination flags
fail cleanly via ``argparse``'s ``SystemExit(2)`` path.

Running the real end-to-end inference in pytest would require model
weights on disk + ~10 s of wall clock per sample; we already cover that
path via :mod:`tests.test_integration_infer` and the nightly sweep.
These tests only guard the shape of the CLI so regressions to the
``--use-msa-server`` / ``--esm-backend`` / ``--use-templates-server`` /
``--constraint-path`` surface are caught offline in milliseconds.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from chai_mlx.cli import infer as _infer_mod


@pytest.fixture
def tiny_fasta() -> Path:
    with tempfile.TemporaryDirectory(prefix="chai_mlx_cli_test_") as tmpdir:
        fasta = Path(tmpdir) / "protein.fasta"
        fasta.write_text(">protein|name=T\nMKWV\n")
        yield fasta


def _parse(argv: list[str]):
    """Invoke ``_parse_args`` directly. Passes argv to avoid mutating sys.argv."""
    return _infer_mod._parse_args(argv)


def test_parser_accepts_minimal_args(tiny_fasta: Path) -> None:
    args = _parse(
        [
            "--weights-dir", "weights",
            "--fasta", str(tiny_fasta),
            "--output-dir", "/tmp/out",
        ],
    )
    assert args.weights_dir == Path("weights")
    assert args.fasta == tiny_fasta
    assert args.output_dir == Path("/tmp/out")
    assert args.esm_backend == "off"
    assert args.use_msa_server is False
    assert args.use_templates_server is False
    assert args.constraint_path is None
    # Exact-length is the default padding strategy.  See
    # ``chai_mlx.data.featurize.featurize_fasta`` for the rationale:
    # the MLX model forward is shape-agnostic in N, so the 7 static
    # chai-lab buckets only exist to preserve the traced bundle's
    # fixed crop sizes -- not for correctness.
    assert args.pad_strategy == "exact"


def test_parser_accepts_pad_strategy_bucket(tiny_fasta: Path) -> None:
    args = _parse(
        [
            "--weights-dir", "weights",
            "--fasta", str(tiny_fasta),
            "--output-dir", "/tmp/out",
            "--pad-strategy", "bucket",
        ],
    )
    assert args.pad_strategy == "bucket"


def test_parser_rejects_invalid_pad_strategy(tiny_fasta: Path) -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "--weights-dir", "weights",
                "--fasta", str(tiny_fasta),
                "--output-dir", "/tmp/out",
                "--pad-strategy", "nonsense",
            ],
        )


def test_parser_rejects_mlx_cache_without_dir(tiny_fasta: Path) -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "--weights-dir", "weights",
                "--fasta", str(tiny_fasta),
                "--output-dir", "/tmp/out",
                "--esm-backend", "mlx_cache",
            ],
        )


def test_parser_rejects_msa_server_plus_msa_directory(tiny_fasta: Path) -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "--weights-dir", "weights",
                "--fasta", str(tiny_fasta),
                "--output-dir", "/tmp/out",
                "--use-msa-server",
                "--msa-directory", "/tmp/does-not-exist",
            ],
        )


def test_parser_rejects_templates_server_without_msa_server(tiny_fasta: Path) -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "--weights-dir", "weights",
                "--fasta", str(tiny_fasta),
                "--output-dir", "/tmp/out",
                "--use-templates-server",
            ],
        )


def test_parser_rejects_missing_fasta() -> None:
    with pytest.raises(SystemExit):
        _parse(
            [
                "--weights-dir", "weights",
                "--fasta", "/tmp/this-file-really-does-not-exist.fasta",
                "--output-dir", "/tmp/out",
            ],
        )


def test_parser_rejects_both_fasta_and_fasta_dir(tiny_fasta: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="chai_mlx_batch_test_") as td:
        tddir = Path(td)
        (tddir / "a.fasta").write_text(">protein|name=A\nMKWV\n")
        with pytest.raises(SystemExit):
            _parse(
                [
                    "--weights-dir", "weights",
                    "--fasta", str(tiny_fasta),
                    "--fasta-dir", str(tddir),
                    "--output-dir", "/tmp/out",
                ],
            )


def test_parser_rejects_missing_both_fasta_flags() -> None:
    with pytest.raises(SystemExit):
        _parse(
            ["--weights-dir", "weights", "--output-dir", "/tmp/out"],
        )


def test_parser_accepts_fasta_dir() -> None:
    with tempfile.TemporaryDirectory(prefix="chai_mlx_batch_test_") as td:
        tddir = Path(td)
        (tddir / "a.fasta").write_text(">protein|name=A\nMKWV\n")
        (tddir / "b.fasta").write_text(">protein|name=B\nMKWV\n")
        args = _parse(
            [
                "--weights-dir", "weights",
                "--fasta-dir", str(tddir),
                "--output-dir", "/tmp/out",
            ],
        )
        assert args.fasta is None
        assert args.fasta_dir == tddir


def test_parser_rejects_empty_fasta_dir() -> None:
    with tempfile.TemporaryDirectory(prefix="chai_mlx_batch_test_") as td:
        # No *.fasta files inside -- should fail.
        with pytest.raises(SystemExit):
            _parse(
                [
                    "--weights-dir", "weights",
                    "--fasta-dir", td,
                    "--output-dir", "/tmp/out",
                ],
            )


def test_parser_accepts_full_surface(tiny_fasta: Path) -> None:
    args = _parse(
        [
            "--weights-dir", "weights",
            "--fasta", str(tiny_fasta),
            "--output-dir", "/tmp/out",
            "--constraint-path", str(tiny_fasta),  # reuse any path that exists
            "--use-msa-server",
            "--use-templates-server",
            "--msa-server-url", "https://example.org/colabfold",
            "--esm-backend", "mlx_cache",
            "--esm-cache-dir", "/tmp/cache",
            "--dtype", "float32",
            "--recycles", "2",
            "--num-steps", "50",
            "--num-samples", "3",
            "--seed", "7",
            "--debug",
            "--save-npz", "/tmp/out/coords.npz",
        ],
    )
    assert args.constraint_path == tiny_fasta
    assert args.use_msa_server is True
    assert args.use_templates_server is True
    assert args.msa_server_url == "https://example.org/colabfold"
    assert args.esm_backend == "mlx_cache"
    assert args.esm_cache_dir == Path("/tmp/cache")
    assert args.dtype == "float32"
    assert args.recycles == 2
    assert args.num_steps == 50
    assert args.num_samples == 3
    assert args.seed == 7
    assert args.debug is True
    assert args.save_npz == Path("/tmp/out/coords.npz")
