"""Argparse-level smoke tests for ``scripts/inference.py``.

These tests exercise the CLI surface without actually running inference:
they load the module, invoke ``_parse_args`` with crafted ``sys.argv``
values, and assert that mutually-exclusive / required-combination flags
fail cleanly via ``argparse``'s ``SystemExit(2)`` path.

Running the real end-to-end inference in pytest would require the
``[featurize]`` extra + weights on disk + ~10 s of wall clock per sample;
we already cover that path via the manual smoke tests documented in the
README and the nightly sweep. These tests only guard the shape of the
CLI so regressions to the ``--use-msa-server`` / ``--esm-backend`` /
``--use-templates-server`` / ``--constraint-path`` surface are caught
offline in milliseconds.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_INFERENCE_PY = _REPO_ROOT / "scripts" / "inference.py"


@pytest.fixture(scope="module")
def inference_module():
    """Load ``scripts/inference.py`` as an importable module."""
    spec = importlib.util.spec_from_file_location("chai_mlx_scripts_inference", _INFERENCE_PY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiny_fasta() -> Path:
    with tempfile.TemporaryDirectory(prefix="chai_mlx_cli_test_") as tmpdir:
        fasta = Path(tmpdir) / "protein.fasta"
        fasta.write_text(">protein|name=T\nMKWV\n")
        yield fasta


def _run_parser(module, argv: list[str]) -> "object":
    """Invoke ``_parse_args`` with a given ``argv`` and return the Namespace."""
    saved = sys.argv
    try:
        sys.argv = ["inference.py", *argv]
        return module._parse_args()
    finally:
        sys.argv = saved


def test_parser_accepts_minimal_args(inference_module, tiny_fasta: Path) -> None:
    args = _run_parser(
        inference_module,
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


def test_parser_rejects_mlx_cache_without_dir(inference_module, tiny_fasta: Path) -> None:
    with pytest.raises(SystemExit):
        _run_parser(
            inference_module,
            [
                "--weights-dir", "weights",
                "--fasta", str(tiny_fasta),
                "--output-dir", "/tmp/out",
                "--esm-backend", "mlx_cache",
            ],
        )


def test_parser_rejects_msa_server_plus_msa_directory(inference_module, tiny_fasta: Path) -> None:
    with pytest.raises(SystemExit):
        _run_parser(
            inference_module,
            [
                "--weights-dir", "weights",
                "--fasta", str(tiny_fasta),
                "--output-dir", "/tmp/out",
                "--use-msa-server",
                "--msa-directory", "/tmp/does-not-exist",
            ],
        )


def test_parser_rejects_templates_server_without_msa_server(inference_module, tiny_fasta: Path) -> None:
    with pytest.raises(SystemExit):
        _run_parser(
            inference_module,
            [
                "--weights-dir", "weights",
                "--fasta", str(tiny_fasta),
                "--output-dir", "/tmp/out",
                "--use-templates-server",
            ],
        )


def test_parser_rejects_missing_fasta(inference_module) -> None:
    with pytest.raises(SystemExit):
        _run_parser(
            inference_module,
            [
                "--weights-dir", "weights",
                "--fasta", "/tmp/this-file-really-does-not-exist.fasta",
                "--output-dir", "/tmp/out",
            ],
        )


def test_parser_accepts_full_surface(inference_module, tiny_fasta: Path) -> None:
    args = _run_parser(
        inference_module,
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
