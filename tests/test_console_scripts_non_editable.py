"""Regression test for the console-script shims.

The original ``chai_mlx/cli/infer.py`` loaded ``scripts/inference.py``
via ``Path(__file__).resolve().parents[2] / "scripts" / "inference.py"``.
That design breaks under ``pip install chai-mlx`` (non-editable) because
``scripts/`` is not shipped in the wheel. This test simulates that
install mode:

1. Copy ``chai_mlx/`` into a tmp directory (``site-packages``-like).
2. Remove the repo root from ``sys.path`` so ``scripts/`` cannot be
   found on the filesystem.
3. Import ``chai_mlx.cli.infer`` from the tmp copy and call ``main``
   with ``--help`` -- which should cleanly return ``SystemExit(0)``,
   proving the canonical implementation lives inside the package.

If this test regresses, the binaries declared in ``pyproject.toml``
are broken for PyPI users even though they work from an editable
clone.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_in_isolated_copy(entry: str) -> subprocess.CompletedProcess:
    """Run ``python -c 'from <entry> import main; main()' --help``
    from a directory that does NOT contain ``scripts/``.

    Using a subprocess is cheaper and safer than monkey-patching
    ``sys.path`` in-process -- we get a clean interpreter whose only
    chai-mlx install is the copy in the tmp directory.
    """
    tmp = tempfile.mkdtemp(prefix="chai_mlx_non_editable_")
    try:
        shutil.copytree(_REPO_ROOT / "chai_mlx", Path(tmp) / "chai_mlx")
        cmd = [
            sys.executable,
            "-c",
            f"from {entry} import main; main()",
            "--help",
        ]
        return subprocess.run(
            cmd,
            cwd=tmp,
            capture_output=True,
            text=True,
            timeout=30,
            # Clear PYTHONPATH so the interpreter cannot see the repo
            # root; rely only on the tmp copy.
            env={"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp},
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.parametrize(
    "entry",
    [
        "chai_mlx.cli.infer",
        "chai_mlx.cli.precompute_esm_impl",
        "chai_mlx.cli.sweep_impl",
        "chai_mlx.io.weights.export_torchscript",
        "chai_mlx.io.weights.convert_torchscript",
        "chai_mlx.io.weights.convert_npz",
    ],
)
def test_cli_main_runs_without_scripts_dir(entry: str) -> None:
    """``chai-mlx-*`` binaries must work without ``scripts/`` on disk."""
    # Some entry points (notably ``sweep_impl`` / ``precompute_esm_impl``)
    # still import optional helpers before argparse runs. That's fine:
    # what we care about here is that every console script is packaged
    # self-sufficiently and does not error with
    # "Could not locate scripts/...".
    result = _run_in_isolated_copy(entry)
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, (
        f"{entry} --help failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "Could not locate scripts/" not in combined, (
        f"{entry} still tries to load scripts/ by path:\n{combined}"
    )
    # Sanity: --help output should contain the usage line.
    assert "usage:" in combined.lower(), (
        f"{entry} --help did not emit argparse help text"
    )


def test_entry_points_declared_in_pyproject_match_modules() -> None:
    """Keep pyproject's [project.scripts] in sync with the real modules."""
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text()
    # Each declared entry MUST be importable and expose a ``main``.
    expected = {
        "chai-mlx-infer": "chai_mlx.cli.infer",
        "chai-mlx-precompute-esm": "chai_mlx.cli.precompute_esm_impl",
        "chai-mlx-sweep": "chai_mlx.cli.sweep_impl",
        "chai-mlx-export-torchscript": "chai_mlx.io.weights.export_torchscript",
        "chai-mlx-convert-torchscript": "chai_mlx.io.weights.convert_torchscript",
        "chai-mlx-convert-npz": "chai_mlx.io.weights.convert_npz",
    }
    for script_name, module_path in expected.items():
        assert f'{script_name} = "{module_path}:main"' in pyproject, (
            f"pyproject [project.scripts] entry for {script_name!r} "
            f"must be '{module_path}:main'; found otherwise."
        )
        module = importlib.import_module(module_path)
        assert callable(getattr(module, "main", None)), (
            f"{module_path}.main must be callable for the console "
            "script to work."
        )
