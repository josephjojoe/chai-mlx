"""Regression test for packaged console scripts.

The shipped CLIs should run from the installed package alone. This test
simulates a non-editable install by copying only ``chai_mlx/`` into a
temporary directory, importing the entry module there, and invoking
``main() --help``.
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
    from a directory that only contains the copied package.

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
    ],
)
def test_cli_main_runs_without_scripts_dir(entry: str) -> None:
    """The packaged CLIs must work from the installed package alone."""
    result = _run_in_isolated_copy(entry)
    combined = (result.stdout or "") + (result.stderr or "")
    assert result.returncode == 0, (
        f"{entry} --help failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
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
