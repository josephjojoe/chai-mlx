"""Smoke test for minimum Python + MLX version floors.

``pyproject.toml`` declares ``python>=3.11`` and ``mlx>=0.16`` as
hard requirements. Without a test, those version floors are never
actually exercised in CI -- a developer on 3.13 / MLX 0.22 can
silently introduce an API usage that breaks on 3.11 / 0.16 and
nobody will notice until a user files a bug.

This test asserts the interpreter and installed MLX satisfy the
declared minimums. It doesn't verify that chai-mlx *runs* on the
minimum versions -- that needs a CI matrix -- but it does mean any
developer running ``pytest -q`` on a below-minimum environment sees
a clean failure pointing at ``pyproject.toml`` instead of a cryptic
downstream import error.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import mlx.core as mx


_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _read_min_python() -> tuple[int, int]:
    text = _PYPROJECT.read_text()
    m = re.search(r'requires-python\s*=\s*">=\s*(\d+)\.(\d+)"', text)
    assert m is not None, (
        "pyproject.toml is missing a requires-python = \">=X.Y\" line"
    )
    return int(m.group(1)), int(m.group(2))


def _read_min_mlx() -> tuple[int, int]:
    text = _PYPROJECT.read_text()
    m = re.search(r'"mlx>=\s*(\d+)\.(\d+)"', text)
    assert m is not None, (
        "pyproject.toml's [project].dependencies is missing an mlx>=X.Y pin"
    )
    return int(m.group(1)), int(m.group(2))


def _parse_version_prefix(v: str) -> tuple[int, int]:
    """Extract the leading ``X.Y`` tuple from a PEP 440-ish version string."""
    m = re.match(r"(\d+)\.(\d+)", v)
    assert m is not None, f"unparseable version string: {v!r}"
    return int(m.group(1)), int(m.group(2))


def test_python_interpreter_meets_declared_minimum() -> None:
    major, minor = _read_min_python()
    current = sys.version_info[:2]
    assert current >= (major, minor), (
        f"This interpreter is Python {current[0]}.{current[1]}, but "
        f"chai-mlx's pyproject.toml declares requires-python "
        f"={major}.{minor}. Either upgrade Python or lower the pin."
    )


def test_mlx_meets_declared_minimum() -> None:
    major_needed, minor_needed = _read_min_mlx()
    # ``mlx.core.__version__`` is the installed wheel version; this
    # lines up with the ``mlx`` name used in [project.dependencies].
    current = _parse_version_prefix(mx.__version__)
    assert current >= (major_needed, minor_needed), (
        f"Installed mlx is {mx.__version__}, but pyproject.toml "
        f"declares mlx>={major_needed}.{minor_needed}. Upgrade mlx or "
        "lower the pin."
    )
