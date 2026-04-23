from __future__ import annotations

import sys
from pathlib import Path

import pytest

from chai_mlx.config import ChaiConfig


# ---------------------------------------------------------------------------
# Make vendored submodules importable without a separate `pip install -e`.
# ---------------------------------------------------------------------------
#
# ``chai-lab/`` ships as a submodule so the MLX featurizer can delegate to
# chai-lab's FASTA + MSA + template pipeline without pulling a second copy
# from PyPI. ``esm-mlx/`` ships as a submodule so the ESM-on-MLX adapter
# can be exercised offline, without an editable install.

_REPO_ROOT = Path(__file__).resolve().parents[1]
for submodule in ("chai-lab", "esm-mlx"):
    path = _REPO_ROOT / submodule
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def cfg() -> ChaiConfig:
    return ChaiConfig()
