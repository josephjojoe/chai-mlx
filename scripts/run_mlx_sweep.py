"""Forwarder to :mod:`chai_mlx.cli.sweep_impl`.

The canonical implementation of the MLX subprocess-per-target sweep
driver lives inside the installed package so ``chai-mlx-sweep`` works
after a plain ``pip install chai-mlx``. This file exists so the
long-documented ``python scripts/run_mlx_sweep.py ...`` invocation
keeps working from a clone without reinstalling.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chai_mlx.cli.sweep_impl import main  # noqa: E402


if __name__ == "__main__":
    main()
