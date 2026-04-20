"""Forwarder to :mod:`chai_mlx.cli.precompute_esm_impl`.

The canonical implementation of the ESM-MLX embedding cache builder
lives inside the installed package so ``chai-mlx-precompute-esm``
works after a plain ``pip install chai-mlx``. This file exists so the
long-documented ``python scripts/precompute_esm_mlx.py ...`` invocation
keeps working from a clone without reinstalling.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# Submodules (chai-lab, esm-mlx) are needed at import time for
# cuda_harness.modal_common and esm_mlx respectively. Match the
# historical shim behaviour so running from a clone continues to
# work even when the submodules are not pip-installed.
for submodule in ("chai-lab", "esm-mlx"):
    p = _REPO_ROOT / submodule
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

from chai_mlx.cli.precompute_esm_impl import main  # noqa: E402


if __name__ == "__main__":
    main()
