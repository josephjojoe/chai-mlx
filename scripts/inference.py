"""Forwarder to :mod:`chai_mlx.cli.infer`.

The canonical implementation of the FASTA→CIF runner lives inside the
installed package (``chai_mlx.cli.infer``) so the ``chai-mlx-infer``
console script works after a plain ``pip install chai-mlx``, not just
after an editable install from a clone. This file exists so the
longstanding ``python scripts/inference.py ...`` clone workflow still
works identically.

No logic lives here; see :mod:`chai_mlx.cli.infer` for the actual code.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support running this file directly out of a clone that hasn't been
# pip-installed. When invoked as ``python scripts/inference.py`` the
# interpreter's ``sys.path[0]`` is the ``scripts/`` directory, which is
# not enough to import ``chai_mlx``; add the repo root ourselves.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chai_mlx.cli.infer import main  # noqa: E402


if __name__ == "__main__":
    main()
