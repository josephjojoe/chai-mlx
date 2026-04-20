"""User-facing command-line entry points.

The three ``chai-mlx-*`` console scripts declared in ``pyproject.toml``
point at the ``main`` functions in this sub-package:

* :mod:`chai_mlx.cli.infer` — ``chai-mlx-infer`` (FASTA → CIFs).
* :mod:`chai_mlx.cli.precompute_esm_impl` — ``chai-mlx-precompute-esm``
  (ESM-MLX embedding cache).
* :mod:`chai_mlx.cli.sweep_impl` — ``chai-mlx-sweep`` (subprocess-per-
  target MLX sweep driver).

The ``scripts/*.py`` files are thin forwarders to these modules so
both ``python scripts/inference.py`` (from a clone) and
``chai-mlx-infer`` (after ``pip install``) exercise exactly the same
code path.  The canonical implementation lives here so the binaries
work after a non-editable install -- the ``scripts/`` directory is
not shipped inside the wheel.
"""
