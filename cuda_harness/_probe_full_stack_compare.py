"""Compare MLX vs CUDA (eager fp32) 48-block pairformer stack outputs.

Reads the per-block NPZ dumps produced by ``_probe_full_stack_mlx.py`` and
``_probe_full_stack_cuda.py`` and prints a compact per-block drift table:
rel_err and max_abs for ``(s_block_i, z_block_i)`` at each of i=0..47.

The goal is to see whether drift grows linearly, quadratically, or saturates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_DIR = Path("/tmp/chai_mlx_cuda/full_stack_probe")


def _rel(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = np.abs(a - b)
    ref_range = float(b.max() - b.min())
    if ref_range == 0:
        return float(diff.max()), float(diff.mean()), float("inf")
    return float(diff.max()), float(diff.mean()), float(diff.max() / ref_range)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx", default=DEFAULT_DIR / "mlx_out_fp32.npz", type=Path)
    ap.add_argument("--cuda", default=DEFAULT_DIR / "cuda_out_fp32.npz", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if not args.mlx.is_file():
        raise SystemExit(f"MLX NPZ not found: {args.mlx}")
    if not args.cuda.is_file():
        raise SystemExit(f"CUDA NPZ not found: {args.cuda}")

    mlx = np.load(args.mlx)
    cud = np.load(args.cuda)

    rows = []
    print(
        f"{'block':6s} | {'z max_abs':>10s} {'z mean':>10s} {'z rel':>10s} "
        f"| {'s max_abs':>10s} {'s mean':>10s} {'s rel':>10s}"
    )
    print("-" * 82)
    for i in range(48):
        z_key = f"z_block_{i:02d}"
        s_key = f"s_block_{i:02d}"
        if z_key not in mlx or z_key not in cud or s_key not in mlx or s_key not in cud:
            print(f"block {i:02d}: missing")
            continue
        z_max, z_mean, z_rel = _rel(mlx[z_key], cud[z_key])
        s_max, s_mean, s_rel = _rel(mlx[s_key], cud[s_key])
        rows.append({"block": i, "z_max": z_max, "z_mean": z_mean, "z_rel": z_rel, "s_max": s_max, "s_mean": s_mean, "s_rel": s_rel})
        print(
            f"{i:6d} | {z_max:10.4e} {z_mean:10.4e} {z_rel:10.4e} "
            f"| {s_max:10.4e} {s_mean:10.4e} {s_rel:10.4e}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
