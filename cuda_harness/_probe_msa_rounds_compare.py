"""Compare MLX vs eager-CUDA MSA-module per-round intermediates.

Reads per-round dumps produced by:
    /tmp/chai_mlx_cuda/msa_module_probe/mlx_rounds_{dtype}.npz
    /tmp/chai_mlx_cuda/msa_module_probe/cuda_post_msa_{dtype}.npz

Both sides have identical key naming, so this script walks every key that
exists on both sides and prints a drift table (max_abs_diff, rel_range,
|diff|/|ref|). The output is the direct MLX-vs-eager-CUDA per-sub-op
attribution we need to localise the drift source inside ``msa_module``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_DIR = Path("/tmp/chai_mlx_cuda/msa_module_probe")


def _rel(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    diff = np.abs(a - b)
    ref_range = float(b.max() - b.min()) or 1.0
    ref_norm = float(np.linalg.norm(b)) or 1.0
    return (
        float(diff.max()),
        float(diff.mean()),
        float(diff.max() / ref_range),
        float(np.linalg.norm(diff) / ref_norm),
    )


# Ordered list of keys we want to show, matching the MSA module's
# sequential structure. Keys not present in either file are skipped.
_ORDERED_KEYS = [
    "post_proj_pair",
    "post_template_pair",
    "msa_init",
    "after_linear_s2m_msa",
    *[
        f"round_{i}.{k}"
        for i in range(4)
        for k in [
            "opm_delta_pair",
            "pair_after_opm",
            "msa_after_transition",
            "msa_after_pw",
            "pair_trans_out",
            "pair_after_tri_mult",
            "pair_after_tri_attn",
        ]
    ],
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--dir", default=DEFAULT_DIR, type=Path)
    args = ap.parse_args()

    mlx_path = args.dir / f"mlx_rounds_{args.dtype}.npz"
    cuda_path = args.dir / f"cuda_post_msa_{args.dtype}.npz"

    mlx = np.load(mlx_path)
    cuda = np.load(cuda_path)

    mlx_keys = set(mlx.files)
    cuda_keys = set(cuda.files)

    print(f"MLX keys: {len(mlx_keys)}, CUDA keys: {len(cuda_keys)}")
    both = mlx_keys & cuda_keys
    only_mlx = mlx_keys - cuda_keys
    only_cuda = cuda_keys - mlx_keys
    print(f"common: {len(both)}, only_mlx: {len(only_mlx)}, only_cuda: {len(only_cuda)}")
    if only_mlx:
        print(f"  only_mlx example: {sorted(only_mlx)[:5]}")
    if only_cuda:
        print(f"  only_cuda example: {sorted(only_cuda)[:5]}")

    print()
    print(f"=== MLX vs eager-CUDA msa_module sub-op drift [{args.dtype}] ===")
    print(
        f"{'key':42s} {'|cuda|':>10s} {'max_diff':>10s} {'mean':>10s}"
        f" {'rel_range':>10s} {'|d|/|c|':>10s}"
    )
    print("-" * 100)
    for k in _ORDERED_KEYS:
        if k not in both:
            continue
        max_abs, mean_abs, rel_rng, rel_norm = _rel(mlx[k], cuda[k])
        ref_norm = float(np.linalg.norm(cuda[k]))
        print(
            f"  {k:40s} {ref_norm:10.3e} {max_abs:10.4e} {mean_abs:10.4e}"
            f" {rel_rng:10.4e} {rel_norm:10.4e}"
        )


if __name__ == "__main__":
    main()
