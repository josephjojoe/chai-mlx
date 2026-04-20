"""Isolate triangular_attention (MSA round 1, the biggest BF16 drift site)
and re-run MLX's and eager-CUDA's versions on **bit-identical** CUDA-side
inputs to measure the intrinsic op-level drift without carrying any
upstream noise.

We use the CUDA dump's ``round_1.pair_after_tri_mult`` as the input and
diff MLX's ``TriangleAttention`` output against CUDA's dumped
``round_1.pair_after_tri_attn``.

Additionally, we sweep MLX's ``_ROW_CHUNK`` tile size to see whether a
different tile size brings MLX's output closer to CUDA's untiled output.

Usage::

    python3 cuda_harness/_probe_tri_attn_isolated.py \\
        --cuda-dump /tmp/chai_mlx_cuda/msa_module_probe/cuda_post_msa_bf16.npz \\
        --round 1 \\
        --weights-dir weights
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from chai_mlx import ChaiMLX
from chai_mlx.nn.layers.triangle import TriangleAttention


def _rel(mlx_out: np.ndarray, cuda_out: np.ndarray) -> dict:
    a = mlx_out.astype(np.float64)
    b = cuda_out.astype(np.float64)
    diff = np.abs(a - b)
    ref_range = float(b.max() - b.min()) or 1.0
    ref_norm = float(np.linalg.norm(b)) or 1.0
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p99_abs": float(np.percentile(diff, 99)),
        "rel_range": float(diff.max() / ref_range),
        "rel_norm": float(np.linalg.norm(diff) / ref_norm),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-dump", type=Path, required=True,
                    help="NPZ produced by cuda_harness._probe_msa_module_cuda (--dtype bf16).")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--row-chunks", type=str, default="8,16,32,64,128,256",
                    help="Comma-separated tile sizes for MLX TriangleAttention row chunking.")
    args = ap.parse_args()

    data = np.load(args.cuda_dump)
    key_in = f"round_{args.round}.pair_after_tri_mult"
    key_out = f"round_{args.round}.pair_after_tri_attn"
    if key_in not in data.files or key_out not in data.files:
        print(f"available keys: {sorted(data.files)[:10]} ... {sorted(data.files)[-5:]}")
        raise KeyError(f"missing {key_in} or {key_out}")

    z_np = data[key_in]  # (1, 256, 256, 256)
    ref_out_np = data[key_out]

    print(f"input  {key_in} shape={z_np.shape} max_abs={np.abs(z_np).max():.3f}")
    print(f"output {key_out} shape={ref_out_np.shape} max_abs={np.abs(ref_out_np).max():.3f}")
    print(f"dtype: {args.dtype}")

    mx_dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float32
    compute_dtype_str = "reference" if args.dtype == "bf16" else "float32"

    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False, compute_dtype=compute_dtype_str)
    mm = model.trunk_module.msa_module
    tri_attn: TriangleAttention = mm.triangular_attention[args.round]
    print(f"loaded MLX TriangleAttention (num_heads={tri_attn.num_heads}, head_dim={tri_attn.head_dim}, _ROW_CHUNK={TriangleAttention._ROW_CHUNK})")

    # token_pair_mask for 1L2Y N=256 with 20 live tokens. Load from intermediates.
    # Actually we can make an all-True mask; we're diffing against CUDA's own output,
    # and the CUDA probe uses the same mask construction as this probe would.
    # For correctness we need the real 1L2Y mask — load it:
    intermediates = Path("/tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz")
    ndata = np.load(intermediates)
    token_exists = ndata["inputs.batch.token_exists_mask"]  # (1, 256)
    pair_mask_np = token_exists[..., :, None] & token_exists[..., None, :]
    pair_mask = mx.array(pair_mask_np)

    z = mx.array(z_np).astype(mx_dtype)

    row_chunks = [int(c) for c in args.row_chunks.split(",")]
    print()
    print(f"{'row_chunk':>10s} {'max_abs':>12s} {'mean':>12s} {'p99':>12s} {'rel_range':>12s} {'|d|/|r|':>12s}")
    print("-" * 78)

    best = None
    for chunk_size in row_chunks:
        # Patch the class-level attribute for this experiment.
        original_chunk = TriangleAttention._ROW_CHUNK
        TriangleAttention._ROW_CHUNK = chunk_size
        try:
            out = tri_attn(z, pair_mask=pair_mask)
            mx.eval(out)
            out_np = np.asarray(out.astype(mx.float32))
        finally:
            TriangleAttention._ROW_CHUNK = original_chunk

        stats = _rel(out_np, ref_out_np)
        print(
            f"{chunk_size:10d} {stats['max_abs']:12.4e} {stats['mean_abs']:12.4e}"
            f" {stats['p99_abs']:12.4e} {stats['rel_range']:12.4e} {stats['rel_norm']:12.4e}"
        )
        if best is None or stats["rel_norm"] < best[1]["rel_norm"]:
            best = (chunk_size, stats)

    print()
    print(f"Best tile size: {best[0]} with rel_norm={best[1]['rel_norm']:.4e}")


if __name__ == "__main__":
    main()
