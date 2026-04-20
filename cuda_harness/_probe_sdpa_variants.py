"""Compare several SDPA implementations inside the triangle-attention sub-op
to find one that matches CUDA's untiled ``F.scaled_dot_product_attention``
the closest at BF16.

Target: use the CUDA dump's ``round_1.pair_after_tri_mult`` (bit-identical
input) and diff each MLX SDPA variant's ``round_1.pair_after_tri_attn``
against the CUDA reference. We measure rel_norm at BF16 (the interesting
regime; fp32 is already at 1e-6 with any reasonable formulation).

Variants tried:
  v1: mx.fast.scaled_dot_product_attention (current MLX default)
  v2: explicit Q @ K^T / sqrt(d) -> softmax -> @ V, one go in fp32 promotion
  v3: explicit attention but computed in full-precision inside
  v4: kernel-level: chunked over keys (reduce K first, not rows)
  v5: custom tiled-K with Kahan summation in fp32 accumulator

This probe reassembles all weights from the MLX model but rewrites the
triangle-attention forward body for each variant.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chai_mlx import ChaiMLX
from chai_mlx.nn.layers.triangle import TriangleAttention
from chai_mlx.utils import make_additive_mask, sigmoid


def _rel(mlx_out: np.ndarray, cuda_out: np.ndarray) -> dict:
    a = mlx_out.astype(np.float64)
    b = cuda_out.astype(np.float64)
    diff = np.abs(a - b)
    ref_norm = float(np.linalg.norm(b)) or 1.0
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_norm": float(np.linalg.norm(diff) / ref_norm),
    }


def run_variant_v1_default(tri: TriangleAttention, z: mx.array, pair_mask: mx.array) -> mx.array:
    """Unchanged default implementation (for baseline)."""
    return tri(z, pair_mask=pair_mask)


def _tri_attn_core_manual(
    tri: TriangleAttention,
    z: mx.array,
    pair_mask: mx.array,
    *,
    softmax_dtype: mx.Dtype,
    matmul_dtype: mx.Dtype,
    accumulate_in_fp32: bool,
) -> mx.array:
    """Fully-materialised triangle attention (no SDPA fusion), so we can
    control softmax/matmul dtypes precisely.

    This is the same math as ``TriangleAttention.__call__`` but without
    the row-chunking and without ``mx.fast.scaled_dot_product_attention``.
    Every intermediate is named so we can audit it later.
    """
    b, n, _, _ = z.shape
    H, D = tri.num_heads, tri.head_dim
    z_ln = tri.pair_norm(z)

    bias_all = tri.pair2b(z_ln)                            # (b, n, n, 2H)
    bias_start = bias_all[..., :H].transpose(0, 3, 1, 2)   # (b, H, n, n)
    bias_end = bias_all[..., H:].transpose(0, 3, 1, 2)

    pm_bool = pair_mask.astype(mx.bool_) if pair_mask is not None else None
    col_mask_bool = pm_bool.transpose(0, 2, 1) if pm_bool is not None else None

    def _direction(proj_linear: nn.Linear, bias_raw: mx.array, row_mask: mx.array | None, *, transpose: bool) -> mx.array:
        if transpose:
            z_rows = z_ln.transpose(0, 2, 1, 3)
        else:
            z_rows = z_ln
        proj = proj_linear(z_rows).reshape(b, n, n, H, 4, D)
        q = proj[..., 0, :]  # (b, n_row, n_col, H, D)
        k = proj[..., 1, :]
        v = proj[..., 2, :]
        g = proj[..., 3, :]

        # Reshape to (b*n_row, H, n_col, D).
        q = q.transpose(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
        k = k.transpose(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
        v = v.transpose(0, 1, 3, 2, 4).reshape(b * n, H, n, D)

        if accumulate_in_fp32:
            q_mm = q.astype(matmul_dtype)
            k_mm = k.astype(matmul_dtype)
            v_mm = v.astype(matmul_dtype)
        else:
            q_mm, k_mm, v_mm = q, k, v

        # Attention logits in matmul dtype
        logits = (q_mm @ k_mm.transpose(0, 1, 3, 2)) * (D ** -0.5)   # (b*n, H, n, n)

        # bias_raw has shape (b, H, n, n); broadcast to (b*n, H, n, n).
        bias_full = mx.broadcast_to(bias_raw[:, None, :, :, :], (b, n, H, n, n)).reshape(b * n, H, n, n)
        if row_mask is not None:
            # build additive mask from pair_mask rows/cols.
            pm_c = row_mask  # (b, n, n) of bool
            attn_mask = (pm_c[:, :, :, None] & pm_c[:, :, None, :])
            bias_full = bias_full + make_additive_mask(attn_mask.reshape(b * n, 1, n, n), dtype=bias_full.dtype)

        logits = logits + bias_full.astype(logits.dtype)
        weights = mx.softmax(logits.astype(softmax_dtype), axis=-1).astype(v_mm.dtype)
        # (b*n, H, n, n) @ (b*n, H, n, D) -> (b*n, H, n, D)
        out = weights @ v_mm
        out = out.reshape(b, n, H, n, D).transpose(0, 1, 3, 2, 4)  # (b, n, n, H, D)
        return out * sigmoid(g)

    out_s = _direction(tri.pair2qkvg1, bias_start, pm_bool, transpose=False)
    out_e = _direction(tri.pair2qkvg2, bias_end, col_mask_bool, transpose=True)
    from chai_mlx.utils import merge_heads

    combined = mx.concatenate([merge_heads(out_s), merge_heads(out_e)], axis=-1)
    out = tri.linear_out(combined) * tri.out_scalers
    return z + out


def run_variant_v2_manual_bf16(tri: TriangleAttention, z: mx.array, pair_mask: mx.array) -> mx.array:
    return _tri_attn_core_manual(
        tri, z, pair_mask,
        softmax_dtype=mx.bfloat16, matmul_dtype=mx.bfloat16,
        accumulate_in_fp32=False,
    )


def run_variant_v3_manual_fp32_softmax(tri: TriangleAttention, z: mx.array, pair_mask: mx.array) -> mx.array:
    return _tri_attn_core_manual(
        tri, z, pair_mask,
        softmax_dtype=mx.float32, matmul_dtype=mx.bfloat16,
        accumulate_in_fp32=False,
    )


def run_variant_v4_manual_fp32_everything(tri: TriangleAttention, z: mx.array, pair_mask: mx.array) -> mx.array:
    return _tri_attn_core_manual(
        tri, z, pair_mask,
        softmax_dtype=mx.float32, matmul_dtype=mx.float32,
        accumulate_in_fp32=True,
    )


VARIANTS = {
    "v1_default_fused_sdpa": run_variant_v1_default,
    "v2_manual_bf16_everywhere": run_variant_v2_manual_bf16,
    "v3_manual_fp32_softmax_bf16_matmul": run_variant_v3_manual_fp32_softmax,
    "v4_manual_all_fp32_internally": run_variant_v4_manual_fp32_everything,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-dump", type=Path, required=True)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    args = ap.parse_args()

    data = np.load(args.cuda_dump)
    z_np = data[f"round_{args.round}.pair_after_tri_mult"]
    ref_np = data[f"round_{args.round}.pair_after_tri_attn"]
    print(f"  input  max={np.abs(z_np).max():.3f}  |input|={np.linalg.norm(z_np):.3e}")
    print(f"  ref    max={np.abs(ref_np).max():.3f}  |ref|={np.linalg.norm(ref_np):.3e}")

    mx_dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float32
    compute_dtype_str = "reference" if args.dtype == "bf16" else "float32"
    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False, compute_dtype=compute_dtype_str)
    tri: TriangleAttention = model.trunk_module.msa_module.triangular_attention[args.round]

    # Load real mask.
    intermediates = Path("/tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz")
    ndata = np.load(intermediates)
    token_exists = ndata["inputs.batch.token_exists_mask"]
    pair_mask_np = token_exists[..., :, None] & token_exists[..., None, :]
    pair_mask = mx.array(pair_mask_np)

    z = mx.array(z_np).astype(mx_dtype)

    print()
    print(f"dtype={args.dtype}  round={args.round}")
    print(f"{'variant':44s} {'max_abs':>12s} {'mean':>12s} {'rel_norm':>12s}")
    print("-" * 90)
    for name, fn in VARIANTS.items():
        out = fn(tri, z, pair_mask)
        mx.eval(out)
        out_np = np.asarray(out.astype(mx.float32))
        stats = _rel(out_np, ref_np)
        print(f"  {name:42s} {stats['max_abs']:12.4e} {stats['mean_abs']:12.4e} {stats['rel_norm']:12.4e}")


if __name__ == "__main__":
    main()
