"""Check whether MLX's ``mx.einsum`` at BF16 accumulates in BF16 or FP32.

Materialised test: take a = N(0,1) with shape (1, 256, 256, 256) and
b = N(0,1) with same shape. Compute ``bikd, bjkd -> bijd`` (the triangle-
multiplication outgoing contraction) in three ways:

1. pure fp32 (reference)
2. cast inputs to bf16, use mx.einsum -> the only ambiguity in MLX
3. cast inputs to bf16, upcast each k-step to fp32 and Kahan-sum -> oracle

If MLX's einsum accumulates in bf16, (2) will diverge from (1) by 0.1-1%.
If it accumulates in fp32, (2) will match (1) to ~1e-5 rel_norm.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np


def main() -> None:
    rng = np.random.default_rng(0)
    B, N, D = 1, 256, 256
    a_np = rng.standard_normal((B, N, N, D), dtype=np.float32)
    b_np = rng.standard_normal((B, N, N, D), dtype=np.float32)

    a32 = mx.array(a_np)
    b32 = mx.array(b_np)
    a16 = a32.astype(mx.bfloat16)
    b16 = b32.astype(mx.bfloat16)

    # (1) fp32 reference
    out_fp32 = mx.einsum("bikd,bjkd->bijd", a32, b32)
    mx.eval(out_fp32)

    # (2) bf16 einsum (MLX's default accumulation policy)
    out_bf16 = mx.einsum("bikd,bjkd->bijd", a16, b16)
    mx.eval(out_bf16)

    # (3) bf16 inputs upcast to fp32 before einsum -> fp32 accumulator guaranteed
    out_bf16_to_fp32 = mx.einsum(
        "bikd,bjkd->bijd", a16.astype(mx.float32), b16.astype(mx.float32)
    )
    mx.eval(out_bf16_to_fp32)

    # Reference from numpy (higher precision)
    out_np = np.einsum("bikd,bjkd->bijd", a_np, b_np)

    def stats(label: str, arr: mx.array):
        cand = np.asarray(arr.astype(mx.float32))
        diff = np.abs(cand.astype(np.float64) - out_np.astype(np.float64))
        ref_norm = float(np.linalg.norm(out_np))
        print(f"  {label:42s} max_abs={diff.max():.4e}  mean={diff.mean():.4e}  rel_norm={np.linalg.norm(diff)/ref_norm:.4e}")

    print("Reference output |np.einsum| =", float(np.linalg.norm(out_np)))
    stats("mx.einsum(fp32)", out_fp32)
    stats("mx.einsum(bf16 -> ?)", out_bf16)
    stats("mx.einsum(bf16->fp32 upcast)", out_bf16_to_fp32)

    # Show MLX output dtype.
    print(f"  dtype of mx.einsum(bf16,bf16): {out_bf16.dtype}")
    print(f"  dtype of mx.einsum(fp32,fp32): {out_fp32.dtype}")


if __name__ == "__main__":
    main()
