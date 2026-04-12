# Kernel and optimization plan

## Fast path (current)

- **Global attention**: `mlx.core.fast.scaled_dot_product_attention` for Pairformer, diffusion transformer, triangle attention, and blocked atom attention when `use_custom_kernel=False`.
- **Diffusion cache** (once per trunk): `z_cond`, 16 per-block pair biases, `blocked_pair_base`, and **`to_atom_cond` output** (projected atom conditioning — not recomputed each of the 398 steps).
- **Triangle multiplication**: chunked over the feature dimension with **chunk size 32** inside `TriangleMultiplication` (`layers/triangle.py`), with `mx.eval` between chunks to cap peak `n×n` intermediate memory.
- **Custom Metal kernels** (`kernels/`):
  - fused SwiGLU activation,
  - fused gate + residual,
  - **fused AdaLN** — LayerNorm + conditional affine in one kernel when `use_custom_kernel=True` (`FUSED_ADALN_SOURCE` + `fused_adaln_full`),
  - experimental blocked 32×128 local attention (naive three-pass path; SDPA is usually preferred).

## Why not hand-roll every attention?

MLX’s SDPA is already tiled with fp32 softmax and additive mask support. Best ROI is caching diffusion pair work and only specializing atom-local attention if profiling says so.

## Optional next steps

1. **Blocked local attention**: replace the per-element three-pass kernel with a threadgroup/shared-memory 32×128 kernel (or keep SDPA default). See `findings/OPTIMIZATIONS.md` §4.5.
2. **Pair-update / gather fusion**, confidence-head distance batching, and other items in `findings/OPTIMIZATIONS.md` §6.

Full platform-agnostic spec: [`findings/OPTIMIZATIONS.md`](../../findings/OPTIMIZATIONS.md) (repo root).
