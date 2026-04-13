# Kernel and optimization plan

This page is contributor-facing. It describes what is already in the fast path
and which optimization ideas are still worth keeping in mind after the repo
cleanup.

## Fast path (current)

- **Global attention**: `mlx.core.fast.scaled_dot_product_attention` for Pairformer, diffusion transformer, triangle attention, and blocked atom attention when `use_custom_kernel=False`.
- **Diffusion cache** (once per trunk): `z_cond`, 16 per-block pair biases, `blocked_pair_base`, and **`to_atom_cond` output** (projected atom conditioning — not recomputed each of the 398 steps).
- **Triangle multiplication**: chunked over the feature dimension with **chunk size 32** inside `TriangleMultiplication` (`chai_mlx/nn/layers/triangle.py`), with `mx.eval` between chunks to cap peak `n×n` intermediate memory.
- **Custom Metal kernels** (`chai_mlx/nn/kernels/`):
  - fused SwiGLU activation,
  - fused gate + residual,
  - **fused AdaLN** — LayerNorm + conditional affine in one kernel when `use_custom_kernel=True` (`FUSED_ADALN_SOURCE` + `fused_adaln_full`),
  - experimental blocked 32×128 local attention (naive three-pass path; SDPA is usually preferred).

## Why not hand-roll every attention?

MLX’s SDPA is already tiled with fp32 softmax and additive mask support. Best ROI is caching diffusion pair work and only specializing atom-local attention if profiling says so.

## Low-priority next steps

1. **Blocked local attention**: replace the per-element three-pass kernel with a threadgroup/shared-memory 32×128 kernel if profiling ever shows SDPA is the bottleneck. Right now it is not worth prioritizing.
2. **Pair-update / gather fusion** and confidence-head distance batching are still possible, but also not worth prioritizing without fresh profiling evidence.

Full platform-agnostic spec: [`findings/OPTIMIZATIONS.md`](../../findings/OPTIMIZATIONS.md) (repo root).
