# Kernel and optimization notes

## Current fast path

- **Attention**: `mx.fast.scaled_dot_product_attention` for all attention ops.
- **Diffusion cache**: `z_cond`, 16 per-block pair biases, `blocked_pair_base`,
  and `to_atom_cond` output precomputed once per trunk pass.
- **Triangle multiplication**: chunked over feature dim (chunk size 32) with
  `mx.eval` between chunks to cap peak N-by-N intermediate memory.
- **Custom Metal kernels** (`chai_mlx/nn/kernels/`): fused SwiGLU, fused
  gate + residual, fused AdaLN (LN + conditional affine in one kernel with
  fp32 reductions).

## Not worth pursuing

- Hand-rolled attention kernels: MLX SDPA is already tiled with fp32 softmax
  and additive mask support.
- Pair-update / gather fusion and confidence-head distance batching: no
  profiling evidence that these are bottlenecks.
