# Kernel and optimization notes

## Current fast path

- **Attention**: `mx.fast.scaled_dot_product_attention` for all attention ops.
- **Diffusion cache**: `z_cond`, 16 per-block pair biases, `blocked_pair_base`,
  and `to_atom_cond` output precomputed once per trunk pass.
- **Triangle multiplication**: chunked over feature dim (chunk size 32) with
  `mx.eval` between chunks to cap peak N-by-N intermediate memory.
- **Activations**: native `mx.sigmoid` / `mlx.nn.silu`. These are themselves
  fused Metal kernels tuned by the MLX team.

## Not worth pursuing

- **Hand-rolled fused Metal kernels for SwiGLU / AdaLN / gate+residual.**
  An earlier iteration shipped custom kernels under `chai_mlx/nn/kernels/`
  behind a `use_kernel=True` flag. Microbenchmarks consistently showed
  parity-or-slower vs the composed MLX ops at every realistic shape, which
  is unsurprising — MLX's built-in ops are already fused and threadgroup-tuned,
  and reimplementing them naively throws that away. The code was removed
  (see git history) to cut a maintenance liability.
- **Hand-rolled attention kernels**: MLX SDPA is already tiled with fp32
  softmax and additive-mask support.
- **Pair-update / gather fusion and confidence-head distance batching**:
  no profiling evidence that these are bottlenecks.
