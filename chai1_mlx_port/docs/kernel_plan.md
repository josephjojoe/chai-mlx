# Kernel plan

## Default fast path

- Global pair-biased attention stays on `mlx.core.fast.scaled_dot_product_attention`.
- Diffusion-time `z_cond`, per-block pair biases, and blocked atom-pair bases are cached.
- Custom Metal kernels are reserved for:
  - fused SwiGLU activation,
  - fused gate + residual,
  - fused AdaLN post-normalization affine,
  - experimental blocked 32x128 atom-local attention.

## Why not replace every attention with a hand-written kernel?

MLX already ships a fused SDPA implementation with float32 softmax and additive-mask support.
For Chai-1, the best return-on-effort is:

1. keep the large dense attentions on MLX SDPA,
2. avoid recomputing pair conditioning inside the 398 diffusion forwards,
3. specialize only the atom-local 32x128 pattern if profiling shows it is still a hotspot.

## Next profiling targets

1. chunked triangle multiplication,
2. blocked pair gather + pair-update fusion,
3. fused confidence-head representative-atom distance binning,
4. batched confidence scoring across diffusion samples.
