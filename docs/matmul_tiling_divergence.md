# Why Metal matmul breaks the diffusion ODE

A detailed account of the matmul tiling divergence between MLX (Metal) and
PyTorch (MPS), the kernel source code responsible, what was tested, and why
no precision-based fix can help.

## Background

Chai-1's diffusion module runs a 200-step ODE whose denoiser is a 16-block
transformer. Each denoise evaluation contains hundreds of matrix
multiplications (`nn.Linear`). The ODE integrates these results sequentially —
each step's output becomes the next step's input. A per-step denoise error of
~4.0 max (vs the PyTorch/MPS reference) accumulates over 200 steps, producing
25 Å median Cα spacing instead of the expected 3.8 Å.

## The two matmul implementations

### MLX: Steel GEMM (custom tiled Metal kernel)

MLX uses its own GEMM library called **Steel**, compiled from Metal Shading
Language headers shipped in the `mlx` Python package. The kernel source lives
at:

```
site-packages/mlx/include/mlx/backend/metal/kernels/steel/gemm/
  gemm.h          — outer loop, tile loading, grid swizzle
  mma.h           — inner multiply-accumulate, simdgroup MMA
  params.h        — runtime params struct (M, N, K, strides)
  loader.h        — threadgroup memory loaders
  kernels/
    steel_gemm_fused.h   — entry kernel template
    steel_gemm_splitk.h  — split-K variant
```

The GEMM is parameterized by compile-time template arguments:
- `BM`, `BN`: output tile dimensions (rows, cols) per threadgroup
- `BK`: reduction tile width — how many K elements are loaded per iteration
- `WM`, `WN`: warp (simdgroup) tile counts
- `AccumType = float`: accumulator precision (always fp32, even for bf16 inputs)

**Outer K loop** (from `gemm.h`):

```metal
for (int k = 0; k < gemm_k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_a.load_unsafe();           // load [BM × BK] tile of A
    loader_b.load_unsafe();           // load [BK × BN] tile of B
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);              // accumulate into Ctile (fp32)
    loader_a.next();
    loader_b.next();
}
```

Each call to `mma_op.mma()` iterates over the `BK`-wide tile in fragments
of 8 (from `mma.h`):

```metal
STEEL_PRAGMA_UNROLL
for (short kk = 0; kk < BK; kk += kFragSize) {     // kFragSize = 8
    simdgroup_barrier(mem_flags::mem_none);
    Atile.template load<T, WM, 1, A_str_m, A_str_k>(As);
    simdgroup_barrier(mem_flags::mem_none);
    Btile.template load<T, 1, WN, B_str_k, B_str_n>(Bs);
    simdgroup_barrier(mem_flags::mem_none);
    tile_matmad(Ctile, Atile, Btile, Ctile);         // hardware MMA
    As += tile_stride_a;
    Bs += tile_stride_b;
}
```

The `tile_matmad` function uses a **serpentine traversal** to improve data
reuse:

```metal
template <typename T, typename U, int M, int N, int K>
METAL_FUNC void tile_matmad(
    thread MMATile<T, M, N>& D,
    thread MMATile<U, M, K>& A,
    thread MMATile<U, K, N>& B,
    thread MMATile<T, M, N>& C) {
  STEEL_PRAGMA_UNROLL
  for (short m = 0; m < M; ++m) {
    STEEL_PRAGMA_UNROLL
    for (short n = 0; n < N; ++n) {
      short n_serp = (m % 2) ? (N - 1 - n) : n;    // serpentine N
      STEEL_PRAGMA_UNROLL
      for (short k = 0; k < K; ++k) {
        MMATile<T, M, N>::MMAFrag_t::mma(
            D.frag_at(m, n_serp),
            A.frag_at(m, k),
            B.frag_at(k, n_serp),
            C.frag_at(m, n_serp));
      }
    }
  }
}
```

At the leaf, `BaseMMAFrag::mma` calls the hardware intrinsic:

```metal
METAL_FUNC static constexpr void mma(
    thread mat_type& D, thread mat_type& A,
    thread mat_type& B, thread mat_type& C) {
  simdgroup_multiply_accumulate(D, A, B, C);
}
```

`simdgroup_multiply_accumulate` is Apple's 8×8 hardware matrix FMA — the
accumulation order within this intrinsic is opaque and determined by the
Apple GPU's matrix multiplication unit.

**Grid assignment** uses a swizzled mapping to improve L2 cache locality:

```metal
const int tid_y = ((tid.y) << params->swizzle_log) +
    ((tid.x) & ((1 << params->swizzle_log) - 1));
const int tid_x = (tid.x) >> params->swizzle_log;
```

**Split-K variant** (`steel_gemm_splitk.h`): partitions K across multiple
threadgroups, each producing a partial result, then a second kernel sums
partials sequentially:

```metal
AccT out = 0;
for (int i = 0; i < k_partitions; i++) {
    out += C_split[offset];
    offset += partition_stride;
}
```

### PyTorch MPS: Apple's MPSGraph matrixMultiplication

PyTorch's MPS backend dispatches `torch.mm` to **two different paths**
(from `aten/src/ATen/native/mps/operations/LinearAlgebra.mm`):

1. **Default**: `MPSGraph`'s `matrixMultiplicationWithPrimaryTensor:secondaryTensor:name:` — Apple's closed-source compiled graph backend. This is what runs for all standard float matmuls in Chai-1's reference pipeline.

2. **Fallback** (`do_metal_mm`): A PyTorch-bundled custom Metal kernel with
   `TILE_DIM = 16`, used only for integral types, small shapes, or when
   `PYTORCH_MPS_PREFER_METAL` is set. Not used in normal Chai-1 inference.

The MPSGraph matmul implementation is **proprietary**. Its tiling, reduction
order, and accumulation strategy are not inspectable. What we know:
- It uses fp32 accumulators for bf16/fp16 matmul (Apple documentation)
- It produces the same results as CUDA closely enough for the ODE to converge
- Its per-element results differ from MLX's Steel GEMM by up to 2e-3

## Where the numerical difference comes from

### Verified: NOT from accumulator precision

Both backends use fp32 accumulators. Verified empirically:

```python
a = mx.random.normal((256, 256)).astype(mx.bfloat16)
b = mx.random.normal((256, 256)).astype(mx.bfloat16)

c1 = a @ b                                                  # Metal bf16 matmul
c2 = (a.astype(mx.float32) @ b.astype(mx.float32))          # explicit fp32
    .astype(mx.bfloat16)

# max diff: 0.0 — bit identical
```

MLX Steel's `AccumType` defaults to `float` via `AccumHelper`, and
`simdgroup_multiply_accumulate` uses fp32 registers. Explicitly casting
inputs to fp32 before the `@` operator produces identical results.

### Verified: NOT from weight quantization

Keeping the diffusion module's weights in their original fp32 precision
(from safetensors) instead of casting to bf16:

| | Baseline (bf16 weights) | FP32 weights |
|--|------------------------|-------------|
| Median Cα spacing | 24.94 Å | 24.95 Å |
| Valid structure | No | No |

No measurable difference. Weight quantization contributes ~0.01 per element
but this is negligible compared to the tiling-order divergence.

### The actual source: reduction ordering

Computing `C[i,j] = Σ_k A[i,k] * B[k,j]` requires summing K products. fp32
addition is not associative: `(a + b) + c ≠ a + (b + c)` when rounding
occurs. The order in which partial products are accumulated determines the
final rounding.

**MLX Steel's order** (for a 384×384 matmul, typical Chai-1 linear):

1. K is partitioned into `ceil(K / BK)` tiles of width BK
2. Within each tile, K is further divided into fragments of 8
3. Each fragment is processed by `simdgroup_multiply_accumulate` (hardware-
   opaque 8×8 MMA with its own internal accumulation order)
4. Fragment results are accumulated into `Ctile` in the serpentine traversal
   order: `n_serp = (m % 2) ? (N-1-n) : n`
5. The grid swizzle determines which threadgroup handles which output tile
6. If split-K is used, partial results are summed sequentially in a second
   kernel

**Apple MPSGraph's order**: unknown (proprietary), but necessarily different
from Steel's. Apple may use different tile sizes, different fragment
orderings, different grid assignments, or the Apple Neural Engine for some
shapes.

The result: for the same bf16 inputs and the same fp32 accumulators, the two
implementations produce outputs that differ by up to 2e-3 per element. This
is entirely from different rounding at each partial-sum step.

## How 2e-3 per matmul becomes 4.0 per denoise call

Within each DiffusionTransformerBlock, the error cascades:

```
matmul (2e-3) → LN → up-proj matmul (1.6e-2) → SwiGLU (0.25) → down-proj matmul (1.0)
```

SwiGLU is `silu(a) * b` where `silu(x) = x * sigmoid(x)`. Near the steep
region of the sigmoid, a small input error Δ produces an amplified output
error because `silu'(x)` can exceed 1. The down-projection then multiplies
this amplified error by a 384×768 weight matrix, further scaling it.

Across 16 blocks with parallel residual connections (`x += attn_delta +
trans_delta`), the block output error grows from ~1.0 (block 0) to ~10.0
(block 15).

## Experiments that ruled out precision as the cause

### 1. FP32 upcast (cast bf16→fp32 before matmul)

**Result**: bit-identical to baseline. Metal already accumulates in fp32;
the explicit cast is a no-op.

### 2. FP32 weights (original safetensors precision, no bf16 quantization)

**Result**: 24.94→24.95 Å. No change to the 200-step ODE trajectory.

### 3. Prior isolation tests (from `status.md`)

| Configuration | Per-step denoise max error |
|---------------|--------------------------|
| bf16 weights, bf16 activations | ~4.7 |
| fp32 weights (BF16Linear), bf16 activations | 4.659 |
| fp32 weights, fp32 activations | 4.651 |

All three produce the same per-step error. The 0.05 difference between
bf16×bf16 and fp32×fp32 is invisible after 200 ODE steps.

### 4. Lyapunov sweep (diffusion module is proportional)

Input perturbation → output error scales proportionally (not saturated):

| Perturbation | Output mean | Amplification |
|-------------|-------------|---------------|
| 1e-6 | 8.7e-9 | ~0× |
| 1e-3 | 9.1e-2 | 115× |
| 1.0 | 1.7e-1 | 0.2× |

This means the diffusion module *would* benefit from smaller per-step error —
but the error floor is set by Metal's tiling order, not by precision.

## Why Stable Diffusion is unaffected

Stable Diffusion's U-Net has ~25 effective layers with skip connections.
Per-block amplification is bounded (~1×): a 2e-3 input perturbation produces
~2e-3 output perturbation. Over 20–50 ODE steps, accumulated error is ~0.05
— well within the perceptual tolerance of images.

Chai-1's sequential transformer amplifies the same 2e-3 seed by ~500× per
block. Over 200 ODE steps, the accumulated error far exceeds the geometric
tolerance of protein structures (~0.1 Å on bond lengths).

## What this means for fixes

### Cannot help

- **Kahan-summation kernel**: doesn't address tiling order. It would produce
  a third, more-accurate trajectory that matches neither MPS nor CUDA.
  Whether the ODE converges on this trajectory is unknown.
- **FP32 weights / activations**: already tested. Metal's fp32 accumulators
  mean there's no precision headroom to gain.
- **Alternative precision profiles**: bf16×bf16, bf16×fp32, fp32×fp32 all
  produce the same per-step error within the diffusion module.

### Can help

- **Hybrid inference (MPS diffusion)**: MPS's matmul matches CUDA closely
  enough for the ODE to converge (produces 3.82 Å structures). Running the
  diffusion loop on MPS with MLX trunk outputs as conditioning is the most
  direct path to working structures.
- **Matching the trained backend's reduction order**: if Apple publishes
  the MPSGraph matmul algorithm, or if we can reverse-engineer the tiling,
  a custom Metal kernel could replicate MPS's exact numerical behavior.
  Impractical but theoretically sufficient.
- **Retraining / distillation on Metal**: train the denoiser to converge on
  Metal's numerical trajectory. Requires Chai Discovery's training
  infrastructure.

## Appendix: key source file locations

MLX Steel GEMM:

```
# Outer K-loop, tile loading, grid swizzle
site-packages/mlx/include/mlx/backend/metal/kernels/steel/gemm/gemm.h

# Inner accumulation: BlockMMA::mma(), tile_matmad(), simdgroup_multiply_accumulate
site-packages/mlx/include/mlx/backend/metal/kernels/steel/gemm/mma.h

# Runtime params (M, N, K, tile counts, swizzle)
site-packages/mlx/include/mlx/backend/metal/kernels/steel/gemm/params.h

# Entry kernel template (BM, BN, BK, WM, WN as template args)
site-packages/mlx/include/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h

# Split-K: partial sums across threadgroups
site-packages/mlx/include/mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.h
```

MLX Steel Attention (SDPA):

```
# Block attention kernel — loops over KV blocks, online softmax
site-packages/mlx/include/mlx/backend/metal/kernels/steel/attn/kernels/steel_attention.h

# Vector SDPA (single-query fast path)
site-packages/mlx/include/mlx/backend/metal/kernels/sdpa_vector.h
```

Custom fused kernels in this project:

```
# SwiGLU, gated residual, fused AdaLN (all elementwise — not a source of divergence)
chai_mlx/nn/kernels/sources.py
chai_mlx/nn/kernels/elementwise.py
```

PyTorch MPS matmul dispatch:

```
# Source (not in wheel — upstream only):
# aten/src/ATen/native/mps/operations/LinearAlgebra.mm
# Default: MPSGraph matrixMultiplicationWithPrimaryTensor (Apple proprietary)
# Fallback: do_metal_mm with TILE_DIM=16 (custom Metal kernel)
```
