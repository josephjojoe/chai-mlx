# Numerical divergence: why the MLX port can't fold proteins

## The problem in one paragraph

Chai-1's MLX/Metal port is structurally identical to the PyTorch/MPS
reference but produces invalid protein structures. The cause is a single
operation — **bf16 matrix multiplication** — where Metal and MPS accumulate
dot-product reductions in different orders, producing ~2e-3 max error per
matmul. This error is amplified by nonlinearities (SwiGLU, sigmoid gates)
within each transformer block, then compounded across 48 sequential
pairformer blocks and 200 sequential ODE steps. The architecture has no
self-correcting mechanism: errors only accumulate.

## Architecture overview

Chai-1 follows the AlphaFold3 pattern:

```
FASTA → Feature Embedding → Trunk (Pairformer) → Diffusion Loop → Confidence → Ranking
                              48 blocks            200 ODE steps
                              3 recycles            ×2 NFE (Heun)
```

**Trunk (Pairformer)**: 48 sequential blocks, each containing triangle
multiplication (pair-to-pair einsums), triangle attention (SDPA with N×N
pair bias), single-track attention with pair bias, and SwiGLU transitions.
All with residual connections. Processes `[B, N, N, 256]` pair and
`[B, N, 384]` single representations.

**Diffusion module**: 16-block transformer with AdaLayerNorm conditioning,
SDPA with precomputed pair bias, and gated SwiGLU transitions. Parallel
residual (`x + attn_delta + transition_delta`). Runs inside a 200-step EDM
ODE solver (Heun second-order) that iteratively denoises atom coordinates.

**Confidence head**: 4 pairformer blocks. Produces pLDDT, PAE, PDE logits.

## The sole error source: bf16 matmul

Tested every elementary operation with identical bf16 inputs on Metal vs MPS:

| Operation | Max error | |
|-----------|-----------|--|
| Einsum (no reduction), sigmoid | 0.0 | Exact |
| Softmax, LayerNorm, SDPA, SiLU (all fp32) | <5e-7 | Negligible |
| **Matmul (bf16 × bf16)** | **2e-3** | **Root cause** |
| Matmul (bf16 × fp32) | 8e-7 | 2400× better |

Both backends are IEEE-compliant; the difference is summation order in the
dot-product reduction. Metal and MPS use different tiling/accumulation
strategies, producing different rounding at each partial sum.

## How 2e-3 becomes catastrophic

Within one transformer block, the error cascades through the MLP:

```
matmul seed (2e-3) → Linear up-proj (1.6e-2) → SwiGLU (0.25) → Linear down-proj (1.0)
```

The amplification is ~500× per block. SwiGLU (`silu(a) * b`) is the key
amplifier: it stretches errors near steep regions of the SiLU curve, then
the down-projection matmul multiplies already-wrong activations by weights.

## Trunk vs diffusion: different regimes

The trunk and diffusion module are both broken, but in fundamentally
different ways.

### Trunk (48 blocks): saturated chaos

Perturbing the trunk input by 1 bf16 LSB (7.8e-3) and measuring output:

| Blocks | Amplification |
|--------|---------------|
| 1 | ~1,200× |
| 4 | ~61,000× |
| 48 | ~102,000× |

The system **saturates**: whether the per-op error is 2e-3 (bf16×bf16) or
8e-7 (bf16×fp32), the trunk output divergence is identical (~1069 max on
pair). Tested explicitly with `BF16Linear` (fp32 weights) and pure fp32 —
all produce the same trunk error. Precision improvements cannot help the
trunk. This is a property of the 48-block depth.

Despite this, **discrete predictions survive**: confidence argmax agreement
is 99.6–100%. The trunk representations are statistically equivalent (same
distribution, different trajectory) — good enough for one-shot classifiers,
not for the iterative ODE.

### Diffusion (16 blocks × 200 steps): proportional, not saturated

A Lyapunov sweep perturbing denoise inputs across magnitudes:

| Input perturbation | Output mean error | Amplification |
|-------------------|-------------------|---------------|
| 1e-6 | 8.7e-9 | ~0× |
| 1e-4 | 9.1e-2 | 1,152× |
| 1e-2 | 1.1e-1 | 14× |
| 1.0 | 1.7e-1 | 0.2× |

**The diffusion module is NOT saturated.** Output error scales with input
perturbation. This means reducing per-operation matmul error will
proportionally reduce per-step denoise error, potentially pushing the
200-step ODE below its convergence threshold.

Per-op trace through the 16-block diffusion transformer (block 0 → 15):

| Operation | Block 0 | Block 15 | Trend |
|-----------|---------|----------|-------|
| AdaLayerNorm | 0.0 | 0.63 | Growing |
| SDPA (re-synced) | 0.27 | 0.13 | Stable |
| SwiGLU | 0.50 | 3.0 | Growing |
| **Down projection** | **2.0** | **7.0** | **Dominant** |
| Block output | 1.0 | 10.0 | Accumulating |

## What has been tried

| Approach | Result | Why |
|----------|--------|-----|
| Match TorchScript precision (bf16 act × fp32 weight) | No change to trunk error | Trunk is saturated — per-op improvement invisible after 48 blocks |
| Pure fp32 (weights + activations) | No change to trunk error | Same saturation; Metal-vs-MPS kernel diff alone sufficient |
| Feed MPS reference trunk → MLX diffusion | Still broken (58–118 Å spacing) | Diffusion module's own amplification is independently fatal |
| Different ODE solvers (Euler, Heun) | No improvement | Error is per-evaluation bias, not discretization error |
| Stochastic noise injection (gamma > 0) | Worse (87→118 Å) | Per-step error too large for stochastic correction |
| FP32 weights for diffusion only | No change | fp32 weights still go through bf16 activations; same matmul seed |

Key negative result: precision profile changes don't help because the
trunk is saturated and the diffusion module's bf16 activations still
produce the 2e-3 matmul seed regardless of weight precision.

## Why Stable Diffusion works on MLX

Image-domain Stable Diffusion has the same per-op Metal-vs-MPS error
(~2e-3 per bf16 matmul) but produces correct images. The differences:

| | Stable Diffusion | Chai-1 |
|--|-----------------|--------|
| Denoiser architecture | U-Net (~25 layers, skip connections) | 16-block transformer (sequential residual) |
| Per-block amplification | ~1× (bounded) | ~115× (exponential) |
| Per-step output error | ~1e-3 | ~4.0 |
| ODE steps | 20–50 | 200 (×2 for Heun = 400 NFE) |
| Accumulated error | ~50 × 1e-3 ≈ 0.05 | ~200 × 4.0 ≈ 800 |
| Output tolerance | Perceptual (pixels are forgiving) | Geometric (bonds must be ~1.5 Å ± 0.1 Å) |

The U-Net is not a chaotic amplifier. A 1e-3 perturbation in produces
~1e-3 out. Chai-1's sequential transformer architecture amplifies the same
seed error by orders of magnitude before passing it to the next ODE step.

## Proposals

### 1. Kahan-summation Metal matmul kernel

**Priority: highest for pure MLX.** Complexity: high.

Write a custom Metal kernel for `C = A @ B` using Kahan compensated
summation in the dot-product accumulation loop. This doesn't match MPS's
proprietary accumulation order — it's more accurate than both backends.

Target: reduce per-matmul error from 2e-3 to <1e-5. The Lyapunov sweep
shows the diffusion module's output error is proportional to input
perturbation, so a 200× reduction in matmul error should produce a ~200×
reduction in per-step denoise error. Whether this crosses the ODE
convergence threshold is the open question.

Throughput cost: ~2× slower per matmul due to the compensator. Could be
applied selectively to the diffusion module only (512 MB weights, ~80% of
inference time), leaving the trunk on standard matmul.

### 2. Hybrid inference (MPS diffusion loop)

**Priority: highest for working structures now.** Complexity: medium.

Run the trunk and confidence head in MLX, serialize the `DiffusionCache`,
run the 200-step ODE on PyTorch/MPS. The MPS reference produces correct
structures (3.82 Å Cα spacing on 1L2Y).

Open question: whether MPS diffusion can tolerate the numerically-divergent
MLX trunk outputs as conditioning. Confidence argmax is 99.6–100%
correct, suggesting the trunk representations are functionally adequate
despite large raw numerical error. This is the next experiment to run.

### 3. Consistency distillation

**Priority: long-term.** Complexity: very high.

Train a consistency model that maps noisy coordinates to clean structures
in 1–4 steps instead of 200. With so few steps, per-step error of ~4.0
doesn't accumulate catastrophically. Would also make inference ~50–100×
faster.

Requires training infrastructure and data that Chai Discovery hasn't
released. This is the theoretically correct solution but practically out
of reach without significant compute investment.

### 4. Physical constraint projection

**Priority: speculative.** Complexity: medium.

After each denoise step, project atom positions back onto the manifold of
physically valid geometries (enforce Cα spacing ~3.8 Å, bond angles,
clash removal). This doesn't fix the ODE — it keeps the trajectory from
wandering into unrecoverable regions.

Risk: fights the model's learned denoising trajectory. The model was
trained without these constraints and may not respond well to mid-loop
corrections.

### Ruled out

- **Alternative ODE solvers**: the error is per-evaluation network bias,
  not discretization error. Higher-order solvers use more evaluations per
  step, increasing accumulated bias.
- **Stochastic samplers**: tested — gamma > 0 makes structures worse.
- **Precision profile matching**: tested exhaustively — no benefit due to
  trunk saturation.
