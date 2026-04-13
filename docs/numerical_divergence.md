# Numerical divergence: why the MLX port can't fold proteins

## The problem in one paragraph

Chai-1's MLX/Metal port is structurally identical to the PyTorch/MPS
reference but produces invalid protein structures. The root cause is
**Metal's matmul tiling order** differing from MPS's, producing ~2e-3
max error per matmul (both backends use fp32 accumulators internally — the
difference is operand summation order, not precision). This error is
amplified ~500× per transformer block by SwiGLU nonlinearities, then
compounded across 48 sequential pairformer blocks and 200 sequential ODE
steps. Neither weight precision (fp32 vs bf16) nor explicit fp32
accumulation changes the outcome.

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

## The error source: matmul tiling order (not precision)

Tested every elementary operation with identical bf16 inputs on Metal vs MPS:

| Operation | Max error | |
|-----------|-----------|--|
| Einsum (no reduction), sigmoid | 0.0 | Exact |
| Softmax, LayerNorm, SDPA, SiLU (all fp32) | <5e-7 | Negligible |
| **Matmul (bf16 × bf16)** | **2e-3** | **Root cause** |
| Matmul (bf16 × fp32) | 8e-7 | 2400× better (less weight quantization noise) |

**Critical finding**: Metal bf16 matmul already uses fp32 accumulators.
Verified: `bf16 @ bf16` produces bit-identical results to
`fp32(bf16) @ fp32(bf16)` on Metal. The 2e-3 error between Metal and MPS
comes from different **tiling/reduction orderings** — each backend tiles
the matrix into blocks and accumulates partial sums in a different
sequence, producing different fp32 rounding at each step.

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
all produce the same trunk error.

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
perturbation. However, this proportionality is academic — the per-step
error (~4.0 max) doesn't come from precision; it comes from Metal's tiling
order. Reducing precision has no effect (see experiments below).

Per-op trace through the 16-block diffusion transformer (block 0 → 15):

| Operation | Block 0 | Block 15 | Trend |
|-----------|---------|----------|-------|
| AdaLayerNorm | 0.0 | 0.63 | Growing |
| SDPA (re-synced) | 0.27 | 0.13 | Stable |
| SwiGLU | 0.50 | 3.0 | Growing |
| **Down projection** | **2.0** | **7.0** | **Dominant** |
| Block output | 1.0 | 10.0 | Accumulating |

## What has been tried

| Approach | Result | Why it didn't help |
|----------|--------|--------------------|
| FP32 upcast (bf16→fp32 before matmul) | **No-op** — bit-identical | Metal bf16 matmul already uses fp32 accumulators |
| FP32 weights (original precision, no bf16 quantization) | 24.94→24.95 Å (no change) | Weight quantization error is negligible vs tiling order error |
| Match TorchScript precision (bf16 act × fp32 weight) | No change | Same tiling-order divergence regardless of weight precision |
| Pure fp32 (weights + activations) | No change | All precision changes are invisible — the error is structural |
| Feed MPS reference trunk → MLX diffusion | Still broken (58–118 Å) | Diffusion module's own Metal divergence is independently fatal |
| MLX trunk → MPS diffusion (cache analysis) | z_cond 50% rel. error, s_static 11%, pair_biases max 5–22 | Trunk chaotic divergence corrupts conditioning tensors too heavily |
| Different ODE solvers (Euler, Heun) | No improvement | Error is per-evaluation bias, not discretization error |
| Stochastic noise injection (gamma > 0) | Worse (87→118 Å) | Per-step error too large for stochastic correction |

**The core negative result**: every precision intervention (bf16 weights,
fp32 weights, fp32 activations, BF16Linear) produces the same ~4.0
per-step denoise error and the same ~25 Å invalid Cα spacing. The error
is not about the number of bits — it's about Metal's matmul producing
systematically different results from the backend the model was trained on
(CUDA, with MPS as a compatible alternative).

## Why Stable Diffusion works on MLX

Image-domain Stable Diffusion has the same per-op Metal-vs-MPS error
(~2e-3 per bf16 matmul) but produces correct images:

| | Stable Diffusion | Chai-1 |
|--|-----------------|--------|
| Denoiser architecture | U-Net (~25 layers, skip connections) | 16-block transformer (sequential residual) |
| Per-block amplification | ~1× (bounded) | ~115× (exponential) |
| Per-step output error | ~1e-3 | ~4.0 |
| ODE steps | 20–50 | 200 (×2 for Heun = 400 NFE) |
| Accumulated error | ~50 × 1e-3 ≈ 0.05 | ~200 × 4.0 ≈ 800 |
| Output tolerance | Perceptual (pixels are forgiving) | Geometric (bonds must be ~1.5 Å ± 0.1 Å) |

The U-Net is not a chaotic amplifier. Chai-1's sequential transformer
amplifies the same 2e-3 seed error by orders of magnitude before passing
it to the next ODE step.

## Proposals (revised after precision experiments)

### 1. Hybrid inference — MPS diffusion loop

**Priority: downgraded after conditioning divergence measurement.**

Run the trunk and confidence head in MLX, serialize the `DiffusionCache`,
run the 200-step ODE on PyTorch/MPS. The MPS reference produces correct
structures (3.82 Å Cα spacing on 1L2Y).

**Resolved**: the L2 distance measurement on cache conditioning tensors
shows that the trunk's chaotic divergence is **not sufficiently attenuated**
by the cache projection. The diffusion conditioning tensors derived from
MLX trunk outputs vs MPS reference trunk outputs (1L2Y, 256 tokens,
1 recycle) diverge as follows:

| Tensor | L2 | RMS | max | rel. RMS |
|--------|-----|-----|-----|----------|
| `single_trunk` (raw) | 6.15e3 | 19.6 | 456 | 0.264 |
| `pair_trunk` (raw) | 2.13e5 | 52.0 | 1056 | 1.024 |
| `s_static` | 3.81e2 | 1.21 | 17.0 | 0.111 |
| `z_cond` | 2.31e3 | 0.563 | 6.83 | 0.497 |
| `blocked_pair_base` | 2.98e3 | 0.858 | 3.02 | 0.457 |
| `atom_cond` | 0.059 | 6.8e-5 | 0.016 | 0.000 |
| `atom_single_cond` | 87.3 | 0.101 | 0.315 | 0.100 |
| `pair_biases` (16 blocks) | 89–317 | 0.087–0.309 | 5.0–21.8 | ~0 (mask-dominated ref) |

Key findings:

- **`z_cond`** has 50% relative RMS error — the pair conditioning is half
  noise. The projection attenuates the raw trunk pair divergence (102% →
  50%) but not enough.
- **`s_static`** has 11% relative RMS error — the projection attenuates
  trunk single divergence (26% → 11%) but 11% is still significant.
- **`pair_biases`** have max errors of 5–22 in attention logit space.
  Their relative RMS appears near zero only because the ref RMS (~9950)
  is dominated by the additive -inf attention mask. The absolute errors
  are large enough to redirect attention.
- **`atom_cond`** is essentially unaffected (derived from structure
  inputs, not trunk). `atom_single_cond` picks up 10% error from
  trunk single via gather.
- `pair_structure` and `single_structure` have near-zero error (they
  come from embeddings, not the trunk).

**Conclusion**: the hybrid approach (MLX trunk → MPS diffusion) is
unlikely to produce valid structures. The conditioning tensors that feed
the diffusion ODE — especially `z_cond` (50% error) and the per-block
pair biases (up to 22 max error) — are too corrupted by the trunk's
chaotic divergence. Even MPS's correct diffusion math cannot compensate
for conditioning inputs that are this far from the trained distribution.

Script: `scripts/cache_conditioning_divergence.py`.

### 2. Custom Metal matmul kernel matching MPS reduction order

**Priority: speculative but high impact if feasible.**

The error is specifically from Metal's tiling order. If we could determine
MPS's reduction order and replicate it in a custom Metal kernel, the
per-matmul error would drop from 2e-3 to ~0. This is reverse-engineering
Apple's proprietary MPS implementation, which is impractical.

Alternative: Kahan compensated summation reduces sensitivity to reduction
ordering (error becomes independent of order). This would produce a THIRD
numerical trajectory, more accurate than both Metal and MPS but matching
neither. Whether this trajectory converges the ODE is an open question —
it was trained on CUDA's trajectory, and MPS happens to be close enough.

### 3. Consistency distillation

**Priority: long-term.** Complexity: very high.

Train a consistency model that maps noisy coordinates to clean structures
in 1–4 steps instead of 200. With so few steps, per-step error of ~4.0
doesn't accumulate catastrophically.

### 4. Physical constraint projection

**Priority: speculative.** Complexity: medium.

After each denoise step, project atom positions back onto the manifold of
physically valid geometries (enforce Cα spacing ~3.8 Å, bond angles,
clash removal). Risk: fights the model's learned denoising trajectory.

### 5. Alternative ODE solvers

**Partially reconsidered.** An implicit or multistep solver that averages
multiple evaluations at nearby points could reduce bias if the Metal kernel
error has input-dependent structure (different tiling hits different
rounding on different inputs). Worth a quick check once the ODE convergence
threshold is known empirically. Not a fix on its own.

### Ruled out

- **FP32 accumulation / fp32 upcast**: Metal already uses fp32 accumulators
  for bf16 matmul. Explicitly upcasting is a no-op (bit-identical results).
- **FP32 weights**: eliminating bf16 weight quantization has no effect on
  the full 200-step ODE (24.94→24.95 Å, both invalid).
- **Precision profile matching**: all precision variants (bf16×bf16,
  bf16×fp32, fp32×fp32) produce the same ~4.0 per-step denoise error.
- **Stochastic samplers**: tested, makes structures worse.
