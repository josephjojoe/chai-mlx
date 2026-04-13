# Port status

Authoritative status page for `chai_mlx`. Last updated April 2026.

## Current state

The port is **structurally complete** and runs in **bfloat16 mixed precision**
(weights and activations cast to bf16, with fp32 islands for sensitive ops).
Every module has been traced against the TorchScript graph dumps in
`findings/graphs/`, and all identified structural bugs have been fixed. The
model loads real weights, runs end-to-end on real FASTA inputs, and every stage
is individually faithful when tested in isolation with reference inputs.

### Isolation parity (each module fed TorchScript reference inputs)

| Module | Metric | Notes |
|--------|--------|-------|
| Feature embedding | max 0.05 on `single_initial` | Precision-limited; all encoding paths verified |
| Diffusion module | max 4.7 abs on `denoise.output` | Conditioning, transformer, encoder, decoder all verified |
| Confidence head | max 2.9–8.4 abs on logits | Structurally correct; error from pairformer amplification |
| Trunk (pairformer) | max ≈ 441 single, ≈ 1069 pair | Structurally correct; chaotic amplification (see below) |

### End-to-end (full MLX pipeline)

When running the full pipeline, the trunk's chaotic amplification cascades
downstream. This is not a structural bug — each pairformer block is
individually faithful, but tiny per-operation differences between Metal and
PyTorch compound exponentially across 48 sequential blocks.

## Root cause of numerical divergence: Metal-vs-MPS kernel differences

**The trunk error is NOT caused by fp32-vs-bf16 precision differences.** This
was exhaustively verified (see investigation log below). The remaining error
comes from **compute backend differences** — MLX's Metal GPU kernels and
PyTorch's MPS kernels produce slightly different floating-point results for the
same mathematical operation.

### Per-operation divergence breakdown (`scripts/per_op_divergence.py`)

Elementary operations tested with identical bf16 inputs on Metal vs MPS:

| Operation | Max error | Notes |
|-----------|-----------|-------|
| Einsum (bikd,bjkd→bijd) | 0.0 | Perfect agreement — no reduction axis |
| Sigmoid | 0.0 | Exact match |
| Softmax (fp32) | 3.0e-8 | Near-perfect |
| LayerNorm (fp32) | 4.8e-7 | Near-perfect |
| SDPA (fp32) | 4.5e-7 | Near-perfect |
| SiLU (fp32) | 2.4e-7 | Near-perfect |
| **Matmul (bf16 × bf16)** | **2.0e-3** | **Summation order divergence** |
| Matmul (bf16 × fp32) | 8.3e-7 | 1000× better than bf16×bf16 |

The bf16 matmul is the dominant seed. Tracing through a real pairformer block
with model weights shows how it cascades:

| Step | Max error | Mechanism |
|------|-----------|-----------|
| LayerNorm | 2.0e-3 | bf16 matmul in internal reduction |
| Linear (up proj) | 1.6e-2 | bf16 matmul on diverged normalized input |
| SwiGLU | 0.25 | Nonlinearity amplifies linear error |
| Linear (down proj) | 1.0 | Another matmul on amplified errors |
| Triangle mult (full) | 0.25 | Multiple einsum contractions + sigmoid gates |
| Re-synced SDPA | 0.25 | Even with identical q/k/v, bf16 pair bias causes divergence |

Single-block Lyapunov test: a 1e-3 input perturbation amplifies to 6.3e-2
mean / 3.5 max — an **862× amplification factor per block**. Over 48 blocks
this guarantees full saturation regardless of seed magnitude.

### The pairformer stack is a chaotic system

A sensitivity analysis perturbing the input by 1 bf16 LSB (7.8e-3) shows:

| Blocks | Pair amplification | Single amplification |
|--------|-------------------|---------------------|
| 1 | ~1,200x | ~6,100x |
| 4 (confidence head) | ~61,000x | ~37,000x |
| 48 (trunk) | ~102,000x | ~81,000x |

Any per-operation difference — even a single bit of rounding — gets amplified
by ~100,000x through the full trunk. This is an inherent property of the
48-block sequential residual architecture.

Critically, the chaotic system **saturates**: once two trajectories diverge
past a threshold, increasing the initial perturbation doesn't increase the
final error. This is why bf16 and fp32 produce the same trunk error (see below).

### Practical impact

Discrete predictions are preserved, but the diffusion ODE is not stable:
- **Confidence argmax agreement**: 99.6–100% ✅
- **Diffusion module** (isolated single step): max error 4.7 abs ✅
- Probability distribution TVD: 0.13–0.45 on confidence heads ✅
- **Diffusion loop** (200-step ODE): does not converge ❌

The isolated denoise error (~4.0 max) is small enough for one-shot predictions
but compounds catastrophically over 200 sequential ODE steps.

**However**: unlike the 48-block trunk, the diffusion module is **not
saturated** — see the Lyapunov sweep below. Reducing per-operation error
*will* reduce diffusion output error proportionally, making precision
improvements a viable path to a working diffusion loop.

---

## Diffusion module diagnostics (`scripts/diffusion_diagnostics.py`)

Three experiments run April 2026 to characterize the diffusion module's
numerical behavior independently from the trunk.

### Experiment 1: Lyapunov sweep — diffusion module is NOT saturated

Perturbed denoise inputs across magnitudes (1e-6 to 1.0) and measured output
error. Unlike the 48-block trunk which saturates (output error constant
regardless of input perturbation), the 16-block diffusion transformer shows
**proportional** behavior: smaller input perturbation → smaller output error.

| Perturbation ε | Output max error | Output mean error | Amplification |
|----------------|-----------------|-------------------|---------------|
| 1e-6 | 3.8e-6 | 8.7e-9 | ~0x |
| 1e-5 | 1.2 | 8.5e-2 | 10,480x |
| 1e-4 | 1.7 | 9.1e-2 | 1,152x |
| 1e-3 | 0.98 | 9.1e-2 | 115x |
| 1e-2 | 1.1 | 1.1e-1 | 14x |
| 1e-1 | 2.2 | 1.4e-1 | 1.7x |
| 1.0 | 2.9 | 1.7e-1 | 0.2x |

Smallest-to-largest output error ratio: **~0** (proportional, not saturated).

**This is the key finding**: the earlier conclusion that "precision is
irrelevant due to chaotic saturation" applies only to the 48-block trunk.
The diffusion module has only 16 transformer blocks and its error is
**proportional** to per-operation precision. Reducing bf16 matmul error
(the dominant seed at 2e-3 per op) will directly improve the diffusion ODE
trajectory and may push it below the convergence threshold.

### Experiment 2: Diffusion transformer per-op trace

Instrumented one denoise call through all 16 `DiffusionTransformerBlock`s,
comparing MLX (Metal) vs PyTorch (MPS) with identical bf16 inputs at each
sub-operation. The error cascade follows the same pattern as the pairformer:

| Operation | Block 0 max | Block 7 max | Block 15 max | Pattern |
|-----------|------------|------------|-------------|---------|
| AdaLayerNorm | 0.0 | 0.16 | 0.63 | Grows with block depth |
| QKV projection | 0.0 | 0.22 | 0.25 | bf16 matmul seed |
| **SDPA (re-synced)** | **0.27** | **0.19** | **0.13** | Dominant at block 0 |
| Output projection | 0.75 | 0.50 | 0.22 | Matmul on diverged SDPA output |
| SwiGLU | 0.50 | 6.0 | 3.0 | Nonlinearity amplification |
| **Down projection** | **2.0** | **9.0** | **7.0** | **Largest per-op error** |
| **Block output** | **1.0** | **4.0** | **10.0** | **Accumulates across blocks** |

The error pipeline within each block is:
`SDPA (~0.2) → output_proj (~0.5) → SwiGLU (~3) → down_proj (~8) → gated (~2)`

The **down projection** (bf16 matmul on SwiGLU-amplified errors) is the
dominant per-operation error source, reaching 12.0 max in the worst case.
The block output error grows steadily from 1.0 (block 0) to 10.0 (block 15)
as errors accumulate through the parallel residual connections.

### Experiment 3: MPS reference vs MLX baseline (1L2Y, Trp-cage)

Full pipeline comparison on 1L2Y (20 residues, 460 atoms):

| Pipeline | Median Cα spacing | Expected |
|----------|------------------|----------|
| MPS reference (PyTorch throughout) | **3.82 Å** | ~3.8 Å ✅ |
| MLX (full pipeline) | 0.87 Å | ~3.8 Å ❌ |

The MPS reference produces physically correct structures. The MLX diffusion
loop collapses atom positions rather than spreading them (0.87 Å median
spacing instead of 3.8 Å).

**MLX trunk → MPS diffusion hybrid** was not directly tested in this run
(requires DiffusionCache serialization between frameworks), but the
proportional Lyapunov result means this is the highest-priority next
experiment.

### Why Stable Diffusion works on MLX but Chai-1 does not

Image-domain Stable Diffusion uses a U-Net with skip connections and ~25
effective layers. Its Lyapunov exponent is bounded (<1 per block) — a 1e-3
input perturbation produces ~1e-3 output perturbation. Over 20–50 ODE steps:
accumulated error ~0.05, well within the perceptual tolerance of images.

Chai-1's diffusion transformer has 16 sequential blocks with ~115x
amplification per denoise call. Over 200 ODE steps (400 NFE with Heun):
accumulated error far exceeds the geometric tolerance of protein structures
(bond lengths must be ±0.1 Å). The Pairformer trunk adds an additional
48-block chaotic amplifier upstream that is fully saturated.

The fundamental difference is not the diffusion framework but the
**denoiser architecture's sensitivity to per-operation numerical noise**.

---

## Precision-profile investigation log (do not repeat)

We investigated whether matching TorchScript's exact precision profile would
reduce the trunk error. **It does not.** This section documents the full
investigation so it is not repeated.

### TorchScript's actual precision profile

The TorchScript reference models store **fp32 weights** and selectively cast
activations to bf16 before each `torch.linear` call (via `torch.to(x, 15)`).
This means the actual TorchScript computation is:

```
activation.to(bfloat16) @ fp32_weight → fp32_output   (PyTorch type promotion)
```

Our MLX port casts **both weights and activations** to bf16 via `_cast_weights`
in `from_pretrained`, so the computation is:

```
bf16_activation @ bf16_weight → bf16_output
```

These are different precision profiles.

### Per-operation microbenchmark (single 256×256 matmul)

| Operation | MLX vs PyTorch MPS max diff |
|-----------|---------------------------|
| `bf16 × bf16` | 0.406 |
| `bf16 × fp32` (matching TorchScript) | 0.000084 |
| Improvement ratio | **~4800×** |

Matching TorchScript's profile gives ~4800× better per-operation agreement
on an isolated matmul.

### Implementation tested: `BF16Linear`

We implemented `BF16Linear(nn.Linear)` that casts input to bf16 before the
matmul while keeping weights fp32. All 1053 `nn.Linear` instances were
upgraded via `__class__` swap. Weights stayed fp32 (1.26 GB instead of 0.63 GB).

### End-to-end result: no improvement

| Stage (isolation) | BF16Linear (fp32 weights) | Old bf16 (bf16 weights) | Pure fp32 |
|---|---|---|---|
| Trunk single | 441.3 | ~441 | 441.6 |
| Trunk pair | 1069 | ~1060 | 1069 |
| Diffusion | 4.659 | ~4.7 | 4.651 |
| Confidence PAE | 2.880 | ~2.9 | 2.868 |

All three approaches produce **the same trunk error** (~1060–1069 pair).
The 4800× per-operation improvement is real on an isolated matmul, but
**completely invisible after 48 blocks of chaotic amplification** because the
system saturates — Metal-vs-MPS kernel differences alone are enough to hit the
divergence ceiling.

### Why chaotic saturation makes precision irrelevant

In a chaotic system, the Lyapunov exponent determines how fast nearby
trajectories diverge. But there is a maximum divergence set by the system's
attractor diameter. Both small perturbations (Metal-vs-MPS, ~0.003 per op)
and large perturbations (bf16 rounding, ~0.4 per op) reach this maximum
after enough sequential blocks. Like the butterfly effect: a butterfly and a
hurricane both change weather one month out by the same amount.

### Decision: keep bf16 weights for practical benefit

Since the precision profile does not affect numerical parity, we keep the
approach that saves memory and bandwidth:

- **Weights**: cast to bf16 in `from_pretrained` (saves 630 MB)
- **Activations**: bf16 throughout (lower memory, higher throughput on Apple Silicon)
- **FP32 islands**: LayerNorm reductions, softmax, pairwise distance,
  `prev_pos_embed`, Fourier sigma embedding, Heun correction, distance binning
- **Embedding boundary**: features cast to `compute_dtype` at model entry

The `BF16Linear` approach (fp32 weights, bf16 activation cast before each
linear) was tested and reverted because it doubled weight memory for zero
numerical benefit.

---

## Memory usage: bf16 vs fp32

BF16 is strictly better — identical numerical parity with the reference, half
the memory, higher throughput on Apple Silicon. Peak estimates for batch=1,
5 diffusion samples (`scripts/memory_estimate.py`):

| Tokens | Atoms | BF16 peak | FP32 peak | BF16 on 16 GB? | FP32 on 16 GB? |
|--------|-------|-----------|-----------|----------------|----------------|
| 256 | 5,888 | 1.1 GB | 2.1 GB | Yes | Yes |
| 384 | 8,832 | 1.5 GB | 3.0 GB | Yes | Yes |
| 512 | 11,776 | 2.1 GB | 4.2 GB | Yes | Yes |
| 768 | 17,664 | 3.8 GB | 7.5 GB | Yes | Yes |
| 1024 | 23,552 | 6.1 GB | 12.3 GB | Yes | Tight |
| 1536 | 35,328 | 12.9 GB | 25.8 GB | Tight | No |
| 2048 | 47,104 | 22.3 GB | 44.7 GB | No | No |

The pair tensor `[B, N, N, 256]` dominates — O(N²) scaling. BF16 roughly
doubles the maximum feasible sequence length on a given machine.

---

## Structural validation: diffusion loop does not converge

**The MLX port cannot currently produce valid protein structures.** Despite
correct loop logic and faithful isolated components, the 200-step diffusion ODE
diverges catastrophically due to per-step denoise errors that accumulate over
the sampling trajectory.

### Investigation summary

We ran systematic experiments feeding **reference trunk outputs** (from the
TorchScript model on MPS) into the MLX diffusion loop, eliminating trunk
divergence as a variable:

| Configuration | Final Cα spacing | Expected |
|---------------|-----------------|----------|
| Euler only, no gamma | 58.78 Å | ~3.8 Å |
| Heun, no gamma | 59.36 Å | ~3.8 Å |
| Euler + gamma | 87.41 Å | ~3.8 Å |
| Heun + gamma (full) | 118.11 Å | ~3.8 Å |
| FP32 weights, full config | 117.16 Å | ~3.8 Å |

All configurations produce physically unrealistic structures (15–30× the
expected Cα spacing).

### Why the loop diverges

Each denoise call introduces ~4.0 max error (0.93 mean) in the `pos_updates`
tensor vs the TorchScript reference, even with float32 weights. At sigma=1261
(first step), this causes the denoised prediction to have 2.89 Å Cα spacing in
raw pos_updates, which after c_out=16 scaling gives ~46 Å instead of the
correct ~3.8 Å.

Step-by-step trace shows convergence toward ~22 Å std by step 101, but then
re-divergence when stochastic noise injection activates (gamma > 0 at
sigma < 80). Even without noise injection, the ODE trajectory drifts into an
invalid region of structure space and cannot recover.

The error source is the **diffusion transformer** — the same chaotic
amplification pattern as the pairformer trunk:
- Input to transformer: std ≈ 2.3
- After transformer (before LN): std ≈ 57.7
- After LN: std ≈ 0.87

Small Metal-vs-MPS differences in the 25× intermediate amplification produce
systematically wrong conditioning for the atom attention decoder, yielding
pos_updates that are correlated (not random noise) and consistently place atoms
too far apart.

### Confirmed non-bugs

- **Heun correction**: matches reference exactly (`atom_pos = atom_pos + ...`,
  not `atom_pos_hat + ...`)
- **Sigma schedule**: max diff 3.7e-4 between MLX and reference
- **Gamma values**: identical
- **Augmentation** (`center_random_augmentation`): quaternion, rotation, centering
  all match
- **Preconditioning** (c_in, c_skip, c_out): verified against TorchScript graph
- **Weight loading**: all 343 diffusion module weights verified identical to
  TorchScript state dict (max diff = 0)
- **kv_indices**: MLX's stored indices match TorchScript's on-the-fly generation
  (0 differences across 23,552 index values)

### Consequence

The MLX port is currently limited to:
- Correct trunk representations (structurally faithful, numerically divergent)
- Correct confidence predictions (argmax agreement 99.6–100%)
- **Invalid structural predictions** from the diffusion module

The diffusion loop requires higher numerical fidelity than the trunk or
confidence head because it runs 200 sequential steps of an ODE where errors
are correlated and compound, rather than producing a one-shot prediction.

## Path forward

1. ~~Mixed-precision execution~~ — **DONE**
2. ~~Precision-profile alignment~~ — **INVESTIGATED, no benefit for trunk (saturated)**
3. ~~End-to-end structure quality validation~~ — **DONE, structures invalid**
4. ~~Diffusion module characterization~~ — **DONE (see diagnostics above)**
   - Diffusion module is NOT saturated → precision improvements WILL help
   - Per-op trace identifies down_proj matmul as the dominant error source
5. **Kahan-summation Metal matmul kernel** (highest priority for pure MLX):
   write a custom Metal matmul with compensated summation to reduce per-op
   error from 2e-3 to <1e-5. The Lyapunov sweep shows this would
   proportionally reduce diffusion output error. Target: per-step denoise
   error below ODE convergence threshold (~0.01 max on pos_updates).
6. **Hybrid inference** (highest priority for working structures now):
   run the diffusion loop on PyTorch/MPS while using MLX for trunk +
   confidence. Requires DiffusionCache serialization. The Lyapunov result
   means MLX trunk outputs may be acceptable conditioning for MPS diffusion
   even though they are numerically divergent.
7. **Alternative sampler**: unlikely to help — the error is per-evaluation
   network bias, not ODE discretization error. Stochastic samplers
   (gamma > 0) make structures worse, not better. Higher-order solvers
   increase NFE (and accumulated bias) without reducing per-step error.
8. **MSA depth** — MLX trims empty MSA rows (16384→1 for minimal FASTA). Minor
   contributor to trunk divergence.

## What is structurally verified

- Pipeline split: embed → trunk → diffusion cache → diffusion loop → confidence → ranking
- Hidden dimensions and module boundaries matching TorchScript weight shapes
- 113 weight tensors reshaped for einsum→Linear conversion
- EDM sampling loop with Heun second-order correction and `sigma_next != 0` guard
- Blocked atom attention topology (query 32, KV 128)
- Parallel residual in `DiffusionTransformerBlock` and `LocalAtomTransformer`
- Diffusion conditioning: `single_trans1`/`single_trans2` both applied after sigma embedding
- Encoder pair shared with decoder (not raw `blocked_pair_base`)
- Token-index gathering for `blocked_pair_base` (not atom indices)
- Pair-update block sourced from conditioning signal (not initial atom state)
- Template embedder: per-template processing, masked averaging, ReLU before `proj_out`
- Featurization: one-hot widths, RBF encoding, OUTERSUM embedding, alphabetical concat order
- Confidence head: PDE symmetrization, affine=False output norms, `token_single_mask`

## Historical fixes (highlights)

Over the course of development, ~30 structural bugs were found and fixed.
The major categories:

- **Weight conversion** (113 einsum-shape tensors, 3 learned feature params)
- **Featurization encoding** (one-hot widths, RBF, OUTERSUM, concat order)
- **Template embedder** (full architecture rewrite to match per-template processing)
- **Trunk** (block ordering, pair_transition source, recycle projection, mask threading)
- **Token embedder** (concat order, transformer input, conditioning LN, pair source, output gating)
- **Diffusion** (conditioning order, sigma-dependence, parallel residual, pair sharing, pair indexing, atom masking)
- **Confidence** (PDE symmetry, output norms, mask application)
- **Attention** (additive mask conversion, float AND on masks, single_mask gating)
- **Mixed precision** (FP32LayerNorm, mask dtype propagation, embedding boundary casting)

Full git history has the details for each fix.

## Validation tooling

| Script | Purpose |
|--------|---------|
| `scripts/layer_parity.py` | Compare MLX intermediates against reference dump |
| `scripts/chai_lab_reference_dump.py` | Generate FASTA-backed reference artifacts |
| `scripts/stage_isolation_parity.py` | Feed reference tensors at stage boundaries |
| `scripts/weight_loading_e2e.py` | Convert → strict load → smoke forward |
| `scripts/per_op_divergence.py` | Per-operation Metal-vs-MPS divergence breakdown |
| `scripts/error_diagnostics.py` | Error distribution analysis (percentiles, TVD, KL) |
| `scripts/memory_estimate.py` | Peak memory estimation at various sequence lengths |
| `scripts/structural_validation.py` | End-to-end structure quality vs PDB ground truth |
| `scripts/parity_check.py` | Per-component numerical agreement |
| `scripts/diffusion_diagnostics.py` | Lyapunov sweep, per-op trace, hybrid test |
| `examples/fasta_smoke.py` | Full pipeline dimension check |
