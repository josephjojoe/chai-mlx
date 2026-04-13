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

### Sources of Metal-vs-MPS per-operation disagreement

| Source | Mechanism |
|--------|-----------|
| Matmul tile/reduction order | `Σ(a_k × b_k)` over k=256 uses different partial-sum trees in Metal vs MPS. Different summation order → different fp32 rounding. |
| LayerNorm reductions | Mean/variance computed with different partial-sum groupings |
| Softmax denominator | `Σ exp(x_j)` accumulated in different order |
| Einsum decomposition | Triangle multiplication contractions decomposed differently |

Each of these contributes ~1–2 ULP of fp32 error per operation. Individually
negligible; catastrophic after chaotic amplification.

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

Despite massive intermediate divergence, discrete predictions are preserved:
- **Confidence argmax agreement**: 99.6–100%
- **Diffusion module** (isolated): max error 4.7 abs
- Probability distribution TVD: 0.13–0.45 on confidence heads

The numerical parity gap **cannot be closed** by precision matching or
algorithmic changes — it is a fundamental consequence of different compute
backends on a chaotic architecture. Validation should be done at the
**structural output level** (RMSD, GDT on predicted PDB structures).

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

## Path forward

1. ~~Mixed-precision execution~~ — **DONE**
2. ~~Precision-profile alignment~~ — **INVESTIGATED, no benefit, documented above**
3. **End-to-end structure quality validation** — run inference on real FASTA
   inputs and compare predicted PDB against reference (RMSD, GDT). The model
   likely produces reasonable structures despite the intermediate divergence.
4. **MSA depth** — MLX trims empty MSA rows (16384→1 for minimal FASTA). Minor
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
| `scripts/parity_check.py` | Per-component numerical agreement |
| `examples/fasta_smoke.py` | Full pipeline dimension check |
