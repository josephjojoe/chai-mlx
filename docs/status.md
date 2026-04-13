# Port status

Authoritative status page for `chai_mlx`. Last updated after the bf16
mixed-precision implementation (April 2026).

## Current state

The port is **structurally complete** and runs in **bfloat16 mixed precision**,
matching the TorchScript reference's precision profile. Every module has been
traced against the TorchScript graph dumps in `findings/graphs/`, and all
identified structural bugs have been fixed. The model loads real weights, runs
end-to-end on real FASTA inputs, and every stage is individually faithful when
tested in isolation with reference inputs.

### Isolation parity (each module fed TorchScript reference inputs)

| Module | Metric | Notes |
|--------|--------|-------|
| Feature embedding | 0.3–0.5% rel | Precision-limited; all encoding paths verified |
| Token embedder | max 0.05 abs on `single_initial` | 5 structural bugs fixed + parallel residual |
| Diffusion module | max 8.4 abs on `denoise.output` | Conditioning, transformer, encoder, decoder all verified |
| Confidence head | max 2.9–8.4 abs on logits; argmax agreement 99.6–100% | Structurally correct; error from pairformer amplification |
| Trunk (pairformer) | max≈1060 on `pair_trunk` | Structurally correct; chaotic amplification (see below) |

### End-to-end (full MLX pipeline)

When running the full pipeline, the trunk's chaotic amplification cascades
downstream. This is not a structural bug — each pairformer block is
individually faithful, but tiny per-operation differences between Metal and
PyTorch compound exponentially across 48 sequential blocks.

## Root cause: chaotic amplification, not precision mismatch

**The trunk error is NOT caused by fp32 vs bf16 precision differences.** This
was empirically confirmed by running the MLX trunk in both bf16 and fp32 against
the same PyTorch bf16 reference:

| Comparison | pair max error |
|---|---|
| MLX bf16 vs PyTorch reference | 1060 |
| MLX fp32 vs PyTorch reference | 1069 |
| MLX bf16 vs MLX fp32 | **36** |

Precision accounts for ~3% of the total error. The remaining ~97% comes from
**compute backend differences** — MLX's Metal GPU kernels and PyTorch's
CPU/MPS kernels produce slightly different floating-point results for the same
operation due to different matmul reduction ordering, different kernel
implementations, etc.

### The pairformer stack is a chaotic system

A sensitivity analysis perturbing the input by 1 bf16 LSB (7.8e-3) shows:

| Blocks | Pair amplification | Single amplification |
|--------|-------------------|---------------------|
| 1 | ~1,200x | ~6,100x |
| 4 (confidence head) | ~61,000x | ~37,000x |
| 48 (trunk) | ~102,000x | ~81,000x |

Any per-operation difference (even a single bit of rounding) gets amplified
by ~100,000x through the full trunk. This is an inherent property of the
48-block sequential residual architecture, not a bug.

### Practical impact

Despite massive intermediate divergence, discrete predictions are preserved:
- **Confidence argmax agreement**: 99.6–100%
- **Diffusion module** (isolated): max error 8.4 abs
- Probability distribution TVD: 0.13–0.45 on confidence heads

The numerical parity gap **cannot be closed** by precision matching or
algorithmic changes — it is a fundamental consequence of different compute
backends on a chaotic architecture. Validation must be done at the **structural
output level** (RMSD, GDT on predicted PDB structures).

## Mixed precision (bf16) implementation

The MLX port now matches TorchScript's precision profile:
- **Weights and activations**: bfloat16 (`compute_dtype` field in `ChaiConfig`)
- **FP32 islands**: LayerNorm reductions (`FP32LayerNorm`), softmax, pairwise
  distance, `prev_pos_embed`, Fourier sigma embedding, Heun correction
  arithmetic, distance binning (confidence head)
- **Boundary casting**: Embedding outputs cast to `compute_dtype` at model entry;
  masks cast to tensor dtype at multiplication sites
- Weight casting happens in `from_pretrained` via `_cast_weights`

## Path forward

1. ~~Mixed-precision execution~~ — **DONE**
2. **End-to-end structure quality validation** — run inference on real FASTA
   inputs and compare predicted PDB against reference (RMSD, GDT). The model
   likely produces reasonable structures despite the intermediate divergence.
3. **MSA depth** — MLX trims empty MSA rows (16384→1 for minimal FASTA). Minor
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
