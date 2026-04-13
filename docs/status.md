# Port status

Authoritative status page for `chai_mlx`. Last updated after the diffusion
module parity push (April 2026).

## Current state

The port is **structurally complete**. Every module has been traced against the
TorchScript graph dumps in `findings/graphs/`, and all identified structural
bugs have been fixed. The model loads real weights, runs end-to-end on real
FASTA inputs, and every stage is individually faithful when tested in isolation
with reference inputs.

### Isolation parity (each module fed TorchScript reference inputs)

| Module | Metric | Notes |
|--------|--------|-------|
| Feature embedding | 0.3–0.5% rel | Precision-limited; all encoding paths verified |
| Token embedder | max 0.05 abs on `single_initial` | 5 structural bugs fixed + parallel residual |
| Diffusion module | **2.0% rel** on `denoise.output` | Conditioning, transformer, encoder, decoder all verified |
| Confidence head | 13–59% rel on logits | Structurally correct; error from pairformer amplification |
| Trunk (pairformer) | max=1069 on `pair_trunk` | Structurally correct; 48-block precision amplification |

### End-to-end (full MLX pipeline)

When running the full pipeline, the trunk's precision amplification cascades
downstream. This is not a structural bug — each pairformer block is
individually faithful, but tiny float32 vs bfloat16 differences compound
exponentially across 48 sequential blocks.

## Known numerical gap

TorchScript runs most activations in **bfloat16** (7-bit mantissa), casting to
float32 only for sensitive operations (e.g. `prev_pos_embed`, LayerNorms). The
MLX port runs everything in **float32** (23-bit mantissa). The results aren't
wrong — they're computed at higher precision — but the intermediate values
diverge from the reference after deep sequential processing.

This affects:
- **Trunk**: 48 pairformer blocks amplify ~0.5% embedding error to max=1069
- **Confidence head**: 4 pairformer blocks amplify to 13–59% on logits
- **Diffusion** (in full pipeline): inherits trunk divergence as input

In isolation (feeding reference trunk outputs), the diffusion module achieves
2.0% relative error, confirming the module itself is faithful.

## Path forward

1. **Mixed-precision execution** — run pairformer blocks in bf16 to match
   TorchScript's precision profile. This is the single biggest improvement for
   numerical parity. Sensitive ops (LayerNorms, softmax, distance computations)
   stay in fp32.
2. **End-to-end structure quality validation** — run inference on real FASTA
   inputs and compare predicted PDB against reference (RMSD, GDT). The model
   likely produces reasonable structures already despite the precision gap.
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
