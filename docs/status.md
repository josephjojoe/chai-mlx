# Port status

Last updated April 2026. For the numerical divergence analysis and
proposals, see [`numerical_divergence.md`](./numerical_divergence.md).

## Current state

The port is **structurally complete** and runs in bfloat16 mixed precision.
Every module has been traced against the TorchScript graph dumps in
`findings/graphs/`. The model loads real weights and runs end-to-end on real
FASTA inputs.

**The port cannot produce valid protein structures** because per-operation
numerical differences between MLX/Metal and PyTorch/MPS compound through the
deep sequential architectures (48-block trunk, 16-block diffusion
transformer, 200-step ODE).  Discrete predictions (confidence argmax) are
99.6–100% correct.

## Isolation parity

Each module fed identical TorchScript reference inputs:

| Module | Max error | Notes |
|--------|-----------|-------|
| Feature embedding | 0.05 | Precision-limited; all paths verified |
| Diffusion module (single step) | 4.7 | Transformer amplification |
| Confidence head | 2.9–8.4 | Structurally correct |
| Trunk (48-block pairformer) | 441 single, 1069 pair | Chaotic amplification (saturated) |

## Memory (bf16, batch=1, 5 diffusion samples)

| Tokens | BF16 peak | FP32 peak | Fits 16 GB? |
|--------|-----------|-----------|-------------|
| 256 | 1.1 GB | 2.1 GB | Yes |
| 512 | 2.1 GB | 4.2 GB | Yes |
| 1024 | 6.1 GB | 12.3 GB | Yes |
| 1536 | 12.9 GB | 25.8 GB | Tight |
| 2048 | 22.3 GB | 44.7 GB | No |

## Structural verification checklist

- Pipeline split: embed → trunk → cache → diffusion loop → confidence → ranking
- 113 weight tensors reshaped for einsum→Linear conversion
- EDM sampling loop with Heun 2nd-order correction
- Blocked atom attention (query 32, KV 128)
- Parallel residual in DiffusionTransformerBlock and LocalAtomTransformer
- Diffusion conditioning order and sigma-dependence
- Template embedder: per-template processing, masked averaging, ReLU before proj_out
- Featurization: one-hot widths, RBF, OUTERSUM, alphabetical concat order
- Confidence head: PDE symmetrization, affine=False output norms

## Historical fixes

~30 structural bugs found and fixed across: weight conversion (113 tensors),
featurization encoding, template embedder, trunk block ordering, token
embedder, diffusion module, confidence head, attention masks, mixed
precision.  Full git history has details.

## Validation scripts

| Script | Purpose |
|--------|---------|
| `scripts/layer_parity.py` | MLX intermediates vs reference dump |
| `scripts/stage_isolation_parity.py` | Reference tensors at stage boundaries |
| `scripts/per_op_divergence.py` | Per-operation Metal-vs-MPS divergence |
| `scripts/diffusion_diagnostics.py` | Lyapunov sweep, per-op trace, hybrid test |
| `scripts/structural_validation.py` | End-to-end structure quality vs PDB |
| `scripts/memory_estimate.py` | Peak memory at various sequence lengths |
| `examples/fasta_smoke.py` | Full pipeline dimension check |
