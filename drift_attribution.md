# MLX vs CUDA drift attribution

This document used to contain a snapshot of historical MLX-vs-CUDA drift
tables and structural comparison results. Those historical numbers have
been intentionally removed because the port, precision policy, and
published defaults have changed enough that the old results should not be
treated as current.

## What this file is for now

Use this page as a checklist for rerunning the attribution work and for
keeping track of which probes answer which question.

## Questions to rerun

1. How closely does the current MLX port track the scripted CUDA
   reference at the embedding, trunk, diffusion, and confidence stages?
2. How much of the remaining gap comes from the trunk versus the
   diffusion loop?
3. How much of the discrepancy is backend accumulation order versus a
   port-level wiring bug?
4. What changes, if any, appear when running `reference` versus
   `float32` on the MLX side?
5. Which target classes in `DEFAULT_TARGETS` still need refreshed
   structural comparisons?

## Probe inventory

The following probes and reports are the main pieces of the attribution
workflow:

- `scripts/cuda_parity.py`:
  Stage-by-stage MLX-vs-CUDA numerical comparison using captured CUDA
  intermediates.
- `scripts/cuda_mlx_diffusion_isolation.py`:
  Separates trunk-conditioned drift from diffusion-loop drift by mixing
  MLX and CUDA trunk outputs.
- `scripts/cuda_error_accumulation.py`:
  Summarises how differences evolve across trunk and diffusion stages.
- `scripts/cuda_structure_sweep.py`:
  Structure-level MLX-vs-CUDA comparison across the target matrix.
- `scripts/cuda_determinism_report.py`:
  Checks CUDA run-to-run variability so MLX-vs-CUDA differences can be
  interpreted in context.
- `cuda_harness/_probe_first_block_mlx.py` and
  `cuda_harness/_probe_first_block_cuda.py`:
  Single-block Pairformer comparison on matched inputs.
- `cuda_harness/_probe_msa_rounds_mlx.py`,
  `cuda_harness/_probe_msa_module_cuda.py`, and
  `cuda_harness/_probe_msa_rounds_compare.py`:
  Round-by-round MSA module attribution.
- `cuda_harness/_probe_tri_attn_isolated.py` and
  `cuda_harness/_probe_sdpa_variants.py`:
  Isolated triangular-attention and SDPA investigations.
- `cuda_harness/_probe_jit_precision.py` and `findings/graphs/`:
  TorchScript graph inspection and precision-policy extraction.

## Recommended rerun order

1. Re-establish the reference precision policy from the TorchScript graph
   dumps.
2. Rerun `scripts/cuda_parity.py` on one representative monomer target
   in both `reference` and `float32` modes.
3. Rerun the trunk-vs-diffusion isolation script on the same target.
4. If needed, drill into the MSA and Pairformer probes to localise any
   remaining disagreement.
5. Refresh the structure-level sweep across the target matrix.

## Notes

- Keep raw probe outputs under `/tmp/chai_mlx_cuda/...` or another
  ignored path rather than checking fresh result artifacts into the repo.
- When new comparisons have been rerun, summarise the regenerated
  findings here without mixing them with older, incompatible numbers.
