# Status / parity notes

This package is an MLX port of the Chai-1 inference pipeline, structurally verified
against the TorchScript `.pt` modules and ARCHITECTURE.md reference.

## Implemented exactly in structure

- Pipeline split (`embed_inputs` -> `trunk` -> `prepare_diffusion_cache` -> diffusion loop -> confidence -> ranking).
- Hidden dimensions and module boundaries matching TorchScript weight shapes.
- Recycle structure (3 recycles, LN -> Linear no-bias projections).
- Diffusion cache split: `z_cond`, 16 precomputed pair biases, `blocked_pair_base`.
- EDM-style sampling loop with Heun second-order correction (non-standard variant).
- Initial noise scaled by `sigma[0]` matching reference.
- Blocked atom attention topology (query 32, KV 128).
- Pair-biased attention interfaces for Pairformer, diffusion, and confidence.
- Confidence head: fused single-projection triangle attention (`pair2qkvgb`).
- Outer-product-mean with `weight_ab: [2, 8, 8, msa_dim]` matching TorchScript.
- TriangleMultiplication with separate `layernorm_out`/`layernorm_in` instances.
- Diffusion conditioning uses structure path (not trunk initial) for concat inputs.
- Confidence head uses `single_initial` for pair outer-sum, `single_trunk` for blocks.
- Diffusion atom encoder includes `LayerNorm` before `token_to_atom_single`.

## Still requiring parity work

- Exact TorchScript-to-MLX weight-name alignment and verified loading.
- Exhaustive per-layer tensor parity checks against the reference runtime.
- Full raw frontend featurization from FASTA/MSA/template/restraint inputs.

## Recommended path to productionize

1. Export and load the real weights via `weights/export_torchscript.py`.
2. Dump intermediate tensors from the reference implementation.
3. Validate one component at a time:
   - feature embedding,
   - token input embedder,
   - one trunk recycle,
   - one diffusion denoise call,
   - confidence head,
4. Only after parity is acceptable, switch on the experimental custom blocked-local kernel.
