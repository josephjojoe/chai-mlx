# Status / parity notes

This package is an MLX port of the Chai-1 inference pipeline, structurally verified
against the TorchScript `.pt` modules and the graph dumps in `findings/graphs/`.

## Implemented exactly in structure

- Pipeline split (`embed_inputs` -> `trunk` -> `prepare_diffusion_cache` -> diffusion loop -> confidence -> ranking).
- Hidden dimensions and module boundaries matching TorchScript weight shapes.
- Recycle structure (3 recycles, LN -> Linear no-bias projections).
- Diffusion cache split: `z_cond`, 16 precomputed pair biases, `blocked_pair_base`.
- EDM-style sampling loop with Heun second-order correction (non-standard variant), including `sigma_next != 0` guard matching reference.
- Initial noise scaled by `sigma[0]` matching reference.
- Blocked atom attention topology (query 32, KV 128).
- Pair-biased attention interfaces for Pairformer, diffusion, and confidence.
- Confidence head: fused single-projection triangle attention (`pair2qkvgb`).
- Outer-product-mean with `weight_ab: [2, 8, 8, msa_dim]` matching TorchScript. Einsum verified against `trunk_forward256.py`: group dim shared, both inner dims kept, producing 8×8×8 = 512 per token pair.
- TriangleMultiplication with separate `layernorm_out`/`layernorm_in` instances.
- Diffusion conditioning uses structure path (`pair_structure`, `single_structure`) for the initial half of the conditioning concatenation, matching the reference (`token_pair_structure_input_feats`, `token_single_structure_input`).
- Confidence head uses `single_initial` for pair outer-sum, `single_trunk` for blocks.
- Diffusion atom encoder includes `LayerNorm` before `token_to_atom_single`.

## Bugs fixed

- **Diffusion conditioning initial representations**: Was using trunk-path initial (`pair_initial`, `single_initial`) instead of structure-path (`pair_structure`, `single_structure`). This would have corrupted `z_cond` and `s_cond` throughout the entire diffusion loop.
- **OPM einsum**: Was contracting both depth and inner dims (`"bmiae,bmjbe->bijab"` → 64-dim), instead of only depth (`"bmige,bmjgf->bijgef"` → 512-dim). Would crash at the reshape to 512. Mask normalization broadcast also corrected (3 trailing dims instead of 2).
- **Second-order correction guard**: Added `sigma_next != 0` check before the Heun correction, matching the reference to avoid division by zero.

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
   - one trunk recycle (especially OPM output shapes),
   - one diffusion denoise call,
   - confidence head,
4. Only after parity is acceptable, switch on the experimental custom blocked-local kernel.
