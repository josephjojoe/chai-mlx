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
- **Memory / diffusion**: `to_atom_cond` is computed in `prepare_cache` (not every denoise step). Triangle multiplication uses internal **feature-chunk size 32** to lower peak `n×n` activations. With `use_custom_kernel=True`, AdaLN uses a **full LN + affine** Metal kernel (`fused_adaln_full`).

## Bugs fixed

- **Weight reshape for einsum-based layers**: TorchScript stores several attention weights as multi-dimensional tensors consumed via `torch.einsum`, while the MLX port uses standard `nn.Linear` (2D). The conversion scripts now reshape these correctly:
  - `input2qkvg.weight`: [in, 4, H, D] → [4\*H\*D, in] (96 trunk + 8 confidence = 104 weights)
  - `output_proj.weight` (AttentionPairBias): [H, D, out] → [out, H\*D] (104 weights, same blocks)
  - `to_qkv.weight` (atom attention): [3, H, D, in] → [3\*H\*D, in] (9 weights across token embedder, diffusion encoder, diffusion decoder)
  - Total: 113 weight tensors. Without this fix the model could not load or would produce incorrect outputs.
- **Template embedder missing ReLU**: TorchScript `proj_out` is `Sequential(ReLU[0], Linear[1])`, but the MLX code omitted the ReLU before the final linear projection. Added `nn.relu()` in `TemplateEmbedder.__call__`.
- **Diffusion conditioning initial representations**: Was using trunk-path initial (`pair_initial`, `single_initial`) instead of structure-path (`pair_structure`, `single_structure`). This would have corrupted `z_cond` and `s_cond` throughout the entire diffusion loop.
- **OPM einsum**: Was contracting both depth and inner dims (`"bmiae,bmjbe->bijab"` → 64-dim), instead of only depth (`"bmige,bmjgf->bijgef"` → 512-dim). Would crash at the reshape to 512. Mask normalization broadcast also corrected (3 trailing dims instead of 2).
- **Second-order correction guard**: Added `sigma_next != 0` check before the Heun correction, matching the reference to avoid division by zero.

## Featurization

Frontend featurization is delegated to the upstream **chai-lab** package via
`featurize_fasta()`.  The thin adapter in `featurize.py` calls chai-lab's
`make_all_atom_feature_context` and `Collate`, then encodes the per-generator
features into the dense tensors that the MLX `FeatureEmbedding` expects.

Features that require learned parameters (TemplateResType embedding, RBF
restraint radii) are kept as raw auxiliary data on `FeatureContext` and encoded
at runtime by `FeatureEmbedding` using its learned weights.

### Encoding bugs fixed in featurize.py audit

- **ONE_HOT mask channel**: All ONE_HOT features with `can_mask=True` now use
  `one_hot(idx, width)` where `width` is the exact value from the TorchScript
  IR (typically `num_classes + 1`), not the generator's `num_classes`.  Fixed
  via a lookup table (`_ONE_HOT_WIDTH`) verified against every `torch.one_hot`
  call in `feature_embedding_forward256.py`.
- **AtomNameOneHot mult=4 flattening**: The 4-character-per-atom one-hot is now
  correctly reshaped from `(B, N, 4, 65)` to `(B, N, 260)` to match the
  TorchScript `reshape(…, 260)`.
- **RBF encoding for TokenDistanceRestraint / TokenPairPocketRestraint**: Raw
  distance values are stored in `FeatureContext.distance_restraint_data` /
  `.pocket_restraint_data`.  `FeatureEmbedding._encode_rbf` applies the
  Gaussian RBF expansion using learned `radii` parameters and config-derived
  scale constants (4.8 and 2.8), producing 7 channels (6 RBF + 1 mask) each.
- **TemplateResType OUTERSUM encoding**: Raw residue type indices are stored in
  `FeatureContext.template_restype_indices`.  `FeatureEmbedding` encodes them
  via `nn.Embedding(33, 32)` with pairwise outer-sum, matching the TorchScript
  embedding-based (not one-hot-based) outer-sum pattern.
- **Feature concatenation order**: Features within each type group are now
  sorted alphabetically by name, matching the TorchScript concatenation order
  verified against `feature_embedding_forward256.py`.
- **Identity feature shape**: `_encode_identity` now guarantees at least 3D
  output (adding trailing dim when needed) to prevent shape mismatches during
  concatenation.
- **Weight map**: Added mappings for the 3 previously-dropped learned
  parameters: `TemplateResType.embedding.weight`, `TokenDistanceRestraint.radii`,
  `TokenPairPocketRestraint.radii`.

## Still requiring parity work

- Verified loading with real weights (name mapping + einsum reshape logic is in place but untested end-to-end).
- Exhaustive per-layer tensor parity checks against the reference runtime.
- Parity test that exercises `_batch_to_feature_context` end-to-end against the reference `feature_embedding.pt` output (existing `validate_parity.py` only tests the Linear projections, not the encoding path).

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
