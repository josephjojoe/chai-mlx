# Status / parity notes

This is the authoritative status page for the current `chai_mlx` port. It tracks
what is structurally implemented, what has already been fixed, and what still
needs validation against the upstream TorchScript artifacts and the graph dumps
in `findings/graphs/`.

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
- **Memory / diffusion**: `to_atom_cond` is computed in `prepare_cache` (not every denoise step). Triangle multiplication in `chai_mlx/nn/layers/triangle.py` uses internal **feature-chunk size 32** to lower peak `n×n` activations. With `use_custom_kernel=True`, AdaLN uses a **full LN + affine** Metal kernel (`fused_adaln_full`).

## Bugs fixed

- **Weight reshape for einsum-based layers**: TorchScript stores several attention weights as multi-dimensional tensors consumed via `torch.einsum`, while the MLX port uses standard `nn.Linear` (2D). The conversion scripts now reshape these correctly:
  - `input2qkvg.weight`: [in, 4, H, D] → [4\*H\*D, in] (96 trunk + 8 confidence = 104 weights)
  - `output_proj.weight` (AttentionPairBias): [H, D, out] → [out, H\*D] (104 weights, same blocks)
  - `to_qkv.weight` (atom attention): [3, H, D, in] → [3\*H\*D, in] (9 weights across token embedder, diffusion encoder, diffusion decoder)
  - Total: 113 weight tensors. Without this fix the model could not load or would produce incorrect outputs.
- **Template embedder architecture rewrite**: The MLX code was concatenating all 4 templates along the feature dim and projecting them once — but the TorchScript processes each template *independently* through shared pairformer blocks starting from `proj_in(pair) + template_feats[:, t]`, then stacks, LayerNorms, masks by per-template validity, averages over valid templates, and projects back with `Sequential(ReLU, Linear)`. The MLX code now matches: per-template processing with per-template masks, masked averaging with `n_templates_non_mask` clamped to 1, and ReLU before `proj_out`.
- **msa_mask / template_input_masks not wired through**: Both masks are now populated in `_batch_to_feature_context` (from `inputs["msa_mask"]` and the outer product of `inputs["template_mask"]`), stored on `StructureInputs`, and threaded through `Trunk.__call__` to `MSAModule` (existing `msa_mask` plumbing) and the rewritten `TemplateEmbedder` (per-template masks AND'd with `token_pair_mask`).
- **Diffusion conditioning initial representations**: Was using trunk-path initial (`pair_initial`, `single_initial`) instead of structure-path (`pair_structure`, `single_structure`). This would have corrupted `z_cond` and `s_cond` throughout the entire diffusion loop.
- **OPM einsum**: Was contracting both depth and inner dims (`"bmiae,bmjbe->bijab"` → 64-dim), instead of only depth (`"bmige,bmjgf->bijgef"` → 512-dim). Would crash at the reshape to 512. Mask normalization broadcast also corrected (3 trailing dims instead of 2).
- **Second-order correction guard**: Added `sigma_next != 0` check before the Heun correction, matching the reference to avoid division by zero.
- **TriangleMultiplication chunked path crash**: The memory-efficient `_forward_chunked` accumulated einsum results with addition (`x_out += chunk`) instead of concatenation. The einsum `"bikd,bjkd->bijd"` treats `d` as a free index, so each chunk produces `(b, n, n, chunk_size)` which cannot be added to `(b, n, n, pair_dim)`. This crashed at runtime for any pairformer/MSA/template block. Fixed to `mx.concatenate(chunks, axis=-1)` and verified numerically identical to the unchunked reference path.

## Featurization

Frontend featurization is delegated to the upstream **chai-lab** package via
`featurize_fasta()`. The thin adapter in `chai_mlx/data/featurize.py` calls chai-lab's
`make_all_atom_feature_context` and `Collate`, then passes per-feature raw
tensors through to `FeatureContext.raw_features`.

### Memory-efficient encoding (matching TorchScript)

The TorchScript `feature_embedding.pt` takes raw per-feature tensors, encodes
each one (one-hot / RBF / embedding outersum / identity), concatenates within
each type group, then immediately projects through a Linear — the wide
concatenated tensor is only transient.

The MLX port now matches this pattern.  When `raw_features` is populated
(the `featurize_fasta()` path), `FeatureEmbedding._forward_raw` encodes,
concatenates, and projects **one group at a time**, so peak memory is the
largest single group rather than the sum of all groups.  This avoids
materialising and copying the multi-GB wide tensors (e.g.
`(1, 2048, 2048, 163)` TOKEN_PAIR) through CPU → numpy → MLX.

A precomputed path (`_forward_precomputed`) is also retained for callers who
bring their own encoded tensors, and is verified bit-identical to the raw path.

### Encoding bugs fixed

- **ONE_HOT widths**: Per-component widths verified against every
  `torch.one_hot` call in `feature_embedding_forward256.py` and stored in
  the `_*_FEATURES` spec tables in `embeddings.py`.
- **AtomNameOneHot mult=4 flattening**: 4-character one-hot `(B, N, 4, 65)` →
  `(B, N, 260)` matching the TorchScript `reshape(…, 260)`.
- **RBF encoding**: `_encode_rbf` applies Gaussian RBF using learned `radii`
  and config scales (4.8 / 2.8), producing 7 channels (6 RBF + 1 mask).
- **TemplateResType OUTERSUM**: Learned `nn.Embedding(33, 32)` with pairwise
  outer-sum, matching TorchScript.
- **Feature concatenation order**: Alphabetical within each type group,
  verified against TorchScript.
- **Identity feature shape**: Guarantees trailing dim for concatenation.
- **Weight map**: 3 learned parameters mapped: `TemplateResType.embedding.weight`,
  `TokenDistanceRestraint.radii`, `TokenPairPocketRestraint.radii`.

## Correctness fixes (post-audit)

- **`make_additive_mask` no-op for float masks**: The function returned float 0/1 masks unchanged instead of converting to additive bias (-10000/0). All attention modules (pair-biased, triangle, MSA pair-weighted averaging) were failing to mask padded tokens. Fixed to always convert through `bool_` before applying `mx.where`.
- **Triangle attention float `&` on masks**: `TriangleAttention` used bitwise AND on float32 `pair_mask` values. Fixed to cast to `mx.bool_` before the AND operation.
- **Confidence head PDE symmetrization missing**: TorchScript symmetrizes the pair representation (`z + z.transpose`) before the PDE projection. Added.
- **Confidence head missing LayerNorms before projections**: TorchScript applies `affine=False` LayerNorms to single and pair representations before pLDDT/PAE/PDE projections. Added `single_output_norm` and `pair_output_norm`.
- **Confidence head missing `token_single_mask`**: TorchScript applies `token_exists_mask` multiplicatively to the single representation after pairformer blocks. Added.
- **Trunk first recycle skipped recycle projection**: The reference always applies `recycle_proj(prev)` even on the first iteration (where `prev = initial`). The port was skipping this. Fixed by initializing `prev_single = single_init, prev_pair = pair_init` and always applying recycle projection at the start of each iteration.
- **Unused import in `diffusion.py`**: Removed unused `make_additive_mask` import.
- **`parity_check.py` attribute name bug**: `TEMPLATES.lower()` gave `templates_proj` but the actual attribute is `template_proj`. Fixed with an explicit name mapping dict.
- **`bond_adjacency` dual-sourcing**: Documented `FeatureContext.bond_adjacency` as canonical, `StructureInputs.bond_adjacency` as legacy fallback.
- **README layout**: Removed reference to nonexistent `blocked_local_attention.py`, added missing files (`convert_npz.py`, `name_map.py`, `validate.py`, `parity_check.py`).

## Architectural fixes (second audit round)

- **Fix A — Pair conditioning concatenation order reversed**: `DiffusionConditioning.prepare_static` concatenated pair features as `[pair_structure, pair_trunk]` but TorchScript (`diffusion_module_forward256.py:1441`) uses `[pair_trunk, pair_structure]`. Fixed — the misaligned ordering would corrupt the subsequent Linear projection.
- **Fix B — `token_to_atom_single` used sigma-dependent input**: The diffusion atom encoder was feeding `s_cond` (sigma-conditioned single tokens) to `token_to_atom_single`, but TorchScript uses the raw trunk single representation (sigma-independent). Fixed by moving this computation to `prepare_cond()` in the cache, using `trunk.single_trunk`.
- **Fix C — Atom encoder merged initial state and conditioning**: TorchScript separates: `x = broadcast(to_atom_cond) + prev_pos_embed(coords)` as initial state, and `cond = LN(to_atom_cond + token_to_atom[indices])` as AdaLN conditioning. The MLX port was merging these into one tensor. Fixed with `prepare_cond()` and a new `cond_layer_norm` (affine=False).
- **Fix D — `post_atom_cond_layernorm` applied to wrong quantity**: Was applied to bare `to_atom_cond`; now applied to the combined conditioning `atom_single_cond` (matching TorchScript line 3108). Decoder now also receives `encoder_atom_repr` as initial state, matching TorchScript's `input18 + token_to_atom(input68)`.
- **Fix G — MSA module block ordering wrong**: Was `OPM → pair_transition → pair_weighted_avg → msa_transition → tri_mult → tri_attn`. TorchScript order is `OPM → msa_transition → pair_weighted_avg → tri_mult ‖ pair_transition → tri_attn` where tri_mult and pair_transition read from the same post-OPM pair.
- **Fix G addendum — pair_transition reads from wrong pair in PairformerBlock**: Both MSA module and PairformerBlock compute `transition_pair(z)` from the original pair before `triangle_multiplication`, then add both deltas. Was feeding tri_mult output to pair_transition.
- **Fix H — `token_single_mask` not threaded through pairformer**: TorchScript applies `token_single_mask * attention_delta` 48 times (once per pairformer block) to gate single attention residuals. Added `single_mask` parameter to `PairformerBlock` and `PairformerStack`, threaded from `Trunk.__call__` and `ConfidenceHead._run_single`.

## Still requiring parity work

- **Artifact-backed validation still needs to be run regularly**: `scripts/weight_loading_e2e.py` now exercises the real convert -> strict load -> smoke-forward path, but it is still a manual/integration check until it is wired into a release process.
- **Reference dump parity now has a harness**: `scripts/layer_parity.py` compares captured MLX intermediate tensors against a reference runtime dump, but it depends on generating those reference dump artifacts out-of-band.
- Parity test that exercises `_batch_to_feature_context` end-to-end against the reference `feature_embedding.pt` output (existing `scripts/parity_check.py` only tests the Linear projections, not the encoding path). The `examples/fasta_smoke.py` smoke script exercises dimensions but not numerical parity.

## Recommended path to productionize

1. Export and load the real weights via `python scripts/weight_loading_e2e.py --torchscript-dir ...`.
2. Run `python scripts/parity_check.py` to verify per-component numerical agreement.
3. Dump intermediate tensors from the reference implementation and compare them with `python scripts/layer_parity.py --weights-dir ... --input-npz ... --reference-npz ...`.
4. Validate one component at a time:
   - feature embedding (use `examples/fasta_smoke.py` for dimensions, `scripts/parity_check.py` for numerics),
   - token input embedder,
   - one trunk recycle (especially OPM output shapes),
   - one diffusion denoise call,
   - confidence head (verify PDE symmetry and LayerNorm effects),
5. Only after parity is acceptable, switch on the experimental custom Metal kernels.
