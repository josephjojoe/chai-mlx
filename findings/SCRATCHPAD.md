# Scratchpad: Chai-1 Codebase Analysis

## Key Files and Their Roles

### Entry Points
- `chai_lab/chai1.py` — Main inference pipeline: feature construction, trunk recycling, diffusion sampling, confidence scoring, CIF export
- `chai_lab/main.py` — Typer CLI entry: `fold`, `a3m-to-pqt`, `citation`

### Neural Network Components (shipped as TorchScript `.pt` files, not source)
- `feature_embedding.pt` — Embeds raw features into representation spaces (TOKEN, TOKEN_PAIR, ATOM, ATOM_PAIR, TEMPLATES, MSA)
- `token_embedder.pt` — Token input embedding: takes single+pair token features and atom features, outputs (token_single_initial_repr, token_single_structure_input, token_pair_initial_repr)
- `bond_loss_input_proj.pt` — Projects bond features into trunk and structure pair representations
- `trunk.pt` — Main trunk (MSA module + Pairformer stack, recycled N times); takes initial + recycled representations, MSA, templates, masks
- `diffusion_module.pt` — Structure diffusion denoiser; takes noised atom coords + trunk representations → denoised atom coords
- `confidence_head.pt` — Predicts PAE (64 bins, 0-32Å), PDE (64 bins, 0-32Å), pLDDT logits per atom

### Model Weights Downloaded From
- URL pattern: `https://chaiassets.com/chai1-inference-depencencies/models_v2/{comp_key}`
- ESM model: `https://chaiassets.com/chai1-inference-depencencies/esm2/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt`
- Found in `chai_lab/utils/paths.py` lines 67-82

### ModuleWrapper and crop sizes
- `AVAILABLE_MODEL_SIZES = [256, 384, 512, 768, 1024, 1536, 2048]` (in `data/collate/utils.py` line 13)
- Each exported module has `forward_{crop_size}` variants for static graph optimization
- Found in `chai1.py` lines 115-148

### Feature Embedding Outputs (from `chai1.py` lines 687-698)
The `feature_embedding.pt` module returns a dict with keys:
- `"TOKEN"` → token_single_input_feats
- `"TOKEN_PAIR"` → split in half: (token_pair_input_feats, token_pair_structure_input_feats)
- `"ATOM"` → split in half: (atom_single_input_feats, atom_single_structure_input_feats)
- `"ATOM_PAIR"` → split in half: (block_atom_pair_input_feats, block_atom_pair_structure_input_feats)
- `"TEMPLATES"` → template_input_feats
- `"MSA"` → msa_input_feats

Key insight: TOKEN_PAIR, ATOM, and ATOM_PAIR each produce TWO outputs by chunking — one for the trunk, one for the structure (diffusion) module.

### Token Input Embedder Outputs (from `chai1.py` lines 720-738)
`token_embedder.pt` takes:
- token_single_input_feats, token_pair_input_feats
- atom_single_input_feats, block_atom_pair_feat, block_atom_pair_mask
- block_indices_h, block_indices_w (local attention block indices)
- atom_single_mask, atom_token_indices
Returns tuple of 3:
- token_single_initial_repr
- token_single_structure_input
- token_pair_initial_repr

This is equivalent to AF3's "InputEmbedder" — it aggregates atom-level information up to token-level.

### Trunk Forward (from `chai1.py` lines 757-777)
`trunk.pt` takes:
- token_single_trunk_initial_repr, token_pair_trunk_initial_repr (from embedder, fixed)
- token_single_trunk_repr, token_pair_trunk_repr (recycled from previous iteration)
- msa_input_feats, msa_mask
- template_input_feats, template_input_masks
- token_single_mask, token_pair_mask
Returns tuple of 2:
- token_single_trunk_repr (updated)
- token_pair_trunk_repr (updated)

This is recycled `num_trunk_recycles` times (default 3).
Per the diagram: trunk contains MSA module (4 blocks) → Input embedding (48 pair-bias attention blocks) → Structure prediction prep

### Diffusion Module (from `chai1.py` lines 806-886)
Takes:
- Static inputs from trunk: token_single_initial_repr, token_pair_initial_repr (structure versions), trunk representations, atom features
- Per-step inputs: atom_noised_coords, noise_sigma
Returns: denoised atom positions

Uses EDM-style sampling (Karras et al.) with:
- `InferenceNoiseSchedule` (power interpolation, p=7.0, sigma_data=16.0, s_max=80, s_min=4e-4)
- S_churn=80, S_tmin=4e-4, S_tmax=80.0, S_noise=1.003
- Second-order Heun corrector
- Center random augmentation at each step (random rotation + translation)
- Default 200 timesteps, 5 diffusion samples

### Confidence Head (from `chai1.py` lines 894-958)
`confidence_head.pt` takes:
- token_single_input_repr, token_single_trunk_repr, token_pair_trunk_repr
- token_single_mask, atom_single_mask
- atom_coords (denoised), token_reference_atom_index
- atom_token_index, atom_within_token_index
Returns tuple of 3:
- pae_logits: (b, n_tokens, n_tokens, 64) — Predicted Aligned Error, bins 0-32Å
- pde_logits: (b, n_tokens, n_tokens, 64) — Predicted Distance Error, bins 0-32Å
- plddt_logits: (b, n_atoms, n_bins) — per-atom predicted lDDT

### Atom Representation Details
- Max 23 atoms per token: `n_atoms = 23 * n_tokens` (in `data/collate/utils.py` line 35)
- Blocked local attention: query block size=32, key block size=128 (in `chai1.py` lines 641-642)
- Block indices computed via `get_qkv_indices_for_blocks()` in `model/utils.py`

### Feature Generators (defined in `chai1.py` lines 172-235)
TOKEN features:
- RelativeSequenceSeparation (TOKEN_PAIR, one-hot, 67 classes)
- RelativeTokenSeparation (TOKEN_PAIR, one-hot, r_max=32 → 67 classes)
- RelativeEntity (TOKEN_PAIR, one-hot, 3 classes)
- RelativeChain (TOKEN_PAIR, one-hot, 6 classes)
- ResidueType (TOKEN, one-hot, 32 classes)
- ESMEmbeddings (TOKEN, ESM encoding)
- MSAProfile (TOKEN, identity, 32 classes)
- MSADeletionMean (TOKEN, identity, 1 class)
- IsDistillation (TOKEN)
- TokenBFactor (TOKEN, include_prob=0.0)
- TokenPLDDT (TOKEN, include_prob=0.0)
- ChainIsCropped (TOKEN)
- MissingChainContact (TOKEN)

ATOM features:
- AtomRefPos (ATOM, identity, 3 coords, scaled by 1/10)
- AtomRefCharge (ATOM, identity, 1)
- AtomRefMask (ATOM, identity, 1)
- AtomRefElement (ATOM, one-hot, 128 classes)
- AtomNameOneHot (ATOM, one-hot)

ATOM_PAIR features:
- BlockedAtomPairDistogram (ATOM_PAIR, one-hot, 11 classes, bins at 0-16Å)
- InverseSquaredBlockedAtomPairDistances (ATOM_PAIR, identity, transform=1/(1+d²))

TOKEN_PAIR features (restraints/constraints):
- TokenDistanceRestraint (TOKEN_PAIR, RBF, 6 radii)
- DockingConstraintGenerator (TOKEN_PAIR, one-hot, 6 classes, bins at 0/4/8/16Å)
- TokenPairPocketRestraint (TOKEN_PAIR, RBF, 6 radii)
- TokenBondRestraint (TOKEN_PAIR, identity, 1) — separate from feature_embedding, goes through bond_loss_input_proj.pt

TEMPLATE features:
- TemplateMask (TEMPLATES, identity, 2 — backbone frame mask + pseudo beta mask)
- TemplateUnitVector (TEMPLATES, identity, 3)
- TemplateResType (TEMPLATES, outersum, 32 classes)
- TemplateDistogram (TEMPLATES, one-hot, 38 bins from 3.25-50.75Å)

MSA features:
- MSAOneHot (MSA, one-hot, 32 classes)
- MSAHasDeletion (MSA, identity, 1)
- MSADeletionValue (MSA, identity, 1, scaled by 2/π * arctan(d/3))
- IsPairedMSA (MSA, identity, 1)
- MSADataSource (MSA, one-hot, 6 classes)

### Ranking/Scoring (from `ranking/rank.py`)
Aggregate score = 0.2 * complex_pTM + 0.8 * interface_pTM - 100 * has_inter_chain_clashes
- pTM uses TM-score d0 normalization: d0 = 1.24 * (n-15)^(1/3) - 1.8
- PAE bins: 64 bins, 0-32Å
- pLDDT: per-atom, then averaged per-token

### ESM Model Details
- ESM2 3B parameter model (esm2_t36_3B_UR50D), traced with SDPA, fp16
- Embedding dimension: 2560 (from `EmbeddingContext.empty()` default d_emb=2560)
- Non-protein tokens get zero embeddings
- Modified residues replaced with parent canonical residue; otherwise "X"
- Found in `data/dataset/embeddings/esm.py`

### Constants
- MAX_MSA_DEPTH = 16,384 (from `data/dataset/all_atom_feature_context.py` line 20)
- MAX_NUM_TEMPLATES = 4 (line 21)
- MSA has pairing keys for cross-chain pairing (species-based)
- Template features are masked to same-chain only (inter-chain distances zeroed)

### Diffusion Schedule Details (from `model/diffusion_schedules.py`)
Power interpolation: σ(t) = σ_data * (t * s_min^(1/p) + (1-t) * s_max^(1/p))^p
- p = 7.0, σ_data = 16.0, s_max = 80.0, s_min = 4e-4
- 200 timesteps evenly spaced in t ∈ [0, 1]
