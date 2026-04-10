# Chai-1 Architecture: Comprehensive Reference

This document describes the full Chai-1 neural architecture as reconstructed from the
[chai-lab](chai-lab/) open-source codebase. Chai-1 closely follows AlphaFold 3 (Abramson
et al. 2024), with key modifications described in the Chai-1 preprint.

> **Important context**: The neural network layers themselves are shipped as pre-compiled
> TorchScript (`.pt`) files — the Python codebase contains the full inference
> orchestration, feature construction, and post-processing, but not the internal layer
> definitions. This document reconstructs the architecture from the inference pipeline
> signatures, feature shapes, and preprint information.

---

## Table of Contents

1. [High-Level Pipeline](#1-high-level-pipeline)
2. [Input Representation](#2-input-representation)
3. [Feature Processing](#3-feature-processing)
4. [Feature Embedding](#4-feature-embedding)
5. [Token Input Embedder](#5-token-input-embedder)
6. [Trunk (MSA Module + Pairformer Stack)](#6-trunk-msa-module--pairformer-stack)
7. [Diffusion Module (Structure Prediction)](#7-diffusion-module-structure-prediction)
8. [Confidence Head](#8-confidence-head)
9. [Ranking and Scoring](#9-ranking-and-scoring)
10. [Key Differences from AlphaFold 3](#10-key-differences-from-alphafold-3)

---

## 1. High-Level Pipeline

The end-to-end inference pipeline, orchestrated by `run_folding_on_context()` in
`chai_lab/chai1.py`, follows these stages:

```
Input (FASTA + optional MSAs/templates/constraints/ESM embeddings)
  │
  ▼
Feature Construction (FeatureFactory + AllAtomFeatureContext)
  │
  ▼
Feature Embedding (feature_embedding.pt)
  │  Produces: TOKEN, TOKEN_PAIR, ATOM, ATOM_PAIR, MSA, TEMPLATE representations
  │  TOKEN_PAIR, ATOM, ATOM_PAIR each split into trunk and structure halves
  │
  ▼
Bond Feature Projection (bond_loss_input_proj.pt)
  │  Adds covalent bond features to TOKEN_PAIR representations
  │
  ▼
Token Input Embedder (token_embedder.pt)
  │  Aggregates atom-level → token-level representations
  │  Outputs: (single_initial, single_structure_input, pair_initial)
  │
  ▼
Trunk (trunk.pt) × num_trunk_recycles (default: 3)
  │  Contains: MSA module (4 blocks) + Pairformer (48 blocks)
  │  Recycling: output single/pair representations fed back as input
  │  Outputs: (single_trunk_repr, pair_trunk_repr)
  │
  ▼
Diffusion Module (diffusion_module.pt) × num_diffn_timesteps (default: 200)
  │  EDM-style denoising with second-order Heun correction
  │  Multiple samples drawn in parallel (default: 5)
  │  Outputs: denoised atom coordinates
  │
  ▼
Confidence Head (confidence_head.pt) × num_diffn_samples
  │  Outputs: PAE logits, PDE logits, pLDDT logits
  │
  ▼
Ranking & Output
  │  Aggregate score = 0.2·pTM + 0.8·ipTM − 100·has_clashes
  │  Export CIF files with per-atom pLDDT as B-factors
```

**Evidence**: The pipeline is defined in `chai_lab/chai1.py` lines 580-1059, with each
stage clearly delineated by comments and module loading via `_component_moved_to()`.

---

## 2. Input Representation

### 2.1 Token-Atom Hierarchy

Chai-1 uses a two-level representation following AF3's "token" abstraction:

- **Tokens**: The primary sequence-level units. For standard residues (amino acids,
  nucleotides), one token = one residue. For ligands and modified residues, one token =
  one atom.
- **Atoms**: The atomic-level representation. Each token maps to up to 23 atoms
  (hardcoded: `n_atoms = 23 * n_tokens` in `data/collate/utils.py:35`).

Key mappings maintained in `AllAtomStructureContext` (`data/dataset/structure/all_atom_structure_context.py`):

| Field | Shape | Description |
|-------|-------|-------------|
| `atom_token_index` | `[n_atoms]` | Maps each atom to its parent token |
| `atom_within_token_index` | `[n_atoms]` | Atom ordering within each token |
| `token_centre_atom_index` | `[n_tokens]` | Representative atom for each token |
| `token_ref_atom_index` | `[n_tokens]` | Reference atom for each token |
| `atom_ref_pos` | `[n_atoms, 3]` | Reference conformer coordinates |
| `atom_ref_space_uid` | `[n_atoms]` | Groups atoms sharing a reference frame |
| `token_backbone_frame_index` | `[n_tokens, 3]` | Three atom indices defining backbone frame |
| `token_backbone_frame_mask` | `[n_tokens]` | Whether backbone frame is defined |

### 2.2 Model Sizes

The model is exported as static graphs for specific token counts:

```python
AVAILABLE_MODEL_SIZES = [256, 384, 512, 768, 1024, 1536, 2048]
```

Inputs are padded to the smallest size that fits. Each exported module has a
`forward_{crop_size}` method variant for each size.

**Evidence**: `data/collate/utils.py` line 13, `chai1.py` lines 115-148 (`ModuleWrapper`).

### 2.3 Input Contexts

The `AllAtomFeatureContext` (`data/dataset/all_atom_feature_context.py`) bundles:

| Context | Description |
|---------|-------------|
| `structure_context` | All-atom structural data (coords, types, masks, bonds) |
| `msa_context` | MSA tokens, deletion matrix, pairing keys, source labels |
| `profile_msa_context` | Separate MSA for computing profile statistics |
| `template_context` | Up to 4 structural templates (distances, unit vectors, types) |
| `embedding_context` | ESM2 per-residue embeddings (dim=2560) |
| `restraint_context` | Docking, contact, and pocket constraints |

**Evidence**: `data/dataset/all_atom_feature_context.py` lines 24-39.

---

## 3. Feature Processing

### 3.1 Feature Factory

The `FeatureFactory` (`data/features/feature_factory.py`) applies a dictionary of
`FeatureGenerator` instances to the batched data, producing typed tensors consumed by
`feature_embedding.pt`. Features are organized by `FeatureType`:

```python
class FeatureType(Enum):
    TOKEN = "TOKEN"          # Per-token single features
    TOKEN_PAIR = "TOKEN_PAIR"  # Token-pair features (n_tokens × n_tokens)
    ATOM = "ATOM"            # Per-atom features
    ATOM_PAIR = "ATOM_PAIR"  # Blocked atom-pair features
    MSA = "MSA"              # MSA features (depth × n_tokens)
    TEMPLATES = "TEMPLATES"  # Template features (n_templates × n_tokens × n_tokens)
```

**Evidence**: `data/features/feature_type.py`, `data/features/feature_factory.py`.

### 3.2 Complete Feature Inventory

The full set of features is defined in `chai1.py` lines 172-235. Each is registered
in the `feature_generators` dict.

#### Token (Single) Features

| Generator | Encoding | Classes | Source |
|-----------|----------|---------|--------|
| `ResidueType` | One-hot | 32 | `token_residue_type` — residue/nucleotide type |
| `ESMEmbeddings` | Float | 2560 | ESM2-3B per-residue embeddings |
| `MSAProfile` | Float | 32 | Distribution over residue types from profile MSA |
| `MSADeletionMean` | Float | 1 | Mean deletion count per position |
| `IsDistillation` | Float | 1 | Whether data is from distillation |
| `TokenBFactor` | Float | 1 | B-factor (include_prob=0.0 at inference) |
| `TokenPLDDT` | Float | 1 | pLDDT (include_prob=0.0 at inference) |
| `ChainIsCropped` | Bool | 1 | Whether chain was cropped |
| `MissingChainContact` | Float | 1 | Missing chain contact indicator |

#### Token Pair Features

| Generator | Encoding | Classes | Description |
|-----------|----------|---------|-------------|
| `RelativeSequenceSeparation` | One-hot | 67 | Residue index separation (±32 bins + inter-chain) |
| `RelativeTokenSeparation` | One-hot | 67 | Token index separation within same residue (r_max=32) |
| `RelativeEntity` | One-hot | 3 | Same/different entity encoding (AF-Multimer Alg. 5) |
| `RelativeChain` | One-hot | 6 | Same/different chain within entity (s_max=2) |
| `TokenDistanceRestraint` | RBF | 6 | Contact restraints between token pairs |
| `DockingConstraintGenerator` | One-hot | 6 | Docking distance bins (0/4/8/16Å + mask) |
| `TokenPairPocketRestraint` | RBF | 6 | Pocket-chain distance restraints |
| `TokenBondRestraint` | Float | 1 | Covalent bond indicator (separate pathway) |

#### Atom Features

| Generator | Encoding | Classes | Description |
|-----------|----------|---------|-------------|
| `AtomRefPos` | Float | 3 | Reference conformer position (÷10 for scale) |
| `AtomRefCharge` | Float | 1 | Reference atom charge |
| `AtomRefMask` | Float | 1 | Whether atom has valid reference data |
| `AtomRefElement` | One-hot | 128 | Atomic number (up to 128) |
| `AtomNameOneHot` | One-hot | varies | Atom name encoding |

#### Atom Pair Features (Blocked)

Atom pairs use **blocked local attention** with query block size 32 and key block size
128, computed via `get_qkv_indices_for_blocks()`.

| Generator | Encoding | Classes | Description |
|-----------|----------|---------|-------------|
| `BlockedAtomPairDistogram` | One-hot | 11 | Distance bins (0–16Å, 10 boundaries) |
| `InverseSquaredBlockedAtomPairDistances` | Float | 2 | 1/(1+d²) + mask channel |

Atom pairs are restricted to atoms sharing the same `atom_ref_space_uid` (i.e. same
residue/ligand reference frame).

**Evidence**: `data/features/generators/blocked_atom_pair_distances.py` lines 155-175.

#### MSA Features

| Generator | Encoding | Classes | Description |
|-----------|----------|---------|-------------|
| `MSAOneHot` | One-hot | 32 | MSA residue type |
| `MSAHasDeletion` | Float | 1 | Binary: deletion to the left |
| `MSADeletionValue` | Float | 1 | Scaled deletion count: 2/π·arctan(d/3) |
| `IsPairedMSA` | Float | 1 | Same species as first token (pairing) |
| `MSADataSource` | One-hot | 6 | Data source (UniRef, BFD, etc.) |

MSA depth limit: **16,384 sequences**. MSA can be subsampled during trunk recycling.

**Evidence**: `data/dataset/all_atom_feature_context.py` line 20, `data/features/generators/msa.py`.

#### Template Features

| Generator | Encoding | Classes | Description |
|-----------|----------|---------|-------------|
| `TemplateMask` | Float | 2 | Backbone frame mask + pseudo-beta mask (per pair) |
| `TemplateUnitVector` | Float | 3 | Unit vector between pseudo-beta positions |
| `TemplateResType` | OuterSum | 32 | Residue type for outer-sum embedding |
| `TemplateDistogram` | One-hot | 38 | Distance bins (3.25–50.75Å) |

Templates are limited to **4 maximum**. All template features are masked to
**intra-chain only** (inter-chain distances zeroed via `same_asym` masks).

**Evidence**: `data/features/generators/templates.py`, `data/dataset/all_atom_feature_context.py` line 21.

### 3.3 Constraint Features

Three types of structural constraints can guide prediction:

1. **Contact constraints**: Pairwise token distance thresholds, encoded as RBF with 6
   radii. Sampled from geometric distribution with p=1/3 during training.
   (`generators/token_dist_restraint.py`)

2. **Pocket constraints**: Token-to-chain proximity constraints. A token i and chain C
   such that min_j∈C ||x_i - x_j|| ≤ θ_P, with θ_P ∈ (6, 20)Å.
   (`generators/token_pair_pocket_restraint.py`)

3. **Docking constraints**: One-hot pairwise distances between chain groups using 4
   bins [0–4Å, 4–8Å, 8–16Å, >16Å]. Chains partitioned into two groups; intra-group
   distances provided, inter-group masked. Supports structure-level and chain-level
   dropout. (`generators/docking.py`)

All constraints include a learnable mask value for when they are absent. During
training, each feature type is included independently with 10% probability.

**Evidence**: Preprint Section 5.1.2, `chai1.py` lines 203-221 (inference config shows
contact include_probability=1.0, docking include_probability=0.0).

---

## 4. Feature Embedding

**Module**: `feature_embedding.pt`

The feature embedding module takes all raw features from the `FeatureFactory` and
projects them into dense representations. It returns a dictionary with six keys:

| Output Key | Shape | Description |
|------------|-------|-------------|
| `"TOKEN"` | `[b, n_tokens, d]` | Token single features |
| `"TOKEN_PAIR"` | `[b, n_tokens, n_tokens, 2d]` | Token pair features (split: trunk + structure) |
| `"ATOM"` | `[b, n_atoms, 2d]` | Atom single features (split: trunk + structure) |
| `"ATOM_PAIR"` | `[b, n_blocks, q_size, kv_size, 2d]` | Blocked atom pair features (split) |
| `"TEMPLATES"` | `[b, n_templates, n_tokens, n_tokens, d]` | Template pair features |
| `"MSA"` | `[b, depth, n_tokens, d]` | MSA features |

**Critical design**: TOKEN_PAIR, ATOM, and ATOM_PAIR outputs are **twice the channel
width** and are split with `.chunk(2, dim=-1)` into two halves:
- First half → feeds the trunk (Pairformer)
- Second half → feeds the structure prediction (diffusion) module

This separation allows the diffusion module to receive features that bypass the trunk
entirely, providing a direct pathway for structural information.

**Evidence**: `chai1.py` lines 679-698.

```python
token_pair_input_feats, token_pair_structure_input_feats = embedded_features[
    "TOKEN_PAIR"
].chunk(2, dim=-1)
atom_single_input_feats, atom_single_structure_input_feats = embedded_features[
    "ATOM"
].chunk(2, dim=-1)
block_atom_pair_input_feats, block_atom_pair_structure_input_feats = (
    embedded_features["ATOM_PAIR"].chunk(2, dim=-1)
)
```

### 4.1 Bond Feature Projection

Covalent bond features are handled separately due to TorchScript export limitations.
The `TokenBondRestraint` generator produces a binary token-pair feature indicating
covalent bonds between tokens. This is projected by `bond_loss_input_proj.pt` and
**added** to both the trunk and structure token-pair representations:

```python
trunk_bond_feat, structure_bond_feat = bond_loss_input_proj.forward(...).chunk(2, dim=-1)
token_pair_input_feats += trunk_bond_feat
token_pair_structure_input_feats += structure_bond_feat
```

**Evidence**: `chai1.py` lines 705-715.

---

## 5. Token Input Embedder

**Module**: `token_embedder.pt`

The token input embedder aggregates atom-level features up to the token level. This
corresponds to AF3's "InputEmbedder" and performs the critical atom→token information
flow using blocked local attention over atom pairs.

### Inputs

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `token_single_input_feats` | `[b, n_tokens, d]` | Token single features |
| `token_pair_input_feats` | `[b, n_tokens, n_tokens, d]` | Token pair features |
| `atom_single_input_feats` | `[b, n_atoms, d]` | Atom single features |
| `block_atom_pair_feat` | `[b, bl, q, kv, d]` | Blocked atom pair features |
| `block_atom_pair_mask` | `[b, bl, q, kv]` | Blocked atom pair mask |
| `block_indices_h` | `[bl, q]` | Query block indices |
| `block_indices_w` | `[bl, kv]` | Key block indices |
| `atom_single_mask` | `[b, n_atoms]` | Atom existence mask |
| `atom_token_indices` | `[b, n_atoms]` | Atom→token mapping |

### Outputs

Returns a tuple of three tensors:
1. **`token_single_initial_repr`**: Initial single representation for the trunk
2. **`token_single_structure_input`**: Single representation for the diffusion module
3. **`token_pair_initial_repr`**: Initial pair representation for the trunk

The atom→token aggregation likely uses the atom transformer architecture from AF3,
where atom-level attention within local blocks processes atom features, and the results
are pooled to the token level using `atom_token_indices`.

**Evidence**: `chai1.py` lines 720-738.

---

## 6. Trunk (MSA Module + Pairformer Stack)

**Module**: `trunk.pt`

The trunk is the central processing module, containing the MSA module and Pairformer
stack. Per the architecture diagram, it processes:
- **MSA module**: 4 blocks of MSA-based attention
- **Pairformer stack**: 48 blocks of pair-bias self-attention

### 6.1 Inputs

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `token_single_trunk_initial_repr` | `[b, n, d]` | Fixed initial single repr (from embedder) |
| `token_pair_trunk_initial_repr` | `[b, n, n, d]` | Fixed initial pair repr (from embedder) |
| `token_single_trunk_repr` | `[b, n, d]` | Recycled single repr |
| `token_pair_trunk_repr` | `[b, n, n, d]` | Recycled pair repr |
| `msa_input_feats` | `[b, depth, n, d]` | MSA features |
| `msa_mask` | `[b, depth, n]` | MSA mask |
| `template_input_feats` | `[b, t, n, n, d]` | Template features |
| `template_input_masks` | `[b, t, n, n]` | Template masks (frame × pseudo-beta) |
| `token_single_mask` | `[b, n]` | Token existence mask |
| `token_pair_mask` | `[b, n, n]` | Token pair mask (outer product of single mask) |

### 6.2 Outputs

Returns a tuple of two tensors:
1. **`token_single_trunk_repr`**: Updated single representation
2. **`token_pair_trunk_repr`**: Updated pair representation

### 6.3 Recycling

The trunk is run `num_trunk_recycles` times (default: 3). On each iteration:
- The **initial** representations (from the embedder) are provided unchanged
- The **recycled** representations from the previous iteration are passed back in
- MSA features can optionally be subsampled per-recycle via
  `subsample_and_reorder_msa_feats_n_mask()`

This follows AF3's recycling strategy where the initial embeddings provide a stable
anchor while the recycled representations carry information from previous iterations.

**Evidence**: `chai1.py` lines 746-777.

### 6.4 MSA Module (4 blocks)

The MSA module processes multiple sequence alignment data. Per the preprint, Chai-1
uses MSAs trained alongside protein language model embeddings. The MSA module takes:
- Embedded MSA features (depth × tokens × channels)
- Pair representations from the previous layer

It likely follows AF3's MSA architecture with:
- Row-wise gated self-attention with pair bias
- Column-wise gated self-attention
- Transition layers
- Outer product mean (MSA → pair update)

### 6.5 Pairformer Stack (48 blocks)

The main body of the trunk consists of 48 blocks of **pair-bias self-attention** — the
key architectural motif of both AF3 and Chai-1. Each block likely contains:
- Single representation update via self-attention with pair bias
- Pair representation update via triangular attention and/or triangular
  multiplicative updates
- Transition feed-forward layers

The pair representation biases the attention weights in the single representation,
creating a tight coupling between sequence-level and pairwise structural information.

### 6.6 Template Processing

Templates (up to 4) provide structural homology information. Template features include:
- Pseudo-beta distances (38 distogram bins, 3.25–50.75Å)
- Unit vectors between pseudo-beta positions
- Residue types (outer-sum encoded)
- Masks (backbone frame + pseudo-beta)

All template features are restricted to **intra-chain pairs only** — inter-chain
template distances are explicitly zeroed. This is a key difference from docking
constraints, which provide inter-chain distance information.

**Evidence**: `data/features/generators/templates.py` lines 52-53, 96-101.

---

## 7. Diffusion Module (Structure Prediction)

**Module**: `diffusion_module.pt`

The diffusion module predicts 3D atomic coordinates using a denoising diffusion process.
It follows an EDM-style framework (Karras et al. 2022) adapted for molecular structure.

### 7.1 Architecture

The diffusion module takes as input:
- **Trunk outputs**: `token_single_trunk_repr`, `token_pair_trunk_repr`
- **Structure-specific features** (bypassing trunk):
  `token_single_structure_input`, `token_pair_structure_input_feats`,
  `atom_single_structure_input_feats`, `atom_block_pair_structure_input_feats`
- **Noised coordinates**: `atom_noised_coords` with shape `[b, ds, n_atoms, 3]`
- **Noise level**: `noise_sigma` with shape `[b, ds]`
- **Structural masks and indices**: atom/token masks, block indices, atom→token mapping

Per the architecture diagram, the diffusion module contains **16 blocks** of
structure prediction and is run inside the denoising loop (×4 recycling of the
confidence head per the diagram notation, though the inference code shows this is
configurable).

**Evidence**: `chai1.py` lines 788-886.

### 7.2 Noise Schedule

The noise schedule uses power interpolation for smooth σ transitions:

```
σ(t) = σ_data · (t · s_min^(1/p) + (1-t) · s_max^(1/p))^p
```

Parameters (from `InferenceNoiseSchedule` and `DiffusionConfig`):
- `σ_data = 16.0` — data standard deviation
- `s_max = 80.0` — maximum noise level
- `s_min = 4e-4` — minimum noise level
- `p = 7.0` — interpolation power

Timesteps are sampled at midpoints of 200 evenly-spaced intervals in [0, 1].

**Evidence**: `model/diffusion_schedules.py` lines 14-48.

### 7.3 Sampling Algorithm

The sampling follows Algorithm 2 from Karras et al. (EDM) with stochastic churn:

```
DiffusionConfig:
  S_churn = 80        # stochastic churn
  S_tmin  = 4e-4      # minimum σ for churn
  S_tmax  = 80.0      # maximum σ for churn
  S_noise = 1.003     # noise amplification
  second_order = True  # enable Heun correction
```

Per-step procedure:
1. **Center and augment**: Random rotation + translation of atom positions
2. **Add churn noise**: σ̂ = σ_curr + γ · σ_curr, inject noise scaled by S_noise
3. **First-order step**: Denoise at σ̂, compute derivative d_i, step to σ_next
4. **Second-order correction** (Heun): Denoise again at σ_next, average derivatives

The centering augmentation (`center_random_augmentation`) at each step:
- Computes the weighted centroid of valid atoms
- Centers coordinates to origin
- Applies a random 3D rotation (uniformly sampled via quaternions)
- Adds random translation (σ=1.0)

This ensures SE(3) invariance during the denoising process.

**Evidence**: `chai1.py` lines 844-886, `model/utils.py` lines 178-194.

### 7.4 Multi-Sample Generation

Multiple diffusion samples (default: 5) are generated in parallel from the **same**
trunk representation. Initial positions are sampled as `σ[0] · N(0, I)` for each atom.

**Evidence**: `chai1.py` lines 839-842.

---

## 8. Confidence Head

**Module**: `confidence_head.pt`

The confidence head predicts per-residue and per-pair quality metrics for each
diffusion sample. Per the architecture diagram, it contains **4 blocks** of pair-bias
attention.

### 8.1 Inputs

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `token_single_input_repr` | `[b, n, d]` | Initial single repr (from embedder) |
| `token_single_trunk_repr` | `[b, n, d]` | Final trunk single repr |
| `token_pair_trunk_repr` | `[b, n, n, d]` | Final trunk pair repr |
| `token_single_mask` | `[b, n]` | Token existence mask |
| `atom_single_mask` | `[b, n_atoms]` | Atom existence mask |
| `atom_coords` | `[1, n_atoms, 3]` | Denoised atom coordinates (one sample) |
| `token_reference_atom_index` | `[b, n]` | Reference atom per token |
| `atom_token_index` | `[b, n_atoms]` | Atom→token mapping |
| `atom_within_token_index` | `[b, n_atoms]` | Atom ordering within token |

### 8.2 Outputs

Returns a tuple of three logit tensors:

| Output | Shape | Description |
|--------|-------|-------------|
| `pae_logits` | `[b, n_tokens, n_tokens, 64]` | Predicted Aligned Error (64 bins, 0–32Å) |
| `pde_logits` | `[b, n_tokens, n_tokens, 64]` | Predicted Distance Error (64 bins, 0–32Å) |
| `plddt_logits` | `[b, n_atoms, n_bins]` | Predicted lDDT per atom |

The confidence head is run **once per diffusion sample**, not once per denoising step.
It takes the final denoised coordinates and predicts quality metrics.

**Evidence**: `chai1.py` lines 894-915, output processing at lines 912-958.

### 8.3 Score Computation

From the logits, expected values are computed:

- **PAE scores**: softmax over 64 bins with centers from 0.0 to 32.0Å, producing
  expected error per token pair
- **PDE scores**: Same binning as PAE
- **pLDDT scores**: Per-atom, then averaged to per-token using `atom_token_index`

**Evidence**: `chai1.py` lines 920-958.

---

## 9. Ranking and Scoring

### 9.1 Aggregate Score

The aggregate ranking score combines three components:

```
aggregate_score = 0.2 × complex_pTM + 0.8 × interface_pTM − 100 × has_inter_chain_clashes
```

**Evidence**: `ranking/rank.py` lines 95-98.

### 9.2 pTM / ipTM Computation

Predicted TM-score uses the standard normalization:

```
d0 = 1.24 · (max(N, 19) − 15)^(1/3) − 1.8
TM_pair(i,j) = 1 / (1 + (PAE_ij / d0)²)
```

- **Complex pTM**: Normalized by total token count, maximized over alignment rows
- **Interface pTM (ipTM)**: For each chain c, computes pTM of c against all other
  chains, takes the maximum over chains. Weighted 4× more than complex pTM in the
  aggregate score
- **Per-chain pTM**: Individual chain quality
- **Per-chain-pair ipTM**: Pairwise interface quality matrix

**Evidence**: `ranking/ptm.py` lines 33-36, 74-115.

### 9.3 Clash Detection

Inter-chain steric clashes are detected and penalized:
- Clash threshold: 1.1Å
- Maximum clashes allowed: 100
- Maximum clash ratio: 0.5
- Any inter-chain clash results in −100 penalty to aggregate score

**Evidence**: `ranking/rank.py` lines 52-55, 95-98.

### 9.4 pLDDT

Per-atom predicted lDDT is computed as the expected value of the pLDDT logits. It is
aggregated to per-chain scores and exported as B-factors (0–100 scale) in CIF files.

**Evidence**: `ranking/plddt.py`, `chai1.py` lines 1028-1031.

---

## 10. Key Differences from AlphaFold 3

Based on the preprint and codebase analysis:

### 10.1 Language Model Embeddings

Chai-1 integrates **ESM2-3B** (esm2_t36_3B_UR50D) protein language model embeddings as
a first-class input alongside MSAs. The model is a 3-billion parameter transformer
producing 2560-dimensional per-residue embeddings.

- Non-protein tokens (DNA, RNA, ligands) receive zero/mask embeddings
- Modified residues use their canonical parent residue; unknown residues get "X"
- At inference, the model can run with **any combination** of MSAs, templates, and ESM
  embeddings

**Evidence**: `data/dataset/embeddings/esm.py` (ESM2-3B download URL at line 21,
tokenizer at lines 54-88, embedding extraction at lines 112-137).

### 10.2 Constraint Features

Chai-1 adds novel constraint/prompting features not present in AF3:

- **Contact constraints**: Token-pair distance thresholds (RBF-encoded)
- **Pocket constraints**: Token-to-chain proximity constraints
- **Docking constraints**: Inter-group distance histograms
- **Covalent bond features**: Explicit bond indicators with dedicated projection

These are trained with 10% inclusion probability and geometric sampling, enabling
flexible conditioning at inference.

### 10.3 Block Counts

From the architecture diagram:
- MSA module: **4 blocks** (vs AF3's 4 blocks — same)
- Pairformer: **48 blocks** (vs AF3's 48 blocks — same)
- Diffusion module: **16 blocks**
- Confidence head: **4 blocks**

### 10.4 Other Notable Details

- **Blocked atom-pair attention**: Query block size 32, key block size 128 (matching
  AF3's atom transformer design)
- **Diffusion sampling**: Second-order Heun correction with stochastic churn (S_churn=80)
- **Template restriction**: Templates provide only intra-chain distances; inter-chain
  structural information comes exclusively from docking constraints
- **Recycling**: Default 3 trunk recycles with optional MSA subsampling per cycle
- **Multi-sample diffusion**: Default 5 parallel samples from a single trunk output

---

## Appendix A: Neural Network Component Summary

| Component | File | Role |
|-----------|------|------|
| `feature_embedding.pt` | — | Raw features → dense representations |
| `bond_loss_input_proj.pt` | — | Bond features → pair representation |
| `token_embedder.pt` | — | Atom→token aggregation + initial representations |
| `trunk.pt` | — | MSA module + Pairformer (recycled) |
| `diffusion_module.pt` | — | Denoising diffusion structure prediction |
| `confidence_head.pt` | — | PAE, PDE, pLDDT prediction |
| ESM2-3B | `esm2_t36_3B_UR50D` | Protein language model embeddings |

All neural network components are downloaded from `chaiassets.com` and cached locally.

## Appendix B: Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| Max tokens | 2048 | `data/collate/utils.py:13` |
| Atoms per token | 23 | `data/collate/utils.py:35` |
| Max MSA depth | 16,384 | `data/dataset/all_atom_feature_context.py:20` |
| Max templates | 4 | `data/dataset/all_atom_feature_context.py:21` |
| ESM embedding dim | 2560 | `data/dataset/embeddings/embedding_context.py:48` |
| Query atom block | 32 | `chai1.py:642` |
| Key atom block | 128 | `chai1.py:641` |
| σ_data | 16.0 | `chai1.py:247` |
| Default recycles | 3 | `chai1.py:513` |
| Default diffusion steps | 200 | `chai1.py:514` |
| Default diffusion samples | 5 | `chai1.py:515` |
| PAE/PDE bins | 64, 0–32Å | `chai1.py:932-939` |
