# Chai-1 Architecture Reference

This document describes the full Chai-1 neural architecture based on the
[chai-lab](../chai-lab/) open-source codebase and the TorchScript (`.pt`) model weights.

**Total model parameters: ~314M** (excluding ESM2-3B). All weights are **float32**.

| Component | File | Parameters | Size |
|-----------|------|-----------|------|
| Feature Embedding | `feature_embedding.pt` | ~1.1M | 4.4 MB |
| Bond Projection | `bond_loss_input_proj.pt` | 512 | 5.4 KB |
| Token Input Embedder | `token_embedder.pt` | ~1.5M | 6.0 MB |
| Trunk | `trunk.pt` | 169,955,072 | 604 MB |
| Diffusion Module | `diffusion_module.pt` | 127,928,768 | 454 MB |
| Confidence Head | `confidence_head.pt` | 14,812,416 | 53 MB |

Models are downloaded from `https://chaiassets.com/chai1-inference-depencencies/models_v2/{comp_key}`
([`utils/paths.py`](../chai-lab/chai_lab/utils/paths.py)).
ESM2-3B: `https://chaiassets.com/chai1-inference-depencencies/esm2/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt`.

---

## Table of Contents

1. [High-Level Pipeline](#1-high-level-pipeline)
2. [Global Hidden Dimensions](#2-global-hidden-dimensions)
3. [Input Representation](#3-input-representation)
4. [Feature Embedding](#4-feature-embedding)
5. [Token Input Embedder](#5-token-input-embedder)
6. [Trunk](#6-trunk)
7. [Diffusion Module](#7-diffusion-module)
8. [Confidence Head](#8-confidence-head)
9. [Ranking and Scoring](#9-ranking-and-scoring)
10. [Handling of Optional Inputs](#10-handling-of-optional-inputs)
11. [Key Differences from AlphaFold 3](#11-key-differences-from-alphafold-3)
12. [Layer Building Blocks](#12-layer-building-blocks)
13. [Inference Implementation Notes](#13-inference-implementation-notes)
14. [Inference Linearity and Parallelism](#14-inference-linearity-and-parallelism)

---

## 1. High-Level Pipeline

The end-to-end inference pipeline is orchestrated by `run_folding_on_context()` in
[`chai_lab/chai1.py:580–1059`](../chai-lab/chai_lab/chai1.py):

```
Input (FASTA + optional MSAs/templates/constraints/ESM embeddings)
  │
  ▼
Feature Construction ─────────── FeatureFactory + AllAtomFeatureContext
  │
  ▼
Feature Embedding ────────────── feature_embedding.pt
  │  TOKEN(384), TOKEN_PAIR(256+256), ATOM(128+128),
  │  ATOM_PAIR(16+16), MSA(64), TEMPLATES(64)
  │
  ▼
Bond Projection ──────────────── bond_loss_input_proj.pt
  │  Linear(1 → 512) split to 256+256, added to TOKEN_PAIR
  │
  ▼
Token Input Embedder ─────────── token_embedder.pt
  │  Atom Transformer (3 blocks) → aggregate atoms to tokens
  │  → (single_initial[384], single_structure[384], pair_initial[256])
  │
  ▼
Trunk (×3 recycles) ─────────── trunk.pt
  │  Template Embedder (2 Pairformer blocks at dim 64)
  │  MSA Module (4 iterations, mixed block counts)
  │  Pairformer Stack (48 blocks)
  │  → (single_trunk[384], pair_trunk[256])
  │
  ▼
Diffusion Module (×200 steps) ── diffusion_module.pt
  │  Diffusion Conditioning (Fourier σ embedding)
  │  Atom Attention Encoder (3 local attn blocks at dim 128)
  │  Diffusion Transformer (16 blocks at dim 768)
  │  Atom Attention Decoder (3 local attn blocks at dim 128)
  │  → denoised atom coordinates [n_atoms, 3]
  │
  ▼
Confidence Head ─────────────── confidence_head.pt
  │  4 Pairformer blocks
  │  → PAE[n×n×64], PDE[n×n×64], pLDDT[n_atoms×50]
  │
  ▼
Ranking ── 0.2·pTM + 0.8·ipTM − 100·has_clashes
```

---

## 2. Global Hidden Dimensions

All dimensions verified from TorchScript weight shapes across the six `.pt` modules.

| Representation | Dimension |
|---------------|-----------|
| Token single (s) | **384** |
| Token pair (z) | **256** |
| MSA | **64** |
| Template pair | **64** |
| Atom single (a) | **128** |
| Atom pair (p) | **16** |
| Diffusion token | **768** |

All modules consistently use **pre-norm** (LayerNorm before transformation) with
additive residual connections: `x = x + sublayer(LayerNorm(x))`.

---

## 3. Input Representation

### 3.1 Token-Atom Hierarchy

- **Tokens**: One token per residue (protein/nucleotide) or per atom (ligands/modified residues)
- **Atoms**: Up to **23 atoms per token** (`n_atoms = 23 * n_tokens` in
  [`data/collate/utils.py:35`](../chai-lab/chai_lab/data/collate/utils.py))
- Model exported for static sizes: **[256, 384, 512, 768, 1024, 1536, 2048]** tokens
  ([`data/collate/utils.py:AVAILABLE_MODEL_SIZES`](../chai-lab/chai_lab/data/collate/utils.py))

### 3.2 Input Contexts

Bundled in `AllAtomFeatureContext`
([`data/dataset/all_atom_feature_context.py`](../chai-lab/chai_lab/data/dataset/all_atom_feature_context.py)):

| Context | Max Size | Description |
|---------|----------|-------------|
| `structure_context` | 2048 tokens | All-atom coordinates, types, masks, bonds |
| `msa_context` | 16,384 depth | MSA tokens, deletion matrix, pairing keys |
| `template_context` | 4 templates | Distances, unit vectors, residue types |
| `embedding_context` | 2560-dim | ESM2-3B per-residue embeddings |
| `restraint_context` | — | Docking, contact, pocket constraints |

### 3.3 Blocked Atom Geometry

Atom-level attention uses a blocked local scheme. The blocking is computed in Python
([`model/utils.py:get_qkv_indices_for_blocks`](../chai-lab/chai_lab/model/utils.py)) before
being passed to the TorchScript modules.

**Block assignment**
([`data/collate/collate.py`](../chai-lab/chai_lab/data/collate/collate.py),
[`model/utils.py`](../chai-lab/chai_lab/model/utils.py)):
- `n_atoms = 23 * n_tokens`, enforced divisible by 32
- `num_blocks = n_atoms / 32`
- Query blocks: non-overlapping consecutive chunks of **32 atoms**
- KV window: **128 atoms**, **centered** on the query block:
  `kv_start = first_query_index + (32 - 128) / 2 = first_query_index - 48`
- The 32 query atoms sit at positions **48–79** within their 128-wide KV window
- Consecutive KV windows overlap by **96 atoms** (shift 32, window 128)

**Boundary handling**:
- `sequence_length` must be exactly divisible by 32 (no partial last block)
- KV indices that fall below 0 or above `sequence_length` are marked invalid in
  `kv_mask`, then taken `% sequence_length` (modular wrap for indexing only)
- Invalid KV positions are excluded from `block_atom_pair_mask` and never attend

**Block pair mask**
([`model/utils.py:get_block_atom_pair_mask`](../chai-lab/chai_lab/model/utils.py)):
- `atom_q_mask = atom_single_mask[:, q_idx]` and `atom_kv_mask = ..[:, kv_idx]`
- Outer product: `mask = atom_q_mask & atom_kv_mask` → `[b, bl, 32, 128]`
- ANDed with `kv_is_wrapped_mask` broadcast over queries
- **Uncited**: Distance features may additionally require same-residue matching
  (the term `atom_ref_space_uid` appears in AF3 but is not found in this codebase)

**Pair tensor layout**: `[b, num_blocks, 32, 128, feature_dim]` — block index, query
slot within block, key slot within KV window, feature channels.

---

## 4. Feature Embedding

**Module**: [`feature_embedding.pt`](../chai-lab/downloads/models_v2/feature_embedding.pt) (~1.1M params)

Internal class: `chai.model.embedding.feature_embedding.FeatureEmbedding`

Features are generated in Python by `FeatureFactory`
([`data/features/feature_factory.py`](../chai-lab/chai_lab/data/features/feature_factory.py))
using generators registered in
[`chai1.py:172–235`](../chai-lab/chai_lab/chai1.py).

### 4.1 Input Feature Dimensions

The feature embedding takes **32 named features** and projects them through type-specific
input projections. All raw dimensions below are verified from the TorchScript weight
shapes in `feature_embedding.pt` (attribute `input_projs.{TYPE}.0.weight`):

| Type | Raw Dim | Projection | Output Dim | Split |
|------|---------|-----------|------------|-------|
| TOKEN | 2638 | `Linear(2638, 384)` | 384 | No |
| TOKEN_PAIR | 163 | `Linear(163, 512)` | 256 + 256 | Trunk + Structure |
| ATOM | 395 | `Linear(395, 256)` | 128 + 128 | Trunk + Structure |
| ATOM_PAIR | 14 | `Linear(14, 32)` | 16 + 16 | Trunk + Structure |
| MSA | 42 | `Linear(42, 64)` | 64 | No |
| TEMPLATES | 76 | `Linear(76, 64)` | 64 | No |

All input projections include bias. TOKEN_PAIR, ATOM, and ATOM_PAIR outputs are
**split with `.chunk(2, dim=-1)`** into trunk and structure halves.

### 4.2 Feature Encoding Details

Details in this section come from inside the `feature_embedding.pt` TorchScript module
unless otherwise noted. The Python generators
([`data/features/generators/`](../chai-lab/chai_lab/data/features/generators/)) pass raw
scalar values; encoding (RBF, one-hot, outer-sum) happens inside the TorchScript.

- **TemplateResType**: `Embedding(33, 32)` for outer-sum encoding, with learned offset
- **RBF encoding** (TokenDistanceRestraint, TokenPairPocketRestraint):
  `encoding = exp(-((radii - raw_data) / width)²)`, clamped at exponent ≤ 16.
  When `raw_data == -1.0` (absent constraint), encoding is zeroed and a `should_mask`
  indicator is concatenated.
  - TokenDistanceRestraint: width **4.8 Å** (constant c0), radii
    `[6.0, 10.8, 15.6, 20.4, 25.2, 30.0]` (buffer `TokenDistanceRestraint.radii`)
  - TokenPairPocketRestraint: width **2.8 Å** (constant c2), radii
    `[6.0, 8.8, 11.6, 14.4, 17.2, 20.0]` (buffer `TokenPairPocketRestraint.radii`)
- **AtomNameOneHot**: One-hot to 65 classes → reshaped to 260 (4 chars × 65)
- **AtomRefElement**: One-hot to 130 classes (128 elements + mask)

### 4.3 Raw Feature Dimension Breakdown

TOKEN (2638 total — verified breakdown: 33+2560+33+1+2+3+4+1+1 = 2638):
- ResidueType: 33 (32 types + mask, one-hot,
  [`generators/residue_type.py`](../chai-lab/chai_lab/data/features/generators/residue_type.py))
- ESMEmbeddings: 2560 (ESM2-3B embedding dim)
- MSAProfile: 33 — masked frequency distribution over 33 residue types, computed via
  `scatter_add` of `main_msa_mask` by token type index, divided by sum
  ([`generators/msa.py:MSAProfileGenerator`](../chai-lab/chai_lab/data/features/generators/msa.py))
- MSADeletionMean: 1 — `masked_mean(deletion_matrix, mask=main_msa_mask, dim=depth)`;
  raw deletion counts from A3M are consecutive lowercase (insertion) chars to the
  left of each match column, capped at 255
- IsDistillation: 2 (one-hot, 1 class + mask, `can_mask=True`)
- TokenBFactor: 3 (one-hot, 1 bin → 2 classes + mask; `include_prob=0.0` → always masked,
  [`generators/structure_metadata.py`](../chai-lab/chai_lab/data/features/generators/structure_metadata.py))
- TokenPLDDT: 4 (one-hot, 2 bins → 3 classes + mask; `include_prob=0.0` → always masked)
- ChainIsCropped: 1, MissingChainContact: 1

TOKEN_PAIR (163 total — verified: 67+67+3+6+6+7+7 = 163):
- RelativeSequenceSeparation: 67 — `sep_bins = arange(-32, 33)` (65 values),
  `searchsorted(sep_bins, rel_sep + 1e-4)` gives indices 0–65 for same-chain;
  inter-chain pairs get class 66. Total: `len(sep_bins) + 2 = 67`
  ([`data/features/generators/relative_sep.py`](../chai-lab/chai_lab/data/features/generators/relative_sep.py))
- RelativeTokenSeparation: 67 — `r_max=32`, only within same residue AND same chain;
  `clamp(token_i - token_j + 32, 0, 65)`, off-mask class 66. Total: `2*32+3 = 67`
  ([`data/features/generators/relative_token.py`](../chai-lab/chai_lab/data/features/generators/relative_token.py))
- RelativeEntity: 3
  ([`data/features/generators/relative_entity.py`](../chai-lab/chai_lab/data/features/generators/relative_entity.py))
- RelativeChain: 6 (`2×s_max+2`, `s_max=2`)
  ([`data/features/generators/relative_chain.py`](../chai-lab/chai_lab/data/features/generators/relative_chain.py))
- DockingConstraintGenerator: 6 (5 dist bins + mask, one-hot)
- TokenDistanceRestraint: 7 (6 RBF radii + 1 mask indicator)
- TokenPairPocketRestraint: 7 (6 RBF radii + 1 mask indicator)

ATOM (395 total — verified: 260+1+130+1+3 = 395):
- AtomNameOneHot: 260 (4 chars × `one_hot(x, 65)`, TorchScript line 63)
- AtomRefCharge: 1
- AtomRefElement: 130 (`one_hot(x, 130)`, TorchScript line 73; 128 elements + 2 mask)
- AtomRefMask: 1
- AtomRefPos: 3

ATOM_PAIR (14 total — verified: 12+2 = 14):
- BlockedAtomPairDistogram: 12 (`one_hot(x, 12)`, TorchScript line 89)
- InverseSquaredBlockedAtomPairDistances: 2 (value + mask)

MSA (42 total — verified from `feature_embedding.pt` TorchScript line 274:
`torch.cat([IsPairedMSA[...,1], MSADataSource_onehot[...,6],
MSADeletionValue[...,1], MSAHasDeletion[...,1], MSAOneHot_onehot[...,33]], -1)`):
- IsPairedMSA: 1 — True where `msa_mask & (pairing_key != NO_PAIRING_KEY)`
- MSADataSource: 6 — `torch.one_hot(x, 6)` (TorchScript line 246). Values 0–4 at
  inference: BFD_UNICLUST(0), MGNIFY(1), UNIREF90(2), UNIPROT(3), NONE(4). QUERY(5) is
  always remapped to NONE(4) in
  [`generators/msa.py:243`](../chai-lab/chai_lab/data/features/generators/msa.py), so
  class 5 is effectively unused. The Python `can_mask=True` sets `mask_value=6`, but this
  value would crash `one_hot(x, 6)` and is never used — masked positions are filled with
  NONE(4) instead
  ([`data/parsing/msas/data_source.py:68–78`](../chai-lab/chai_lab/data/parsing/msas/data_source.py))
- MSADeletionValue: 1 — transformed as `(2/π) * arctan(deletion_count / 3)`
- MSAHasDeletion: 1
- MSAOneHot: 33 — `torch.one_hot(x, 33)` (TorchScript line 266). Vocab =
  20 AA + X + 5 RNA + 5 DNA + gap + non-existent
  ([`residue_constants.py:519–526`](../chai-lab/chai_lab/data/residue_constants.py))

TEMPLATES (76 total — verified: 2+3+32+39 = 76):
- TemplateMask: 2 (backbone + pseudo-beta)
- TemplateUnitVector: 3
- TemplateResType: 32 (via outer-sum `Embedding(33, 32)`)
- TemplateDistogram: 39 (`one_hot(x, 39)`, TorchScript line 276) — bin edges from
  `linspace(3.25, 50.75, 38)[1:]` (37 edges, spacing ~1.284 Å, first edge ~4.53 Å,
  last 50.75 Å). Distances are **pseudo-beta** (CB for protein, C2/C4 for nucleic
  acids, sole atom for atom tokens) via Euclidean `cdist`. Cross-asym pairs set to
  mask class 38. Absent/masked positions filled with 100.0 Å (last bin).

### 4.4 Bond Feature Projection

**Module**: [`bond_loss_input_proj.pt`](../chai-lab/downloads/models_v2/bond_loss_input_proj.pt)
(512 params)

A single `Linear(1, 512, bias=False)`. Output is split 256+256 and **added** to the
trunk and structure token-pair representations respectively
([`chai1.py:705–715`](../chai-lab/chai_lab/chai1.py)).

**Important**: Despite the name "bond_loss", this is NOT a bond prediction head. Covalent
bonds are **input features**, not predicted outputs. The `TokenBondRestraint` generator
([`data/features/generators/token_bond.py`](../chai-lab/chai_lab/data/features/generators/token_bond.py))
creates a binary `[n_tokens, n_tokens]` adjacency feature from the input structure's
covalent bond indices. The name "bond_loss" reflects training-time usage (a bond
reconstruction loss); at inference, this module simply projects bond topology features
into the pair representation.

---

## 5. Token Input Embedder

**Module**: [`token_embedder.pt`](../chai-lab/downloads/models_v2/token_embedder.pt) (~1.5M params)

Internal class: `chai.model.af3.token_input_emb.TokenInputEmbedding`

Called at [`chai1.py:721–738`](../chai-lab/chai_lab/chai1.py).

This module aggregates atom-level information to token-level representations using an
atom transformer with blocked local attention.

### 5.1 Architecture

```
token_single_input_feats [b, n_tokens, 384]
atom_single_input_feats  [b, n_atoms, 128]    ─┐
block_atom_pair_feat     [b, bl, q, kv, 16]   ─┤
                                                ▼
                                    AtomAttentionBlockedEncoder
                                    ├─ to_atom_cond: Linear(128, 128)
                                    ├─ Pair Update Block
                                    │   ├─ proj_h: Linear(128, 16)
                                    │   ├─ proj_w: Linear(128, 16)
                                    │   └─ MLP: Linear(16,16) → Linear(16,16)
                                    ├─ LocalDiffusionTransformer (3 blocks)
                                    │   ├─ blocked_pairs2blocked_bias:
                                    │   │    LN(16) → [3, 4, 16] per-head bias
                                    │   ├─ 3× LocalAttentionPairBiasBlock
                                    │   │    (4 heads, 32 head_dim, AdaLN)
                                    │   └─ 3× ConditionedTransitionBlock
                                    │        (SwiGLU, 4× expansion)
                                    └─ to_token_single: Linear(128, 384)
                                                │
                                                ▼ aggregate via atom_token_indices
                           concat [token_single(384) + atom_agg(384)] = 768
                                                │
                                    ┌───────────┴───────────┐
                    token_single_proj_in_trunk        token_single_proj_in_structure
                    Linear(768, 384, no bias)         Linear(768, 384, no bias)
                           │                                    │
                           ▼                                    ▼
              token_single_initial_repr [384]     token_single_structure_input [384]

              ┌─── token_pair_proj_in_trunk: Linear(256, 256, no bias) ───┐
              │    + outer_sum from single_to_pair: Linear(384, 512)      │
              ▼                                                           │
         token_pair_initial_repr [256]                                    │
```

### 5.2 Atom Transformer Details

The atom-level transformer uses **blocked local attention** (query block=32, key block=128):

**Local Attention Blocks** (3 blocks, each):
- **AdaLayerNorm**: Conditioned by atom_single. `lin_s_merged: Linear(128, 256)` produces
  scale and shift (128 each) for the 128-dim atom representation.
- **QKV projection**: `to_qkv.weight: [3, 4, 32, 128]` — 4 heads, 32 head_dim, producing
  q, k, v from 128-dim input
- **Query bias**: `q_bias: [4, 32]` — learned bias added to queries
- **Pair bias**: `blocked_pairs2blocked_bias`: `LN(16) → Linear(16, 3×4)` = 3 attention
  blocks × 4 heads, pair features provide per-head attention bias
- **Output projection**: `Linear(128, 128, bias=True)`
- **Attention scale**: 1/√32

**Conditioned Transition Blocks** (3 blocks, each):
- **AdaLayerNorm**: `lin_s_merged: Linear(128, 256)` for scale+shift
- **SwiGLU expansion**: `linear_a_nobias_double: Linear(128, 512)` — split to 256+256
  (one half through SiLU, then element-wise multiply)
- **Down projection**: `linear_b_nobias: Linear(256, 128)`
- **Gating**: `linear_s_biasinit_m2: Linear(128, 128, bias=True)` — sigmoid gate from
  conditioning signal, bias initialized to -2 (gate starts near sigmoid(-2) ≈ 0.12)

---

## 6. Trunk

**Module**: [`trunk.pt`](../chai-lab/downloads/models_v2/trunk.pt) (169,955,072 params)

The trunk consists of three sub-modules: template embedder, MSA module, and
Pairformer stack. Recycling is orchestrated in Python at
[`chai1.py:746–777`](../chai-lab/chai_lab/chai1.py).

TorchScript submodule hierarchy (from `trunk.pt`):
```
trunk
├── token_single_recycle_proj   (LN → Linear)
├── token_pair_recycle_proj     (LN → Linear)
├── template_embedder
│   ├── proj_in                 (LN → Linear)
│   ├── pairformer              (2 blocks)
│   ├── template_layernorm
│   └── proj_out
├── msa_module
│   ├── linear_s2m
│   ├── outer_product_mean      (4 blocks)
│   ├── msa_pair_weighted_averaging (3 blocks)
│   ├── msa_transition          (3 blocks)
│   ├── pair_transition         (4 blocks)
│   ├── triangular_multiplication (4 blocks)
│   └── triangular_attention    (4 blocks)
└── pairformer_stack
    ├── blocks                  (48 blocks)
    └── squash_norm             (no-op at inference)
```

### 6.1 Recycle Projections

Before each recycle iteration, the previous output is projected and added to the
initial representations:

- `token_single_recycle_proj`: `LayerNorm(384)` → `Linear(384, 384, no bias)`
- `token_pair_recycle_proj`: `LayerNorm(256)` → `Linear(256, 256, no bias)`

Default: **3 recycles**.

### 6.2 Template Embedder

Processes up to 4 structural templates through a small Pairformer:

```
Template features [b, 4, n, n, 64]
    │
    ▼
proj_in: LayerNorm(256) → Linear(256, 64, no bias)
    │
    ▼
2× PairformerBlock (at dim 64):
    ├─ transition_pair: LN(64) → SwiGLU(64→256→128→64)
    ├─ triangle_multiplication:
    │   ├─ layernorm_z_in: LN(64)
    │   ├─ merged_linear_p: Linear(64, 256, no bias)
    │   ├─ merged_linear_g: Linear(64, 320, no bias) → sigmoid gated
    │   └─ linear_z_out: Linear(64, 64, no bias)
    └─ triangle_attention:
        ├─ pair2b: Linear(64, 8, no bias) → 8 heads bias
        ├─ pair2qkvg1: Linear(64, 512, no bias) → starting node
        ├─ pair2qkvg2: Linear(64, 512, no bias) → ending node
        ├─ linear_out: Linear(256, 64, no bias)
        └─ out_scalers: [64] learned per-channel output scaling
    │
    ▼
template_layernorm: LayerNorm(64) → proj_out: Linear(64, 256, no bias)
    │
    ▼
Added to pair representation [b, n, n, 256]
```

### 6.3 MSA Module

The MSA module converts MSA information into pair representation updates. It runs
**4 iterations** with an asymmetric block structure:

| Sub-module | Count | Operates On |
|-----------|-------|-------------|
| `linear_s2m` | 1 | Single → MSA projection |
| `outer_product_mean` | 4 | MSA → Pair update |
| `msa_pair_weighted_averaging` | 3 | Pair → MSA (pair-biased attention) |
| `msa_transition` | 3 | MSA → MSA (feed-forward) |
| `pair_transition` | 4 | Pair → Pair (feed-forward) |
| `triangular_multiplication` | 4 | Pair → Pair (triangle update) |
| `triangular_attention` | 4 | Pair → Pair (triangle attention) |

#### Iteration Order

```
linear_s2m: project single → MSA (once at start)

For i in 0..3:
    outer_product_mean[i]: MSA → pair update
    pair_transition[i]: pair feed-forward

    if i < 3:
        msa_pair_weighted_averaging[i]: pair-biased MSA attention
        msa_transition[i]: MSA feed-forward

    triangular_multiplication[i]: pair triangle update (both directions)
    triangular_attention[i]: pair triangle attention (both directions)
```

The final iteration (i=3) only updates the pair representation — the MSA
representation is discarded after being consumed into pairs via the outer product mean.

#### 6.3.1 Single-to-MSA Projection

`linear_s2m: Linear(384, 64, no bias)` — projects token single representation to MSA
dimension, added to the first row of the MSA.

#### 6.3.2 Outer Product Mean (4 blocks)

Each block:
- `weight_ab: [2, 8, 8, 64]` — two projections (a, b) from MSA dim 64 to 8-dim
- Einsum patterns: `"abc, defc -> abdef"` and `"abcde, afcdg -> cegabf"`
- `ln_out: LayerNorm(512)` — normalizes the outer product
- `linear_out: Linear(512, 256, bias=True)` — projects to pair dimension

#### 6.3.3 MSA Pair-Weighted Averaging (3 blocks)

Each block:
- `layernorm_msa: LayerNorm(64)` — normalize MSA
- `linear_msa2vg: Linear(64, 512, no bias)` — **8 heads × (32 value + 32 gate)**
- `layernorm_pair: LayerNorm(256)` — normalize pair representation
- `linear_pair: Linear(256, 8, no bias)` — per-head attention weights from pair
- `linear_out_no_bias: Linear(256, 64, no bias)` — project gated values (8×32=256) to MSA dim

#### 6.3.4 MSA Transition (3 blocks)

Each: `LayerNorm(64)` → `Linear(64, 512, no bias)` → SwiGLU → `Linear(256, 64)`.
Expansion: **4×**.

#### 6.3.5 Pair Transition (4 blocks)

Each: `LayerNorm(256)` → `Linear(256, 2048, no bias)` → SwiGLU → `Linear(1024, 256)`.
Expansion: **4×**.

#### 6.3.6 Triangular Multiplication (4 blocks)

Both **outgoing and incoming** contractions are computed in every block:

```python
z_normed = LayerNorm(z)
p = merged_linear_p(z_normed)           # [b, n, n, 4c] no bias
g = sigmoid(merged_linear_g(z_normed))  # [b, n, n, 5c] no bias

ab = p * g[..., :4c]                    # gate the projections
ab_left, ab_right = chunk(ab, 2)        # split into two edge groups

# Apply pair masks (rows for left, transposed for right)
a1, b1 = chunk(masked(ab_left), 2)
a2, b2 = chunk(masked(ab_right), 2)

x_out = einsum("... i k d, ... j k d -> ... i j d", a1, b1)  # outgoing
x_in  = einsum("... k i d, ... k j d -> ... i j d", a2, b2)  # incoming

output = linear_z_out(LayerNorm(x_out) + LayerNorm(x_in)) * g[..., 4c:]
z += feature_dropout(output)
```

Weight shapes: `merged_linear_p: Linear(256, 1024)`, `merged_linear_g: Linear(256, 1280)`,
`linear_z_out: Linear(256, 256)`.

#### 6.3.7 Triangular Attention (4 blocks)

Both **starting-node and ending-node** attention are computed in every block via two
`scaled_dot_product_attention` calls per block:

- `pair2qkvg1: Linear(256, 1024, no bias)` — starting node projection: **8 heads × (q32 + k32 + v32 + g32)**
- `pair2qkvg2: Linear(256, 1024, no bias)` — ending node projection: same decomposition
- `pair2b: Linear(256, 8, no bias)` — per-head attention bias
- `linear_out: Linear(512, 256, no bias)` — 512 = 8 heads × 32 v_dim × 2 directions
- `out_scalers: [256]` — learned per-channel output scaling

The pair2qkvg outputs are split into q, k, v, g via `unbind` (4 equal components).
Both directions' attention outputs are combined and gated by sigmoid(g).

#### 6.3.8 MSA Pairing Algorithm

MSAs from multiple chains are paired by species before being merged
([`data/dataset/msas/preprocess.py`](../chai-lab/chai_lab/data/dataset/msas/preprocess.py)):

1. **Pairing keys**: Each MSA row has a string pairing key (typically a species/taxonomy
   identifier; for ColabFold paired outputs, the row index). Keys are hashed to stable
   ints via SHA-256 (first 7 hex chars → int). `NO_PAIRING_KEY = -999991` marks
   unpaired/absent/padding rows.

2. **Ranking**: For each chain's MSA, compute edit distance (Hamming) of every row vs
   the query sequence (row 0). Rows sharing the same pairing key hash are ranked by
   ascending edit distance, producing a unique key `(hash, rank)`.

3. **Cross-chain matching**: A `(hash, rank)` is selected for pairing only if it appears
   in **every** non-empty chain MSA. At most `MAX_PAIRED_DEPTH = 8,192` unique keys are
   kept.

4. **Merge**: Per-chain MSAs are reordered with **paired rows first** (same order of
   selected keys), then unpaired rows, truncated to `FULL_DEPTH = 16,384`.
   `merge_main_msas_by_chain` concatenates along the token dimension, padding to common
   depth.

5. **Inference subsampling**: Optional (`recycle_msa_subsample`), keeps up to **4,096**
   rows via biased random scoring.

#### 6.3.9 Mask Propagation

Masks are constructed in Python and passed into TorchScript modules:

- **Token pair mask**: `und_self(token_exists_mask, "b i, b j -> b i j")` — outer
  product of the token exists mask. Passed to `trunk.forward()` as `token_pair_mask`.
- **MSA mask**: `main_msa_mask` (bool, `[b, depth, tokens]`) gates profile computation
  via `scatter_add` and deletion mean via `masked_mean`. An all-False mask (empty MSA)
  zeros out all MSA-derived features.
- **Atom-level masks**: `block_atom_pair_mask` (see §3.3) and `atom_single_mask` from
  `atom_exists_mask` are passed to the token embedder and diffusion module.
- **No causal masking** is used anywhere in the pipeline.

### 6.4 Pairformer Stack (48 blocks)

Each of the 48 identical blocks executes 5 sub-layers in this order:

```
z, s = pair_repr, single_repr

1. z += PairTransition(z)              # SwiGLU on pair repr
2. s += AttentionPairBias(s, z)        # single self-attn biased by pair
3. s += SingleTransition(s)            # SwiGLU on single repr
4. z += TriangularMultiplication(z)    # both outgoing + incoming
5. z += TriangleAttention(z)           # both starting + ending node
```

Total SDPA calls per block: **3** (1 attention_pair_bias + 2 triangle_attention).

A `SquashNorm` module is called between blocks but is a **no-op at inference** (all
methods return `None`; active only during training for gradient/activation management).

#### 6.4.1 Pair Transition

`LayerNorm(256)` → `Linear(256, 1024, no bias)` → SwiGLU → `Linear(512, 256)`.
Expansion: **4×**.

#### 6.4.2 Attention with Pair Bias

- `single_layer_norm: LayerNorm(384)`, `pair_layer_norm: LayerNorm(256)`
- `pair_linear: Linear(256, 16, no bias)` — pair → per-head attention bias
- `input2qkvg.weight: [384, 4, 16, 24]` — **16 heads, 24 head_dim**, producing q, k, v, g
- `query_bias: [16, 24]` — learned query bias per head
- `output_proj.weight: [16, 24, 384]` — multi-head output → single dim

Attention: softmax(q·k^T / √24 + pair_bias) · v, gated by sigmoid(g).

#### 6.4.3 Single Transition

`LayerNorm(384)` → `Linear(384, 1536, no bias)` → SwiGLU → `Linear(768, 384)`.
Expansion: **4×**.

#### 6.4.4 Triangle Multiplication

Same architecture as MSA module (§6.3.6). Operates at pair dim 256.

#### 6.4.5 Triangle Attention

Same architecture as MSA module (§6.3.7). 8 heads, 32 head_dim, at pair dim 256.

### 6.5 Summary Table

| Block Type | Count | Key Dimensions |
|-----------|-------|---------------|
| Template Pairformer | 2 | pair=64, 8 heads, 16 head_dim |
| MSA outer product mean | 4 | MSA=64 → pair=256, 8×8 outer |
| MSA pair-weighted avg | 3 | 8 heads, 32 value_dim |
| MSA transition | 3 | 64→512→256→64 (SwiGLU 4×) |
| Pair transition (MSA) | 4 | 256→2048→1024→256 (SwiGLU 4×) |
| Triangular mult (MSA) | 4 | pair=256, sigmoid gate, both dirs |
| Triangular attn (MSA) | 4 | pair=256, 8 heads, 32 head_dim, both dirs |
| Pairformer blocks | 48 | single=384, pair=256 |
| — attention pair bias | 48 | 16 heads, 24 head_dim, gated |
| — transition single | 48 | 384→1536→768→384 (SwiGLU 4×) |
| — transition pair | 48 | 256→1024→512→256 (SwiGLU 4×) |
| — triangle mult | 48 | pair=256, sigmoid gate, both dirs |
| — triangle attn | 48 | pair=256, 8 heads, 32 head_dim, both dirs |

---

## 7. Diffusion Module

**Module**: [`diffusion_module.pt`](../chai-lab/downloads/models_v2/diffusion_module.pt)
(127,928,768 params)

The diffusion module predicts 3D atomic coordinates via denoising diffusion. It consists
of four sub-systems: conditioning, atom attention encoder, diffusion transformer, and
atom attention decoder. The diffusion loop is orchestrated in Python at
[`chai1.py:844–886`](../chai-lab/chai_lab/chai1.py).

TorchScript submodule hierarchy (from `diffusion_module.pt`):
```
diffusion_module
├── diffusion_conditioning
│   ├── token_pair_proj         (LN → Linear(512, 256))
│   ├── token_in_proj           (LN → Linear(768, 384))
│   ├── single_trans1, single_trans2   (SwiGLU 384→1536→768→384)
│   ├── pair_trans1, pair_trans2       (SwiGLU 256→1024→512→256)
│   ├── fourier_embedding       (weights[256], bias[256])
│   ├── fourier_proj            (LN(256) → Linear(256, 384))
│   ├── single_ln               (LN(384))
│   └── pair_ln                 (LN(256))
├── atom_attention_encoder      (same arch as token embedder's encoder)
│   ├── to_atom_cond            (Linear(128, 128))
│   ├── token_to_atom_single    (LN(384) → Linear(384, 128))
│   ├── prev_pos_embed          (Linear(3, 128, no bias))
│   ├── pair_update_block
│   ├── token_pair_to_atom_pair (LN(256) → Linear(256, 16))
│   ├── atom_transformer        (3 local attn blocks + 3 transitions)
│   └── to_token_single         (Linear(128, 768, no bias))
├── diffusion_transformer
│   └── blocks                  (16 blocks, 768-dim)
├── atom_attention_decoder
│   ├── token_to_atom           (Linear(768, 128, no bias))
│   ├── atom_transformer        (3 local attn blocks + 3 transitions)
│   └── to_pos_updates          (LN(128) → Linear(128, 3, no bias))
├── structure_cond_to_token_structure_proj (Linear(384, 768, no bias))
├── post_attn_layernorm         (LN(768))
└── post_atom_cond_layernorm    (LN(128))
```

### 7.1 Diffusion Conditioning

Prepares conditioning signals from trunk outputs and noise level σ:

```
token_single_initial[384] ++ token_single_trunk[384] = 768
    │
    ▼
token_in_proj: LayerNorm(768) → Linear(768, 384, no bias)
    │
    ▼
single_trans1: SwiGLU transition (384, 4×)
    │
    ▼
 + fourier_embedding(σ) → fourier_proj: LayerNorm(256) → Linear(256, 384)
    │
    ▼
single_trans2: SwiGLU transition (384, 4×)
    │
    ▼
single_ln: LayerNorm(384) → s_cond [b, n, 384]


token_pair_initial[256] ++ token_pair_trunk[256] = 512
    │
    ▼
token_pair_proj: LayerNorm(512) → Linear(512, 256, no bias)
    │
    ▼
pair_trans1 → pair_trans2: 2× SwiGLU transition (256, 4×)
    │
    ▼
pair_ln: LayerNorm(256) → z_cond [b, n, n, 256]
```

**Fourier Embedding**: `weights: [256], bias: [256]` — random Fourier features for σ.
Produces `cos((weights · ln(σ)/4 + bias) · 2π)` → 256-dim, projected to 384 and added
to the single conditioning signal between the two transition blocks. See §7.6.0 for the
full EDM preconditioning constants.

### 7.2 Atom Attention Encoder

Same architecture as the token embedder's atom encoder (§5.2), but **weights are not
shared** — `token_embedder.pt` and `diffusion_module.pt` are separate artifacts loaded
independently. The diffusion encoder has additional components and a different output
width (768 vs 384). The atom attention decoder (§7.4) is likewise a separate submodule
within `diffusion_module.pt`, not weight-tied to either encoder.

Additions beyond the token embedder's encoder:

- `to_atom_cond: Linear(128, 128)` — condition atoms from structure features
- `token_to_atom_single: LayerNorm(384) → Linear(384, 128)` — broadcast token→atom
- `prev_pos_embed: Linear(3, 128, no bias)` — embeds noised atom positions
- `pair_update_block`: same as token embedder (proj_h/w + MLP at dim 16)
- `token_pair_to_atom_pair: LayerNorm(256) → Linear(256, 16)` — pair→atom pair
- 3 local attention blocks + 3 transitions (4 heads, 32 head_dim)
- `to_token_single: Linear(128, 768, no bias)` — aggregate to token at **768** dim

### 7.3 Diffusion Transformer (16 blocks)

Each of the **16 blocks** performs conditioned pair-bias self-attention at **768-dim**:

```python
# x: [b, n, 768]  s_cond: [b, n, 384]  z_cond: [b, n, n, 256]

# 1. Conditioned attention
scale, shift = chunk(lin_s_merged(s_cond), 2)   # AdaLN
x_norm = LayerNorm(x) * (1 + scale) + shift

q, k, v = chunk(to_qkv(x_norm), 3)              # 16 heads × 48 head_dim
q += q_bias
pair_bias = pair_linear(LayerNorm(z_cond))       # per-head bias [b, n, n, 16]

attn_out = SDPA(q, k, v, bias=pair_bias)
gate = sigmoid(gate_proj(s_cond))                # [b, n, 768]
x += to_out(attn_out) * gate

# 2. Conditioned transition
scale, shift = chunk(ada_ln(s_cond), 2)
x_norm = LayerNorm(x) * (1 + scale) + shift

a, b = chunk(linear_a(x_norm), 2)               # SwiGLU 4× expansion
transition_out = linear_b(silu(a) * b)
gate = sigmoid(linear_s(s_cond))                 # bias init -2
x += transition_out * gate
```

**Configuration**: **16 heads, 48 head_dim, 768 working dim**, conditioned by 384-dim
signals. The 768 working dim is 2× the single representation width.

### 7.4 Atom Attention Decoder

Mirrors the encoder to decode token-level predictions back to atomic positions:

- `token_to_atom: Linear(768, 128, no bias)` — broadcast token→atom
- 3 local attention blocks + 3 transitions (4 heads, 32 head_dim)
- `to_pos_updates: LayerNorm(128) → Linear(128, 3, no bias)` — output xyz deltas

### 7.5 Post-Processing

- `structure_cond_to_token_structure_proj: Linear(384, 768, no bias)` — initial
  projection of structure conditioning to 768 working dim
- `post_attn_layernorm: LayerNorm(768)`, `post_atom_cond_layernorm: LayerNorm(128)`

### 7.6 Noise Schedule and Sampling

EDM (Karras et al. 2022) Algorithm 2 with stochastic churn and second-order Heun
correction. Implementation in
[`model/diffusion_schedules.py`](../chai-lab/chai_lab/model/diffusion_schedules.py) and
[`chai1.py:821–886`](../chai-lab/chai_lab/chai1.py).

### 7.6.0 EDM Parameterization (Preconditioning)

The network output is preconditioned following Karras et al. 2022 Table 1 (VP column
adapted for protein coordinates). Constants extracted from `diffusion_module.pt` constants
archive:

| Constant | Value | Role |
|----------|-------|------|
| c0 | 0.25 | `c_noise = ln(σ) × 0.25` — Fourier log-sigma scaling |
| c1 | 1 (int) | Identity multiplier for TorchScript size math |
| c2 | 2π | Fourier frequency factor (`cos(w·ln(σ) + b) × 2π`) |
| c3 | 256.0 | σ_data² = 16² |
| c4 | 2 (int) | Block index centering divisor |
| c5 | 16 (int) | Number of heads (diffusion transformer) |
| c6 | 16.0 | σ_data |

The denoising output formula (visible at `diffusion_module.pt` TorchScript lines
3502–3509):

```
c_in   = (σ² + σ_data²)^(-0.5)              # input scaling
c_skip = σ_data² / (σ² + σ_data²)           # skip connection weight
c_out  = σ · σ_data / √(σ² + σ_data²)       # output scaling
c_noise = ln(σ) / 4                          # noise conditioning

D(x; σ) = c_skip · x + c_out · F_θ(c_in · x; c_noise(σ))
```

Where `F_θ` is the full encoder → transformer → decoder network. The network predicts
a "unit noise error" which is scaled by `c_out` and combined with the skip-connected
noised input scaled by `c_skip`.

#### 7.6.1 Noise Schedule

```
σ(t) = σ_data · (t · s_min^(1/p) + (1-t) · s_max^(1/p))^p
```

| Parameter | Value |
|-----------|-------|
| σ_data | 16.0 |
| s_max | 80.0 |
| s_min | 4e-4 |
| p | 7.0 |
| S_churn | 80 |
| S_tmin | 4e-4 |
| S_tmax | 80.0 |
| S_noise | 1.003 |
| Timesteps | 200 |
| 2nd order | True (Heun) |

#### 7.6.2 Timestep Spacing

200 time points via `torch.linspace(0, 1, 2*200+1)[1::2]` — **uniformly spaced in
t ∈ (0, 1)**, strictly interior (never hits 0 or 1). σ(t) follows the power-7 curve
above, so spacing is **nonlinear in σ** (denser at low noise). The sampling loop
iterates over **199 consecutive σ pairs** (`zip(sigmas[:-1], sigmas[1:], gammas[:-1])`).

#### 7.6.3 Stochastic Churn

```
gamma = min(S_churn / num_timesteps, sqrt(2) - 1)   if S_tmin ≤ σ ≤ S_tmax
gamma = 0                                            otherwise
```

Since early σ values (after scaling by σ_data=16) can exceed S_tmax=80, the first
high-noise steps have gamma=0 (no stochastic inflation).

#### 7.6.4 Sampling Loop (per step)

```python
# 1. SE(3) augmentation (before noise injection)
atom_pos = center_random_augmentation(atom_pos, atom_single_mask)

# 2. Stochastic noise inflation
sigma_hat = sigma_curr + gamma * sigma_curr
noise = S_noise * randn_like(atom_pos)
atom_pos_hat = atom_pos + noise * sqrt(clamp(sigma_hat² - sigma_curr², min=1e-6))

# 3. First denoising + Euler step
denoised = denoise(atom_pos_hat, sigma_hat)
d_i = (atom_pos_hat - denoised) / sigma_hat
atom_pos = atom_pos_hat + (sigma_next - sigma_hat) * d_i

# 4. Second-order Heun correction (if second_order=True and sigma_next != 0)
denoised = denoise(atom_pos, sigma_next)
d_i_prime = (atom_pos - denoised) / sigma_next
atom_pos = atom_pos_hat + (sigma_next - sigma_hat) * (d_i + d_i_prime) / 2
```

The Heun correction replaces the Euler result with a trapezoid average of two score
estimates. It is skipped at the final step (sigma_next=0).

#### 7.6.5 SE(3) Augmentation

Applied at the **start of each diffusion step**, before noise injection
([`model/utils.py:center_random_augmentation`](../chai-lab/chai_lab/model/utils.py),
called at [`chai1.py:849–856`](../chai-lab/chai_lab/chai1.py)):

1. **Centering**: Compute one centroid per batch item via masked mean over all atoms
   (**global**, not per-chain). Subtract centroid. Denominator clamped at 1e-4.
2. **Rotation**: Uniform SO(3) via random quaternions → rotation matrix, applied as
   `einsum("b i j, b a j -> b a i", R, centered_coords)`.
3. **Translation**: `s_trans * randn_like(centroid)` with `s_trans = 1.0`, added to
   all atom positions.

---

## 8. Confidence Head

**Module**: [`confidence_head.pt`](../chai-lab/downloads/models_v2/confidence_head.pt)
(14,812,416 params)

Called once per diffusion sample in a sequential loop at
[`chai1.py:894–910`](../chai-lab/chai_lab/chai1.py).

TorchScript submodule hierarchy (from `confidence_head.pt`):
```
confidence_head
├── single_to_pair_proj         (Linear(384, 512, no bias))
├── atom_distance_bins_projection (Linear(16, 256, no bias))
├── blocks                      (4 Pairformer blocks)
├── plddt_projection            (Linear(384, 1850))
├── pae_projection              (Linear(256, 64))
└── pde_projection              (Linear(256, 64))
```

### 8.1 Input Processing

- `single_to_pair_proj: Linear(384, 512, no bias)` → chunk to 2×256, outer sum for
  pair initialization
- Predicted atom positions are selected at representative atoms via
  `token_reference_atom_index`, then pairwise distances computed via `torch.cdist`
  (visible at `confidence_head.pt` TorchScript line 521)
- Distances binned via `searchsorted` with **15 bin edges** (from `confidence_head.pt`
  attribute `atom_distance_v_bins`):
  `[3.375, 4.661, 5.946, 7.232, 8.518, 9.804, 11.089, 12.375, 13.661, 14.946,
   16.232, 17.518, 18.804, 20.089, 21.375]` — uniformly spaced at ~1.286 Å
- One-hot to **16 classes** (15 bins + 1 overflow)
- `atom_distance_bins_projection: Linear(16, 256, no bias)` → added to pair repr

### 8.2 Pairformer Blocks (4 blocks)

Same execution order as trunk Pairformer blocks (§6.4):

**Identical to trunk**:
- `transition_pair`: SwiGLU at dim 256 (4×)
- `triangle_multiplication`: Same as trunk, sigmoid-gated, both directions
- `transition_single`: SwiGLU at dim 384 (4×)
- `attention_pair_bias`: 16 heads, 24 head_dim (identical to trunk)

**Different triangle attention variant**:
- `pair2qkvgb: Linear(256, 2056, no bias)` — fused single-projection:
  **8 heads × (q64 + k64 + v64 + g64 + bias1) = 2056**
- `linear_out: Linear(512, 512, no bias)` — 8 heads × 64 v_dim
- Both directions computed via chunk(2), producing 2 SDPA calls per block

This uses **8 heads at 64 head_dim** (vs the trunk's 8 heads at 32 head_dim), and
fuses q, k, v, g, and per-head bias into a single projection.

### 8.3 Output Heads

| Head | Projection | Shape | Bins |
|------|-----------|-------|------|
| pLDDT | `Linear(384, 1850)` | `[b, n_tok, 37, 50]` → `[b, n_atoms, 50]` | 50 bins, [0, 1] |
| PAE | `Linear(256, 64)` | `[b, n_tok, n_tok, 64]` | 64 bins, [0, 32]Å |
| PDE | `Linear(256, 64)` | `[b, n_tok, n_tok, 64]` | 64 bins, [0, 32]Å |

pLDDT: 1850 = **37 atom positions × 50 bins** per token, scattered to atom-level via
`atom_within_token_index`.

---

## 9. Ranking and Scoring

Ranking is computed in Python at
[`chai1.py:988–1015`](../chai-lab/chai_lab/chai1.py), delegating to
[`ranking/rank.py`](../chai-lab/chai_lab/ranking/rank.py).

### 9.1 Aggregate Score

```
aggregate_score = 0.2 × pTM + 0.8 × ipTM − 100 × has_inter_chain_clashes
```

([`ranking/rank.py:94–99`](../chai-lab/chai_lab/ranking/rank.py))

### 9.2 TM-Score Normalization

```
d0 = 1.24 · (max(N, 19) − 15)^(1/3) − 1.8
TM_pair(i,j) = 1 / (1 + (PAE_ij / d0)²)
```

- **pTM**: Max over alignment rows of sum of pairwise TM scores, normalized by total tokens
- **ipTM**: Max over chains c of pTM(c vs rest), weighted 4× in aggregate score
- **Clash detection**
  ([`ranking/clashes.py`](../chai-lab/chai_lab/ranking/clashes.py)): Pairwise atom
  distances via `cdist`;
  clash if distance < **1.1 Å** (valid atom pairs, not self). Aggregated to chain–chain
  matrices. `has_inter_chain_clashes` uses thresholds `max_clashes=100`,
  `max_clash_ratio=0.5`, **polymer–polymer chain pairs only**.

### 9.3 Post-Processing

**No energy minimization, relaxation, or structural correction is applied.** Clashes
only affect the ranking score (−100 penalty); they are not repaired. All diffusion
samples are written to separate CIF files regardless of quality.

CIF export ([`data/io/cif_utils.py`](../chai-lab/chai_lab/data/io/cif_utils.py)) uses the
`modelcif` library. B-factors are set
from pLDDT scores. Ligand atoms receive unique names via a counter suffix (`_1`, `_2`,
etc.) to avoid duplicate atom IDs.

Atom symmetries (intra-residue, computed via RDKit at tokenization time) are stored in
`AllAtomStructureContext` but are **not applied** as post-processing during inference.
They are relevant only for training loss computation.

---

## 10. Handling of Optional Inputs

Chai-1 can run with any combination of MSAs, templates, and ESM embeddings.

### 10.1 ESM Embeddings (absent)

Zero tensor of shape `[n_tokens, 2560]`
([`data/dataset/embeddings/embedding_context.py:EmbeddingContext.empty`](../chai-lab/chai_lab/data/dataset/embeddings/embedding_context.py)),
concatenated into the TOKEN feature vector and projected through `Linear(2638, 384)`.
No learned mask embedding — absence is zeros.

### 10.2 MSA (absent)

[`MSAContext.create_empty`](../chai-lab/chai_lab/data/dataset/msas/msa_context.py)
fills tokens with the non-existent character `":"` (index for "should get masked", per
[`residue_constants.py:525`](../chai-lab/chai_lab/data/residue_constants.py)),
sets mask to all-False, deletion matrix to zeros, pairing keys to
`NO_PAIRING_KEY = -999991`, and source to `MSADataSource.NONE`. The all-False mask
zeros out MSA profile, deletion mean, and pair-weighted averaging attention.

### 10.3 Templates (absent)

[`TemplateContext.empty`](../chai-lab/chai_lab/data/dataset/templates/context.py) fills
residue types with gap `"-"` and all masks to False. The template embedder processes
these but the all-zero masks prevent information flow.

### 10.4 Constraints (absent)

- **Contact / pocket constraints**: Feature filled with `-1.0`. The RBF encoding
  detects this and produces zeroed encoding with a concatenated `should_mask` indicator.
- **Docking constraints**: Filled with `mask_value = 6` (special masked one-hot class).

At inference, contact and pocket have `include_probability=1.0` (always included when
provided); docking has `include_probability=0.0` (only from explicit specifications).

### 10.5 Non-Protein ESM Tokens

DNA, RNA, and ligand chains receive zero-vector ESM embeddings. Modified residues use
their canonical parent residue; unknown residues use "X" (token ID 24).

---

## 11. Key Differences from AlphaFold 3

Chai-1 values are verified from the TorchScript `.pt` files. AF3 values are from
Abramson et al. 2024 (Nature) and the
[preprint modifications document](../auxiliary/preprint-af3-modifications.md).

### 11.1 Architectural Differences

1. **ESM2-3B language model** embeddings (2560-dim) as first-class TOKEN input
2. **Constraint features**: Contact (RBF 6), pocket (RBF 6), docking (one-hot 6 bins)
3. **Covalent bond features**: Separate projection pathway
4. **MSA module asymmetry**: 4 OPM blocks but only 3 MSA attention/transition blocks
5. **Confidence head triangle attention**: Fused single-projection (2056-dim) with
   8 heads at 64 head_dim, vs trunk's dual-projection with 32 head_dim
6. **pLDDT output**: 37 atom positions × 50 bins = 1850 per token
7. **Triangular operations compute both directions** in every block (not alternating)

### 11.2 Dimension Comparison

| Component | Chai-1 | AF3 |
|-----------|--------|-----|
| Token single dim | 384 | 384 |
| Token pair dim | **256** | 128 |
| MSA dim | 64 | 64 |
| Atom single dim | 128 | 128 |
| Atom pair dim | 16 | 16 |
| Pairformer blocks | 48 | 48 |
| Pairformer heads | 16 | 16 |
| Pairformer head dim | 24 | 24 |
| Diffusion blocks | **16** | 24 |
| Diffusion heads | 16 | 16 |
| Diffusion head dim | **48** | 64 |
| Diffusion working dim | 768 | 768 |
| Confidence blocks | 4 | 4 |
| Template blocks | 2 | 2 |
| Atom attn blocks | 3 | 3 |
| Atom attn heads | 4 | 4 |
| Atom attn head dim | 32 | 32 |

Notable: Chai-1 uses **pair dim 256** (vs AF3's 128), **16 diffusion blocks**
(vs 24), and **48 diffusion head dim** (vs 64).

---

## 12. Layer Building Blocks

### 12.1 SwiGLU Transition

```python
x_norm = LayerNorm(x)
a, b = chunk(Linear_no_bias(x_norm), 2)
x = x + Linear(SiLU(a) * b)
```

Expansion factor is **4×** across all modules.

### 12.2 AdaLayerNorm

```python
scale, shift = chunk(Linear(conditioning), 2)
x_norm = LayerNorm(x) * (1 + scale) + shift
```

Used in diffusion transformer and atom transformer.

### 12.3 Gated Output (bias init -2)

```python
gate = sigmoid(Linear(conditioning, bias=True))  # bias init -2 → starts ~0.12
output = gate * value
```

Used in conditioned transitions and diffusion attention.

### 12.4 Blocked Local Attention

Atom-level attention uses a blocked local structure (see §3.3 for geometry):

- **Query block**: 32 atoms (non-overlapping, consecutive)
- **KV window**: 128 atoms, **centered** on the query block (Q at positions 48–79)
- **Overlap**: consecutive KV windows share 96 atoms
- **Boundary**: modular wrap with invalid positions masked out
- **Pair bias**: `blocked_pairs2blocked_bias` projects `[b, bl, 32, 128, 16]` pair
  features through `LayerNorm(16) → Linear(16, n_blocks × n_heads)` to produce
  per-head attention bias for each local block. For the atom transformer: 3 attention
  blocks × 4 heads = 12 bias channels, reshaped per-block per-layer.

### 12.5 Normalization Summary

| Location | Type |
|----------|------|
| Transitions, attention | Standard LayerNorm (weight + bias) |
| Atom transformer, diffusion transformer | AdaLayerNorm (conditioned scale+shift) |
| Triangular multiplication (intermediate) | Standard LayerNorm (weight + bias) |
| Post-diffusion | Standard LayerNorm |

---

## 13. Inference Implementation Notes

Details relevant to porting inference to MLX.

### 13.1 Weight Storage and Loading

- Each component is a separate TorchScript artifact loaded via `torch.jit.load`
  ([`chai1.py:139–148`](../chai-lab/chai_lab/chai1.py))
- A `_component_cache` (dict keyed by filename) reuses loaded modules across calls
  (e.g., `trunk.pt` is loaded once and reused across 3 recycle iterations)
  ([`chai1.py:151–166`](../chai-lab/chai_lab/chai1.py))
- **No cross-file weight sharing**: each `.pt` file has fully independent parameters
- Components are moved to/from GPU on demand via `_component_moved_to` context manager
  (low-memory mode moves back to CPU after each use)
- Each `.pt` file exports **size-specific forward methods**: `forward_256`, `forward_384`,
  `forward_512`, `forward_768`, `forward_1024`, `forward_1536`, `forward_2048`. The
  `ModuleWrapper` class dispatches to the correct method via
  `getattr(jit_module, f"forward_{crop_size}")`
  ([`chai1.py:115–136`](../chai-lab/chai_lab/chai1.py))

### 13.2 Numerical Precision

All six Chai-1 `.pt` modules store weights in **float32**. ESM2-3B is stored as **fp16**.

- Tensors passed to diffusion and confidence modules are explicitly cast to fp32
- Rigid transforms ([`tools/rigid.py`](../chai-lab/chai_lab/tools/rigid.py)) force fp32
  with `torch.autocast("cuda", enabled=False)`
- Softmax uses explicit fp32 upcast: `logits.float().softmax(dim=-1)`
- No global `torch.autocast` wrapping the main inference loop

**Numerical stability constants**:

| Location | Value | Purpose |
|----------|-------|---------|
| `calc_centroid` | `clamp(min=1e-4)` | Denominator for masked mean |
| Diffusion sigma sqrt | `clamp_min(1e-6)` | Inside `sqrt(σ_hat² - σ_curr²)` |
| Pairwise distances | `eps=1e-10` | Inside sqrt in `_naive_pairwise_distances` |
| Template unit vectors | `eps=1e-12` | `torch.rsqrt(eps + ...)` |
| Rigid transforms | `1e-8` to `1e-20` | Various quaternion/rotation operations |
| `searchsorted` epsilon | `+1e-4` | Relative sequence separation bin assignment |

### 13.3 Dropout

All dropout in the exported models is effectively **disabled**. The trunk and confidence
head contain `aten::feature_dropout` calls with `probability=0.0` and `training=True`
baked in — these are no-ops. No other dropout calls exist in any module.

- **Docking constraint masking** (Python-side): `structure_dropout_prob=0.75`,
  `chain_dropout_prob=0.75` — stochastic feature zeroing in the data pipeline, not
  neural dropout.
- **ESM2**: `.eval()` is called, disabling any internal dropout.

### 13.4 SquashNorm

Called between Pairformer blocks in the trunk. Two calls per block (one for single repr,
one for pair repr). In the exported model, all implementations return `None`
unconditionally — a **pure no-op at inference**. Not present in any other module.

### 13.5 The 32 Named Features

The complete feature list, as registered in `chai1.py` lines 172–235:

| # | Generator | Type | Encoding |
|---|-----------|------|----------|
| 1 | `RelativeSequenceSeparation` | TOKEN_PAIR | ONE_HOT |
| 2 | `RelativeTokenSeparation(r_max=32)` | TOKEN_PAIR | ONE_HOT |
| 3 | `RelativeEntity` | TOKEN_PAIR | ONE_HOT |
| 4 | `RelativeChain` | TOKEN_PAIR | ONE_HOT |
| 5 | `ResidueType` | TOKEN | ONE_HOT |
| 6 | `ESMEmbeddings` | TOKEN | IDENTITY |
| 7 | `BlockedAtomPairDistogram` | ATOM_PAIR | ONE_HOT |
| 8 | `InverseSquaredBlockedAtomPairDistances` | ATOM_PAIR | IDENTITY |
| 9 | `AtomRefPos` | ATOM | IDENTITY |
| 10 | `AtomRefCharge` | ATOM | IDENTITY |
| 11 | `AtomRefMask` | ATOM | IDENTITY |
| 12 | `AtomRefElement` | ATOM | ONE_HOT |
| 13 | `AtomNameOneHot` | ATOM | ONE_HOT |
| 14 | `TemplateMask` | TEMPLATES | IDENTITY |
| 15 | `TemplateUnitVector` | TEMPLATES | IDENTITY |
| 16 | `TemplateResType` | TEMPLATES | OUTER_SUM |
| 17 | `TemplateDistogram` | TEMPLATES | ONE_HOT |
| 18 | `TokenDistanceRestraint` | TOKEN_PAIR | RBF |
| 19 | `DockingConstraintGenerator` | TOKEN_PAIR | ONE_HOT |
| 20 | `TokenPairPocketRestraint` | TOKEN_PAIR | RBF |
| 21 | `MSAProfileGenerator` | TOKEN | IDENTITY |
| 22 | `MSADeletionMeanGenerator` | TOKEN | IDENTITY |
| 23 | `IsDistillation` | TOKEN | ONE_HOT |
| 24 | `TokenBFactor` | TOKEN | ONE_HOT |
| 25 | `TokenPLDDT` | TOKEN | ONE_HOT |
| 26 | `ChainIsCropped` | TOKEN | IDENTITY |
| 27 | `MissingChainContact` | TOKEN | IDENTITY |
| 28 | `MSAOneHotGenerator` | MSA | ONE_HOT |
| 29 | `MSAHasDeletionGenerator` | MSA | IDENTITY |
| 30 | `MSADeletionValueGenerator` | MSA | IDENTITY |
| 31 | `IsPairedMSAGenerator` | MSA | IDENTITY |
| 32 | `MSADataSourceGenerator` | MSA | ONE_HOT |

RBF and OUTER_SUM encodings are applied **inside** `feature_embedding.pt`, not in
Python. Python passes raw feature values; the TorchScript module handles the encoding.

### 13.6 Attention Masking

All modules use `masked_fill` with value **`-10000`** (not `-inf`) on attention bias
tensors. This is applied before softmax; after softmax, `exp(-10000) ≈ 0` effectively
zeroes out masked positions.

Token-level padding is propagated via `token_pair_mask` (§6.3.9); atom-level via
`block_atom_pair_mask` (§3.3). Invalid positions are also zero-filled with
`masked_fill(..., 0.)` on single representations.

---

## 14. Inference Linearity and Parallelism

### 14.1 Is the Pipeline Linear?

**Yes — the Chai-1 inference pipeline is fundamentally sequential.** There are no
parallel prediction branches, no concurrent heads, and no fork-join topology. Each stage
depends on the previous stage's output:

```
Feature Construction → Feature Embedding → Bond Projection → Token Input Embedder
    → Trunk (×3 recycles) → Diffusion Module (×200 steps) → Confidence Head → Ranking
```

This is visible in
[`run_folding_on_context()`](../chai-lab/chai_lab/chai1.py) (lines 580–1059), which
executes each stage in strict sequence using Python `with` blocks that load/unload
modules.

### 14.2 Where Parallelism Does Exist

#### 14.2.1 Diffusion Sample Batching

The only architectural parallelism is in the diffusion module: all `num_diffn_samples`
(default 5) noise trajectories are processed **simultaneously** via a `ds` batch
dimension. Inside the diffusion module, tensors have shape `[b, ds, ...]`:

- `atom_noised_coords`: `[b, ds, n_atoms, 3]` — different noise per sample
- `noise_sigma`: `[b, ds]` — same σ for all samples at a given timestep

The `_denoise` helper at [`chai1.py:806–819`](../chai-lab/chai_lab/chai1.py) reshapes the
`(b*ds)` batch into `[b, ds, ...]` before passing to the module:

```python
atom_noised_coords = rearrange(atom_pos, "(b ds) ... -> b ds ...", ds=ds)
noise_sigma = repeat(sigma, " -> b ds", b=batch_size, ds=ds)
```

Inside `diffusion_module.pt`, the conditioning signals are handled asymmetrically:
- **Pair conditioning** (`z_cond`): `[b, n, n, 256]` — shared across all `ds` samples
  (computed once from trunk outputs, independent of σ)
- **Single conditioning** (`s_cond`): `[b, ds, n, 384]` — includes the Fourier σ
  embedding, so conceptually per-sample. However, since all ds samples share the same σ
  at a given timestep, the conditioning is actually identical across samples.

The practical implication: while the conditioning tensors are formally broadcast across
the `ds` dimension, the only thing that truly differs across samples is the atom
coordinates themselves (different random noise realizations).

#### 14.2.2 Confidence Head — No Parallelism

The confidence head is called **sequentially** per diffusion sample in a Python for-loop:

```python
confidence_outputs = [
    confidence_head.forward(..., atom_coords=atom_pos[ds:ds+1], ...)
    for ds in range(num_diffn_samples)
]
```

([`chai1.py:895–910`](../chai-lab/chai_lab/chai1.py))

This is not batched — each sample is processed independently. This is likely because
the confidence head computes `torch.cdist` on predicted atom positions
(`confidence_head.pt` TorchScript line 521), producing an O(n²) distance matrix that
would be expensive to batch across multiple samples.

#### 14.2.3 Ranking — No Parallelism

Ranking is also sequential per sample ([`chai1.py:984–1015`](../chai-lab/chai_lab/chai1.py)),
computing PTM, clashes, and pLDDT for each sample independently
([`ranking/rank.py`](../chai-lab/chai_lab/ranking/rank.py)).

### 14.3 There Is No Bond Prediction

**Covalent bonds are input features, not predicted outputs.** There is no bond prediction
head anywhere in the architecture. The `bond_loss_input_proj.pt` module is a simple
`Linear(1, 512)` that projects a binary bond adjacency matrix (from input constraints)
into the pair representation. The name "bond_loss" refers to a training-time bond
reconstruction loss, not an inference prediction.

The bond adjacency feature is generated by `TokenBondRestraint`
([`data/features/generators/token_bond.py`](../chai-lab/chai_lab/data/features/generators/token_bond.py)),
which reads `atom_covalent_bond_indices` from the input structure context and maps atom-
level bond pairs to token-level pairs via `atom_token_index`. The resulting binary
`[n_tokens, n_tokens]` feature is projected and split 256+256 into trunk and structure
pair representations.

The model's outputs are exclusively:
1. **Atom coordinates** `[n_atoms, 3]` — from the diffusion module
2. **PAE logits** `[n_tokens, n_tokens, 64]` — from the confidence head
3. **PDE logits** `[n_tokens, n_tokens, 64]` — from the confidence head
4. **pLDDT logits** `[n_atoms, 50]` — from the confidence head

### 14.4 Diffusion Step Budget

With the default settings, the diffusion module runs:

| Configuration | Value | Forward Passes |
|---------------|-------|----------------|
| Timesteps | 200 | |
| Heun correction | Yes (except last step) | |
| Steps with 2 passes | 199 | |
| Steps with 1 pass | 1 (final) | |
| **Total per trajectory** | | **399** |
| Diffusion samples | 5 | |
| **Total diffusion calls** | | **399** (batched) |

Each forward pass processes all 5 samples simultaneously via the `ds` dimension, so the
total number of diffusion module invocations is 399, not 399×5.

### 14.5 Overall Inference Call Counts

| Module | Calls | Notes |
|--------|-------|-------|
| `feature_embedding.pt` | 1 | Once |
| `bond_loss_input_proj.pt` | 1 | Once |
| `token_embedder.pt` | 1 | Once |
| `trunk.pt` | 3 | Once per recycle |
| `diffusion_module.pt` | 399 | 200 steps × ~2 passes (Heun) |
| `confidence_head.pt` | 5 | Once per diffusion sample |
| **Total** | **410** | |

The diffusion module dominates inference time (399/410 calls, ~97%).

