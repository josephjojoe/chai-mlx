# Chai-1 Architecture: Comprehensive Reference

This document describes the full Chai-1 neural architecture as reconstructed from the
[chai-lab](../chai-lab/) open-source codebase **and direct inspection of the TorchScript
(`.pt`) model weights**. Every hidden dimension, head count, layer count, and activation
pattern documented here is verified against the actual model parameter shapes.

**Total model parameters: ~312.7M** (excluding ESM2-3B)

| Component | File | Parameters | Size |
|-----------|------|-----------|------|
| Feature Embedding | `feature_embedding.pt` | ~1.1M | 4.4 MB |
| Bond Projection | `bond_loss_input_proj.pt` | 512 | 5.4 KB |
| Token Input Embedder | `token_embedder.pt` | ~1.5M | 6.0 MB |
| Trunk | `trunk.pt` | 169,955,072 | 604 MB |
| Diffusion Module | `diffusion_module.pt` | 127,928,768 | 454 MB |
| Confidence Head | `confidence_head.pt` | 14,812,416 | 53 MB |

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

---

## 1. High-Level Pipeline

The end-to-end inference pipeline is orchestrated by `run_folding_on_context()` in
`chai_lab/chai1.py`:

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

Extracted from TorchScript model parameter shapes:

| Representation | Dimension | Evidence |
|---------------|-----------|---------|
| Token single (s) | **384** | `pairformer.attention.input2qkvg.weight: [384, 4, 16, 24]` |
| Token pair (z) | **256** | `pairformer.transition_pair.layer_norm.weight: [256]` |
| MSA | **64** | `msa_module.msa_transition.linear_no_bias_ab.weight: [512, 64]` |
| Template pair | **64** | `template_embedder.pairformer.blocks.0.transition_pair.layer_norm.weight: [64]` |
| Atom single (a) | **128** | `atom_encoder.to_atom_cond.weight: [128, 128]` |
| Atom pair (p) | **16** | `atom_pair_mlp.0.weight: [16, 16]` |
| Diffusion token | **768** | `diffusion_transformer.blocks.0.to_out.weight: [768, 768]` |

---

## 3. Input Representation

### 3.1 Token-Atom Hierarchy

- **Tokens**: One token per residue (protein/nucleotide) or per atom (ligands/modified residues)
- **Atoms**: Up to **23 atoms per token** (`n_atoms = 23 * n_tokens` in `data/collate/utils.py:35`)
- Model exported for static sizes: **[256, 384, 512, 768, 1024, 1536, 2048]** tokens

### 3.2 Input Contexts

Bundled in `AllAtomFeatureContext` (`data/dataset/all_atom_feature_context.py`):

| Context | Max Size | Description |
|---------|----------|-------------|
| `structure_context` | 2048 tokens | All-atom coordinates, types, masks, bonds |
| `msa_context` | 16,384 depth | MSA tokens, deletion matrix, pairing keys |
| `template_context` | 4 templates | Distances, unit vectors, residue types |
| `embedding_context` | 2560-dim | ESM2-3B per-residue embeddings |
| `restraint_context` | — | Docking, contact, pocket constraints |

---

## 4. Feature Embedding

**Module**: `feature_embedding.pt` (~1.1M params)

Internal class: `chai.model.embedding.feature_embedding.FeatureEmbedding`

### 4.1 Input Feature Dimensions

The feature embedding takes **32 named features** and projects them through type-specific
input projections. The total raw feature dimensions per type (verified from projection
input sizes):

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

Special embedding operations within `feature_embedding.pt`:

- **TemplateResType**: `Embedding(33, 32)` for outer-sum encoding, with learned offset
- **TokenDistanceRestraint**: RBF encoding with 6 learned radii (buffer)
- **TokenPairPocketRestraint**: RBF encoding with 6 learned radii (buffer)
- **AtomNameOneHot**: One-hot to 65 classes → reshaped to 260 (4 chars × 65)
- **AtomRefElement**: One-hot to 130 classes (128 elements + mask)

### 4.3 Raw Feature Dimension Breakdown

TOKEN (2638 total):
- ResidueType: 33 (32 types + mask, one-hot)
- ESMEmbeddings: 2560 (ESM2-3B embedding dim)
- MSAProfile: 32 (distribution over residue types)
- MSADeletionMean: 1
- IsDistillation: 1, TokenBFactor: 1, TokenPLDDT: 1
- ChainIsCropped: 1, MissingChainContact: 1
- Residual from encodings: ~7

TOKEN_PAIR (163 total):
- RelativeSequenceSeparation: 67 (±32 bins + 2 special + 1)
- RelativeTokenSeparation: 67 (r_max=32, 2×32+3)
- RelativeEntity: 3
- RelativeChain: 6 (2×2+2)
- DockingConstraintGenerator: 6 (5 dist bins + mask, one-hot)
- TokenDistanceRestraint: 6 (RBF radii) + mask
- TokenPairPocketRestraint: 6 (RBF radii) + mask
- Residual from concatenation: ~2

ATOM (395 total):
- AtomNameOneHot: 260 (4 chars × 65 classes)
- AtomRefCharge: 1
- AtomRefElement: 130 (128 + mask, one-hot)
- AtomRefMask: 1
- AtomRefPos: 3

ATOM_PAIR (14 total):
- BlockedAtomPairDistogram: 12 (11 classes + mask, one-hot)
- InverseSquaredBlockedAtomPairDistances: 2 (value + mask)

MSA (42 total):
- MSAOneHot: 32 (residue types, one-hot)
- MSAHasDeletion: 1
- MSADeletionValue: 1
- IsPairedMSA: 1
- MSADataSource: 7 (6 classes + mask, one-hot)

TEMPLATES (76 total):
- TemplateMask: 2 (backbone + pseudo-beta)
- TemplateUnitVector: 3
- TemplateResType: 32 (via outer-sum embedding → 32-dim)
- TemplateDistogram: 39 (38 bins + mask, one-hot)

**Evidence**: `feature_embedding.pt` parameter shapes, `chai1.py` lines 172-235.

### 4.4 Bond Feature Projection

**Module**: `bond_loss_input_proj.pt` (512 params)

A single `Linear(1, 512, bias=False)`. Output is split 256+256 and **added** to the
trunk and structure token-pair representations respectively.

**Evidence**: `bond_loss_input_proj.pt` weight shape `[512, 1]`.

---

## 5. Token Input Embedder

**Module**: `token_embedder.pt` (~1.5M params)

Internal class: `chai.model.af3.token_input_emb.TokenInputEmbedding`

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
- **Attention scale**: 1/√32 = 1/√head_dim
- **Dropout**: 0.1 (from graph constant `0.10000000000000001`)

**Conditioned Transition Blocks** (3 blocks, each):
- **AdaLayerNorm**: `lin_s_merged: Linear(128, 256)` for scale+shift
- **SwiGLU expansion**: `linear_a_nobias_double: Linear(128, 512)` — split to 256+256
  (one half through SiLU, then element-wise multiply)
- **Down projection**: `linear_b_nobias: Linear(256, 128)`
- **Gating**: `linear_s_biasinit_m2: Linear(128, 128, bias=True)` — sigmoid gate from
  conditioning signal, bias initialized to -2 (so gate starts near-zero)

**Evidence**: `token_embedder.pt` parameter names and shapes.

---

## 6. Trunk

**Module**: `trunk.pt` (169,955,072 params)

The trunk is the central processing module, consisting of three sub-modules:
template embedder, MSA module, and Pairformer stack.

### 6.1 Recycle Projections

Before each recycle iteration, the previous output is projected:

- `token_single_recycle_proj`: `LayerNorm(384)` → `Linear(384, 384, no bias)`
- `token_pair_recycle_proj`: `LayerNorm(256)` → `Linear(256, 256, no bias)`

The recycled representations are added to the initial representations and fed back
into the trunk. Default: **3 recycles**.

**Evidence**: Last parameters in `trunk.pt`.

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
    │   ├─ merged_linear_p: Linear(64, 256, no bias) → 4×64, split to a,b pairs
    │   ├─ merged_linear_g: Linear(64, 320, no bias) → 5×64, includes gate
    │   └─ linear_z_out: Linear(64, 64, no bias)
    └─ triangle_attention:
        ├─ pair2b: Linear(64, 8, no bias) → 8 heads bias
        ├─ pair2qkvg1: Linear(64, 512, no bias) → starting node
        ├─ pair2qkvg2: Linear(64, 512, no bias) → ending node
        ├─ linear_out: Linear(256, 64, no bias)
        └─ out_scalers: [64] learned per-channel output scaling
        (8 heads, 64 head_dim)
    │
    ▼
template_layernorm: LayerNorm(64)
    │
    ▼
proj_out: Linear(64, 256, no bias)
    │
    ▼
Added to pair representation [b, n, n, 256]
```

**Evidence**: `trunk.pt` parameters prefixed `template_embedder.*`.

### 6.3 MSA Module

The MSA module converts MSA information into pair representation updates. It has an
asymmetric block structure (not all sub-modules have the same number of blocks):

| Sub-module | Count | Operates On |
|-----------|-------|-------------|
| `linear_s2m` | 1 | Single → MSA projection |
| `outer_product_mean` | 4 | MSA → Pair update |
| `msa_pair_weighted_averaging` | 3 | Pair → MSA (pair-biased attention) |
| `msa_transition` | 3 | MSA → MSA (feed-forward) |
| `pair_transition` | 4 | Pair → Pair (feed-forward) |
| `triangular_multiplication` | 4 | Pair → Pair (triangle update) |
| `triangular_attention` | 4 | Pair → Pair (triangle attention) |

#### Iteration Order (verified from TorchScript graph source lines)

The MSA module runs **4 iterations** with the following per-iteration structure, where
the MSA self-attention/transition only runs in the first 3:

```
linear_s2m: project single → MSA (once at start)

For i in 0..3:
    outer_product_mean[i]: MSA → pair update
    pair_transition[i]: pair feed-forward

    if i < 3:
        msa_pair_weighted_averaging[i]: pair-biased MSA attention
        msa_transition[i]: MSA feed-forward

    triangular_multiplication[i]: pair triangle update
    triangular_attention[i]: pair triangle attention
```

The final iteration (i=3) only updates the pair representation — it computes the
outer product mean from MSA to pairs, runs a pair transition, triangular multiplication,
and triangular attention, but does NOT run MSA self-attention or MSA transition. This
makes sense: after the final iteration, the MSA representation is discarded and only
the pair representation continues to the Pairformer stack.

**Evidence**: Source line numbers in `trunk.py` from TorchScript graph: lines 9408→9572
(OPM_0 + pair_trans_0 + MSA_avg_0 + MSA_trans_0 + tri_mult_0 + tri_attn_0), repeating
with MSA_avg/trans absent in the last iteration ending ~11403.

#### 6.3.1 Single-to-MSA Projection

`linear_s2m: Linear(384, 64, no bias)` — projects token single representation to MSA
dimension, added to the first row of the MSA.

#### 6.3.2 Outer Product Mean (4 blocks)

Each block:
- `weight_ab: [2, 8, 8, 64]` — two projections (a, b) from MSA dim 64 to 8-dim,
  producing outer products summed over MSA depth
- `ln_out: LayerNorm(512)` — normalizes the 8×8=64 flattened outer product
  (though stored as 512 — likely includes padding or concatenation with pair repr)
- `linear_out: Linear(512, 256, bias=True)` — projects to pair dimension

#### 6.3.3 MSA Pair-Weighted Averaging (3 blocks)

Each block:
- `layernorm_msa: LayerNorm(64)` — normalize MSA
- `linear_msa2vg: Linear(64, 512, no bias)` — project to value+gate: **8 heads × (32 value + 32 gate)**
- `layernorm_pair: LayerNorm(256)` — normalize pair representation
- `linear_pair: Linear(256, 8, no bias)` — per-head attention weights from pair
- `linear_out_no_bias: Linear(256, 64, no bias)` — project gated values (8×32=256) back to MSA dim

**Attention**: 8 heads, 32 value_dim, gated, pair-biased.

#### 6.3.4 MSA Transition (3 blocks)

Each: `LayerNorm(64)` → `Linear(64, 512, no bias)` → SwiGLU → `Linear(256, 64)`.
Expansion factor: **4×** (512 = 2×256, SwiGLU halves).

#### 6.3.5 Pair Transition (4 blocks)

Each: `LayerNorm(256)` → `Linear(256, 2048, no bias)` → SwiGLU → `Linear(1024, 256)`.
Expansion factor: **4×**.

#### 6.3.6 Triangular Multiplication (4 blocks)

Each block:
- `layernorm_z_in: LayerNorm(256)`
- `merged_linear_p: Linear(256, 1024, no bias)` — projects to 4×256 (a_left, a_right,
  b_left, b_right for outgoing/incoming edges)
- `merged_linear_g: Linear(256, 1280, no bias)` — projects to 5×256 (includes gate)
- `linear_z_out: Linear(256, 256, no bias)` — output projection

#### 6.3.7 Triangular Attention (4 blocks)

Each block:
- `out_scalers: [256]` — learned per-channel scaling
- `pair2b: Linear(256, 8, no bias)` — per-head bias (8 heads)
- `pair2qkvg1: Linear(256, 1024, no bias)` — starting node: q+k+v+g projection
- `pair2qkvg2: Linear(256, 1024, no bias)` — ending node: q+k+v+g projection
- `linear_out: Linear(512, 256, no bias)` — output from gated attention

**Configuration**: 8 heads. Per head: 1024/8 = 128 per projection, decomposed as
q(32)+k(32)+v(32)+g(32) or similar. Output per head: 512/8 = 64.

**Evidence**: `trunk.pt` parameters prefixed `msa_module.*`.

### 6.4 Pairformer Stack (48 blocks)

Each of the 48 identical blocks contains 5 sub-layers:

#### 6.4.1 Transition Pair

`LayerNorm(256)` → `Linear(256, 1024, no bias)` → SwiGLU → `Linear(512, 256)`.
Expansion: **4×** (1024 = 2×512, SwiGLU halves to 512, project to 256).

#### 6.4.2 Triangle Multiplication

Same architecture as MSA module triangular multiplication (see §6.3.6).
Operates at pair dim 256.

#### 6.4.3 Triangle Attention

Same architecture as MSA module triangular attention (see §6.3.7).
8 heads operating at pair dim 256.

#### 6.4.4 Transition Single

`LayerNorm(384)` → `Linear(384, 1536, no bias)` → SwiGLU → `Linear(768, 384)`.
Expansion: **4×**.

#### 6.4.5 Attention with Pair Bias

The core pair-bias self-attention mechanism:

- `single_layer_norm: LayerNorm(384)` — normalize single representation
- `pair_layer_norm: LayerNorm(256)` — normalize pair representation
- `pair_linear: Linear(256, 16, no bias)` — pair → per-head attention bias

**Attention**:
- `input2qkvg.weight: [384, 4, 16, 24]` — from single dim 384, produces 4 outputs
  (q, k, v, g) for **16 heads** at **24 head_dim** each
- `query_bias: [16, 24]` — learned query bias per head
- `output_proj.weight: [16, 24, 384]` — projects multi-head output back to 384

**Configuration**: **16 heads, 24 head_dim, gated output**

The attention computation is:
1. Project single → q, k, v, g (each [b, n, 16, 24])
2. Add query_bias to q
3. Compute attention: softmax(q·k^T / √24 + pair_bias) · v
4. Gate output with sigmoid(g)
5. Project back: [b, n, 16, 24] → [b, n, 384]

**Evidence**: `trunk.pt`, e.g. `pairformer_stack.blocks.0.attention_pair_bias.*`.

### 6.5 Summary Table

| Block Type | Count | Key Dimensions |
|-----------|-------|---------------|
| Template Pairformer | 2 | pair=64, 8 heads |
| MSA outer product mean | 4 | MSA=64 → pair=256, 8×8 outer |
| MSA pair-weighted avg | 3 | 8 heads, 32 value_dim |
| MSA transition | 3 | 64→512→256→64 (SwiGLU 4×) |
| Pair transition (MSA) | 4 | 256→2048→1024→256 (SwiGLU 4×) |
| Triangular mult (MSA) | 4 | pair=256 |
| Triangular attn (MSA) | 4 | pair=256, 8 heads |
| Pairformer blocks | 48 | single=384, pair=256 |
| — attention pair bias | 48 | 16 heads, 24 head_dim |
| — transition single | 48 | 384→1536→768→384 (SwiGLU 4×) |
| — transition pair | 48 | 256→1024→512→256 (SwiGLU 4×) |
| — triangle mult | 48 | pair=256 |
| — triangle attn | 48 | pair=256, 8 heads |

---

## 7. Diffusion Module

**Module**: `diffusion_module.pt` (127,928,768 params)

The diffusion module predicts 3D atomic coordinates via denoising diffusion. It consists
of four sub-systems: conditioning, atom attention encoder, diffusion transformer, and
atom attention decoder.

### 7.1 Diffusion Conditioning

Prepares conditioning signals from trunk outputs and noise level σ:

```
token_single_initial[384] ++ token_single_trunk[384] = 768
    │
    ▼
token_in_proj: LayerNorm(768) → Linear(768, 384, no bias)
    │
    ▼
single_trans1: SwiGLU transition (384→1536→768→384, 4×)
    │
    ▼
 + fourier_embedding(σ) → fourier_proj: LayerNorm(256) → Linear(256, 384)
    │
    ▼
single_trans2: SwiGLU transition (384→1536→768→384, 4×)
    │
    ▼
single_ln: LayerNorm(384) → s_cond [b, n, 384]


token_pair_initial[256] ++ token_pair_trunk[256] = 512
    │
    ▼
token_pair_proj: LayerNorm(512) → Linear(512, 256, no bias)
    │
    ▼
pair_trans1: SwiGLU transition (256→1024→512→256, 4×)
    │
    ▼
pair_trans2: SwiGLU transition (256→1024→512→256, 4×)
    │
    ▼
pair_ln: LayerNorm(256) → z_cond [b, n, n, 256]
```

**Fourier Embedding**: `weights: [256], bias: [256]` — random Fourier features for σ.
Projects log(σ) through `sin(weights * log_σ + bias)` to produce 256-dim embedding,
then projected to 384 via `LayerNorm(256) → Linear(256, 384)`.

**Evidence**: `diffusion_module.pt` parameters prefixed `diffusion_conditioning.*`.

### 7.2 Atom Attention Encoder

Same architecture as the token embedder's atom encoder, with additions for diffusion:

- `to_atom_cond: Linear(128, 128)` — condition atoms from structure features
- `token_to_atom_single: LayerNorm(384) → Linear(384, 128)` — broadcast token→atom
- **`prev_pos_embed: Linear(3, 128, no bias)`** — embeds noised atom positions (unique to diffusion)
- `pair_update_block`: same as token embedder (proj_h/w + MLP at dim 16)
- `token_pair_to_atom_pair: LayerNorm(256) → Linear(256, 16)` — pair→atom pair
- 3 local attention blocks + 3 transitions (identical to token embedder, 4 heads, 32 head_dim)
- `to_token_single: Linear(128, 768, no bias)` — aggregate to token, output dim **768**
  (double the single dim, matching diffusion transformer width)

**Evidence**: `diffusion_module.pt` parameters prefixed `atom_attention_encoder.*`.

### 7.3 Diffusion Transformer (16 blocks)

The core denoising network. Each of the **16 blocks** performs conditioned pair-bias
self-attention at the **768-dim** working width:

```
For each block:
    x [b, n, 768]  (token-level representation)
    s_cond [b, n, 384]  (conditioning signal)
    z_cond [b, n, n, 256]  (pair conditioning)
        │
        ▼
    AdaLayerNorm: norm_in.lin_s_merged: Linear(384, 1536, no bias)
        → produces scale[768] + shift[768] from 384-dim conditioning
        │
        ▼
    to_qkv: Linear(768, 2304, no bias)
        → 2304 = 3 × 768 = 3 × 16_heads × 48_head_dim
        │
        ▼
    q_bias: [16, 48] (added to queries)
        │
        ▼
    pair_layer_norm: LayerNorm(256)
    pair_linear: Linear(256, 16, no bias) → per-head pair bias
        │
        ▼
    Attention: softmax(q·k^T / √48 + pair_bias) · v
        │
        ▼
    to_out: Linear(768, 768, no bias) — output projection
        │
        ▼
    gate: gate_proj: Linear(384, 768, bias=True) → sigmoid gate from conditioning
        │
        ▼
    x += gated_attention_output
        │
        ▼
    ConditionedTransitionBlock:
        AdaLN: Linear(384, 1536) → scale+shift for 768-dim
        SwiGLU: Linear(768, 3072) → split → SiLU × identity → Linear(1536, 768)
        Gate: Linear(384, 768, bias=True) → sigmoid, bias init -2
        │
        ▼
    x += gated_transition_output
```

**Configuration**: **16 heads, 48 head_dim, 768 working dim, conditioned by 384-dim**

Key architectural detail: The diffusion transformer operates at **2× the single
representation width** (768 vs 384). This allows more capacity for the denoising task
while keeping the trunk's single representation compact.

**Evidence**: `diffusion_module.pt` parameters prefixed `diffusion_transformer.blocks.*`.

### 7.4 Atom Attention Decoder

Mirrors the encoder to decode token-level predictions back to atomic positions:

- `token_to_atom: Linear(768, 128, no bias)` — broadcast token→atom
- 3 local attention blocks + 3 transitions (same as encoder, 4 heads, 32 head_dim)
- `to_pos_updates: LayerNorm(128) → Linear(128, 3, no bias)` — output xyz deltas

### 7.5 Post-Processing Layers

- `structure_cond_to_token_structure_proj: Linear(384, 768, no bias)` — initial projection
  of structure conditioning to the 768 working dim
- `post_attn_layernorm: LayerNorm(768)` — normalize after diffusion transformer
- `post_atom_cond_layernorm: LayerNorm(128)` — normalize atom representations

### 7.6 Noise Schedule and Sampling

**Noise Schedule** (`InferenceNoiseSchedule` in `model/diffusion_schedules.py`):

```
σ(t) = σ_data · (t · s_min^(1/p) + (1-t) · s_max^(1/p))^p
```

| Parameter | Value | Source |
|-----------|-------|--------|
| σ_data | 16.0 | `chai1.py:247` |
| s_max | 80.0 | `chai1.py:243` |
| s_min | 4e-4 | `chai1.py:244` |
| p | 7.0 | `diffusion_schedules.py:24` |
| S_churn | 80 | `chai1.py:243` |
| S_tmin | 4e-4 | `chai1.py:244` |
| S_tmax | 80.0 | `chai1.py:245` |
| S_noise | 1.003 | `chai1.py:246` |
| Timesteps | 200 | Default |
| 2nd order | True | Heun corrector |

**Sampling**: EDM (Karras et al. 2022) Algorithm 2 with stochastic churn and
second-order Heun correction. At each step, coordinates are centered, randomly
rotated (uniform SO(3) via quaternions), and translated (σ=1.0) for SE(3) invariance.

**Evidence**: `chai1.py` lines 821-886, `model/diffusion_schedules.py`.

---

## 8. Confidence Head

**Module**: `confidence_head.pt` (14,812,416 params)

Internal class: `chai.model.modules.af3_confidence_head`

### 8.1 Input Processing

The confidence head receives trunk outputs and denoised coordinates, then constructs
initial representations:

- `single_to_pair_proj: Linear(384, 512, no bias)` — projects single representation,
  then constructs pair representation via outer sum (512 = 2×256, split for i and j)
- `atom_distance_bins_projection: Linear(16, 256, no bias)` — embeds pairwise atom
  distances (16 distance bins) into pair space, added to pair representation

### 8.2 Pairformer Blocks (4 blocks)

Each block has the same structure as the trunk Pairformer blocks (§6.4) with one
difference in triangle attention:

**Standard sub-layers** (same as trunk):
- `transition_pair`: SwiGLU at dim 256 (4× expansion)
- `triangle_multiplication`: Same as trunk (dim 256)
- `transition_single`: SwiGLU at dim 384 (4× expansion)
- `attention_pair_bias`: 16 heads, 24 head_dim (identical to trunk)

**Triangle Attention** (different variant):
- `pair_layer_norm: LayerNorm(256)`
- **`pair2qkvgb: Linear(256, 2056, no bias)`** — fused q+k+v+g+bias projection
  - 2056 = 2048 + 8 = 8 heads × (q32 + k32 + v64 + g64 + bias1)
- `linear_out: Linear(512, 512, no bias)` — output: 8 heads × 64

This is a **simplified single-projection triangle attention** compared to the trunk's
dual-projection variant (pair2qkvg1 + pair2qkvg2).

### 8.3 Output Heads

| Head | Projection | Output Shape | Bins |
|------|-----------|-------------|------|
| pLDDT | `Linear(384, 1850, no bias)` | `[b, n_tokens, 37, 50]` → `[b, n_atoms, 50]` | 50 bins, [0, 1] |
| PAE | `Linear(256, 64, no bias)` | `[b, n_tokens, n_tokens, 64]` | 64 bins, [0, 32]Å |
| PDE | `Linear(256, 64, no bias)` | `[b, n_tokens, n_tokens, 64]` | 64 bins, [0, 32]Å |

**pLDDT detail**: The 1850-dim output is reshaped to `[n_tokens, 37, 50]` — **37 atom
positions per token, 50 bins each**. This is then scattered to atom-level using
`atom_within_token_index`. Graph constants confirm: `37` and `50` appear at line 1497
of the JIT source.

**Evidence**: `confidence_head.pt` parameter shapes and graph constants.

---

## 9. Ranking and Scoring

### 9.1 Aggregate Score

```
aggregate_score = 0.2 × pTM + 0.8 × ipTM − 100 × has_inter_chain_clashes
```

### 9.2 TM-Score Normalization

```
d0 = 1.24 · (max(N, 19) − 15)^(1/3) − 1.8
TM_pair(i,j) = 1 / (1 + (PAE_ij / d0)²)
```

- **pTM**: Max over alignment rows of sum of pairwise TM scores, normalized by total tokens
- **ipTM**: Max over chains c of pTM(c vs rest), weighted 4× in aggregate score
- **Clash detection**: Threshold 1.1Å, max 100 clashes, max ratio 0.5

**Evidence**: `ranking/rank.py`, `ranking/ptm.py`, `ranking/clashes.py`.

---

## 10. Handling of Optional Inputs

Chai-1 can run with any combination of MSAs, templates, and ESM embeddings. Here is
how each optional input is handled when absent:

### 10.1 ESM Embeddings (absent)

When `use_esm_embeddings=False`, `EmbeddingContext.empty(n_tokens)` creates a **zero
tensor** of shape `[n_tokens, 2560]`. These zeros are concatenated into the TOKEN
feature vector (which has total dim 2638) and projected through `Linear(2638, 384)`.
There is **no learned mask embedding** — absence is represented as zeros.

**Evidence**: `data/dataset/embeddings/embedding_context.py:48-51`, `data/dataset/embeddings/esm.py:158-161`.

### 10.2 MSA (absent)

When no MSA is provided, `MSAContext.create_empty(n_tokens, depth=MAX_MSA_DEPTH)` creates:
- **Tokens**: filled with gap character `":"` (residue type index for gap)
- **Mask**: all `False` (no valid MSA rows)
- **Deletion matrix**: all zeros
- **Pairing keys**: all `NO_PAIRING_KEY = -999991`
- **Sequence source**: all `MSADataSource.NONE`

The MSA features (MSAOneHot, MSAHasDeletion, MSADeletionValue, IsPairedMSA,
MSADataSource) are computed from this empty context and produce gap-encoded / zero
features. The MSA mask being all-False means:
- MSA profile (TOKEN feature) is zero (no sequences to average over)
- MSA deletion mean is zero
- The MSA module's pair-weighted averaging uses the mask to zero out attention weights

**Evidence**: `data/dataset/msas/msa_context.py:153-167`.

### 10.3 Templates (absent)

When no templates are available, `TemplateContext.empty(n_templates=4, n_tokens=n)` creates:
- **Residue types**: filled with gap character `"-"`
- **Pseudo-beta mask**: all `False`
- **Backbone frame mask**: all `False`
- **Distances**: all zeros
- **Unit vectors**: all zeros

The template mask (`template_restype != "-"`) is all-False, so the template input
masks (`template_input_masks`) are all-zero. The template embedder processes these
through its 2 Pairformer blocks but the all-zero masks ensure no information flows.

**Evidence**: `data/dataset/templates/context.py:87-107`.

### 10.4 Constraints (absent)

Each constraint type has its own absent representation:
- **Contact constraints**: Feature filled with `-1.0` (the `ignore_idx`). The RBF
  encoding checks for `raw_data == -1.0` and creates a `should_mask` indicator that
  is concatenated with the RBF output. The feature embedding then zeros the RBF values
  for masked positions (verified from graph: `aten::eq(%raw_data, -1.0)` → multiply
  encoding by `~should_mask`).
- **Pocket constraints**: Same -1.0 / RBF masking as contacts.
- **Docking constraints**: Feature filled with the `mask_value` (= `num_classes` for
  one-hot = 6), which maps to a special "masked" one-hot class.

At inference, contact and pocket constraints have `include_probability=1.0` (always
included when provided), while docking has `include_probability=0.0` (never included
from random sampling — only from explicit `RestraintGroup` specifications).

**Evidence**: `chai1.py:203-221`, `data/features/generators/token_dist_restraint.py`,
feature_embedding.pt graph lines 345-378.

### 10.5 Non-Protein ESM Tokens

For non-protein chains (DNA, RNA, ligands), ESM embeddings are set to zero vectors.
Within protein chains, modified residues use their canonical parent residue for ESM
tokenization; unknown residues use "X" (token ID 24).

**Evidence**: `data/dataset/embeddings/esm.py:155-161`.

---

## 11. Key Differences from AlphaFold 3

### 11.1 Verified Differences from Model Inspection

1. **ESM2-3B language model** embeddings (2560-dim) as first-class TOKEN input
2. **Constraint features**: Contact (RBF 6), pocket (RBF 6), docking (one-hot 6 bins)
3. **Covalent bond features**: Separate projection pathway
4. **MSA module asymmetry**: 4 outer-product-mean blocks but only 3 MSA attention/transition
   blocks — the last iteration only updates the pair representation
5. **Confidence head triangle attention**: Uses a single fused qkvgb projection (2056-dim)
   instead of the trunk's dual-projection (qkvg1 + qkvg2) triangle attention
6. **pLDDT output**: 37 atom positions × 50 bins = 1850 per token (vs AF3's per-atom approach)

### 11.2 Architecture Dimensions Summary

| Component | Chai-1 | AF3 (for reference) |
|-----------|--------|---------------------|
| Token single dim | 384 | 384 |
| Token pair dim | 256 | 128 |
| MSA dim | 64 | 64 |
| Atom single dim | 128 | 128 |
| Atom pair dim | 16 | 16 |
| Pairformer blocks | 48 | 48 |
| Pairformer heads | 16 | 16 |
| Pairformer head dim | 24 | 24 |
| Diffusion blocks | 16 | 24 |
| Diffusion heads | 16 | 16 |
| Diffusion head dim | 48 | 64 |
| Diffusion working dim | 768 | 768 |
| Confidence blocks | 4 | 4 |
| MSA module blocks | 4 (mixed) | 4 |
| Template blocks | 2 | 2 |
| Atom attn blocks | 3 | 3 |
| Atom attn heads | 4 | 4 |
| Atom attn head dim | 32 | 32 |

Notable: Chai-1 uses **pair dim 256** (vs AF3's 128), and **16 diffusion blocks**
(vs AF3's 24), with **48 diffusion head dim** (vs AF3's 64).

---

## Appendix A: Layer Building Blocks

### A.1 SwiGLU Transition

Used throughout (pair transitions, single transitions, MSA transitions):

```python
x = LayerNorm(x)
ab = Linear_no_bias(x)  # expansion: dim → 2 * expansion * dim
a, b = ab.chunk(2, dim=-1)
x = x + Linear(SiLU(a) * b)  # down projection
```

Expansion factor is consistently **4×** across all modules.

### A.2 AdaLayerNorm (Adaptive Layer Normalization)

Used in diffusion transformer and atom transformer:

```python
scale_shift = Linear(conditioning)  # dim → 2 * dim
scale, shift = scale_shift.chunk(2, dim=-1)
x = LayerNorm(x) * (1 + scale) + shift
```

### A.3 Gated Output with Bias Init -2

Used in conditioned transitions and diffusion attention:

```python
gate = sigmoid(Linear(conditioning, bias=True))  # bias initialized to -2
output = gate * value  # gate starts near sigmoid(-2) ≈ 0.12
```

### A.4 Blocked Local Attention

Atom-level attention uses blocked structure:
- Query block size: **32 atoms**
- Key/value block size: **128 atoms**
- Pair bias from blocked atom pairs projected per-head

---

## Appendix B: Complete Parameter Counts

| Module | Params (exact) |
|--------|---------------|
| feature_embedding.pt | ~1,100,000 |
| bond_loss_input_proj.pt | 512 |
| token_embedder.pt | ~1,500,000 |
| trunk.pt | 169,955,072 |
| diffusion_module.pt | 127,928,768 |
| confidence_head.pt | 14,812,416 |
| **Total (excl. ESM)** | **~314M** |
| ESM2-3B (external) | ~3,000,000,000 |
