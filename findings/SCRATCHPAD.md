# Scratchpad: Chai-1 Codebase Analysis

## Key Files and Their Roles

### Entry Points
- `chai_lab/chai1.py` — Main inference pipeline: feature construction, trunk recycling, diffusion sampling, confidence scoring, CIF export
- `chai_lab/main.py` — Typer CLI entry: `fold`, `a3m-to-pqt`, `citation`

### Neural Network Components (shipped as TorchScript `.pt` files)
Downloaded from: `https://chaiassets.com/chai1-inference-depencencies/models_v2/{comp_key}`

| Component | Size | Params | Internal Class |
|-----------|------|--------|---------------|
| `feature_embedding.pt` | 4.4 MB | ~1.1M | `chai.model.embedding.feature_embedding.FeatureEmbedding` |
| `bond_loss_input_proj.pt` | 5.4 KB | 512 | `torch.nn.modules.linear.Linear` |
| `token_embedder.pt` | 6.0 MB | ~1.5M | `chai.model.af3.token_input_emb.TokenInputEmbedding` |
| `trunk.pt` | 604 MB | 169,955,072 | (see below) |
| `diffusion_module.pt` | 454 MB | 127,928,768 | (see below) |
| `confidence_head.pt` | 53 MB | 14,812,416 | `chai.model.modules.af3_confidence_head` |

ESM model: `https://chaiassets.com/chai1-inference-depencencies/esm2/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt`

## Internal Class Names Discovered from TorchScript Graphs

### Model Module Hierarchy (from graph type annotations)
```
chai.model.embedding.feature_embedding.FeatureEmbedding
├── feature_embeddings (ModuleDict)
│   ├── TEMPLATES.TemplateResType: chai.model.embedding.outer_sum.OuterSumProj
│   │   └── embedding: torch.nn.Embedding
│   └── TOKEN_PAIR
│       ├── TokenDistanceRestraint: chai.model.embedding.rbf.RBF
│       └── TokenPairPocketRestraint: chai.model.embedding.rbf.RBF
└── input_projs (ModuleDict)
    ├── ATOM: Sequential[Linear]
    ├── ATOM_PAIR: Sequential[Linear]
    ├── TOKEN: Sequential[Linear]
    ├── TOKEN_PAIR: Sequential[Linear]
    ├── MSA: Sequential[Linear]
    └── TEMPLATES: Sequential[Linear]

chai.model.af3.token_input_emb.TokenInputEmbedding
├── token_single_input_emb: chai.model.modules.af3_input_embedder.AF3InputEmbedder
│   └── atom_encoder: chai.model.modules.af3_atom_attention_encoder_blocked.AtomAttentionBlockedEncoder
│       ├── to_atom_cond: Linear
│       ├── pair_update_block
│       │   ├── atom_single_to_atom_pair_proj_h: [?, Linear(128,16)]
│       │   ├── atom_single_to_atom_pair_proj_w: [?, Linear(128,16)]
│       │   └── atom_pair_mlp: [Linear(16,16), ?, Linear(16,16)]
│       ├── atom_transformer: chai.model.modules.af3_atom_transformer.AtomTransformer
│       │   └── local_diffn_transformer: chai.model.modules.af3_diffusion_transformer.LocalDiffusionTransformer
│       │       ├── local_attentions: ModuleList[LocalAttentionPairBiasBlock × 3]
│       │       ├── transitions: ModuleList[ConditionedTransitionBlock_v1 × 3]
│       │       └── blocked_pairs2blocked_bias: [LayerNorm(16), EinMix([3,4,16])]
│       └── to_token_single: Sequential[Linear(128,384)]
├── token_single_proj_in_trunk: Linear(768,384)
├── token_single_proj_in_structure: Linear(768,384)
├── token_pair_proj_in_trunk: Linear(256,256)
└── token_single_to_token_pair_outer_sum_proj: Linear(384,512)

trunk.pt
├── template_embedder
│   ├── proj_in: [LayerNorm(256), Linear(256,64)]
│   ├── pairformer: 2× PairformerBlock at dim 64
│   ├── template_layernorm: LayerNorm(64)
│   └── proj_out: [?, Linear(64,256)]
├── msa_module
│   ├── linear_s2m: Linear(384,64)
│   ├── outer_product_mean: 4× blocks
│   ├── msa_pair_weighted_averaging: 3× blocks (not 4!)
│   ├── msa_transition: 3× blocks
│   ├── pair_transition: 4× blocks
│   ├── triangular_multiplication: 4× blocks
│   └── triangular_attention: 4× blocks
├── pairformer_stack.blocks: 48× PairformerBlock
│   each with:
│   ├── transition_pair
│   ├── triangle_multiplication
│   ├── triangle_attention
│   ├── transition_single
│   └── attention_pair_bias
├── token_single_recycle_proj: [LayerNorm(384), Linear(384,384)]
└── token_pair_recycle_proj: [LayerNorm(256), Linear(256,256)]

diffusion_module.pt
├── diffusion_conditioning
│   ├── token_pair_proj: [LayerNorm(512), Linear(512,256)]
│   ├── token_in_proj: [LayerNorm(768), Linear(768,384)]
│   ├── single_trans1/2: SwiGLU transitions at 384
│   ├── pair_trans1/2: SwiGLU transitions at 256
│   ├── fourier_embedding: weights[256], bias[256]
│   ├── fourier_proj: [LayerNorm(256), Linear(256,384)]
│   ├── single_ln: LayerNorm(384)
│   └── pair_ln: LayerNorm(256)
├── atom_attention_encoder
│   ├── to_atom_cond: Linear(128,128)
│   ├── token_to_atom_single: [LayerNorm(384), Linear(384,128)]
│   ├── prev_pos_embed: Linear(3,128) ← UNIQUE to diffusion
│   ├── pair_update_block (same as token_embedder)
│   ├── atom_transformer (3 attn + 3 transitions, same dims)
│   ├── to_token_single: Linear(128,768) ← outputs 768, not 384!
│   └── token_pair_to_atom_pair: [LayerNorm(256), Linear(256,16)]
├── diffusion_transformer.blocks: 16× DiffusionTransformerBlock
│   each with:
│   ├── q_bias: [16, 48]
│   ├── norm_in: AdaLayerNorm(384→1536)
│   ├── to_qkv: Linear(768, 2304)  [= 3 × 16 × 48]
│   ├── gate_proj: Linear(384, 768, bias=True)
│   ├── pair_layer_norm: LayerNorm(256)
│   ├── pair_linear: Linear(256, 16)
│   ├── to_out: Linear(768, 768)
│   └── transition: ConditionedTransition(768, conditioned by 384)
├── atom_attention_decoder
│   ├── token_to_atom: Linear(768,128)
│   ├── atom_transformer (3 attn + 3 transitions)
│   └── to_pos_updates: [LayerNorm(128), Linear(128,3)]
├── structure_cond_to_token_structure_proj: Linear(384,768)
├── post_attn_layernorm: LayerNorm(768)
└── post_atom_cond_layernorm: LayerNorm(128)

confidence_head.pt
├── single_to_pair_proj: Linear(384,512)
├── atom_distance_bins_projection: Linear(16,256)
├── blocks: 4× PairformerBlock
│   each with:
│   ├── transition_pair, triangle_multiplication, transition_single
│   ├── attention_pair_bias (16 heads, 24 head_dim)
│   └── triangle_attention: DIFFERENT variant
│       ├── pair_layer_norm: LayerNorm(256)
│       ├── pair2qkvgb: Linear(256, 2056) ← fused qkvg+bias
│       └── linear_out: Linear(512, 512)
├── plddt_projection: Linear(384, 1850) [= 37 atoms × 50 bins]
├── pae_projection: Linear(256, 64)
└── pde_projection: Linear(256, 64)
```

## Key Architectural Constants Discovered

### Attention Configurations
| Module | Heads | Head Dim | Total Dim | Type |
|--------|-------|----------|-----------|------|
| Pairformer single attention | 16 | 24 | 384 | Pair-biased self-attn |
| Diffusion transformer | 16 | 48 | 768 | Conditioned pair-bias |
| Atom transformer (local) | 4 | 32 | 128 | Blocked local attn |
| MSA pair-weighted avg | 8 | 32 (value) | 256 (output) | Gated, pair-biased |
| Triangle attention (trunk) | 8 | varies | 256 | Dual-projection |
| Triangle attention (conf) | 8 | varies | 256 | Fused single-projection |
| Template triangle attention | 8 | varies | 64 | Same as trunk variant |

### SwiGLU Transition Expansion Factors (all 4×)
| Module | Input | Expanded | Output |
|--------|-------|----------|--------|
| Single transition | 384 | 1536 (→768 after SwiGLU) | 384 |
| Pair transition | 256 | 1024 (→512 after SwiGLU) | 256 |
| MSA transition | 64 | 512 (→256 after SwiGLU) | 64 |
| Diffusion transition | 768 | 3072 (→1536 after SwiGLU) | 768 |
| Template pair transition | 64 | 256 (→128 after SwiGLU) | 64 |

### Normalization
- **LayerNorm with bias**: Used for standard layer norms (weight + bias parameters)
- **AdaLayerNorm**: Used in atom transformer and diffusion transformer
  - Takes conditioning signal, produces scale+shift via linear projection
  - `lin_s_merged: Linear(cond_dim, 2 * target_dim)` → chunk to scale, shift

### Activation Functions
- **SiLU/Swish**: Used in SwiGLU gates
- **Sigmoid**: Used for output gating (gate_proj, linear_s_biasinit_m2)
  - Gating bias initialized to **-2** (sigmoid(-2) ≈ 0.12) for near-zero initial gating

### Dropout
- Attention dropout: **0.1** (from graph constant in token_embedder)

## MSA Module Block Count Asymmetry

The MSA module does NOT have uniform block counts:
- 4× outer_product_mean (MSA → pair)
- 3× msa_pair_weighted_averaging (pair-biased MSA attention)
- 3× msa_transition (MSA feed-forward)
- 4× pair_transition (pair feed-forward)
- 4× triangular_multiplication (pair triangle update)
- 4× triangular_attention (pair triangle attention)

This means the last MSA iteration only updates the pair representation (via OPM,
pair transition, tri mult, tri attn) without further MSA self-attention/transition.

## pLDDT Output Structure

From confidence_head.pt graph inspection:
- `plddt_projection: Linear(384, 1850)` outputs per-token
- Constants `37` and `50` appear at line 1497 of JIT source
- **1850 = 37 × 50**: 37 atom positions per token, 50 bins each
- Reshaped to [b, n_tokens, 37, 50], then scattered to [b, n_atoms, 50]
  using `atom_within_token_index`

## Triangle Multiplication Details

From parameter shapes:
- `merged_linear_p: Linear(256, 1024)` → 4 × 256: left_a, left_b, right_a, right_b
- `merged_linear_g: Linear(256, 1280)` → 5 × 256: includes gate channel
- `linear_z_out: Linear(256, 256)` → output after triangle product

The factor of 5 in merged_linear_g (vs 4 in merged_linear_p) suggests one additional
gate channel beyond the four projection channels.

## Confidence Head Triangle Attention Variant

The confidence head uses `pair2qkvgb: Linear(256, 2056)` instead of the trunk's
dual-projection `pair2qkvg1 + pair2qkvg2`.

Decomposition: 2056 = 2048 + 8
- 2048 = 8 heads × 256 per head = 8 × (q32 + k32 + v64 + g64) ← possible decomposition
- 8 = 1 per head for attention bias
- linear_out: (512, 512) = 8 heads × 64 output dim

This is a **single fused projection** variant, vs the trunk's **starting/ending node**
dual projection for proper triangular attention.
