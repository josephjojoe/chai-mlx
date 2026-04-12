# Porting Chai-1 to MLX: Grounded Proposal

This document specifies a concrete plan for porting the Chai-1 protein structure prediction model (~314M parameters, fp32) to [MLX](https://ml-explore.github.io/mlx/) for optimized inference on Apple Silicon. It is grounded in the verified architecture ([ARCHITECTURE.md](./ARCHITECTURE.md)), the platform-agnostic optimization spec ([OPTIMIZATIONS.md](./OPTIMIZATIONS.md)), and the TorchScript graph dumps in `findings/graphs/`.

**Goal**: A standalone MLX package (`chai1_mlx`) that loads Chai-1 weights, accepts featurized inputs, and produces atom coordinates + confidence scores — with no PyTorch runtime dependency.

---

## Table of Contents

1. [Scope and constraints](#1-scope-and-constraints)
2. [Weight conversion pipeline](#2-weight-conversion-pipeline)
3. [Module hierarchy and naming](#3-module-hierarchy-and-naming)
4. [Layer-by-layer porting notes](#4-layer-by-layer-porting-notes)
5. [Custom Metal kernels](#5-custom-metal-kernels)
6. [Graph-level optimizations (caching)](#6-graph-level-optimizations-caching)
7. [Diffusion loop implementation](#7-diffusion-loop-implementation)
8. [Input featurization boundary](#8-input-featurization-boundary)
9. [Implementation phases](#9-implementation-phases)
10. [Validation strategy](#10-validation-strategy)
11. [Memory budget](#11-memory-budget)

---

## 1. Scope and constraints

### What ships

- Pure MLX inference: `mlx>=0.16`, `numpy>=1.26`, Python ≥ 3.11.
- One-time offline weight conversion script (requires `torch` to run, but torch is **not** a runtime dependency).
- Custom Metal kernels via `mx.fast.metal_kernel` for fused ops and tiled attention.
- Full pipeline: featurized inputs → coordinates + PAE/PDE/pLDDT → ranking scores.
- fp32 throughout (matching the reference; fp16/quantized paths are future work).

### What does not ship

- Feature *construction* from raw FASTA/MSA/templates (Python data pipeline, not a neural module). Callers provide pre-encoded feature tensors matching the 32-feature spec ([ARCHITECTURE.md §13.5](./ARCHITECTURE.md#135-the-32-named-features)).
- ESM2-3B embedding inference. Users pass precomputed 2560-dim embeddings (or zeros).
- CIF/PDB output formatting (pure Python post-processing, orthogonal to the model port).
- Training / gradient computation.

### Design principles

1. **Name-aligned weights**: MLX `nn.Module` parameter paths mirror the TorchScript `named_parameters()` hierarchy as closely as possible, so weight loading is a flat name-map with minimal renaming.
2. **Lazy evaluation**: MLX's computation graph is lazily evaluated. We call `mx.eval()` at explicit synchronization points (end of each diffusion step, end of trunk recycle) rather than after every op.
3. **Standalone**: No torch import at runtime. The conversion script is the only file that touches PyTorch.

---

## 2. Weight conversion pipeline

### 2.1 TorchScript → safetensors

The reference model consists of 6 TorchScript artifacts:

| File | Params | Size (fp32) |
|------|--------|-------------|
| `feature_embedding.pt` | ~1.1M | 4.4 MB |
| `bond_loss_input_proj.pt` | 512 | 5.4 KB |
| `token_embedder.pt` | ~1.5M | 6.0 MB |
| `trunk.pt` | ~170M | 604 MB |
| `diffusion_module.pt` | ~128M | 454 MB |
| `confidence_head.pt` | ~15M | 53 MB |

**Conversion script** (`weights/export_torchscript.py`): For each `.pt` file:

1. `torch.jit.load(path, map_location="cpu")`
2. Iterate `named_parameters()` and `named_buffers()`
3. Convert each tensor to NumPy
4. Save as `.npz` (or `.safetensors` for MLX's native fast-path `mx.load()`)

### 2.2 Name mapping strategy

TorchScript `named_parameters()` produces dot-separated paths like:
```
pairformer_stack.blocks.47.triangle_multiplication.merged_linear_p.weight
diffusion_conditioning.fourier_proj.1.weight
atom_attention_encoder.to_token_single.0.weight
```

The MLX module tree must reproduce these paths exactly. This means:

- **Sequential submodules** (e.g., `to_token_single.0` = Linear, `to_token_single.1` = ReLU) must be wrapped in an indexable container. Since ReLU has no parameters, only the `.0.weight` key matters — the MLX `nn.Linear` can be named with `.0` as a dict key or we flatten the Sequential into a plain Linear and apply ReLU functionally in `__call__`.
- **Numbered block lists** (e.g., `pairformer_stack.blocks.47`) must use Python lists or dicts indexed by string number.
- **Buffers** like `fourier_embedding.weights`, `fourier_embedding.bias`, `atom_distance_v_bins`, and various `CONSTANTS.*` are loaded as frozen arrays (not `nn.Module` parameters). MLX supports this via direct attribute assignment.

### 2.3 Weight shape conventions

- TorchScript `nn.Linear` stores weight as `[out_features, in_features]`.
- MLX `nn.Linear` stores weight as `[out_features, in_features]` by default.
- **No transpose needed** for standard linear layers.
- Special projections with reshaped weights (e.g., `input2qkvg.weight: [384, 4, 16, 24]` in the Pairformer attention) must be loaded with the exact stored shape and reshaped at call time, matching the TorchScript reshape/permute sequence.

### 2.4 Conversion validation

After conversion, for each module:
1. Load weights into the MLX module.
2. Feed a random input through the MLX module.
3. Compare output against PyTorch reference (tolerance: max abs diff < 1e-5 for single-layer, < 1e-3 cumulative for full modules).

---

## 3. Module hierarchy and naming

The MLX package mirrors the 6 TorchScript components as composable `nn.Module` subclasses:

```
Chai1MLX (nn.Module)          — top-level API
├── feature_embedding          — FeatureEmbedding
│   ├── feature_embeddings.*   — per-feature encoders (Embedding, RBF params)
│   └── input_projs.*          — Linear(raw_dim, hidden_dim) per type
├── bond_projection            — BondProjection (single Linear)
├── token_embedder             — TokenInputEmbedding
│   ├── token_single_input_emb
│   │   └── atom_encoder       — AtomAttentionBlockedEncoder
│   │       ├── to_atom_cond, pair_update_block, atom_transformer
│   │       └── to_token_single (Linear → ReLU)
│   ├── token_single_proj_in_trunk, token_single_proj_in_structure
│   └── token_pair_proj_in_trunk, token_single_to_token_pair_outer_sum_proj
├── trunk                      — Trunk
│   ├── token_single_recycle_proj, token_pair_recycle_proj
│   ├── template_embedder      — 2 PairformerBlocks at dim 64
│   ├── msa_module             — MSA processing (OPM, pair-weighted avg, transitions, tri-mult/attn)
│   └── pairformer_stack       — 48 PairformerBlocks
│       └── blocks[0..47]
├── diffusion_module           — DiffusionModule
│   ├── diffusion_conditioning — FourierEmbedding + SwiGLU transitions + LN
│   ├── atom_attention_encoder — AtomAttentionBlockedEncoder (768-dim output)
│   ├── diffusion_transformer  — 16 DiffusionTransformerBlocks
│   ├── atom_attention_decoder — 3 local attn + to_pos_updates
│   └── structure_cond_to_token_structure_proj, post_*_layernorm
├── confidence_head            — ConfidenceHead
│   ├── single_to_pair_proj, atom_distance_bins_projection
│   ├── blocks[0..3]           — 4 PairformerBlocks (ConfidenceTriangleAttention variant)
│   └── plddt_projection, pae_projection, pde_projection
└── ranker                     — Ranker (pure computation, no learned weights)
```

---

## 4. Layer-by-layer porting notes

### 4.1 Linear layers

Most projections are `nn.Linear` with **no bias** (verified from TorchScript: `torch.linear(input, weight)` without bias argument). Only a few have bias:

| Has bias | Examples |
|----------|---------|
| **Yes** | Feature embedding `input_projs`, local attention `out_proj`, gating `linear_s_biasinit_m2`, `gate_proj.0` (diffusion transformer), outer product mean `linear_out` |
| **No** | All `to_atom_cond`, `to_token_single`, `to_qkv`, `to_out`, `merged_linear_p/g`, `linear_z_out`, `pair_linear`, `lin_s_merged`, SwiGLU up/down projections, confidence head output projections, bond projection |

MLX: `nn.Linear(in, out, bias=False)` for no-bias layers.

### 4.2 LayerNorm

Standard `nn.LayerNorm` with learned `weight` and `bias` parameters. MLX provides `nn.LayerNorm(dims)`. All uses are **pre-norm** (LN before transformation).

### 4.3 AdaLayerNorm

Conditional layer normalization used in atom transformer and diffusion transformer blocks:

```python
def adaln(x, cond, ln_weight, ln_bias, lin_weight):
    x_norm = mx.fast.layer_norm(x, ln_weight, ln_bias, eps=1e-5)
    scale, shift = mx.split(cond @ lin_weight.T, 2, axis=-1)
    return x_norm * (1 + scale) + shift
```

The conditioning projection (`lin_s_merged`) has **no bias**. This is a prime candidate for a fused Metal kernel (§5.1).

### 4.4 SwiGLU transition

```python
x_norm = layer_norm(x)
u = x_norm @ up_weight.T          # [*, 2d] — no bias
a, b = mx.split(u, 2, axis=-1)
x = x + (mx.sigmoid(a) * a * b) @ down_weight.T   # SiLU(a) * b, then project
```

The activation `SiLU(a) * b` is a fused kernel target (§5.2).

### 4.5 Gated output

```python
gate = mx.sigmoid(cond @ gate_weight.T + gate_bias)  # bias init -2
output = gate * sublayer_output
```

Fused with the residual add: `x = x + sigmoid(gate_proj) * sublayer_out` (§5.3).

### 4.6 Attention with pair bias (Pairformer, Diffusion Transformer)

MLX provides `mx.fast.scaled_dot_product_attention(q, k, v, scale, mask)`. The pair bias from `z_cond` or pair representation is an additive logits bias. MLX's SDPA supports this via the `mask` parameter (which is additive, matching the reference's `+bias` before softmax).

**Layout**: The reference stores QKV projections in various packed formats:
- Pairformer: `input2qkvg.weight: [384, 4, 16, 24]` → reshape to `[b, n, 4, 16, 24]`, unbind dim 2 → q, k, v, g.
- Diffusion transformer: `to_qkv.weight` → reshape to `[b, n, 3, 16, 48]`, chunk → q, k, v.
- Local attention: `to_qkv.weight: [3, 4, 32, 128]` → reshape + unbind.

Each must be replicated exactly to match the stored weight layout.

### 4.7 Triangle multiplication

Both outgoing and incoming contractions per block. The key operation is:

```python
p = z_norm @ merged_linear_p.T    # [b, n, n, 4c]
g = mx.sigmoid(z_norm @ merged_linear_g.T)  # [b, n, n, 5c]
ab = p * g[..., :4c]
# ... einsum contractions
output = (ln(x_out) + ln(x_in)) @ linear_z_out.T * g[..., 4c:]
```

MLX supports `mx.einsum`. For memory control at large `n`, implement chunked evaluation along the contraction dimension (§6, [OPTIMIZATIONS.md §5](./OPTIMIZATIONS.md#5-chunked-triangle-multiplication-memory)).

### 4.8 Triangle attention

Two SDPA calls per block (starting-node and ending-node). The trunk uses `TriangleAttentionUpdate_v2a` with separate projections (`pair2qkvg1`, `pair2qkvg2`, `pair2b`) and learned `out_scalers`. The confidence head uses `TriangleAttentionUpdate_v1` with a fused single projection (`pair2qkvgb: Linear(256, 2056)`).

Both variants use 4 heads × 64 head_dim. The ending-node direction transposes the pair dimension before attention.

### 4.9 Blocked local attention (atom transformer)

Fixed-size blocks: 32 query atoms × 128 KV atoms. Per-block pair bias from `blocked_pairs2blocked_bias`. This is the primary target for a custom Metal kernel (§5.5) since the small fixed sizes don't benefit from FlashAttention's streaming approach but do benefit from fused bias+softmax+matmul.

### 4.10 Scatter-reduce aggregation

Atom-to-token aggregation via masked scatter sum:

```python
# MLX equivalent using mx.scatter and segment operations
pooled = mx.zeros([b, n_tokens, d])
pooled = segment_sum(masked_repr, atom_token_indices, num_segments=n_tokens)
count = segment_sum(atom_mask, atom_token_indices, num_segments=n_tokens)
aggregated = pooled / mx.maximum(count[..., None], 1)
```

MLX doesn't have a direct `scatter_reduce`. Implement via `mx.scatter_add` (available since MLX 0.16) or a custom segment-sum kernel if performance is critical.

### 4.11 `to_token_single` — the hidden ReLU

Both the token embedder and diffusion encoder apply **ReLU** after the linear projection in `to_token_single` (verified from TorchScript — the module is a `Sequential(.0=Linear, .1=ReLU)`). This is easily missed since the architecture description originally omitted it. Implementation:

```python
x = mx.maximum(x @ to_token_single_weight.T, 0)  # Linear(128, 384/768) → ReLU
```

---

## 5. Custom Metal kernels

All kernels use `mx.fast.metal_kernel` with Metal Shading Language source. The existing port already has initial implementations; this section specifies production-quality versions.

### 5.1 `fused_adaln` — AdaLayerNorm

**Fuses**: LayerNorm(x) + conditional affine `(1 + scale) * x_norm + shift` where `(scale, shift) = split(W @ cond, 2)`.

**Why fuse**: Eliminates materializing the intermediate `x_norm` tensor (768-dim × sequence length, written and immediately re-read). In the diffusion transformer (16 blocks × 2 AdaLN each × 398 steps), this saves ~20,000 unnecessary buffer round-trips.

**Metal kernel sketch**:
- One threadgroup per row (last dimension).
- Phase 1: Parallel reduction for mean and variance (using shared memory).
- Phase 2: Normalize, apply learned LN weight/bias, then apply `(1 + scale) * x + shift`.
- Single write to output.

**Shapes**: `x: [B, T, 768]`, `cond: [B, T, 384]`, `W: [1536, 384]` (precomputed `scale, shift = W @ cond`).

**Existing state**: The current `ADALN_APPLY_SOURCE` only fuses the affine step (`x_norm * (1 + scale) + shift`), not the LayerNorm itself. The production kernel should fuse the full LN + affine.

### 5.2 `fused_swiglu` — SwiGLU activation

**Fuses**: `SiLU(a) * b` where `(a, b) = split(u, 2)`.

**Why fuse**: Avoids materializing `SiLU(a)` as a separate tensor. Called in every SwiGLU transition (48 + 16 + 4 + ... blocks).

**Metal kernel**: Element-wise: read `u[2*i]` and `u[2*i + d]`, compute `silu(a) * b`, write one output element. One thread per output element.

**Existing state**: Already implemented in `SWIGLU_SOURCE`. Adequate for production.

### 5.3 `fused_gated_residual` — Gate + residual add

**Fuses**: `y = x + sigmoid(gate) * sublayer_output`.

**Existing state**: Already implemented in `GATED_RESIDUAL_SOURCE`. Adequate.

### 5.4 `fused_attention_pair_bias` — Tiled SDPA with additive bias

**Purpose**: Multi-head self-attention with additive per-head pair bias, without materializing the full `[B, H, T, T]` logits matrix.

**Why critical**: The Pairformer (48 blocks × 3 SDPA per block), diffusion transformer (16 blocks × 1 SDPA), and confidence head (4 blocks × 3 SDPA) together account for the majority of compute. At `n=512`, a naive implementation allocates `512 × 512 × 16 heads × 4 bytes ≈ 16 MiB` per attention layer just for logits — avoidable with tiled online softmax.

**Strategy**: MLX's built-in `mx.fast.scaled_dot_product_attention` already implements a tiled kernel. The pair bias is passed as the `mask` parameter (MLX treats it as additive logits bias). **Use the built-in SDPA** unless profiling reveals a gap, in which case implement a custom FlashAttention-2 kernel in Metal with:
- Tile sizes tuned for Apple Silicon threadgroup memory (32KB on M1/M2, 64KB on M3+).
- Per-tile online softmax with running max/sum statistics.
- Bias tile loaded alongside K tiles.

**Call-site parameter matrix** (from [ARCHITECTURE.md](./ARCHITECTURE.md)):

| Location | Heads | d_h | Bias source | Calls per forward |
|----------|-------|-----|-------------|-------------------|
| Pairformer self-attn | 16 | 24 | `pair_linear(LN(z))` → `[b,n,n,16]` | 48 |
| Triangle attn (trunk) | 4 | 64 | `pair2b(z)` → `[b,n,n,8]` split 2×4 | 48 × 2 dirs |
| Triangle attn (template) | 4 | 32 | Same structure at dim 64 | 2 × 2 dirs |
| Diffusion transformer | 16 | 48 | `LN+Linear(z_cond)` → `[b,n,n,16]` | 16 |
| Confidence triangle | 4 | 64 | Fused from `pair2qkvgb` | 4 × 2 dirs |

### 5.5 `fused_blocked_local_attention` — Atom-level 32×128 attention

**Purpose**: Local attention for atom transformer blocks (32-query × 128-KV fixed window) with per-block additive pair bias and mask.

**Why separate kernel**: The fixed small sizes (32×128=4096 logits per block) fit entirely in threadgroup shared memory. No need for multi-tile streaming — a single threadgroup computes the full attention for one (batch, block, head) tuple.

**Metal kernel design**:
1. One threadgroup per `(batch_idx, block_idx, head_idx)`.
2. Load Q tile (32 × d_h) and K tile (128 × d_h) into shared memory.
3. Compute 32×128 logit matrix in shared memory.
4. Add bias tile (32×128) from `blocked_pairs2blocked_bias`.
5. Apply mask (set masked positions to -10000).
6. Row-wise softmax (32 rows of 128 elements each).
7. Multiply by V tile (128 × d_h) → output (32 × d_h).

**Shapes**: Q `[M, H, 32, D]`, K/V `[M, H, 128, D]`, bias `[M, H, 32, 128]`, where `M = batch × num_blocks`, `H = 4`, `D = 32`.

**Existing state**: The current `BLOCKED_LOCAL_ATTENTION_SOURCE` is a naive per-element kernel that recomputes Q·K three times (for max, for denom, for accumulation). The production version should use threadgroup shared memory for single-pass tiled attention.

### 5.6 `fused_triangle_mult_chunk` — Chunked triangle multiplication

**Purpose**: Compute both outgoing and incoming triangle contractions without materializing full `[b, n, n, 4c]` intermediates.

**Strategy**: Process the contraction dimension in chunks of size `C` (e.g., C=64 out of d=256):
1. For each chunk `[b, n, n, C]` of the gated projection:
2. Compute partial outgoing: `einsum("...ikd, ...jkd -> ...ijd", a_chunk, b_chunk)`
3. Accumulate into output buffer.
4. Repeat for incoming direction.

This reduces peak intermediate memory from ~11× the pair tensor to ~3-4×.

**Implementation**: Pure MLX with `mx.einsum` per chunk (no custom kernel needed — the contractions are large GEMMs that benefit from MLX's MPS matmul). Wrap in a utility that manages buffer allocation.

---

## 6. Graph-level optimizations (caching)

These are the highest-ROI changes — they remove redundant O(n²) work across 398 diffusion steps.

### 6.1 Diffusion cache structure

```python
@dataclass
class DiffusionCache:
    z_cond: mx.array               # [b, n, n, 256] — conditioned pair repr
    s_cond_static: mx.array        # [b, n, 384] — single conditioning (pre-Fourier)
    pair_biases: list[mx.array]    # 16 × [b, n, n, 16] — per-block transformer bias
    blocked_pair_base: mx.array    # [b, n_bl, 32, 128, 16] — atom pair base
    atom_cond: mx.array            # [b, n_atoms, 128] — to_atom_cond output
    token_to_atom_cond: mx.array   # [b, n_atoms, 128] — token_to_atom_single output
```

**Computed once** after the trunk finishes (method: `DiffusionModule.prepare_cache(trunk_outputs, ...)`).

### 6.2 Cache computation

```python
def prepare_cache(self, trunk_single, trunk_pair, initial_single, initial_pair,
                  structure_single, atom_single_input, structure_inputs):
    # 1. Pair conditioning (two SwiGLU transitions on [n, n, 256])
    z_cat = mx.concatenate([initial_pair, trunk_pair], axis=-1)  # [b,n,n,512]
    z = self.conditioning.token_pair_proj(z_cat)
    z = self.conditioning.pair_trans1(z)
    z = self.conditioning.pair_trans2(z)
    z_cond = self.conditioning.pair_ln(z)

    # 2. Static single conditioning (before Fourier — shared across σ values)
    s_cat = mx.concatenate([initial_single, trunk_single], axis=-1)  # [b,n,768]
    s = self.conditioning.token_in_proj(s_cat)
    s_cond_static = self.conditioning.single_trans1(s)

    # 3. Per-block pair biases
    pair_biases = [block.precompute_pair_bias(z_cond) for block in self.transformer.blocks]

    # 4. Blocked atom pair base
    p_token = self.encoder.token_pair_to_atom_pair(z_cond)  # [b,n,n,16]
    blocked_pair_base = gather_blocked(p_token, structure_inputs.atom_kv_indices)

    # 5. Atom conditioning (invariant across σ)
    atom_cond = self.encoder.to_atom_cond(atom_single_input)
    token_to_atom = self.encoder.token_to_atom_single(structure_single)

    return DiffusionCache(z_cond, s_cond_static, pair_biases,
                          blocked_pair_base, atom_cond, token_to_atom)
```

### 6.3 Per-step forward (uses cache)

Each diffusion step only computes:
1. Fourier embedding of σ → add to `s_cond_static` → `single_trans2` → `single_ln` → `s_cond`.
2. Atom encoder: start from `blocked_pair_base`, add outer-sum terms + MLP, run 3 local attention blocks.
3. Diffusion transformer: 16 blocks with pre-cached `pair_biases[k]`.
4. Atom decoder → position updates.

**Savings at n=512**: Eliminates ~2 pair SwiGLU transitions (each ~500 MFLOP) × 398 steps = ~400 GFLOP. Plus 16 × (LN + Linear) on `[512, 512, 256]` × 398 = ~200 GFLOP. Total ~600 GFLOP saved, plus the memory traffic reduction.

---

## 7. Diffusion loop implementation

### 7.1 EDM sampling

The diffusion loop follows EDM Algorithm 2 ([ARCHITECTURE.md §7.6](./ARCHITECTURE.md#76-noise-schedule-and-sampling)):

```python
def fold(self, features, num_steps=200, num_samples=5, num_recycles=3):
    # 1. Embed + trunk
    embeddings = self.embed_inputs(features)
    trunk_out = self.run_trunk(embeddings, num_recycles)

    # 2. Prepare diffusion cache (ONCE)
    cache = self.diffusion_module.prepare_cache(trunk_out, embeddings, features)
    mx.eval(cache)  # force evaluation before the loop

    # 3. Initialize noise
    sigmas = edm_sigmas(num_steps)  # power-7 schedule, 200 points
    gammas = edm_gammas(sigmas)     # stochastic churn
    coords = init_noise(num_samples, features.structure_inputs)  # [b*ds, n_atoms, 3]

    # 4. Sampling loop (199 σ-pairs × 2 Heun passes = 398 forward calls)
    for sigma_curr, sigma_next, gamma in zip(sigmas[:-1], sigmas[1:], gammas):
        coords = center_random_augmentation(coords, atom_mask)

        # Stochastic noise inflation
        sigma_hat = sigma_curr + gamma * sigma_curr
        noise = 1.003 * mx.random.normal(coords.shape)
        coords_hat = coords + noise * mx.sqrt(mx.maximum(sigma_hat**2 - sigma_curr**2, 1e-6))

        # First denoising (Euler step)
        denoised = self.diffusion_module.denoise(coords_hat, sigma_hat, cache)
        d_i = (coords_hat - denoised) / sigma_hat
        coords = coords_hat + (sigma_next - sigma_hat) * d_i

        # Second-order correction (non-standard Heun: d_i weighted 3/2, d_i' weighted 1/2)
        denoised = self.diffusion_module.denoise(coords, sigma_next, cache)
        d_i_prime = (coords - denoised) / sigma_next
        coords = coords + (sigma_next - sigma_hat) * (d_i_prime + d_i) / 2

        mx.eval(coords)  # sync point per step

    # 5. Confidence + ranking
    confidence = self.confidence_head(trunk_out, coords, features)
    ranking = self.ranker(confidence, features)
    return coords, confidence, ranking
```

### 7.2 EDM preconditioning constants

From [ARCHITECTURE.md §7.6.0](./ARCHITECTURE.md#760-edm-parameterization-preconditioning): `σ_data = 16.0`.

```python
c_in   = (sigma**2 + 256.0)**(-0.5)            # input scaling (σ_data² = 256)
c_skip = 256.0 / (sigma**2 + 256.0)            # skip connection weight
c_out  = sigma * 16.0 / mx.sqrt(sigma**2 + 256.0)  # output scaling
c_noise = mx.log(sigma) / 4                     # noise conditioning for Fourier embedding
```

### 7.3 SE(3) augmentation

Applied at the start of each step:
1. Center atoms (global centroid, denominator clamped at 1e-4).
2. Random SO(3) rotation via uniform quaternion → rotation matrix.
3. Random translation (scale 1.0).

---

## 8. Input featurization boundary

The MLX port accepts pre-encoded feature tensors. The caller is responsible for constructing these from raw inputs (FASTA, MSAs, templates, etc.) using the upstream `chai-lab` Python pipeline or equivalent.

### 8.1 Required inputs

```python
@dataclass
class FeatureContext:
    # Per-type feature tensors (pre-encoded, matching ARCHITECTURE.md §4.3 dims)
    token_features: mx.array        # [b, n_tokens, 2638]
    token_pair_features: mx.array   # [b, n_tokens, n_tokens, 163]
    atom_features: mx.array         # [b, n_atoms, 395]
    atom_pair_features: mx.array    # [b, n_blocks, 32, 128, 14]
    msa_features: mx.array          # [b, n_depth, n_tokens, 42]
    template_features: mx.array     # [b, 4, n_tokens, n_tokens, 76]

    # Structure metadata
    structure_inputs: StructureInputs  # masks, indices, blocking info

    # Optional
    bond_adjacency: mx.array | None  # [b, n_tokens, n_tokens, 1]
```

### 8.2 Standalone usage

```python
import chai1_mlx

model = chai1_mlx.Chai1MLX()
chai1_mlx.load_model_weights(model, weights_dir="./weights/")

features = chai1_mlx.featurize(inputs)  # validates + wraps
coords, confidence, ranking = model.fold(features)
```

No PyTorch, no CUDA, no internet access required at inference time.

---

## 9. Implementation phases

### Phase 1: Core layers and weight loading (1-2 weeks)

**Goal**: Every layer exists as an `nn.Module`, weights load correctly, single-module parity tests pass.

| Task | Details |
|------|---------|
| `nn.Linear` wrappers | With correct bias/no-bias per layer |
| `LayerNorm`, `AdaLayerNorm` | Standard MLX implementations |
| `SwiGLU`, `Transition`, `ConditionedTransition` | With gated output |
| `AttentionPairBias` | Using `mx.fast.scaled_dot_product_attention` |
| `TriangleMultiplication` | Both directions, sigmoid gating |
| `TriangleAttention` (v2a and v1) | Trunk and confidence head variants |
| `PairformerBlock`, `PairformerStack` | Composing the above |
| `AtomAttentionBlockedEncoder/Decoder` | Including `to_token_single` with ReLU |
| `PairUpdateBlock` | Outer-sum + residual MLP |
| Weight export script | TorchScript → NPZ/safetensors |
| Weight loading | `load_model_weights` with name mapping |
| Per-module parity tests | Compare MLX vs PyTorch outputs |

### Phase 2: Full pipeline integration (1 week)

| Task | Details |
|------|---------|
| `FeatureEmbedding` | Including RBF, one-hot, outer-sum encoders |
| `BondProjection` | Trivial single Linear |
| `TokenInputEmbedding` | Atom encoder + scatter aggregation + projections |
| `Trunk` | Recycle loop + template + MSA + Pairformer |
| `DiffusionModule` | Full conditioning + encoder + transformer + decoder |
| `ConfidenceHead` | With v1 triangle attention + output heads |
| `Ranker` | pTM, ipTM, clash detection |
| End-to-end `fold()` | With EDM sampling loop |

### Phase 3: Caching optimizations (3-5 days)

| Task | Details |
|------|---------|
| `prepare_cache()` | Split diffusion conditioning into static/dynamic |
| Per-block pair bias precomputation | 16 cached tensors |
| `blocked_pair_base` cache | Gather from `token_pair_to_atom_pair` |
| `atom_cond` cache | Single Linear, compute once |
| Per-step `denoise()` using cache | Only dynamic path |
| Benchmark: cached vs uncached | Profile wall-time and memory |

### Phase 4: Metal kernel optimization (1-2 weeks)

| Task | Priority | Expected impact |
|------|----------|-----------------|
| `fused_adaln` (full LN + affine) | High | Moderate — 32 calls per diffusion step |
| `fused_blocked_local_attention` (shared memory) | High | Moderate — 6 blocks per diffusion step (encoder+decoder) |
| `fused_swiglu` (already done) | Low | Already implemented |
| `fused_gated_residual` (already done) | Low | Already implemented |
| Chunked triangle multiplication | Medium | Large memory reduction at large n |
| Profile `mx.fast.scaled_dot_product_attention` | Medium | Determine if custom FlashAttention needed |

### Phase 5: Validation and polish (1 week)

| Task | Details |
|------|---------|
| End-to-end structural test | Known protein → RMSD vs reference output |
| Numerical tolerance audit | Max abs diff per module, accumulated drift |
| Memory profiling | Peak allocation at n=256, 512, 768 |
| Performance benchmarks | Wall-time per diffusion step, per trunk recycle |
| API documentation | Usage examples, weight conversion guide |

---

## 10. Validation strategy

### 10.1 Golden outputs

Generate reference intermediate tensors from the PyTorch implementation:

```python
# For each module, save inputs and outputs
torch.save({
    "input": input_tensor,
    "output": module(input_tensor),
}, f"golden/{module_name}_{n_tokens}.pt")
```

Convert to NumPy for loading in MLX tests.

### 10.2 Per-layer tolerance

| Layer type | Max abs diff (fp32) | Notes |
|-----------|-------------------|-------|
| Single Linear | < 1e-6 | Should be near-exact |
| LayerNorm + Linear | < 1e-5 | Small LN variance differences |
| Full Pairformer block | < 1e-4 | Accumulated through 5 sublayers |
| SDPA (tiled vs naive) | < 1e-5 | Online softmax rounding |
| Full trunk (48 blocks) | < 1e-2 | Accumulated across 48 blocks |
| Full diffusion step | < 1e-2 | Acceptable for structural output |

### 10.3 End-to-end structural test

Run inference on a known protein (e.g., 1CRN, 46 residues) and compare:
1. Backbone RMSD vs reference PyTorch output (target: < 0.1 Å).
2. pLDDT correlation (Pearson r > 0.99).
3. PAE matrix cosine similarity (> 0.99).

### 10.4 Numerical gotchas

- **Attention masking**: Reference uses `-10000` (not `-inf`). MLX's SDPA uses the `mask` parameter as additive bias — must ensure the same constant.
- **Scatter-reduce**: MLX's `scatter_add` may accumulate in different order than PyTorch's `scatter_reduce("sum")`. This is expected and acceptable.
- **Einsum contraction order**: MLX's `mx.einsum` may use different internal contraction order than PyTorch, causing small fp32 differences in triangle multiplication. Acceptable.
- **Random number generation**: SE(3) augmentation and noise injection use per-step random draws. For reproducible comparison, seed both frameworks identically.

---

## 11. Memory budget

### 11.1 Model weights

| Component | fp32 | fp16 (future) |
|-----------|------|---------------|
| Feature embedding | 4.4 MB | 2.2 MB |
| Bond projection | 5.4 KB | 2.7 KB |
| Token embedder | 6.0 MB | 3.0 MB |
| Trunk | 604 MB | 302 MB |
| Diffusion module | 454 MB | 227 MB |
| Confidence head | 53 MB | 26.5 MB |
| **Total** | **~1.12 GB** | **~561 MB** |

### 11.2 Activation memory (n=512, b=1, ds=5)

| Tensor | Shape | Size |
|--------|-------|------|
| Token pair (z) | `[1, 512, 512, 256]` | 256 MiB |
| Token single (s) | `[1, 512, 384]` | 0.75 MiB |
| MSA | `[1, 16384, 512, 64]` | 2 GiB (peak, during MSA module only) |
| Atom coordinates | `[5, 11776, 3]` | 0.67 MiB |
| Atom single | `[5, 11776, 128]` | 28.8 MiB |
| Blocked atom pairs | `[5, 368, 32, 128, 16]` | 460 MiB |

### 11.3 Diffusion cache (§6)

| Cached tensor | Shape | Size |
|---------------|-------|------|
| `z_cond` | `[1, 512, 512, 256]` | 256 MiB |
| `pair_biases` ×16 | `[1, 512, 512, 16]` each | 256 MiB total |
| `blocked_pair_base` | `[1, 368, 32, 128, 16]` | ~92 MiB |
| `atom_cond` + `token_to_atom` | `[1, 11776, 128]` × 2 | ~11.5 MiB |
| **Cache total** | | **~616 MiB** |

### 11.4 Total estimated peak (n=512)

| Category | Estimate |
|----------|----------|
| Model weights | 1.12 GB |
| Diffusion cache | 0.6 GB |
| Active pair tensor | 0.25 GB |
| Active blocked atom pairs | 0.45 GB |
| Attention intermediates | ~0.1 GB |
| **Total** | **~2.5 GB** |

This fits comfortably within Apple Silicon unified memory (16 GB minimum on M1, 32-192 GB on M2/M3/M4 Pro/Max/Ultra). For larger sequences (n=1024+), chunked triangle multiplication (§5.6) becomes essential.

---

## References

- [ARCHITECTURE.md](./ARCHITECTURE.md) — Full Chai-1 architecture reference with verified dimensions, block counts, and TorchScript details.
- [OPTIMIZATIONS.md](./OPTIMIZATIONS.md) — Platform-agnostic caching and kernel specifications.
- [MLX Documentation](https://ml-explore.github.io/mlx/) — Framework API reference.
- Karras et al. 2022, "Elucidating the Design Space of Diffusion-Based Generative Models" — EDM sampling algorithm.

---

*Document version: 1.0 — Grounded porting proposal for Chai-1 → MLX inference.*
