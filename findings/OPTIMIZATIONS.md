# Chai-1 Inference Optimizations (Platform-Agnostic)

This document specifies **graph-level caching**, **memory layout changes**, and **custom compute kernels** for a fully optimized Chai-1 inference implementation. It is intentionally **agnostic to CUDA vs Metal** (and to PyTorch vs MLX): algorithms, tensor shapes, invariants, and interfaces are described in generic terms; each backend maps them to its own language, launch APIs, and math libraries.

**Scope**

- Targets the Chai-1 pipeline as documented in [ARCHITECTURE.md](./ARCHITECTURE.md) (TorchScript `.pt` semantics, dimensions, call counts).
- Assumes **fp32 end-to-end** for weights and compute unless stated otherwise (no mixed-precision path in this document).
- Goal: **maximum speed and memory reduction** while preserving **numerically equivalent behavior** to the reference implementation within normal floating-point non-associativity (see §8).

---

## Table of contents

1. [Reference call pattern and invariants](#1-reference-call-pattern-and-invariants)
2. [Graph-level optimizations (caching)](#2-graph-level-optimizations-caching)
3. [Compiler / fusion (no custom kernels)](#3-compiler--fusion-no-custom-kernels)
4. [Custom kernels — specification](#4-custom-kernels--specification)
5. [Chunked triangle multiplication (memory)](#5-chunked-triangle-multiplication-memory)
6. [Implementation order](#6-implementation-order)
7. [Validation and acceptance criteria](#7-validation-and-acceptance-criteria)
8. [Numerical equivalence notes](#8-numerical-equivalence-notes)
9. [Memory and performance summary](#9-memory-and-performance-summary)

---

## 1. Reference call pattern and invariants

### 1.1 Dominant loop

Default inference runs the **diffusion module** **398 times** (199 σ-pairs × 2 Heun evaluations). Per [ARCHITECTURE.md §14.4–14.5](./ARCHITECTURE.md), this dominates wall time (~97% of module invocations).

### 1.2 What changes across diffusion steps

Within one diffusion step, these **depend on** the current noised coordinates and/or σ:

- `atom_noised_coords` / `prev_pos_embed` path in the atom encoder.
- `s_cond`: single conditioning includes **Fourier(σ)** after `single_trans1`; therefore **recomputed every step** (even when all `ds` samples share the same σ).
- All atom-level activations that consume noised positions or per-step `s_cond`.
- Diffusion transformer **token** activations `x` (768-dim sequence state).

### 1.3 What is invariant across diffusion steps (same trunk output, same crop size)

These depend only on **trunk outputs** and **static embeddings** from earlier stages, not on σ or noised coordinates:

- **Pair grid from trunk**: `z_trunk` and merged pair inputs that feed **pair conditioning** inside the diffusion module.
- **`z_cond`**: output of pair conditioning (`pair_ln` after `token_pair_proj` and pair SwiGLU stack) — function of trunk pair + initial pair only.
- **Per-block diffusion transformer pair biases**: for each of the 16 blocks, `pair_bias_k = f_k(LayerNorm_k(z_cond))` (bias from conditioned pair only).
- **`token_pair_to_atom_pair(z_cond)`**: `LayerNorm(256) → Linear(256, 16)` on `[b, n, n, 256]`, then **gather** into blocked atom-pair layout `[b·ds, num_blocks, 32, 128, 16]` using fixed `qkv_indices` (same every step for a given crop).
- **`to_atom_cond` on structure atom input**: `Linear(128, 128)` on atom features derived from trunk/structure path (invariant across σ if trunk is fixed).

The **pair update block** in the atom encoder **must still run every step**: it adds atom-dependent outer-sum terms and an MLP on blocked pairs; it **starts from** the cached base blocked pair features.

---

## 2. Graph-level optimizations (caching)

These are **scheduling / IR changes**, not device kernels. They remove redundant work identically to recomputation (up to §8).

### 2.1 Cache `z_cond` (once per trunk output)

**Operation (conceptual)**

- Inputs: `token_pair_initial`, `token_pair_trunk` (and any concat rules as in ARCHITECTURE §7.1).
- Output: `z_cond[b, n, n, 256]`.

**Policy**

- Compute **once** after trunk is final for the folding job (or once per recycle if your port recomputes trunk per recycle — match reference semantics).
- Pass `z_cond` as an input to each of the 398 diffusion forwards instead of recomputing inside the module.

**Effect**

- Removes **two full SwiGLU pair transitions** on an `[n, n, 256]` tensor every step — the largest single redundant cost in diffusion (order **hundreds of GFLOPs per forward** at moderate `n`, ×397 redundant steps).

### 2.2 Cache per-block diffusion transformer pair biases (16 tensors)

For each transformer block `k ∈ {0..15}`:

**Operation (conceptual)**

- `pair_bias_k = Linear_k(LayerNorm_k(z_cond))` → shape `[b, n, n, H]` with `H = 16` (heads).

**Policy**

- Precompute all 16 after `z_cond` is known.
- Each diffusion step passes `pair_bias_k` into block `k` instead of recomputing from `z_cond`.

**Effect**

- Removes **16 × (LN + Linear)** on the full pair grid per step.
- **Memory**: 16 × `[b, n, n, 16]` fp32. At `n = 512`, `b = 1`: **16 × 16 MiB ≈ 256 MiB** (plus `z_cond` ~268 MiB if kept resident).

### 2.3 Cache blocked atom-pair base from `token_pair_to_atom_pair`

**Operation (conceptual)**

1. `p_token[b, n, n, 16] = token_pair_to_atom_pair(z_cond)` (LN + Linear on last dim).
2. `blocked_base[b, num_blocks, 32, 128, 16] = gather(p_token, qkv_indices)` (same blocking rules as [ARCHITECTURE §3.3, §5.2](./ARCHITECTURE.md)).

**Policy**

- Compute once per cached `z_cond`.
- Per step: run **pair update block** starting from `blocked_base` (add `h`, `w`, MLP), then local attention.

**Effect**

- Removes **O(n²)** LN + small linear + gather **398×** → saves on the order of **~1 TFLOP** over the full run at `n = 512` (small vs §2.1–2.2, but cheap to wire once caching exists).

### 2.4 Cache `to_atom_cond` output (optional, small)

**Operation**

- `atom_cond = to_atom_cond(atom_single_structure_input)` with shapes as in the atom encoder.

**Policy**

- If `atom_single_structure_input` is fixed for the job, compute once; add per-step deltas only where the reference does.

**Effect**

- **O(n_atoms)**; minor FLOPs but removes redundant launches and memory traffic.

### 2.5 Confidence head batching (optional graph change)

Reference Python runs the confidence head **per diffusion sample** to avoid batched `cdist` cost. A port may:

- Keep sequential calls (simplest parity), or
- Implement a **fused or chunked distance** op and batch — not covered as a single “kernel” here but listed for completeness.

---

## 3. Compiler / fusion (no custom kernels)

### 3.1 Element-wise fusion pass

Many blocks are **pre-norm + sublayer + residual + gating**:

- LayerNorm / AdaLN tails, SiLU, sigmoid, element-wise multiply, adds.

**Recommendation**

- Use whatever **graph fuser** or **kernel compiler** your stack provides to merge **chains of element-wise ops** on the same tensor shapes into one pass, reducing **launch overhead** and **round-trips through device memory**.

**Note (PyTorch reference)**

- Upstream `chai-lab` sets `torch.jit.set_fusion_strategy([("STATIC", 0), ("DYNAMIC", 0)])`, which **disables** this class of fusion for TorchScript. A port that stays on TorchScript should **re-enable** non-zero fusion limits; an MLX/other port should rely on the local compiler’s fusion rules.

### 3.2 What fusion does *not* replace

- **Tiled attention** with **online softmax** (memory proportional to tile size, not full `n×n` per head).
- **Chunked triangle multiplication** peak-memory control (§5).

---

## 4. Custom kernels — specification

Each kernel below is defined by **inputs**, **outputs**, **math**, **numerical strategy** (fp32), and **complexity**. Names are descriptive, not API-final.

### 4.1 `fused_adaln`

**Purpose**

Fuse **LayerNorm** on `x` with **conditional affine** from `cond`:

\[
\text{AdaLN}(x \mid c) = \mathrm{LN}(x) \odot (1 + s) + t,\quad (s,t) = \mathrm{split}(W c + b)
\]

**Typical shapes**

- Diffusion transformer: `x: [B, T, 768]`, `cond: [B, T, 384]`; reference uses `lin_s_merged: Linear(384, 1536)` then `chunk(..., 2)` into `scale, shift` each `[B, T, 768]` (see [ARCHITECTURE.md §7.3](./ARCHITECTURE.md)).
- Atom transformer: `x: [B, L, 128]`, conditioning atom single `[B, L, 128]`; reference `lin_s_merged: Linear(128, 256)` → `scale, shift` each `[B, L, 128]` (see [ARCHITECTURE.md §5.2](./ARCHITECTURE.md)).

**Reads / writes**

- Read `x`, `cond`; write `y` same shape as `x`.
- **Avoid** materializing full `LN(x)` if followed immediately by scale/shift in the same kernel (single write).

**Algorithm sketch**

1. For each normalized row (last dimension): compute mean, variance (fp32), normalize, apply γ, β from LN params.
2. Compute `(s, t) = split(W c + b)` with the **exact** weight layout as the reference submodule (fused matmul+split in one kernel, or precompute `s,t` in a separate GEMM then fuse only LN+affine — **must match reference op order** if splitting).

**Complexity**

- O(batch × rows × d) time; O(1) extra memory per row beyond inputs/output if `s,t` fused.

---

### 4.2 `fused_swiglu_activation`

**Purpose**

Fuse the **gated linear unit** activation **only** (not the up/down projections), for an already computed up-projection output `u`:

\[
\text{SwiGLU}(u) = \mathrm{SiLU}(u_{:,1:d}) \odot u_{:,d+1:2d}
\]

**Typical shapes**

- `u: [..., 2d]` contiguous split into two halves.

**Algorithm**

- Single pass: read `u`, write `v: [..., d]`.

**Complexity**

- O(elements) time; no extra large temporaries.

---

### 4.3 `fused_gated_residual`

**Purpose**

Fuse **gate** and **residual add** after a sublayer:

\[
y = x + \sigma(W c + b_{\text{gate}}) \odot \mathrm{sub}(x)
\]

Reference uses **sigmoid** gates with bias init −2 in several places (starts ~0.12).

**Typical shapes**

- `x`, `sub`, `gate_proj_out` broadcast-compatible per token/atom.

**Algorithm**

- Single pass over elements: compute `sigmoid(gate)`, multiply `sub`, add `x`.

**Complexity**

- O(elements); optionally fused with §4.2 output in one launch if data locality matches.

---

### 4.4 `fused_attention_pair_bias` (tiled SDPA + additive bias)

**Purpose**

Compute **multi-head attention** with **additive logits bias** (no separate softmax kernel materializing full `n×n` for all heads at once):

\[
\mathrm{Attn}(Q,K,V,B) = \mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{d_h}} + B\right) V
\]

**Bias layout** (match reference heads)

- `B` per head: effectively `[batch, heads, T, T]` or broadcast rules identical to reference `pair_bias` (diffusion: `[b, n, n, H]` → broadcast to heads).

**Masking**

- Reference uses **additive mask** with large negative constant (e.g. −10000) before softmax. Kernel must support **dense bias tensor** already containing mask contribution, or a separate boolean/byte mask applied in the logits tile — **match reference**.

**Algorithm sketch (FlashAttention-style)**

- Tile `Q` and `K/V` along sequence dimension.
- For each tile of `Q` against strips of `K`:
  - Compute logits block **in registers / fast local memory**.
  - Add **bias tile** for the same index ranges.
  - Update **online softmax** statistics (running max, scaled sum) so **full `T×T` logits never stored**.
  - Accumulate contribution to output tile from `V`.

**Numerics**

- Use **fp32** for softmax statistics and logits in the tile; match reference softmax policy (reference sometimes uses fp32 softmax explicitly — align).

**Call-site parameter sets** (heads × head_dim — verify against `.pt`)

| Location | Heads | \(d_h\) | Notes |
|----------|-------|---------|--------|
| Pairformer token self-attn | 16 | 24 | Pair bias from `z` |
| Triangle attention (trunk / MSA / template) | 8 | 32 | Bias from pair; batching over triangle axis as in reference |
| Diffusion transformer | 16 | 48 | Bias = cached `pair_bias_k` or on-the-fly |
| Confidence triangle variant | 4 | 64 | Fused single-projection, no out_scalers; as in confidence `.pt` |

**Complexity**

- Time: same asymptotic as naive attention O(batch · heads · T² · d_h).
- Memory: O(tile_size² · heads) per wavefront, not O(heads · T²) logits materialized.

---

### 4.5 `fused_blocked_local_attention_pair_bias`

**Purpose**

**Local** attention for atom blocks: queries are **32** atoms; keys/values are a **128**-wide window; **per-block** mask and **per-block** pair bias (from blocked pair features).

**Typical shapes**

- `Q: [B·ds, num_blocks, 32, d_h·H]` or equivalent layout after projection.
- `K,V: [B·ds, num_blocks, 128, d_h·H]`.
- Bias: from `LN(16) → Linear(16, heads·layers)` on blocked pairs — match `blocked_pairs2blocked_bias` in [ARCHITECTURE §5.2, §12.4](./ARCHITECTURE.md).
- Mask: `[B·ds, num_blocks, 32, 128]` boolean or additive.

**Why separate from §4.4**

- Fixed **small** Q/K sizes (32×128) per block → different tiling, no need for full-sequence FlashAttention machinery; still benefit from **fused softmax×V** and **bias in logits**.

**Algorithm sketch**

- One threadgroup / warp-group per `(batch, block)` tile.
- Compute 32×128 logits, add bias, apply mask, softmax, matmul with `V`.

**Complexity**

- O(B · num_blocks · 32 · 128 · H · d_h) time; small peak memory per block.

---

## 5. Chunked triangle multiplication (memory)

**Problem**

Triangular multiplication expands `z[b,n,n,256]` into **large intermediate projections** before einsum-like contractions (see [ARCHITECTURE §6.3.6](./ARCHITECTURE.md)). A naive implementation holds **multiple** full `[b,n,n,·]` tensors simultaneously → **peak memory ~ O(n²)** with a large constant (multiple × the pair tensor).

**Approach (platform-agnostic)**

1. **Do not** materialize full `p` and `g` at once if possible.
2. **Chunk** along the **channel / contraction** dimension (the inner dim `d` in the einsums):
   - For each chunk of `d`:
     - Compute the slice of gated projections for that chunk.
     - Accumulate partial results into the output buffer for the `n×n` slice.
3. Use **high-throughput GEMM** (vendor BLAS or native matmul) for each chunk’s contraction where it reduces to batched matrix multiply.

**Interface**

- Same math as reference; only **evaluation order** and **buffer reuse** change.

**Effect**

- **Peak VRAM** drops from ~O(11×) pair-tensor footprint toward ~O(3–4×) (exact factor depends on chunk size and which activations are retained).
- **Compute** is similar; may be slightly slower due to repeated passes unless tuned.

---

## 6. Implementation order

### Phase A — Graph (highest ROI, no custom kernels)

1. **Split diffusion forward** into:
   - `prepare_diffusion_cache(trunk_outputs, …) → { z_cond, pair_bias[0:16], blocked_pair_base, … }`
   - `diffusion_step(caches, coords, σ, …)`
2. Implement caches §2.1–2.3 (and §2.4 if desired).
3. **Enable** generic element-wise fusion / graph fusion §3.1.

### Phase B — “Easy” fused kernels (build infrastructure)

4. §4.2 `fused_swiglu_activation`
5. §4.3 `fused_gated_residual`
6. §4.1 `fused_adaln`

### Phase C — Attention (hardest, largest kernel wins)

7. §4.4 `fused_attention_pair_bias` — implement for **one** configuration first (e.g. Pairformer 16×24), validate §7, then generalize head dims.
8. §4.5 `fused_blocked_local_attention_pair_bias`

### Phase D — Memory peak for large `n`

9. §5 Chunked triangular multiplication

### Phase E — Second backend

10. Port kernels in **the same order**, swapping only the **runtime / intrinsics / matmul backend**.

---

## 7. Validation and acceptance criteria

### 7.1 Per-kernel unit tests

- **Small shapes** (e.g. `T` or `n` = 8, 16, 32): compare against **naive fp32 reference** (separate ops) with tolerance (§8).
- **Edge masks**: all-valid, single valid row/column, full mask (should match reference masking constant).

### 7.2 Integration

- Run **one full forward** of each TorchScript module (or golden outputs) with **intermediate tensor dumps** if available; otherwise end-to-end structural tests (RMSD of coords, confidence logits stats).

### 7.3 Performance

- Profile **before/after** each phase: wall time, peak allocated device memory, kernel launch count.

---

## 8. Numerical equivalence notes

- **Caching** §2: **bit-identical** to recomputation if the same fp32 ops run once (same order as reference for that subgraph).
- **Element-wise fusion** §3–4.2–4.3: **bit-identical** if evaluation order per element is unchanged; fused kernels should preserve scalar operation order where required for strict match.
- **Tiled attention** §4.4: **not bit-identical** to naive triple matmul + softmax in general — **associativity** of sums changes rounding. Expect **small fp32 diffs** (~1e⁻⁶ relative or tighter depending on depth). Accept **tolerance-based** parity vs reference unless you implement “slow path” for exact debug.
- **Chunked triangle mult** §5: same as **well-implemented** reference — if each chunk mathematically sums into the same accumulator order as unchunked, can be close; often minor diffs if accumulation order differs.

---

## 9. Memory and performance summary

### 9.1 Cache residency (order of magnitude, `b=1`, fp32)

| Object | Shape (conceptual) | `n=512` (order) |
|--------|-------------------|-----------------|
| `z_cond` | `[1, n, n, 256]` | ~268 MiB |
| `pair_bias_k` ×16 | `[1, n, n, 16]` each | ~256 MiB total |
| `blocked_pair_base` | `[1, n_bl, 32, 128, 16]` | tens of MiB (depends on `n_atoms = 23·n`) |

Total **~0.5–0.6 GiB** extra device memory for the caches — usually acceptable vs recomputation cost.

### 9.2 Expected impact (qualitative)

| Change | Compute | Peak memory | Notes |
|--------|---------|-------------|--------|
| §2.1–2.2 Caching | **Largest win** on diffusion | +moderate | Removes most redundant O(n²) pair work |
| §2.3 `token_pair_to_atom_pair` cache | Small but cheap | +small blocked buffer | Nice add-on |
| §3 Fusion | Moderate (launch + BW) | tiny | Low risk |
| §4.4 Fused attention | Moderate–large | **Large reduction** (no full logits) | Critical for large `n` |
| §4.5 Blocked atom attention | Moderate | small | Many invocations |
| §5 Chunked tri-mult | similar FLOPs | **Large reduction** | Enables large `n` on device |

---

## References

- [ARCHITECTURE.md](./ARCHITECTURE.md) — dimensions, block counts, diffusion loop, module hierarchy.
- Original orchestration and JIT loading: `chai-lab` `chai1.py` (paths cited in architecture doc).

---

*Document version: 1.0 — platform-agnostic kernel and caching specification for Chai-1 inference optimization.*
