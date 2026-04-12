# Remaining Optimizations — Implementation Plan

This plan covers the four gaps identified in the optimization scorecard. Each section specifies the exact code changes, affected files, and validation strategy.

---

## 1. Cache `to_atom_cond` output (§2.4 — trivial fix)

### Problem

`DiffusionAtomAttentionEncoder.__call__` (line 216 of `layers/atom_attention.py`) calls `self.to_atom_cond(atom_single_structure)` every diffusion step. This is a `Linear(128, 128, no bias)` applied to a tensor that is invariant across all 398 steps — `cache.atom_cond` stores the raw `trunk.atom_single_structure_input`, not the projected result.

**Wasted work**: 398 × matmul on `[b, n_atoms, 128]` × `[128, 128]`. At n=512 (n_atoms ≈ 11776): ~398 × 3 MFLOP ≈ 1.2 GFLOP total, plus 398 unnecessary kernel launches and memory traffic.

### Changes

#### 1a. Compute the projection once in `prepare_cache`

**File**: `diffusion.py`, method `DiffusionModule.prepare_cache`

Replace:
```python
atom_cond=trunk.atom_single_structure_input,
```

With:
```python
atom_cond=self.atom_attention_encoder.to_atom_cond(trunk.atom_single_structure_input),
```

This stores the **projected** result (the output of the Linear) instead of the raw input.

#### 1b. Remove the per-step projection from the encoder

**File**: `layers/atom_attention.py`, class `DiffusionAtomAttentionEncoder.__call__`

Replace:
```python
atom_cond = self.to_atom_cond(atom_single_structure)
```

With:
```python
atom_cond = atom_single_structure  # already projected in prepare_cache
```

The parameter `atom_single_structure` is renamed to `atom_cond_projected` for clarity (update `denoise` call site accordingly).

#### 1c. Do the same for `TokenInputAtomEncoder` if applicable

The token embedder's `TokenInputAtomEncoder.__call__` (line 165) also calls `self.to_atom_cond(atom_single_input)` but this runs only once during embedding (not in the diffusion loop), so it doesn't need caching. Leave it as-is.

### Validation

- Unit test: compute `to_atom_cond(trunk.atom_single_structure_input)` once manually, then run `denoise` with the cached version vs. the original code. Max abs diff should be 0.0 (bit-identical — same ops, same order, just computed once).

### Risk

None. This is a pure scheduling change — mathematically identical output.

---

## 2. Fully fused AdaLN kernel (§4.1 — medium effort)

### Problem

The current `ADALN_APPLY_SOURCE` kernel only fuses the affine step: `x_norm * (1 + scale) + shift`. The LayerNorm itself is a separate `nn.LayerNorm` call, which materializes the intermediate `x_norm` tensor (written to device memory, then immediately re-read by the affine kernel).

In the diffusion transformer, this happens 16 blocks × 2 AdaLN per block × 398 steps = **12,736 times**. Each materialization is a full `[B×ds, n_tokens, 768]` tensor write + read — roughly 6 KB per token pair × n × steps of unnecessary memory traffic.

### Design

A single Metal kernel that:
1. Reads `x` and `cond` (or precomputed `scale`, `shift`).
2. Computes row-wise mean and variance via parallel reduction in threadgroup shared memory.
3. Normalizes, applies LN `weight`/`bias`, then applies `(1 + scale) * x_normalized + shift`.
4. Writes one output tensor — `x_norm` is never materialized.

### Metal kernel specification

```
Input:  x[B, T, D], ln_weight[D], ln_bias[D], scale[B, T, D], shift[B, T, D]
Output: y[B, T, D]

Dispatch: one threadgroup per row (B×T rows total)
Threadgroup size: min(D, 1024) threads
Shared memory: 2 floats per thread (for mean/var reduction)

Algorithm per row:
  1. Each thread loads x[row, tid] into register (strided if D > threadgroup_size).
  2. Parallel sum → mean. Parallel sum of (x - mean)² → variance.
  3. inv_std = rsqrt(variance + eps).
  4. x_norm = (x[row, tid] - mean) * inv_std * ln_weight[tid] + ln_bias[tid]
  5. y[row, tid] = x_norm * (1 + scale[row, tid]) + shift[row, tid]
  6. Write y.
```

### Implementation steps

#### 2a. Write the Metal kernel source

**File**: `kernels/sources.py` — add `FUSED_ADALN_SOURCE`

Key considerations:
- The kernel must handle arbitrary last-dimension sizes (768 for diffusion, 128 for atom transformer) with strided loading when D > threadgroup size.
- Use `threadgroup float` shared memory for the two-pass reduction (sum, then sum-of-squares).
- `eps` is passed as a scalar input (1e-5).

#### 2b. Register and wrap the kernel

**File**: `kernels/elementwise.py` — add `_fused_adaln_kernel()` and `fused_adaln_full(x, ln_weight, ln_bias, scale, shift, eps)`.

The dispatch grid is `(B * T, 1, 1)` with threadgroup size `(min(D, 1024), 1, 1)`.

#### 2c. Update `AdaLayerNorm.__call__` to use the full-fused path

**File**: `layers/common.py`

```python
def __call__(self, x, cond, *, use_kernel=False):
    scale, shift = chunk_last(self.to_scale_shift(cond), 2)
    if use_kernel:
        return fused_adaln_full(
            x, self.norm.weight, self.norm.bias, scale, shift,
            eps=self.norm.eps,
        )
    return self.norm(x) * (1.0 + scale) + shift
```

Keep the old `fused_adaln` (affine-only) available as a fallback for debugging.

#### 2d. Enable `use_kernel=True` in the diffusion transformer path

Currently `LocalAttentionPairBiasBlock` calls `self.adaln(x, cond)` without `use_kernel=True`. The `use_custom_kernel` flag propagates from `denoise` through the encoder/decoder but doesn't reach `AdaLayerNorm`. Thread it through:

- `DiffusionTransformerBlock.__call__` → `self.attn(...)` and `self.transition(...)` should forward `use_kernel`.
- `DiffusionSelfAttention.__call__` → its internal `AdaLayerNorm`.
- `ConditionedTransition.__call__` → its `self.adaln(...)` call.
- `LocalAttentionPairBiasBlock.__call__` → its `self.adaln(...)` call.

### Validation

- **Unit test**: For random `x [4, 128, 768]` and `cond [4, 128, 384]`, compare `fused_adaln_full(...)` against `nn.LayerNorm(x) * (1 + scale) + shift`. Tolerance: max abs diff < 1e-5 (fp32 reduction order may differ slightly).
- **Integration**: Run one full `denoise` step with the fused kernel enabled vs. disabled. Max abs diff on output coordinates < 1e-4.

### Risk

Medium. The parallel reduction for mean/variance must be numerically stable. Use Welford's online algorithm or two-pass (sum then variance) to avoid catastrophic cancellation. Test with both small and large D values.

---

## 3. Shared-memory blocked local attention kernel (§4.5 — high effort)

### Problem

The current `BLOCKED_LOCAL_ATTENTION_SOURCE` dispatches one thread per output element `(m, h, q_idx, d)`. Each thread independently recomputes the full Q·K dot product three times (once for `max_logit`, once for `denom`, once for the weighted accumulation). With D=32 and K=128:
- Each thread does 3 × 128 × 32 = 12,288 multiply-adds just for Q·K.
- There is no data reuse between threads processing the same `(m, h, q_idx)` — they all redundantly load the same K row.
- The SDPA fallback (`mx.fast.scaled_dot_product_attention`) is likely faster because it uses shared memory internally.

### Design

Replace with a threadgroup-cooperative kernel where:
- **One threadgroup** handles one `(m, h)` tuple (one batch-block, one head).
- Q tile `[32, D]`, K tile `[128, D]`, V tile `[128, D]`, and bias `[32, 128]` are loaded into **threadgroup shared memory** once.
- The 32×128 logit matrix is computed cooperatively (each thread computes a tile of the logit matrix).
- Softmax (max, exp-sum, normalize) is done per query row using shared memory reductions.
- Output `[32, D]` is accumulated from softmax weights × V.

### Metal kernel specification

```
Inputs:
  q[M, H, 32, D], k[M, H, 128, D], v[M, H, 128, D],
  bias[M, H, 32, 128], scale (scalar)

Output:
  out[M, H, 32, D]

Dispatch: grid = (M * H, 1, 1), one threadgroup per (m, h)
Threadgroup size: 256 threads (8 warps of 32)
Shared memory layout:
  q_shared[32][D]        — 32 × 32 × 4 = 4 KB
  k_shared[128][D]       — 128 × 32 × 4 = 16 KB
  v_shared[128][D]       — 128 × 32 × 4 = 16 KB
  logits[32][128]         — 32 × 128 × 4 = 16 KB
  Total: ~52 KB (fits in 64 KB threadgroup memory on M3+; for M1/M2
         with 32 KB limit, split into two passes: compute logits+softmax
         first, then accumulate V)

Algorithm:
  1. Cooperative load: threads load q, k, v, bias into shared memory.
  2. Compute logit matrix: thread (i, j) computes dot(q[i], k[j]) * scale + bias[i][j].
     With 256 threads and 32×128 = 4096 logits, each thread computes 16 logits.
  3. Row-wise max: parallel reduction across 128 columns per query row.
  4. Subtract max, exponentiate, row-wise sum.
  5. Normalize: weights[i][j] = exp(logit[i][j] - max[i]) / sum[i].
  6. Accumulate output: out[i][d] = Σ_j weights[i][j] * v[j][d].
     With 256 threads, 32×32 = 1024 output elements, each thread handles 4.
  7. Write out[m, h, :, :] to global memory.
```

### Implementation steps

#### 3a. Write the Metal kernel

**File**: `kernels/sources.py` — replace `BLOCKED_LOCAL_ATTENTION_SOURCE` with the new shared-memory version, keeping the old source as `BLOCKED_LOCAL_ATTENTION_NAIVE_SOURCE` for fallback/testing.

Key design decisions:
- **M1/M2 compatibility**: If 52 KB shared memory is too much for 32 KB threadgroup limit, use a two-phase approach: (1) compute logits + softmax in shared memory (needs Q + K + logits = 36 KB — still tight), (2) reload V and accumulate. Alternatively, reduce to processing 16 query rows at a time (halves shared memory).
- **Thread assignment**: Assign threads to `(query_row, key_col)` pairs for logit computation, then reassign to `(query_row, d)` pairs for V accumulation.
- **Numerical stability**: Use the standard online softmax (max subtraction before exp) exactly as the naive kernel does, but computed once cooperatively rather than three times independently.

#### 3b. Update the kernel registration

**File**: `kernels/blocked_local_attention.py` — update the kernel launch to use threadgroup-level dispatch:
- `grid = (M * H, 1, 1)` (one threadgroup per batch-block-head)
- `threadgroup = (256, 1, 1)`

The old dispatch was `grid = (M * H * 32 * D, 1, 1)` with per-element threads.

#### 3c. Add an M1/M2 fallback path

Query the device's max threadgroup memory size. If < 52 KB, either:
- Use the two-phase shared memory approach, or
- Fall back to `mx.fast.scaled_dot_product_attention` (current behavior when `use_custom_kernel=False`).

### Validation

- **Unit test**: For Q `[8, 4, 32, 32]`, K/V `[8, 4, 128, 32]`, random bias `[8, 4, 32, 128]`:
  - Compare new kernel output against naive triple-pass kernel output. Max abs diff < 1e-5.
  - Compare against `mx.fast.scaled_dot_product_attention`. Max abs diff < 1e-5.
- **Edge cases**: Test with all-masked rows (bias = -10000 everywhere for some rows). Output should be zero or near-zero for masked rows.
- **Performance**: Benchmark new kernel vs. SDPA fallback on M1/M2/M3. Only enable the custom kernel by default if it beats SDPA.

### Risk

High. Shared memory kernels in Metal via `mx.fast.metal_kernel` have constraints:
- The `metal_kernel` API may not expose threadgroup shared memory directly. Verify that `threadgroup float` declarations work in the source string.
- If `mx.fast.metal_kernel` doesn't support shared memory, this optimization requires a compiled Metal library (`.metallib`) loaded via a custom MLX extension — significantly more infrastructure.
- **Recommendation**: Before writing the full kernel, write a minimal test kernel that allocates shared memory via `mx.fast.metal_kernel` to verify feasibility.

---

## 4. Chunked triangle multiplication (§5 — medium effort)

### Problem

`TriangleMultiplication.__call__` (in `layers/triangle.py`) computes:

```python
p = self.merged_linear_p(z_normed)       # [b, n, n, 4*pair_dim]  = [1, n, n, 1024]
g = sigmoid(self.merged_linear_g(z_normed))  # [b, n, n, 5*pair_dim]  = [1, n, n, 1280]
ab = p * g[..., :4*pair_dim]             # [b, n, n, 1024]
```

At n=512, pair_dim=256:
- `z_normed`: 256 MiB
- `p`: 1024 MiB
- `g`: 1280 MiB
- `ab`: 1024 MiB
- Plus `a1, b1, a2, b2` (each 256 MiB): 1024 MiB

**Peak intermediate memory**: ~3.5 GiB just for triangle multiplication intermediates, on top of the pair tensor itself. At n=1024, this 4×'s to ~14 GiB.

The einsum contractions are:
```python
x_out = einsum("bikd,bjkd->bijd", a1, b1)   # outgoing
x_in  = einsum("bkid,bkjd->bijd", a2, b2)   # incoming
```

These contract over the `k` dimension (sequence length n), producing `[b, n, n, pair_dim]`.

### Design: chunk over the feature (d) dimension

Instead of materializing full `[b, n, n, 4*pair_dim]` intermediates, process `C` channels of `a` and `b` at a time (e.g., C=64 out of pair_dim=256):

```
for c in range(0, pair_dim, C):
    a1_chunk = a1[..., c:c+C]   # [b, n, n, C]
    b1_chunk = b1[..., c:c+C]
    x_out += einsum("bikd,bjkd->bijd", a1_chunk, b1_chunk)
```

This reduces peak memory from holding all of `a1 [b,n,n,256]` + `b1 [b,n,n,256]` simultaneously to holding only `a1_chunk [b,n,n,C]` + `b1_chunk [b,n,n,C]`.

But the real memory hog is the **projection** step (`merged_linear_p` produces `[b,n,n,1024]` all at once). To truly reduce peak memory, we must also chunk the linear projection.

### Full chunked algorithm

```python
def __call__(self, z, pair_mask=None):
    z_normed = self.layernorm_z_in(z)
    pair_dim = z.shape[-1]  # 256
    C = chunk_size  # e.g. 64

    # Weight slicing: merged_linear_p.weight is [4*pair_dim, pair_dim]
    # Split into 4 sub-weights for a1, b1, a2, b2 (each [pair_dim, pair_dim])
    w_p = self.merged_linear_p.weight  # [1024, 256]
    w_g = self.merged_linear_g.weight  # [1280, 256]

    x_out = mx.zeros_like(z)
    x_in = mx.zeros_like(z)

    for c in range(0, pair_dim, C):
        # Project only the chunk we need: slice the weight rows
        # a1 uses rows [0:pair_dim], b1 uses [pair_dim:2*pair_dim], etc.
        a1_chunk = z_normed @ w_p[c:c+C].T
        b1_chunk = z_normed @ w_p[pair_dim+c:pair_dim+c+C].T
        a2_chunk = z_normed @ w_p[2*pair_dim+c:2*pair_dim+c+C].T
        b2_chunk = z_normed @ w_p[3*pair_dim+c:3*pair_dim+c+C].T

        # Corresponding gate slices
        g1_chunk = sigmoid(z_normed @ w_g[c:c+C].T)
        g2_chunk = sigmoid(z_normed @ w_g[pair_dim+c:pair_dim+c+C].T)
        g3_chunk = sigmoid(z_normed @ w_g[2*pair_dim+c:2*pair_dim+c+C].T)
        g4_chunk = sigmoid(z_normed @ w_g[3*pair_dim+c:3*pair_dim+c+C].T)

        a1_chunk = a1_chunk * g1_chunk
        b1_chunk = b1_chunk * g2_chunk
        a2_chunk = a2_chunk * g3_chunk
        b2_chunk = b2_chunk * g4_chunk

        if pair_mask is not None:
            # apply masks as before
            ...

        x_out += einsum("bikd,bjkd->bijd", a1_chunk, b1_chunk)
        x_in  += einsum("bkid,bkjd->bijd", a2_chunk, b2_chunk)

        mx.eval(x_out, x_in)  # force eval to free chunk intermediates

    # Output gating (still needs full g5 slice — pair_dim channels)
    g5 = sigmoid(z_normed @ w_g[4*pair_dim:].T)
    out = self.linear_z_out(self.layernorm_out(x_out) + self.layernorm_in(x_in))
    return z + out * g5
```

**Wait — this approach requires slicing into the weight matrix of `merged_linear_p`**, which packs a1/b1/a2/b2 weights together. The actual split is:

```
merged_linear_p.weight: [4 * pair_dim, pair_dim]
  rows [0 : pair_dim]         → left half of "ab" (contains a1+b1 interleaved)
  rows [pair_dim : 2*pair_dim] → right half of "ab" (contains a2+b2 interleaved)
```

Actually, looking at the current code more carefully, the split works like this:

```python
p = merged_linear_p(z_normed)     # [b, n, n, 4*d]
ab = p * g[..., :4*d]
ab_left, ab_right = chunk_last(ab, 2)  # each [b, n, n, 2*d]
a1, b1 = chunk_last(ab_left, 2)        # each [b, n, n, d]
a2, b2 = chunk_last(ab_right, 2)       # each [b, n, n, d]
```

So the weight layout (in rows of `merged_linear_p.weight [4d, d]`) is:
- Rows `[0:d]` → a1 projection weights
- Rows `[d:2d]` → b1 projection weights
- Rows `[2d:3d]` → a2 projection weights
- Rows `[3d:4d]` → b2 projection weights

Similarly for `merged_linear_g.weight [5d, d]`:
- Rows `[0:d]` → gate for a1
- Rows `[d:2d]` → gate for b1
- Rows `[2d:3d]` → gate for a2
- Rows `[3d:4d]` → gate for b2
- Rows `[4d:5d]` → output gate (g5)

### Implementation steps

#### 4a. Add a chunked `__call__` method to `TriangleMultiplication`

**File**: `layers/triangle.py`

Add a `chunk_size` parameter (default `None` = unchunked). When set, use the chunked algorithm. This allows easy A/B testing.

```python
def __call__(self, z, pair_mask=None, *, chunk_size=None):
    if chunk_size is None:
        return self._forward_unchunked(z, pair_mask)
    return self._forward_chunked(z, pair_mask, chunk_size)
```

Move the current implementation to `_forward_unchunked`. Implement `_forward_chunked` as described above.

#### 4b. Chunk the projection by slicing weights directly

Rather than calling `self.merged_linear_p(z_normed)` which produces the full `[b,n,n,4d]` tensor, manually slice the weight matrix and compute `z_normed @ w_slice.T` for each chunk. This avoids the large intermediate.

Key detail: `mx.eval()` must be called after each chunk's einsum accumulation to ensure MLX frees the chunk intermediates. Without this, lazy evaluation would hold all chunks in memory simultaneously, defeating the purpose.

#### 4c. Thread `chunk_size` through the Pairformer and trunk

**Files**: `layers/pairformer.py`, `trunk.py`

Add a `triangle_chunk_size` parameter to `PairformerBlock`, `PairformerStack`, and `Trunk`. Default to `None` (unchunked). Pass through to `TriangleMultiplication.__call__`.

Expose at the top-level API:
```python
model.fold(features, triangle_chunk_size=64)  # for large sequences
```

#### 4d. Choose default chunk size based on sequence length

In the top-level `fold()` method, auto-select:
```python
if triangle_chunk_size is None and n_tokens >= 768:
    triangle_chunk_size = 64  # reduce peak memory for large inputs
```

### Memory analysis

At n=512, pair_dim=256, chunk_size=64:

| | Unchunked peak | Chunked peak |
|---|---|---|
| `z_normed` | 256 MiB | 256 MiB |
| `p` (4d) | 1024 MiB | — |
| `g` (5d) | 1280 MiB | — |
| `ab` (4d) | 1024 MiB | — |
| Per-chunk intermediates | — | ~4 × 64 MiB = 256 MiB |
| `x_out + x_in` accumulators | 512 MiB | 512 MiB |
| **Total intermediates** | **~4 GiB** | **~1 GiB** |

At n=1024: unchunked would need ~16 GiB of intermediates; chunked needs ~4 GiB.

### Validation

- **Numerical**: Run unchunked vs. chunked (C=64) on the same input. Max abs diff should be < 1e-5 (same ops, slightly different accumulation order in the einsum due to chunking — expect small fp32 rounding differences).
- **Memory**: Profile peak allocation with `mx.metal.get_peak_memory()` before/after. Expect ~3-4× reduction in triangle multiplication peak.
- **Performance**: Chunked will be slightly slower due to multiple smaller GEMMs and `mx.eval()` sync points. Measure the wall-time overhead — acceptable if < 10% slowdown for > 3× memory reduction.

### Risk

Low-medium. The algorithm is straightforward (same math, different evaluation order). The main risk is:
- `mx.eval()` sync points adding latency. Profile to find the right chunk size (too small = too many syncs, too large = not enough memory savings).
- Weight slicing producing non-contiguous tensors. May need explicit `.copy()` on weight slices to avoid performance issues in the matmul.

---

## Implementation order

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| **1** | Cache `to_atom_cond` (§1) | 30 min | Small compute, removes 398 redundant kernel launches |
| **2** | Fully fused AdaLN kernel (§2) | 1-2 days | Moderate — eliminates ~12,736 buffer round-trips in diffusion |
| **3** | Chunked triangle multiplication (§4) | 1-2 days | Large memory reduction at n≥768, enables larger sequences |
| **4** | Shared-memory blocked attention (§3) | 3-5 days | Moderate compute, but high risk; verify `mx.fast.metal_kernel` shared memory support first |

Tasks 1 and 2 are independent and can be done in parallel.
Task 3 is independent of 1 and 2.
Task 4 should be attempted last since it has the highest risk and the SDPA fallback is adequate.

### Before starting task 4: feasibility check

Run this test to verify shared memory works in `mx.fast.metal_kernel`:

```python
import mlx.core as mx

test_src = r'''
threadgroup float shared_buf[256];
uint tid = thread_index_in_threadgroup.x;
shared_buf[tid] = x[tid];
threadgroup_barrier(mem_flags::mem_threadgroup);
out[tid] = shared_buf[tid];
'''

kernel = mx.fast.metal_kernel(
    name="shared_mem_test",
    input_names=["x"],
    output_names=["out"],
    source=test_src,
)
x = mx.arange(256, dtype=mx.float32)
result = kernel(
    inputs=[x],
    template=[("T", mx.float32)],
    grid=(1, 1, 1),
    threadgroup=(256, 1, 1),
    output_shapes=[(256,)],
    output_dtypes=[mx.float32],
)[0]
mx.eval(result)
assert mx.allclose(x, result).item()
```

If this fails, task 4 requires a compiled `.metallib` extension instead.

---

## Validation summary

| Optimization | Test type | Tolerance | Pass criteria |
|---|---|---|---|
| `to_atom_cond` cache | Bit-identical check | 0.0 | Exact match (same ops, same order) |
| Fused AdaLN | Per-row comparison | < 1e-5 max abs | Reduction order may differ |
| Chunked tri-mult | Full-tensor comparison | < 1e-5 max abs | Accumulation order may differ |
| Blocked local attn | vs. SDPA reference | < 1e-5 max abs | Online softmax rounding |

End-to-end: after all four optimizations, run one full `denoise` call and compare output coordinates against the unoptimized path. Max abs diff < 1e-3 (accumulated through many layers).
