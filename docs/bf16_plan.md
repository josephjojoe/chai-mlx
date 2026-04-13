# Mixed-precision (bfloat16) conversion plan

## Goal

Add a `compute_dtype` option to `ChaiMLX` so the model can run in either
float32 or bfloat16, matching the TorchScript reference's mixed-precision
profile. Sensitive operations (LayerNorm reductions, softmax, distance
geometry) stay in float32 regardless of the chosen compute dtype.

---

## Memory and throughput analysis

### Model weights

Total safetensors size: **1.26 GB** (~315M parameters in float32).

| Dtype | Weight memory | Savings |
|-------|--------------|---------|
| float32 | 1.26 GB | — |
| bfloat16 | 0.63 GB | 0.63 GB (50%) |

### Activation memory

The pair representation `(B, N, N, 256)` dominates memory. It scales
quadratically with sequence length and dwarfs all linear-in-N tensors
(single, MSA, atom-level) at every supported token count.

**Single pair tensor:**

| N tokens | Elements | fp32 | bf16 | Saved |
|----------|----------|------|------|-------|
| 256 | 16.8M | 64 MB | 32 MB | 32 MB |
| 512 | 67.1M | 256 MB | 128 MB | 128 MB |
| 768 | 150.9M | 576 MB | 288 MB | 288 MB |
| 1024 | 268.4M | 1.00 GB | 512 MB | 512 MB |
| 1536 | 603.9M | 2.25 GB | 1.13 GB | 1.13 GB |
| 2048 | 1073.7M | 4.00 GB | 2.00 GB | 2.00 GB |

**Peak within one pairformer block (~4x pair tensor):**

During a single trunk block, the pair representation is needed for the
residual connection while triangle multiplication or triangle attention
allocates 2–3 additional pair-sized intermediates (gate/proj or QKV, since
`H * head_dim = 4 * 64 = 256 = pair_dim`). Estimated peak ≈ 4x base pair.

| N | Trunk peak fp32 | Trunk peak bf16 | Saved |
|---|-----------------|-----------------|-------|
| 256 | 256 MB | 128 MB | 128 MB |
| 512 | 1.0 GB | 512 MB | 512 MB |
| 1024 | 4.0 GB | 2.0 GB | 2.0 GB |
| 2048 | 16.0 GB | 8.0 GB | 8.0 GB |

**Diffusion cache (persistent across 200 denoising steps):**

- `z_cond`: 1x pair tensor
- 16 precomputed pair biases: `16 * (B, N, N, 16)` = 1x pair tensor
- `blocked_pair_base`, `atom_cond`, `atom_single_cond`: O(N), negligible

Total cache ≈ 2x pair tensor. At N=1024: 2.0 GB (fp32) → 1.0 GB (bf16).

### Combined peak estimate (weights + activations)

| N | Total peak fp32 | Total peak bf16 | Reduction |
|---|-----------------|-----------------|-----------|
| 256 | 1.6 GB | 0.8 GB | 50% |
| 512 | 2.5 GB | 1.3 GB | 50% |
| 1024 | 6.3 GB | 3.2 GB | 50% |
| 2048 | 21.3 GB | 10.6 GB | 50% |

The savings are consistently ~50%. At N=2048, this is the difference between
requiring a 24 GB M-series chip and fitting on a 16 GB one.

### Throughput

Apple Silicon GPUs process 16-bit operands (fp16/bf16) at **2x the
throughput** of fp32 for both ALU and memory bandwidth. Since inference is
predominantly memory-bandwidth bound:

| Bottleneck | Expected speedup |
|-----------|-----------------|
| Memory-bound ops (most of inference) | ~2x |
| Compute-bound matmuls | ~2x |
| fp32 islands (LN, softmax, distances) | 1x (no change) |
| **Estimated end-to-end** | **1.5–1.8x** |

The fp32 islands are a small fraction of total compute, so the practical
speedup is close to 2x.

**FLOPs are unchanged** — the same number of multiply-add operations execute
regardless of precision. The gains come from processing more ops per second
(higher ALU throughput) and moving less data (lower memory bandwidth per
element).

---

## Implementation plan

### Phase 0: Infrastructure (config + helpers)

**Files:** `config.py`, `utils.py`, `model/core.py`

1. Add `compute_dtype: str = "bfloat16"` to `ChaiConfig` (stored as string
   for JSON serialization, resolved to `mx.bfloat16` / `mx.float32` at
   runtime).

2. Add a `resolve_dtype(cfg) -> mx.Dtype` helper in `utils.py`.

3. Add `FP32LayerNorm` wrapper in `nn/layers/common.py`:

   ```python
   class FP32LayerNorm(nn.Module):
       def __init__(self, dims, eps=1e-5, affine=True):
           super().__init__()
           self.ln = nn.LayerNorm(dims, eps=eps, affine=affine)

       def __call__(self, x):
           orig_dtype = x.dtype
           y = self.ln(x.astype(mx.float32))
           return y.astype(orig_dtype)
   ```

   When `compute_dtype=float32`, the casts are no-ops.

4. In `from_pretrained`, after `load_safetensors`, cast all parameters:

   ```python
   if dtype != mx.float32:
       model.apply(lambda k, v: v.astype(dtype) if isinstance(v, mx.array) else v)
   ```

### Phase 1: Trunk + Confidence (highest impact)

**Files:** `model/trunk.py`, `model/confidence.py`, `nn/layers/attention.py`,
`nn/layers/triangle.py`, `nn/layers/common.py`

The pairformer is where precision amplification matters most. This phase
converts the ~32 LayerNorm instances inside pairformer/confidence blocks to
`FP32LayerNorm`, and ensures matmul inputs are in `compute_dtype`.

1. Replace all `nn.LayerNorm(...)` in pairformer-related code with
   `FP32LayerNorm(...)`. The weight map doesn't change — `FP32LayerNorm.ln`
   has the same parameter names as `nn.LayerNorm`.

2. Triangle attention: pair bias already upcasts to fp32 for softmax. No
   change needed.

3. Triangle multiplication: the einsum contraction is safe in bf16 (it's a
   sum over a small dimension, `d=256`). No fp32 island needed.

4. `ConditionedTransition.swiglu`: SwiGLU activation is numerically stable.
   No fp32 island needed.

5. Validate: run `scripts/layer_parity.py` and
   `scripts/stage_isolation_parity.py` in bf16 mode. Per-block parity should
   now closely match TorchScript.

### Phase 2: Diffusion module

**Files:** `model/diffusion.py`, `nn/layers/atom_attention.py`,
`nn/layers/common.py`

1. `DiffusionConditioning`: Fourier embedding and sigma arithmetic stay in
   fp32 (small tensors, transcendental functions). Cast `s_cond` output to
   `compute_dtype` at the boundary.

2. `DiffusionTransformer`: AdaLN already has fp32 reductions in the Metal
   kernel. Matmuls in `DiffusionSelfAttention` and `ConditionedTransition`
   follow input dtype — just ensure inputs are bf16.

3. Atom attention encoder/decoder: `prev_pos_embed` (coordinate projection)
   stays in fp32 per reference. `LocalAtomTransformer` matmuls run in
   `compute_dtype`.

4. Noise schedule math in `diffusion_step` (`sigma` scaling, `sqrt`, Heun
   correction): keep in fp32. These are scalar/small-tensor ops with
   negligible memory impact.

5. Validate: run diffusion isolation parity. Target: <2% relative error in
   bf16 (should improve vs current fp32 baseline since we now match the
   reference precision profile).

### Phase 3: Embeddings + featurization

**Files:** `model/embeddings.py`, `data/featurize.py`

1. Feature encoding (one-hot, RBF, OUTERSUM) runs once on CPU-ish tensors.
   Keep in fp32 — the cost is negligible and there's no benefit to bf16 here.

2. `FeatureEmbedding` Linear projections: cast inputs to `compute_dtype`
   after encoding. The projection weights will already be bf16 from Phase 0.

3. `TokenInputEmbedding` / `TokenInputAtomEncoder`: matmuls and atom
   transformer follow `compute_dtype`. `segment_mean` is a simple average —
   stable in bf16.

4. `BondProjection`: single Linear, follows weight dtype automatically.

### Phase 4: End-to-end validation

1. Run full `fold()` pipeline on real FASTA in bf16 mode.
2. Compare against TorchScript reference dumps with `layer_parity.py`.
3. Expected outcome: trunk and confidence head parity should dramatically
   improve (errors should drop from 1000s to single digits).
4. Run structure quality check: predict a protein with known structure,
   compute RMSD. This validates that the biology is correct, not just the
   numbers.
5. Memory profiling: measure actual peak Metal memory at N=1024 and N=2048 in
   both dtype modes. Compare against the estimates above.

---

## Sensitive operations catalog

These operations must always execute in float32, regardless of
`compute_dtype`. The `FP32LayerNorm` wrapper and `ensure_fp32` helper handle
all of these.

| Operation | Location | Why fp32 |
|-----------|----------|----------|
| LayerNorm mean/var/rsqrt | ~32 sites across trunk, diffusion, confidence | Reduction over large dims loses precision in bf16 |
| AdaLayerNorm reductions | `nn/layers/common.py` (also fused Metal kernel) | Same — Metal kernel already uses fp32 internally |
| Softmax | `nn/layers/attention.py`, inside SDPA | Exp overflow at bf16 range; already upcasts |
| `stable_softmax` | `utils.py` → `ranking.py` | Already uses `ensure_fp32` |
| `pairwise_distance` | `utils.py` → `ranking.py`, `confidence.py` | Squared differences + sqrt amplify rounding |
| `cdist` | `utils.py` → `confidence.py` | Same |
| `normalize_quaternion` | `utils.py` | Sum-of-squares + rsqrt |
| Fourier sigma embedding | `model/diffusion.py` | Transcendentals (sin/cos/exp) on small tensors |
| `prev_pos_embed` | `nn/layers/atom_attention.py` | Reference explicitly keeps this in fp32 |
| Heun correction arithmetic | `model/diffusion.py` | Division by small sigma differences |

---

## Custom Metal kernel compatibility

All four kernels (`swiglu`, `gated_residual`, `adaln_apply`, `fused_adaln`)
template on dtype `T` and pass `template=[("T", x.dtype)]`. They do not
hardcode float32 for storage or elementwise ops.

The fused AdaLN kernel already performs LayerNorm reductions in Metal `float`
(fp32) and casts the result back to `T`. This is exactly the right behavior
for mixed precision.

**Required work:** Integration test with `T=bfloat16` to confirm Metal
dispatch works correctly. No kernel source changes expected — just validation.

---

## Risk assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| MLX SDPA precision differs from PyTorch SDPA in bf16 | Low | MLX SDPA uses fp32 accumulation internally; validate with parity harness |
| Metal bf16 dispatch issues on older Apple Silicon | Low | Test on M1 (minimum target); bf16 is supported since Metal 2.0 |
| Weight name mismatch from `FP32LayerNorm` wrapper | Medium | Use `self.ln = nn.LayerNorm(...)` inside wrapper so names stay `*.ln.weight`, `*.ln.bias`; update name map |
| Some op silently promotes to fp32, hiding bf16 savings | Medium | Profile with `mx.metal.peak_memory()` before/after to confirm memory reduction |
| Numerical divergence in ops assumed stable in bf16 | Low | OPM einsum contracts over small dims (MSA depth); triangle mult contracts over `d=256` — both safe |

---

## Effort estimate

| Phase | Scope | Estimated time |
|-------|-------|---------------|
| Phase 0: Infrastructure | 3 files, ~50 lines | 2 hours |
| Phase 1: Trunk + Confidence | ~32 LN replacements, threading dtype | 4 hours |
| Phase 2: Diffusion | fp32 islands for sigma/coords, dtype threading | 3 hours |
| Phase 3: Embeddings | Minimal — just dtype casts at boundaries | 1 hour |
| Phase 4: Validation | Run parity harness, profile memory, structure check | 4 hours |
| **Total** | | **~2 days** |

The conversion is mechanical — no architectural changes, no new algorithms.
The time goes mostly into validation (running the parity harness at each
stage) and chasing any edge cases where bf16 behaves unexpectedly.
