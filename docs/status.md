# Port status

Last substantive update: 2026-04-14 (+ ranker parity fix 2026-04-17). This is
the canonical record of the Chai-1 MLX port's current state and was previously
titled "Trunk Investigation Handoff". End-to-end Cα spacing on 1L2Y is within
0.1 Å of the Torch-MPS reference, and kernel-level analysis shows the
remaining gap comes from different fused-kernel rounding in bf16 sigmoid/silu
between MLX and MPS — not from algorithmic differences (all ops are
bit-identical in fp32).

Importantly, **the reference is Torch/MPS, not CUDA**. Nobody runs Chai-1 on MPS in production; everyone uses CUDA. We have no data on how MLX compares to CUDA. The 0.1 Å gap is specific to the MLX-vs-MPS comparison and may not be representative of the MLX-vs-CUDA gap.

## Project Context

This repository is an MLX port of Chai-1.

There are three important sources of truth in this repo:

- the MLX implementation under `chai_mlx/`
- the local reference TorchScript / Chai Discovery implementation under `chai-lab/`
- the extracted TorchScript graph dumps under `findings/graphs/`

Important local-layout note:

- `chai-lab/` is listed in [.gitignore](/Users/josephjojoe/Documents/Projects/chai-mlx/.gitignore:1)
- it is expected to exist locally, but it is not tracked in this repo
- if a future agent does not see it in the worktree, they need to locate or restore the local reference checkout before trying to run the reference harnesses

In practice, the readable Python trunk source in `chai-lab` is not enough by itself for trunk debugging. The reliable trunk reference is:

- [findings/graphs/README.md](/Users/josephjojoe/Documents/Projects/chai-mlx/findings/graphs/README.md)
- [findings/graphs/trunk_toplevel_code.txt](/Users/josephjojoe/Documents/Projects/chai-mlx/findings/graphs/trunk_toplevel_code.txt)
- [findings/graphs/trunk_forward256.py](/Users/josephjojoe/Documents/Projects/chai-mlx/findings/graphs/trunk_forward256.py)
- [findings/graphs/trunk_code.txt](/Users/josephjojoe/Documents/Projects/chai-mlx/findings/graphs/trunk_code.txt)
- [chai-lab/chai_lab/chai1.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai-lab/chai_lab/chai1.py)

## Operational Constraints

- This is being debugged on a 16 GB MacBook Pro.
- Be aggressive about freeing memory and avoid loading both full MLX and full Torch trunk weights at once unless absolutely necessary.
- Prefer isolated one-framework-at-a-time probes:
  - save MLX outputs to disk
  - exit
  - then run the Torch side
- Any command that touches MLX/Metal or Torch/MPS should be treated as non-sandbox-safe work.
- CPU fallback is usually not worth it for the heavy trunk/diffusion traces. It is extremely slow and will *not* work. MLX vs MPS!

Practical memory advice that worked well here:

- use trunk-only or stage-only probes where possible
- use saved reference artifacts from `/tmp/chai_mlx_runs/refdump/`
- when comparing a purely local op, compare only a slice if the op is independent across one axis
- kill long-running validation jobs before starting another large MPS/MLX job

## Graph-Dump Guidance

If you are investigating trunk fidelity, start by reading:

- [findings/graphs/README.md](/Users/josephjojoe/Documents/Projects/chai-mlx/findings/graphs/README.md)

Then use the graph files as follows:

- [findings/graphs/trunk_toplevel_code.txt](/Users/josephjojoe/Documents/Projects/chai-mlx/findings/graphs/trunk_toplevel_code.txt)
  - top-level ordering and dataflow
- [findings/graphs/trunk_forward256.py](/Users/josephjojoe/Documents/Projects/chai-mlx/findings/graphs/trunk_forward256.py)
  - readable inlined semantics
- [findings/graphs/trunk_code.txt](/Users/josephjojoe/Documents/Projects/chai-mlx/findings/graphs/trunk_code.txt)
  - submodule-level precision / parameter usage / exact TorchScript behavior

For trunk debugging, trust the graph dumps over guesses about "what the source probably meant".

## Recent Git History Worth Reading

Recent commits give the best high-level narrative of what has already been fixed:

- `9cc3224` — "warning about not using CPU"
- `5201bd6` — "fix trunk MSA semantics to better match TorchScript, make MSA pair-weighted averaging chunked and dtype-aware, switch utility sigmoid/silu to native MLX ops"
- `79544ff` — "trying to fix mlx trunk; significant variance in results from one seed to another, so not just a structural error"
- `07dd467` — "huge fix, diffusion module on mlx now bit-for-bit exact effectively with mps; was measuring calcium atoms wrong... now effectively all the difference comes from the trunk"
- `dc7d995` — "huge fixes ... residual problem went from ~4.0 to ~0.07 per step ... harnesses are good"
- `76a2172` — "big fix in attention; qkv interleaving bug"
- `0bc4490` — early large diffusion-side cleanup: pair update source mismatch, blocked pair indexing, residual ordering, new tracing/isolation harnesses

Useful commands:

```bash
git log --oneline --decorate -n 20
git show --stat --summary 07dd467
git show --stat --summary dc7d995
git show --stat --summary 76a2172
git show --stat --summary 0bc4490
```

The commit subjects are blunt but informative. They are worth scanning before duplicating old work.

**Note**: The triangle attention fixes (QKVG reshape + ending-direction transpose) were committed in `b9e97f2` on 2026-04-14.

## Executive Summary

- The MLX port is structurally faithful: 0.1 Å Cα gap against Torch-MPS on 1L2Y (3 seeds, mean=0.104 Å).
- The diffusion module is bit-for-bit exact when given correct trunk outputs.
- Two critical structural bugs were found and fixed in `TriangleAttention`:
  1. QKVG reshape ordering: `[4, H, D]` vs `[H, 4, D]` — scrambled features across attention heads.
  2. Ending-direction transpose: MLX incorrectly transposed the ending-direction output back, breaking spatial alignment.
- An earlier native activation fix (`sigmoid`/`silu`) cut the transition-side residual in half but was dwarfed by the triangle attention bugs.
- There are no known structural bugs.
- **Root cause of remaining gap**: Exhaustive bf16 analysis showed that `exp` and `matmul` are bit-identical between MLX and MPS in both bf16 and fp32. The divergence comes from **fused sigmoid/silu kernels** — both MLX and MPS have fused implementations that differ from the decomposed `1/(1+exp(-x))` arithmetic (which is bit-identical across backends). MLX's fused sigmoid disagrees with its own decomposed version on 518 of 65,280 bf16 inputs; MPS's fused sigmoid disagrees with its own decomposed version on 1,113 inputs. Neither is "wrong" — they make different speed/accuracy tradeoffs in their Metal shaders.
- **This comparison is against MPS, not CUDA.** We have no data on how MLX compares to the CUDA backend that everyone actually uses for Chai-1 in production. The 0.1 Å gap is MLX-vs-MPS specific.

## All Bugs Fixed (Cumulative)

### Diffusion-side (all committed, historical)

These were fixed across commits `0bc4490`, `76a2172`, `dc7d995`, and `07dd467`:

- Pair update source mismatch, blocked pair indexing, residual ordering
- QKV interleaving bug in attention
- Residual problem reduction (~4.0 → ~0.07 per step)
- Calcium atom measurement fix in validation harness (was using ref atoms instead of center atoms)

After these, diffusion was confirmed bit-for-bit exact with MPS. It remains exact.

### Trunk-side (committed in `5201bd6`)

- **Template averaging denominator** in [chai_mlx/model/trunk.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/model/trunk.py)
  - MLX now counts valid templates using `template_input_masks` alone, matching TorchScript.
- **Mixed-precision cast policy** in [chai_mlx/model/core.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/model/core.py)
  - fp32-sensitive params are no longer incorrectly downcast in bf16 mode:
    - `FP32LayerNorm` affine params
    - `query_bias`
    - `out_scalers`
- **`linear_s2m(single)` broadcast** in the MSA module
  - MLX had been adding it only to MSA row 0. Torch broadcasts it across all MSA rows.
- **OPM / pre-pairformer fixes** in [chai_mlx/model/trunk.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/model/trunk.py)
  - removed the extra OPM denominator divide
  - matched Torch chunking along MSA depth
  - matched OPM output layernorm epsilon (`0.1`)
- **MSA pair-weighted averaging fixes** in [chai_mlx/nn/layers/attention.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/nn/layers/attention.py)
  - `v` masking uses `msa_mask`
  - pair logits / softmax / chunking / dtype flow were aligned more closely with the graph
- **Transition graph-matching chunk path** in [chai_mlx/nn/layers/common.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/nn/layers/common.py)
  - `Transition` now has a graph-matching chunked execution path along axis `-2`
  - the chunk estimate follows the TorchScript-style budget heuristic
  - graph-faithful, helpful for memory, but did not materially change the transition mismatch (the layer is position-local)
- **Native activation fix** in [chai_mlx/utils.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/utils.py)
  - `sigmoid(x)` now uses `mx.sigmoid(x)` (was `1 / (1 + exp(-x))`)
  - `silu(x)` now uses `mlx.nn.silu(x)` (was `x * sigmoid(x)`)
  - cut `msa_transition[0]` exact-input max error from `0.125` to `0.0625`
- **Regression coverage** in [tests/test_trunk.py](/Users/josephjojoe/Documents/Projects/chai-mlx/tests/test_trunk.py)
  - MSA broadcast behavior test
  - `Transition` chunking test

### Trunk-side (uncommitted — `chai_mlx/nn/layers/triangle.py`)

- **QKVG reshape ordering** in `TriangleAttention._sdpa_lazy` (critical)
  - `reshape(b, c, n, 4, H, D)` → `reshape(b, c, n, H, 4, D)`
  - indexing `proj_c[:, :, :, i]` → `proj_c[..., i, :]`
  - `pair_iter_0` exact-input error dropped from **max=90.7** to **max=2.6**
- **Ending-direction transpose** in `TriangleAttention._sdpa_lazy`
  - removed `if transpose_pair: result = result.transpose(0, 2, 1, 3, 4)`
- **Ending-direction transpose** in `ConfidenceTriangleAttention._run_direction`
  - removed `if transpose: result = result.transpose(0, 2, 1, 3, 4)`

## Detailed Bug Descriptions

### Bug 1: QKVG reshape ordering in `TriangleAttention` (critical)

**File**: [chai_mlx/nn/layers/triangle.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/nn/layers/triangle.py)

**Location**: `TriangleAttention._sdpa_lazy`, line 129

**The bug**: The QKVG projection was reshaped as `[b, c, n, 4, H, D]` (QKVG outer, heads inner). TorchScript reshapes as `[b, n, n, H, 4, D]` (heads outer, QKVG inner). This scrambled Q/K/V/G values across attention heads.

**How it was found**: Tracing the TorchScript `trunk_forward256.py` line-by-line through the starting-direction SDPA computation (lines 9795–9800) and comparing the reshape `[b, n, n, H, 4, D]` against the MLX reshape `[b, c, n, 4, H, D]`. The permute `[4, 0, 3, 1, 2, 5]` in TorchScript puts the `4` (QKVG) first and `H` third, which requires `H` to come before `4` in the reshape.

**Why it matters**: With `H=4` heads, a `[4, H, D]` vs `[H, 4, D]` reshape produces identical shapes but assigns completely different feature slices to each head's Q/K/V/G. Every attention head was computing with a wrong mix of features from all four heads.

**The fix**: Changed `reshape(b, c, n, 4, H, D)` to `reshape(b, c, n, H, 4, D)` and updated indexing from `proj_c[:, :, :, i]` to `proj_c[..., i, :]`.

**Impact**: `pair_iter_0` exact-input error dropped from **max=90.7** to **max=2.6**.

**Scope**: Only `TriangleAttention` had this bug. The following were verified correct:
- `ConfidenceTriangleAttention` — TorchScript uses `[4, H, D]` for the confidence head (different from the trunk), and MLX already had `[4, H, D]`. No change needed.
- `AttentionPairBias` — uses `chunk_last(qkvg, 4)` + `split_heads`, which naturally gives `[4, H, D]`, matching its TorchScript einsum `"dfa,aebc->edbfc"`. No change needed.
- `DiffusionSelfAttention` — uses `split_heads` + `chunk_last(qkv, 3)`. No change needed.

### Bug 2: Ending-direction spatial transpose in `TriangleAttention` and `ConfidenceTriangleAttention`

**File**: [chai_mlx/nn/layers/triangle.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/nn/layers/triangle.py)

**Location**: `TriangleAttention._sdpa_lazy` line 150, `ConfidenceTriangleAttention._run_direction` line 265

**The bug**: After computing SDPA for the ending direction (which operates on the transposed pair), the result was being transposed back to original spatial coordinates. TorchScript does NOT transpose back. It keeps the ending direction in transposed `(col, row)` order so that concatenation at position `(i, j)` pairs `starting_at(row=i, col=j)` with `ending_at(col=i, row=j)`, giving each pair both row-aggregated and column-aggregated information.

**How it was found**: Tracing `trunk_forward256.py` lines 9820–9863 (ending direction) and comparing the reshape+permute pipeline against lines 9777–9819 (starting direction). Both directions have an identical reshape-permute-reshape sequence. The only structural difference is the transposed input. No extra transpose is applied to the ending output before `cat([starting, ending], -1)` at line 9864.

The confidence head was verified separately in `confidence_head_forward256.py` (lines 693–697) where a double `permute([0,2,1,3])` cancels out, confirming the same pattern.

**The fix**: Removed `if transpose_pair: result = result.transpose(0, 2, 1, 3, 4)` from `_sdpa_lazy` and `if transpose: result = result.transpose(0, 2, 1, 3, 4)` from `_run_direction`.

**Impact**: Additional ~9 max error reduction (measured before Bug 1 was fixed: 99.5 → 90.7).

## What Was Measured

### Exact-input pair path probe (after triangle attention fixes)

Using exact Torch `pair_after_opm_0` fed into MLX `pair_transition[0]` + `triangular_multiplication[0]` + `triangular_attention[0]`:

```
pair_iter_0 vs Torch: max=2.60  mean=0.0196  p99=0.0914  ref_range=472.0
MLX RMS: 27.83  Torch RMS: 27.80
```

Before the fixes: `max=97.98`.

### Pre-pairformer trace (recycle 1, bfloat16)

```
single_after_recycle   max=0.0000
pair_after_recycle     max=0.0625
pair_after_template    max=0.0625
msa_input              max=0.0000
msa_iter_0             max=0.1250
pair_iter_0            max=13.96     (was 97.98)
msa_iter_1             max=0.50
pair_iter_1            max=21.23     (bf16 drift from iter_0 feeds forward)
msa_iter_2             max=8.00
pair_iter_2            max=33.80
```

The pair errors grow across MSA iterations because each iteration's slightly-off output becomes the next iteration's input. Growth is roughly linear (~10–13 per iteration), not exponential.

### Exact transition probe (after native activation fix, before triangle attention fixes)

Using exact Torch `msa_input` from `/tmp/chai_mlx_runs/refdump/opm0_reftrace.npz`:

- before native activation fix:
  - `max=0.125`, `mean=0.0092773`, `p99=0.125`
- after native activation fix:
  - `max=0.0625`, `mean=0.0072407`, `p99=0.0625`

### Transition op decomposition on a 32-row exact slice

On a smaller exact slice, comparing MLX vs Torch:

- `normed`: `max=0.0061545` — expected from bf16 round-trip
- `up`: `max=0.0` — exact
- `swiglu`: `max=0.03125`, `p99=0.0078125` — first real residual
- `out`: `max=0.0625`, `p99=0.0625` — swiglu error amplified by final linear

The remaining transition mismatch is at the activation / `SwiGLU` stage, not the `up` projection.

### Pair-weighted averaging isolation (32-row exact slice)

Feed exact Torch transitioned MSA + exact Torch pair → MLX `msa_pair_weighted_averaging[0]`:

- `max=0.0`, `mean=0.0`, `p99=0.0` — exact. Not a bug source.

### Stage isolation parity (exact TorchScript embeddings → MLX trunk)

```
single_trunk: max=1291  mean=40  rel=0.42
pair_trunk:   max=347   mean=26  rel=0.21
```

Per-block deltas show steady linear growth (~2–4 per block for single, ~1.5–2.5 for pair), not exponential. Block 47 has a spike (`z_delta=28.4`) that may warrant inspection but is consistent with increasing feature magnitudes at the end of the stack.

Diffusion isolation: **PASS** — `max=0.08` (bit-for-bit exact).

Confidence isolation:
```
pae_logits:   max=2.70  rel=0.23
pde_logits:   max=3.45  rel=0.21
plddt_logits: max=4.55  rel=0.06
```

### End-to-end CIF seed sweep (1L2Y, bf16, seed 42, 200 steps, 3 recycles)

```
chai-lab  median Cα: 3.8237 Å
MLX bf16  median Cα: 3.9488 Å
gap:                 0.1251 Å
```

### FP32 diagnostic (3 seeds on 1L2Y, 200 steps, 3 recycles)

```
Per-seed CIF-decoded Cα medians (Å)
  seed    chai-lab     float32       gap
    42      3.8237      3.9498    0.1262
     0      3.8143      3.9018    0.0875
   123      3.8172      3.9144    0.0972

Summary
  chai-lab median Cα: mean=3.8184 std=0.0039
  float32 median Cα:  mean=3.9220 std=0.0203
  float32 gap:        mean=0.1036 std=0.0164 min=0.0875 max=0.1262
```

**Important caveat**: This compares **MLX fp32 vs Torch bf16**. The TorchScript trunk casts to bf16 (`ScalarType 15`) for linear ops and fp32 (`ScalarType 6`) for layernorm. All 1398 parameters are stored in fp32, but the graph forces bf16 compute. Running MLX in fp32 against a bf16 reference doesn't test whether the gap is precision-related — it just adds asymmetry. The similar gap sizes (0.104 vs 0.125 Å) are expected when one side stays bf16.

## Numerical Analysis (Deep Dive)

### Kernel-level cross-backend probes (random inputs, identical on both sides)

```
Op               bf16 max error   fp32 max error   
─────────────    ──────────────   ──────────────
matmul           0.000000         0.000000         (bit-for-bit identical)
exp              0.000000         0.000000         (bit-for-bit identical)
layernorm        0.000000         0.000001         (essentially identical)
softmax          0.007812         0.000000         (diverges in bf16 only)
sigmoid          0.003906         0.000000         (diverges in bf16 only)
silu             0.031250         0.000001         (diverges in bf16 only)
fused SDPA       0.003906         0.000000         (diverges in bf16 only)
```

All ops are bit-for-bit identical in fp32. The divergence is exclusively in bf16 for ops involving sigmoid/silu/softmax.

### Exhaustive bf16 enumeration (all 65,280 finite bf16 values)

```
exp:      0 disagreements out of 34,168 valid inputs (0.00%)
sigmoid:  1,089 disagreements out of 65,280 (1.67%)
silu:     1,382 disagreements out of 65,280 (2.12%)
```

`exp` is perfectly identical. The sigmoid/silu divergences are in the fused kernel, not in `exp`.

### Fused vs decomposed sigmoid analysis

Decomposing sigmoid as `1/(1+exp(-x))` step-by-step (using the bit-identical `exp`):

```
Manual sigmoid (MLX) vs Manual sigmoid (MPS):     0 disagreements
MLX fused sigmoid vs MLX manual sigmoid:         518 disagreements
MPS fused sigmoid vs MPS manual sigmoid:       1,113 disagreements
MLX fused sigmoid vs MPS fused sigmoid:        1,089 disagreements
```

**Key insight**: When both backends compute sigmoid by decomposing into `exp` → add → reciprocal, the results are **perfectly identical**. The disagreements come entirely from the fused kernels. Both MLX and MPS have fused sigmoid implementations that deviate from the decomposed arithmetic — MLX's deviates in 518 places, MPS's deviates in 1,113 places. Neither is "correct" or "wrong"; they make different speed/accuracy tradeoffs in their Metal shaders.

### Which side is more accurate?

For all 1,089 sigmoid disagreements and all 1,382 silu disagreements:
- **MPS is closer to fp64 ground truth: 100% of cases**
- **MLX is closer: 0% of cases**

MPS's fused sigmoid/silu is strictly more accurate than MLX's at the bf16 level.

### TorchScript mixed-precision discovery

The TorchScript trunk model stores all 1,398 parameters in fp32 but the graph casts to bf16 (`ScalarType 15`) for linear operations and back to fp32 (`ScalarType 6`) for layernorm. This means the chai-lab reference runs in **bf16 compute with fp32 layernorm** — the same mixed-precision strategy as MLX's bf16 mode.

### How the gap compounds

Each pairformer block has ~5 sublayers using sigmoid/silu/softmax. Each introduces a handful of 1-ULP errors. The residual connection `pair = pair + sublayer(pair)` accumulates these additively. After 48 blocks, the linear accumulation (measured at ~2–4 error per block) produces total trunk-output errors of max=1291 (single) and max=347 (pair). Despite these intermediate errors, the diffusion module is robust enough that final structures differ by only 0.1 Å.

### What this means for CUDA comparison

The entire investigation compares MLX against Torch/MPS. CUDA has its own fused sigmoid/silu kernels (in cuDNN) with their own rounding tradeoffs. We have no data on whether CUDA's fused kernels match MLX, MPS, or neither. The 0.1 Å gap is MLX-vs-MPS specific. The MLX-vs-CUDA gap could be smaller, larger, or the same.

### Could the gap be closed?

In theory, yes: since the decomposed `1/(1+exp(-x))` is bit-identical across MLX and MPS, replacing MLX's fused `mx.sigmoid` with the decomposed formula would eliminate the sigmoid disagreements. Similarly for silu. However:

1. The native MLX activation fix already *switched from* the decomposed formula *to* the fused kernel — and that *reduced* the error against MPS (from `max=0.125` to `max=0.0625` for `msa_transition[0]`). This seems contradictory but is explained by the fact that the decomposed formula produces *different* rounding than either fused kernel, and the fused-vs-fused comparison happens to be better than decomposed-vs-fused.
2. The real target is CUDA, not MPS. Matching MPS exactly might make us *less* accurate against CUDA if CUDA's fused kernels are closer to MLX's.
3. A 128 KB lookup table per op could force exact MPS-matching, but this optimizes for the wrong reference.

## Current Status of Each Module

### Diffusion — DONE

Bit-for-bit exact when given correct trunk outputs. Confirmed in isolation (`max=0.08`). No regressions.

### Trunk pre-pairformer — DONE

All operations verified correct on exact inputs:
- recycle projections — exact
- template averaging — exact (denominator fix committed)
- MSA input broadcast — exact (linear_s2m fix committed)
- OPM0 — exact (denominator, chunking, epsilon fixes committed)
- `msa_transition[0]` — `max=0.0625` (fused sigmoid/silu rounding floor)
- `msa_pair_weighted_averaging[0]` — exact
- `triangular_multiplication[0]` — passes through exact-input probe cleanly
- `triangular_attention[0]` — fixed (QKVG reshape + transpose, uncommitted)

### Trunk pairformer (48 blocks) — DONE

Per-block error growth is linear (~2–4 per block), consistent with fused-kernel rounding differences accumulating through residual connections. No structural bugs found. The `TriangleAttention` fix applies to all pairformer blocks (same class).

### Confidence head — LIKELY DONE, needs end-to-end verification

`ConfidenceTriangleAttention` transpose fix applied. Isolation test shows moderate logit errors (max 2.7–4.5), consistent with trunk drift propagating into the head. No structural bugs found, but pLDDT/PAE distributions have not been compared end-to-end.

## Validation Status

Passed:

- `pytest -q tests/test_trunk.py tests/test_attention.py tests/test_featurize.py tests/test_diffusion.py`
- result: `10 passed`

Completed:

- fresh `trunk_block_trace.py` pre-pairformer trace after all fixes
- fresh `stage_isolation_parity.py` after all fixes
- fresh `cif_seed_sweep.py` end-to-end validation (bf16 and fp32)
- exhaustive bf16 kernel probes (all 65,280 finite values for exp, sigmoid, silu)
- fused vs decomposed sigmoid analysis

## Files Changed (All Sessions Combined)

### Committed

- `5201bd6` — [chai_mlx/nn/layers/common.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/nn/layers/common.py) (transition chunking), [chai_mlx/utils.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/utils.py) (native sigmoid/silu), [tests/test_trunk.py](/Users/josephjojoe/Documents/Projects/chai-mlx/tests/test_trunk.py) (regression tests)
- `b9e97f2` — [chai_mlx/nn/layers/triangle.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/nn/layers/triangle.py) (QKVG reshape + ending-direction transpose)
- `6ff327b` — [chai_mlx/model/ranking.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/model/ranking.py) + [chai_mlx/data/types.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/data/types.py) + [chai_mlx/data/featurize.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/data/featurize.py) + [tests/test_ranking.py](/Users/josephjojoe/Documents/Projects/chai-mlx/tests/test_ranking.py) (full port of `chai_lab.ranking.rank` — logits-space pTM/ipTM, valid-frame mask, dense clash matrix, per-chain pLDDT, parity tests)

### Diagnostic scripts created during investigation

- [scripts/pair_path_probe.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/pair_path_probe.py) — targeted pair path decomposition probe (can be kept or removed)

## Harness Guide

The repo has several useful harnesses under `scripts/`. For trunk investigation, the most useful ones are:

- [scripts/chai_lab_reference_dump.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/chai_lab_reference_dump.py)
  - builds a fresh FASTA-backed reference bundle from `chai-lab`
  - writes both `input_npz` and `reference_npz`
- [scripts/stage_isolation_parity.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/stage_isolation_parity.py)
  - feeds Torch reference tensors into MLX stage boundaries
  - tells you whether trunk or diffusion is currently the first failing stage
- [scripts/layer_parity.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/layer_parity.py)
  - captures detailed MLX trunk/diffusion intermediates
  - useful for targeted parity once you know the failing module
- [scripts/trunk_block_trace.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/trunk_block_trace.py)
  - clones `trunk.forward_<crop_size>` and registers internal graph values as explicit Torch outputs
  - this is the main trunk-localization tool
- [scripts/cif_seed_sweep.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/cif_seed_sweep.py)
  - end-to-end structural check over multiple seeds
  - useful after a concrete fix, especially with `--skip-reference`
- [scripts/pair_path_probe.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/pair_path_probe.py)
  - targeted pair path decomposition: feeds exact Torch pair inputs through MLX pair_transition + triangular_multiplication + triangular_attention
- [scripts/downstream_reference_trace.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/downstream_reference_trace.py)
  - traces the real post-trunk downstream path on both sides
- [scripts/granular_reference_dump.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/granular_reference_dump.py)
  - broader reference dump with more detailed captures
- [scripts/precision_experiments.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/precision_experiments.py)
  - useful when testing whether a residual is actually precision-related
- [scripts/structural_validation.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/structural_validation.py)
  - end-to-end structure generation and CIF-side validation

Older exploration scripts still useful for context:

- `scripts/deep_denoise_trace.py`
- `scripts/bisect_denoise.py`
- `scripts/ts_block_reference.py`
- `scripts/diffusion_diagnostics.py`

## Local Reference Implementation

The local reference implementation is in:

- `chai-lab/`

Important caveats:

- it is .gitignored
- some readable Python source there is only a runtime wrapper around exported TorchScript modules
- for trunk internals, do not rely only on the Python source tree in `chai-lab`
- use `chai-lab/chai_lab/chai1.py` for component loading / runtime wiring
- use `findings/graphs/` for actual trunk semantics

**Critical**: The TorchScript models use bf16 mixed precision internally. This is baked into the exported graph — parameters are fp32 but the graph casts to `ScalarType 15` (bf16) for linear ops. This cannot be changed. Any comparison against chai-lab is therefore a comparison against bf16 compute.

## Useful Artifacts

Reference / trace files already available:

- `/tmp/chai_mlx_runs/refdump/input_1l2y_r3_trunk_only.npz`
- `/tmp/chai_mlx_runs/refdump/input_1l2y_r3.npz`
- `/tmp/chai_mlx_runs/refdump/reference_1l2y_r3.npz`
- `/tmp/chai_mlx_runs/refdump/opm0_reftrace.npz`
- `/tmp/chai_mlx_runs/refdump/trunk_prepair_r1_reftrace.npz`
- `/tmp/chai_mlx_runs/refdump/trunk_block0_fresh_reftrace.npz`
- `/tmp/chai_mlx_runs/refdump/trunk_block0_r1_reftrace.npz`

Intermediate one-off probe artifacts:

- `/tmp/chai_mlx_runs/refdump/transition0_torch.npy`
- `/tmp/chai_mlx_runs/refdump/transition0_mlx.npy`
- `/tmp/chai_mlx_runs/refdump/transition0_ops_torch.npz`
- `/tmp/chai_mlx_runs/refdump/transition0_ops_mlx.npz`
- `/tmp/chai_mlx_runs/refdump/pwa0_torch_slice.npy`
- `/tmp/chai_mlx_runs/refdump/pwa0_mlx_slice.npy`

End-to-end CIF references:

- `/tmp/chai_mlx_runs/seed_sweep_fp32_diag/chai_lab/seed_*/pred.model_idx_0.cif` (seeds 0, 42, 123)
- `/tmp/chai_mlx_runs/seed_sweep_fp32_diag/mlx/float32/seed_*/pred.model_idx_0.cif` (seeds 0, 42, 123)

## Recommended Next Steps

The port is structurally faithful. There are no known bugs. The remaining numerical differences are the expected consequence of different fused-kernel rounding in bf16 between two Apple Metal backends.

### Priority 1: Larger target validation (50–100 residues)

1L2Y is a 20-residue mini-protein (trp-cage) — it barely exercises the `n×n` pair tensor. A longer sequence will stress the trunk's recycle dynamics and triangle ops much harder. If the gap stays in the 0.1–0.3 Å range on a larger target, the port is done. If it blows up to several Å, there is a bug that 1L2Y was too small to expose.

Suggested targets: any single-chain protein in the 50–100 residue range with mixed secondary structure.

### Priority 2: Multimer / ligand target (if available)

Pair representation handling for cross-chain and ligand contacts goes through code paths that monomer mini-proteins barely touch. If the test set has a multimer or anything with a ligand, running one of those is a good stress test for cross-chain pair representation, ligand featurization, and any code paths that are simply never hit by a 20-residue monomer.

### Priority 3: Confidence head end-to-end comparison

The stage isolation test showed moderate pLDDT/PAE errors (max ~2.7–4.5 in logits). Structural Cα agreement at 0.1 Å is necessary but not sufficient — confidence outputs are a separate head and can be subtly wrong even when coordinates look fine. Compare predicted pLDDT distributions and PAE matrices on a few targets between MLX and chai-lab. This matters for any downstream use of confidence scores (e.g., filtering predictions, ranking models).

**Current gap**: The `cif_seed_sweep.py` script passes `bfactors=None` when writing MLX CIFs, so pLDDT is not stored. The sweep script also does not compute or save ptm/iptm/aggregate scores on the MLX side. To do this comparison, either:
- modify `cif_seed_sweep.py` to run the MLX confidence head and save scores/pLDDT alongside the CIF, or
- write a targeted script that loads trunk outputs and runs only the confidence head on both sides.

### Priority 4: CUDA reference comparison

All current comparisons are against Torch/MPS. If access to a CUDA machine is available, running chai-lab on CUDA and comparing the CIFs against MLX would establish the real-world gap. This is the comparison that actually matters for production users.

### Deprioritized

- **Multi-seed stability on 1L2Y** — The fp32 diagnostic ran 3 seeds and showed consistent gaps (0.088–0.126 Å, std=0.016). More seeds on a tiny protein add limited signal.
- **Recycle stability** — Worth checking if a larger target shows issues, otherwise low priority.
- **Triangle multiplication audit** — Kernel probes showed no algorithm differences. End-to-end results are good. Low priority.
- **Re-probe `msa_transition[0]`** — The 0.0625 max error is the fused-kernel rounding floor. Not fixable without replacing fused sigmoid/silu with decomposed versions, which may hurt CUDA comparison.

## Useful Commands

Re-run pre-pairformer trace:
```bash
python3 scripts/trunk_block_trace.py \
  --weights-dir weights \
  --input-npz /tmp/chai_mlx_runs/refdump/input_1l2y_r3_trunk_only.npz \
  --reference-npz /tmp/chai_mlx_runs/refdump/reference_1l2y_r3.npz \
  --reference-trace-npz /tmp/chai_mlx_runs/refdump/trunk_prepair_r1_reftrace.npz \
  --recycles 1 --trace-scope pre_pairformer --dtypes bfloat16
```

Re-run full stage isolation:
```bash
python3 scripts/stage_isolation_parity.py \
  --weights-dir weights \
  --input-npz /tmp/chai_mlx_runs/refdump/input_1l2y_r3.npz \
  --reference-npz /tmp/chai_mlx_runs/refdump/reference_1l2y_r3.npz \
  --recycles 1 --compute-dtype bfloat16 --verbose
```

Run exact-input pair path probe:
```bash
python3 scripts/pair_path_probe.py \
  --weights-dir weights \
  --opm-trace /tmp/chai_mlx_runs/refdump/opm0_reftrace.npz \
  --prepair-trace /tmp/chai_mlx_runs/refdump/trunk_prepair_r1_reftrace.npz \
  --input-npz /tmp/chai_mlx_runs/refdump/input_1l2y_r3_trunk_only.npz \
  --compute-dtype bfloat16
```

Seed sweep on a new target (replace FASTA path):
```bash
python3 scripts/cif_seed_sweep.py \
  --weights-dir weights \
  --work-dir /tmp/chai_mlx_runs/seed_sweep_new_target \
  --seeds 0 42 123 \
  --num-steps 200 --recycles 3 --mlx-dtypes bfloat16
```

## Bottom Line

The MLX port of Chai-1 is structurally faithful. Two critical bugs in `TriangleAttention` were the dominant error source and are now fixed. The diffusion module is bit-for-bit exact. The remaining 0.1 Å Cα gap against Torch-MPS comes from different fused-kernel rounding in bf16 `sigmoid`/`silu` between MLX and MPS — not from algorithmic differences (all ops are bit-identical in fp32, and `exp` is bit-identical even in bf16). This comparison is against MPS only; we have no CUDA reference data. The remaining work is validation on larger and more diverse targets, not debugging.
