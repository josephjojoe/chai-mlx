# Trunk Investigation Handoff

Date: 2026-04-14

This document summarizes the Chai-1 MLX port investigation status after the diffusion-side fixes and the latest trunk-side tracing work.

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
- CPU fallback is usually not worth it for the heavy trunk/diffusion traces.

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

For trunk debugging, trust the graph dumps over guesses about “what the source probably meant”.

## Recent Git History Worth Reading

Recent commits give the best high-level narrative of what has already been fixed:

- `79544ff`
  - “trying to fix mlx trunk; significant variance in results from one seed to another, so not just a structural error”
- `07dd467`
  - “huge fix, diffusion module on mlx now bit-for-bit exact effectively with mps; was measuring calcium atoms wrong... now effectively all the difference comes from the trunk”
- `dc7d995`
  - “huge fixes ... residual problem went from ~4.0 to ~0.07 per step ... harnesses are good”
- `76a2172`
  - “big fix in attention; qkv interleaving bug”
- `0bc4490`
  - early large diffusion-side cleanup: pair update source mismatch, blocked pair indexing, residual ordering, new tracing/isolation harnesses

Useful commands:

```bash
git log --oneline --decorate -n 20
git show --stat --summary 07dd467
git show --stat --summary dc7d995
git show --stat --summary 76a2172
git show --stat --summary 0bc4490
```

The commit subjects are blunt but informative. They are worth scanning before duplicating old work.

## Executive Summary

- The remaining first structural mismatch is trunk-side, not diffusion-side.
- The first bad checkpoint was previously localized to:
  - `trunk.recycle_0.pre_pairformer.msa_iter_0`
- That mismatch is now narrowed further:
  - `msa_pair_weighted_averaging[0]` is exact when fed exact Torch transition inputs.
  - The remaining error is transition-side inside `msa_transition[0]`.
- A new concrete bug was fixed this turn:
  - native MLX `sigmoid` / `silu` are now used instead of the hand-rolled activation formulas in [chai_mlx/utils.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/utils.py)
- That activation fix materially reduced the exact transition mismatch:
  - `msa_transition[0]` exact-input max error: `0.125 -> 0.0625`
  - exact `msa_iter_0` chain max error: `0.25 -> 0.125`

## What Is Already Fixed

These fixes were already in place before the latest activation work:

- Template averaging denominator in [chai_mlx/model/trunk.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/model/trunk.py)
  - MLX now counts valid templates using `template_input_masks` alone, matching TorchScript.
- Mixed-precision cast policy in [chai_mlx/model/core.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/model/core.py)
  - fp32-sensitive params are no longer incorrectly downcast in bf16 mode:
    - `FP32LayerNorm` affine params
    - `query_bias`
    - `out_scalers`
- `linear_s2m(single)` broadcast in the MSA module
  - MLX had been adding it only to MSA row 0.
  - Torch broadcasts it across all MSA rows.
- OPM / pre-pairformer fixes in [chai_mlx/model/trunk.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/model/trunk.py)
  - removed the extra OPM denominator divide
  - matched Torch chunking along MSA depth
  - matched OPM output layernorm epsilon (`0.1`)
- MSA pair-weighted averaging fixes in [chai_mlx/nn/layers/attention.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/nn/layers/attention.py)
  - `v` masking uses `msa_mask`
  - pair logits / softmax / chunking / dtype flow were aligned more closely with the graph

These fit the broader repo history from `git log`:

- diffusion-side issues were heavily reduced across `0bc4490`, `76a2172`, `dc7d995`, and `07dd467`
- the repo is now in the phase where the remaining first blocker is trunk fidelity, not the old diffusion blow-ups

## New Work In This Turn

### 1. Transition graph-matching chunk path

Updated [chai_mlx/nn/layers/common.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/nn/layers/common.py):

- `Transition` now has a graph-matching chunked execution path along axis `-2`
- the chunk estimate follows the TorchScript-style budget heuristic

Important note:

- This change is graph-faithful and helpful for memory behavior.
- It did **not** materially change the transition mismatch by itself, because the layer is position-local.

### 2. Native activation fix

Updated [chai_mlx/utils.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/utils.py):

- `sigmoid(x)` now uses `mx.sigmoid(x)`
- `silu(x)` now uses `mlx.nn.silu(x)`

This was a real fidelity bug. The old implementation:

- `sigmoid(x) = 1 / (1 + exp(-x))`
- `silu(x) = x * sigmoid(x)`

was measurably worse than Torch/MPS bf16 in the transition path.

### 3. New regression coverage

Updated [tests/test_trunk.py](/Users/josephjojoe/Documents/Projects/chai-mlx/tests/test_trunk.py):

- existing test still covers MSA broadcast behavior
- new test verifies `Transition` chunks along axis `-2` when the budget is exceeded

## What Was Measured

### Exact transition probe

Using:

- exact Torch `msa_input` from `/tmp/chai_mlx_runs/refdump/opm0_reftrace.npz`
- TorchScript `trunk.msa_module.msa_transition.0`
- MLX `msa_transition[0]`

Results:

- before native activation fix:
  - `max=0.125`
  - `mean=0.0092773`
  - `p99=0.125`
- after native activation fix:
  - `max=0.0625`
  - `mean=0.0072407`
  - `p99=0.0625`

### Exact `msa_iter_0` chain

Feeding exact Torch inputs into:

- `msa_transition[0]`
- then `msa_pair_weighted_averaging[0]`

Results:

- before native activation fix:
  - `max=0.25`
  - `mean=0.013046`
  - `p99=0.25`
- after native activation fix:
  - `max=0.125`
  - `mean=0.0089482`
  - `p99=0.125`

### Transition op decomposition on a 32-row exact slice

On a smaller exact slice, comparing MLX vs Torch:

- `normed`
  - `max=0.0061545`
  - expected from bf16 round-trip before explicit cast into the linear
- `up`
  - exact
  - `max=0.0`
- `swiglu`
  - still the first real residual
  - `max=0.03125`
  - `p99=0.0078125`
- `out`
  - `max=0.0625`
  - `p99=0.0625`

Interpretation:

- the remaining transition mismatch is not in the `up` projection
- it is entering at the activation / `SwiGLU` stage and then amplified by the final linear

### Pair-weighted averaging isolation

On a 32-row exact slice:

- feed exact Torch transitioned MSA input
- feed exact Torch `pair_after_opm_0`
- compare MLX `msa_pair_weighted_averaging[0]` vs Torch

Result:

- exact
  - `max=0.0`
  - `mean=0.0`
  - `p99=0.0`

Interpretation:

- `msa_pair_weighted_averaging[0]` is not the remaining bug
- the current `msa_iter_0` residual is transition error propagating forward

## What Still Works

### Diffusion

Current evidence says diffusion is no longer the first blocker.

- Earlier work already fixed the meaningful diffusion-side fidelity issues.
- The remaining first structural mismatch has been consistently localized upstream into the trunk pre-pairformer path.
- I did **not** finish a fresh full downstream diffusion revalidation after the latest transition activation fix in this interrupted turn.

Practical conclusion:

- Treat diffusion as no longer the lead failure site.
- The active blocker is trunk-side.

That conclusion is also consistent with recent commit history:

- `07dd467` explicitly reports the MLX diffusion module as effectively bit-for-bit exact with MPS after the major downstream fixes
- current tracing still points the first live mismatch upstream into trunk pre-pairformer logic

### OPM and pre-pairformer setup

The following are now in good shape based on exact-input checks:

- recycle projections
- template averaging path
- MSA input broadcast
- OPM0
- pair-weighted averaging, when given exact transition inputs

## What Still Does Not Work

The remaining first unresolved trunk mismatch is still transition-side:

- `trunk.recycle_0.pre_pairformer.msa_iter_0`
- specifically inside `msa_transition[0]`
- more specifically at `SwiGLU` / transition output, not the `up` projection

At this point the remaining transition residual looks like one of:

- a remaining MLX-vs-Torch bf16 activation/kernel mismatch
- a Torch/MPS bf16 `silu` behavior that is not exactly reproduced by MLX

What it does **not** currently look like:

- a mask bug
- a template bug
- an OPM bug
- a pair-weighted averaging bug
- a simple “run SiLU in fp32” fix

I tested fp32-style SiLU formulations directly against Torch bf16 behavior:

- bf16-native SiLU variants were best:
  - `max=0.03125`
- fp32 SiLU then cast back to bf16 was worse:
  - `max=0.0625`

So fp32 SiLU is not the right fix here.

## Validation Status

Passed:

- `pytest -q tests/test_trunk.py tests/test_attention.py tests/test_featurize.py tests/test_diffusion.py`
- result: `10 passed`

Not completed in this interrupted turn:

- fresh rerun of `scripts/trunk_block_trace.py` after the latest activation fix
- fresh rerun of `scripts/stage_isolation_parity.py` after the latest activation fix
- fresh MLX-only seed sweep after the latest activation fix

I stopped and killed the in-flight trace / stage-isolation processes when this handoff was requested.

## Files Changed In This Turn

- [chai_mlx/nn/layers/common.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/nn/layers/common.py)
- [chai_mlx/utils.py](/Users/josephjojoe/Documents/Projects/chai-mlx/chai_mlx/utils.py)
- [tests/test_trunk.py](/Users/josephjojoe/Documents/Projects/chai-mlx/tests/test_trunk.py)

## Harness Guide

The repo already has several useful harnesses under `scripts/`. For this specific trunk investigation, the most useful ones are:

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
- [scripts/downstream_reference_trace.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/downstream_reference_trace.py)
  - traces the real post-trunk downstream path on both sides
- [scripts/granular_reference_dump.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/granular_reference_dump.py)
  - broader reference dump with more detailed captures
- [scripts/precision_experiments.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/precision_experiments.py)
  - useful when testing whether a residual is actually precision-related
- [scripts/structural_validation.py](/Users/josephjojoe/Documents/Projects/chai-mlx/scripts/structural_validation.py)
  - end-to-end structure generation and CIF-side validation

There are also older exploration scripts that are still useful for context:

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

## Useful Artifacts

Reference / trace files already available:

- `/tmp/chai_mlx_runs/refdump/input_1l2y_r3_trunk_only.npz`
- `/tmp/chai_mlx_runs/refdump/reference_1l2y_r3.npz`
- `/tmp/chai_mlx_runs/refdump/opm0_reftrace.npz`
- `/tmp/chai_mlx_runs/refdump/trunk_prepair_r1_reftrace.npz`

Intermediate one-off probe artifacts from this turn:

- `/tmp/chai_mlx_runs/refdump/transition0_torch.npy`
- `/tmp/chai_mlx_runs/refdump/transition0_mlx.npy`
- `/tmp/chai_mlx_runs/refdump/transition0_ops_torch.npz`
- `/tmp/chai_mlx_runs/refdump/transition0_ops_mlx.npz`
- `/tmp/chai_mlx_runs/refdump/pwa0_torch_slice.npy`
- `/tmp/chai_mlx_runs/refdump/pwa0_mlx_slice.npy`

## Recommended Next Steps

1. Re-run the pre-pairformer trace after the native activation fix:
   - `python3 scripts/trunk_block_trace.py --weights-dir weights --input-npz /tmp/chai_mlx_runs/refdump/input_1l2y_r3_trunk_only.npz --reference-npz /tmp/chai_mlx_runs/refdump/reference_1l2y_r3.npz --reference-trace-npz /tmp/chai_mlx_runs/refdump/trunk_prepair_r1_reftrace.npz --recycles 1 --trace-scope pre_pairformer --dtypes bfloat16 --mlx-device gpu --torch-device mps --jump-threshold 1e-3`
2. Re-run trunk stage isolation:
   - `python3 scripts/stage_isolation_parity.py --weights-dir weights --input-npz /tmp/chai_mlx_runs/refdump/input_1l2y_r3_trunk_only.npz --reference-npz /tmp/chai_mlx_runs/refdump/reference_1l2y_r3.npz --recycles 3 --compute-dtype bfloat16 --mlx-device gpu --verbose`
3. Re-run the MLX-only seed sweep against the existing chai-lab CIFs:
   - `python3 scripts/cif_seed_sweep.py --weights-dir weights --work-dir /tmp/chai_mlx_runs/seed_sweep_r3_s0123 --seeds 0 42 123 --num-steps 200 --recycles 3 --mlx-dtypes bfloat16 --skip-reference --torch-device mps`
4. If the first jump is still transition-side after those reruns, focus only on the transition activation/kernel behavior.
5. Do **not** spend time re-debugging OPM or pair-weighted averaging unless a fresh trace contradicts the exact-input isolations above.

## Bottom Line

The remaining first bad trunk checkpoint is still `msa_iter_0`, but it has now been narrowed to the transition path itself. The latest real fix was the native MLX activation change, which cut the exact transition and `msa_iter_0` errors in half. Pair-weighted averaging is exact on exact transition inputs. The next question is no longer “where is the bug?”; it is “is the remaining transition-side bf16 activation mismatch fixable in user code, or is it the residual backend-level difference we have to live with?”
