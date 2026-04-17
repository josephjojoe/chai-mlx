"""Precision-targeting experiments for the MLX diffusion module.

Priority-ordered experiments to validate whether higher-precision matmul
can rescue the diffusion ODE, and to find the exact convergence threshold.

Experiments:

  1. fp32_all    — FP32 matmul in ALL diffusion module linears
  2. fp32_down   — FP32 matmul ONLY in down-projection linears
  3. ode_thresh  — Inject noise into denoise outputs to find the ODE
                   convergence threshold empirically
  4. inject_bisect — Run fp32 and bf16 in lockstep, swap at step k to
                     separate "catastrophic early divergence" from
                     "uniform slow drift"

Usage::

    python3 scripts/precision_experiments.py --weights-dir weights/ \\
        --experiment fp32_all

Experiments 3-4 depend on experiment 1 producing valid structures (they
use the fp32-upcast model as the reference trajectory).
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import warnings
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists():
    sys.path.insert(0, str(LOCAL_CHAI_LAB))


# ── FP32 Linear monkey-patch ─────────────────────────────────────────────


def _walk_modules(module, prefix=""):
    """Yield (full_path, leaf_name, child_module) for all descendants."""
    for rel_path, child in module.named_modules():
        if not rel_path:
            continue
        full_path = f"{prefix}.{rel_path}" if prefix else rel_path
        leaf_name = rel_path.rsplit(".", 1)[-1]
        yield full_path, leaf_name, child


def _cast_module_weights(module, dtype):
    """Cast all parameters of a module (and its descendants) to dtype."""
    if not hasattr(module, 'parameters'):
        return
    from mlx.utils import tree_flatten
    params = module.parameters()
    if not params:
        return
    pairs = []
    for k, v in tree_flatten(params):
        if isinstance(v, mx.array) and v.dtype != dtype:
            pairs.append((k, v.astype(dtype)))
    if pairs:
        module.load_weights(pairs, strict=False)


# ── Cα extraction and metrics ────────────────────────────────────────────


def extract_ca(coords_np, structure_inputs):
    """Extract Cα coords [N_residues, 3] from all-atom coords."""
    mask = np.array(structure_inputs.atom_exists_mask.astype(mx.float32))[0]
    centre_idx = getattr(structure_inputs, "token_centre_atom_index", None)
    if centre_idx is None:
        warnings.warn(
            "structure_inputs.token_centre_atom_index missing; falling back to "
            "token_reference_atom_index for Cα extraction. Regenerate the NPZ bundle.",
            stacklevel=2,
        )
        centre_idx = structure_inputs.token_reference_atom_index
    centre_idx = np.array(centre_idx)[0]
    tok_mask = np.array(structure_inputs.token_exists_mask.astype(mx.float32))[0]

    n_tok = int(tok_mask.sum())
    ca = []
    for t in range(n_tok):
        if tok_mask[t] < 0.5:
            continue
        ri = int(centre_idx[t])
        if 0 <= ri < len(coords_np) and mask[ri] > 0.5:
            ca.append(coords_np[ri])
    return np.array(ca, dtype=np.float64) if ca else np.zeros((0, 3))


def ca_spacing(ca):
    """Median and mean consecutive Cα spacing."""
    if len(ca) < 2:
        return float("nan"), float("nan")
    d = np.sqrt(np.sum(np.diff(ca, axis=0) ** 2, axis=-1))
    return float(np.median(d)), float(np.mean(d))


def is_valid_structure(ca, lo=3.0, hi=5.0):
    med, _ = ca_spacing(ca)
    return lo < med < hi


# ── Pipeline runner ──────────────────────────────────────────────────────


def featurize_1l2y(tmpdir):
    """Featurize the 1L2Y Trp-cage miniprotein (20 residues)."""
    from chai_mlx.data.featurize import featurize_fasta

    fasta = Path(tmpdir) / "1L2Y.fasta"
    fasta.write_text(">protein|name=1L2Y\nNLYIQWLKDGGPSSGRPPPS\n")
    feat_dir = Path(tmpdir) / "features"
    feat_dir.mkdir(exist_ok=True)
    return featurize_fasta(
        fasta, output_dir=feat_dir,
        use_esm_embeddings=False, use_msa_server=False,
        use_templates_server=False,
    )


def run_diffusion_loop(
    model,
    cache,
    structure,
    *,
    num_steps=200,
    seed=42,
    denoise_hook=None,
    swap_coords_at=None,
    swap_trajectory=None,
    log_interval=25,
):
    """Custom diffusion loop with optional denoise hook and coordinate swapping.

    denoise_hook:  callable(denoised, step_idx) → modified_denoised
    swap_coords_at: set of step indices where coords should be replaced
    swap_trajectory: list of coords arrays indexed by step
    """
    mx.random.seed(seed)
    coords = model.init_noise(1, 1, structure)
    mx.eval(coords)
    dm = model.diffusion_module
    schedule = list(model.schedule(num_steps))

    orig_denoise = dm.denoise.__func__
    step_counter = [0]

    def hooked_denoise(self, c, co, si):
        out = orig_denoise(self, c, co, si)
        if denoise_hook is not None:
            mx.eval(out)
            out = denoise_hook(out, step_counter[0])
        return out

    import types
    dm.denoise = types.MethodType(hooked_denoise, dm)

    try:
        for i, (s_curr, s_next, gamma) in enumerate(schedule):
            step_counter[0] = i
            if swap_coords_at and i in swap_coords_at and swap_trajectory:
                coords = mx.array(swap_trajectory[i])
            coords = dm.diffusion_step(cache, coords, s_curr, s_next, gamma)
            mx.eval(coords)
            if log_interval and (i % log_interval == 0 or i == len(schedule) - 1):
                std = float(mx.sqrt(mx.mean(coords * coords)).item())
                print(f"      step {i+1:>3}/{num_steps}  coord_rms={std:.2f}")
    finally:
        dm.denoise = types.MethodType(orig_denoise, dm)

    return np.array(coords.astype(mx.float32))


def prepare_pipeline(model, ctx):
    """Embed + trunk + cache. Returns (cache, structure, emb)."""
    emb = model.embed_inputs(ctx)
    trunk_out = model.trunk(emb, recycles=3)
    cache = model.prepare_diffusion_cache(trunk_out)
    mx.eval(
        cache.s_static, cache.z_cond, cache.blocked_pair_base,
        cache.atom_cond, cache.atom_single_cond, *cache.pair_biases,
    )
    return cache, emb.structure_inputs, emb


# ── Experiment 1 & 2: FP32 upcast ────────────────────────────────────────


def run_fp32_weights_experiment(weights_dir, mode, num_steps=200, seed=42):
    """Keep diffusion module weights in original fp32 from safetensors.

    Metal bf16 matmul already uses fp32 accumulators internally, so casting
    bf16→fp32 at runtime is a no-op. The real test is keeping the original
    fp32 weight precision to eliminate bf16 quantization error (~0.01 per
    element, ~0.03 mean per matmul output).

    mode="all":  fp32 weights in ALL diffusion linears
    mode="down": fp32 weights only in down-projection linears
    """
    from chai_mlx import ChaiMLX

    label = f"fp32w_{mode}"
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {label}")
    print(f"  Original fp32 weights in {'ALL' if mode == 'all' else 'DOWN-PROJ ONLY'} diffusion linears")
    print(f"  (Metal bf16 matmul already uses fp32 accumulators — the gain here")
    print(f"   comes from eliminating bf16 weight quantization error)")
    print(f"{'=' * 70}")

    # Load model with fp32 weights (no bf16 quantization)
    print("  Loading model with fp32 weights...")
    model = ChaiMLX.from_pretrained(weights_dir, strict=False, compute_dtype="float32")

    if mode == "all":
        # Cast everything EXCEPT diffusion_module to bf16
        _cast_module_weights(model.input_embedder, mx.bfloat16)
        _cast_module_weights(model.trunk_module, mx.bfloat16)
        _cast_module_weights(model.confidence_head, mx.bfloat16)
        _cast_module_weights(model.ranker, mx.bfloat16)
        n_fp32 = sum(1 for _, n, m in _walk_modules(model.diffusion_module, "dm")
                     if isinstance(m, nn.Linear))
        print(f"  Diffusion module: {n_fp32} linears kept in fp32")
        print(f"  All other modules: cast to bf16")
    else:
        # Cast everything to bf16, then selectively promote down-proj back to fp32
        _cast_module_weights(model.input_embedder, mx.bfloat16)
        _cast_module_weights(model.trunk_module, mx.bfloat16)
        _cast_module_weights(model.confidence_head, mx.bfloat16)
        _cast_module_weights(model.ranker, mx.bfloat16)
        _cast_module_weights(model.diffusion_module, mx.bfloat16)

        # Reload fp32 weights only for down-projections
        from chai_mlx.io.weights.load import load_safetensors
        model_fp32 = ChaiMLX.from_pretrained(weights_dir, strict=False, compute_dtype="float32")
        promoted = []
        for path, name, mod in _walk_modules(model.diffusion_module, "diffusion_module"):
            if isinstance(mod, nn.Linear) and name == "down":
                fp32_mod = dict(_walk_modules(model_fp32.diffusion_module, "diffusion_module"))
                # Walk the fp32 model with the same path structure
                for p2, _, m2 in _walk_modules(model_fp32.diffusion_module, "diffusion_module"):
                    if p2 == path:
                        mod.weight = m2.weight
                        if "bias" in mod:
                            mod.bias = m2.bias
                        promoted.append(path)
                        break
        del model_fp32
        gc.collect()
        mx.clear_cache()
        print(f"  {len(promoted)} down-projection linears promoted to fp32")

    with tempfile.TemporaryDirectory() as tmpdir:
        print("  Featurizing 1L2Y...")
        ctx = featurize_1l2y(tmpdir)

        print("  Running trunk (bf16)...")
        t0 = time.time()
        cache, structure, emb = prepare_pipeline(model, ctx)
        trunk_time = time.time() - t0
        print(f"  Trunk done in {trunk_time:.1f}s")

        print(f"\n  Running diffusion loop — {label.upper()}, {num_steps} steps...")
        t0 = time.time()
        coords_fp32 = run_diffusion_loop(
            model, cache, structure, num_steps=num_steps, seed=seed,
        )
        fp32_time = time.time() - t0
        ca_fp32 = extract_ca(coords_fp32[0, 0], structure)
        med_fp32, mean_fp32 = ca_spacing(ca_fp32)
        valid = is_valid_structure(ca_fp32)

        # Also run baseline for comparison (fully bf16)
        print(f"\n  Running diffusion loop — BASELINE (bf16), {num_steps} steps...")
        _cast_module_weights(model.diffusion_module, mx.bfloat16)
        t0 = time.time()
        coords_base = run_diffusion_loop(
            model, cache, structure, num_steps=num_steps, seed=seed,
        )
        base_time = time.time() - t0
        ca_base = extract_ca(coords_base[0, 0], structure)
        med_base, mean_base = ca_spacing(ca_base)

        print(f"\n  {'─' * 55}")
        print(f"  RESULTS ({label})")
        print(f"  {'─' * 55}")
        print(f"    {'':30} {'Baseline':>12} {'FP32 wts':>12}")
        print(f"    {'Median Cα spacing (Å)':30} {med_base:>12.2f} {med_fp32:>12.2f}")
        print(f"    {'Mean Cα spacing (Å)':30} {mean_base:>12.2f} {mean_fp32:>12.2f}")
        print(f"    {'Wall time':30} {base_time:>11.1f}s {fp32_time:>11.1f}s")
        print(f"    {'Structure valid':30} {'NO ❌':>12} {'YES ✅' if valid else 'NO ❌':>12}")
        print()

        if valid:
            print("  ✅ FP32 weights produce valid structures!")
            print("     → Weight quantization was the bottleneck, not accumulation order.")
            print("     → Keep diffusion weights in fp32 (+630 MB) as an immediate fix.")
        elif med_fp32 > med_base * 1.2:
            improvement = med_fp32 / max(med_base, 0.01)
            print(f"  ⚠️  Improved {improvement:.1f}× ({med_base:.2f} → {med_fp32:.2f}) but not valid.")
            print("     → Weight precision helps partially. May need to combine with other fixes.")
        else:
            print("  ❌ FP32 weights did NOT improve structures.")
            print("     → The error source is Metal backend divergence, not weight quantization.")
            print("     → Next: test hybrid inference (MLX trunk → MPS diffusion).")

    del model
    gc.collect()
    mx.clear_cache()
    return {
        "mode": mode, "valid": valid,
        "baseline_median": med_base, "fp32w_median": med_fp32,
        "baseline_time": base_time, "fp32w_time": fp32_time,
        "fp32_coords": coords_fp32,
    }


# ── Experiment 3: ODE convergence threshold ───────────────────────────────


def run_ode_threshold(weights_dir, num_steps=200, seed=42):
    """Inject gaussian noise into denoise outputs to find the ODE threshold.

    Uses the fp32-upcast model as the reference (must produce valid structures).
    For each noise level σ, adds N(0, σ) to every denoise output and checks
    whether the final structure is still valid.  The largest σ that preserves
    valid geometry is the ODE's convergence threshold — the target for per-step
    denoise error.
    """
    from chai_mlx import ChaiMLX

    print(f"\n{'=' * 70}")
    print("EXPERIMENT: ode_threshold")
    print("  Inject noise into denoise outputs to find convergence cliff")
    print(f"{'=' * 70}")

    # Load with fp32 weights for diffusion (the precision that matters)
    model = ChaiMLX.from_pretrained(weights_dir, strict=False, compute_dtype="float32")
    _cast_module_weights(model.input_embedder, mx.bfloat16)
    _cast_module_weights(model.trunk_module, mx.bfloat16)
    _cast_module_weights(model.confidence_head, mx.bfloat16)
    _cast_module_weights(model.ranker, mx.bfloat16)

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = featurize_1l2y(tmpdir)
        cache, structure, _ = prepare_pipeline(model, ctx)

        # Verify fp32-weights reference first
        print("  Verifying fp32-weights reference produces valid structure...")
        coords_ref = run_diffusion_loop(
            model, cache, structure, num_steps=num_steps, seed=seed,
            log_interval=50,
        )
        ca_ref = extract_ca(coords_ref[0, 0], structure)
        med_ref, _ = ca_spacing(ca_ref)
        ref_valid = is_valid_structure(ca_ref)
        print(f"  FP32-weights reference: median Cα = {med_ref:.2f} Å  {'✅' if ref_valid else '❌'}")

        if not ref_valid:
            print("\n  ❌ FP32-weights reference doesn't produce valid structures.")
            print("     Cannot determine ODE threshold without a working reference.")
            print("     Try running the MPS reference via chai-lab instead.")
            del model; gc.collect(); mx.clear_cache()
            return {"threshold": None, "ref_valid": False}

        noise_sigmas = [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
        results = []

        print(f"\n  {'σ_noise':>10}  {'Median Cα':>10}  {'Valid':>6}")
        print(f"  {'─' * 10}  {'─' * 10}  {'─' * 6}")

        for ns in noise_sigmas:
            def noise_hook(denoised, step_idx, _ns=ns):
                return denoised + _ns * mx.random.normal(denoised.shape).astype(mx.float32)

            coords = run_diffusion_loop(
                model, cache, structure, num_steps=num_steps, seed=seed,
                denoise_hook=noise_hook, log_interval=0,
            )
            ca = extract_ca(coords[0, 0], structure)
            med, _ = ca_spacing(ca)
            valid = is_valid_structure(ca)
            results.append({"sigma": ns, "median": med, "valid": valid})
            tag = "✅" if valid else "❌"
            print(f"  {ns:>10.1e}  {med:>9.2f} Å  {tag:>6}")

        valid_sigmas = [r["sigma"] for r in results if r["valid"]]
        threshold = max(valid_sigmas) if valid_sigmas else 0.0

        print(f"\n  ODE convergence threshold: σ ≤ {threshold:.1e}")
        if threshold > 0:
            print(f"  Current bf16 matmul per-step error: ~4.0 max, ~0.93 mean")
            print(f"  FP32 matmul per-step error: ~8e-7 (well below threshold)")
            if threshold >= 1e-2:
                print("  → Threshold is generous. Even modest precision improvement should work.")
            elif threshold >= 1e-4:
                print("  → Threshold is tight. Need Kahan-level precision (< 1e-5 per matmul).")
            else:
                print("  → Threshold is extremely tight. May need fp32 throughout or custom kernels.")
        else:
            print("  → No noise level preserved valid geometry. ODE is extremely fragile.")

    del model; gc.collect(); mx.clear_cache()
    return {"threshold": threshold, "ref_median": med_ref, "results": results}


# ── Experiment 4: Reference-injection bisection ──────────────────────────


def run_injection_bisection(weights_dir, num_steps=200, seed=42):
    """Run fp32 and bf16 diffusion in lockstep; at step k swap bf16 coords
    to the fp32 reference and continue on bf16.  Sweep k to find where the
    damage happens.

    If swapping early (k small) is enough → early steps are critical.
    If you must swap every step → uniform per-evaluation bias.
    """
    from chai_mlx import ChaiMLX

    print(f"\n{'=' * 70}")
    print("EXPERIMENT: injection_bisection")
    print("  Swap bf16 → fp32 reference coords at step k, continue on bf16")
    print(f"{'=' * 70}")

    # Load with fp32 weights for diffusion
    model = ChaiMLX.from_pretrained(weights_dir, strict=False, compute_dtype="float32")
    _cast_module_weights(model.input_embedder, mx.bfloat16)
    _cast_module_weights(model.trunk_module, mx.bfloat16)
    _cast_module_weights(model.confidence_head, mx.bfloat16)
    _cast_module_weights(model.ranker, mx.bfloat16)

    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = featurize_1l2y(tmpdir)
        cache, structure, _ = prepare_pipeline(model, ctx)

        # Collect fp32-weights reference trajectory
        print("  Collecting fp32-weights reference trajectory...")
        schedule = list(model.schedule(num_steps))
        mx.random.seed(seed)
        ref_coords = model.init_noise(1, 1, structure)
        mx.eval(ref_coords)
        ref_trajectory = [np.array(ref_coords.astype(mx.float32)).copy()]

        dm = model.diffusion_module
        for i, (s_curr, s_next, gamma) in enumerate(schedule):
            ref_coords = dm.diffusion_step(cache, ref_coords, s_curr, s_next, gamma)
            mx.eval(ref_coords)
            ref_trajectory.append(np.array(ref_coords.astype(mx.float32)).copy())

        ca_ref = extract_ca(ref_trajectory[-1][0, 0], structure)
        med_ref, _ = ca_spacing(ca_ref)
        ref_valid = is_valid_structure(ca_ref)
        print(f"  FP32-weights ref: median Cα = {med_ref:.2f} Å  {'✅' if ref_valid else '❌'}")

        # Switch diffusion to bf16 for the bf16 sweep runs
        _cast_module_weights(model.diffusion_module, mx.bfloat16)

        if not ref_valid:
            print("\n  ❌ FP32-weights reference isn't valid — bisection is meaningless.")
            del model; gc.collect(); mx.clear_cache()
            return {"ref_valid": False}

        # Sweep swap points
        swap_points = [0, 5, 10, 25, 50, 75, 100, 125, 150, 175, 190, 199]
        swap_points = [k for k in swap_points if k < len(schedule)]
        results = []

        print(f"\n  {'Swap at':>8}  {'Median Cα':>10}  {'Valid':>6}")
        print(f"  {'─' * 8}  {'─' * 10}  {'─' * 6}")

        for k in swap_points:
            coords = run_diffusion_loop(
                model, cache, structure, num_steps=num_steps, seed=seed,
                swap_coords_at={k}, swap_trajectory=ref_trajectory,
                log_interval=0,
            )
            ca = extract_ca(coords[0, 0], structure)
            med, _ = ca_spacing(ca)
            valid = is_valid_structure(ca)
            results.append({"swap_at": k, "median": med, "valid": valid})
            tag = "✅" if valid else "❌"
            print(f"  {k:>8}  {med:>9.2f} Å  {tag:>6}")

        # Also test: swap at EVERY step (equivalent to running fp32 throughout)
        coords_every = run_diffusion_loop(
            model, cache, structure, num_steps=num_steps, seed=seed,
            swap_coords_at=set(range(len(schedule))),
            swap_trajectory=ref_trajectory, log_interval=0,
        )
        ca_every = extract_ca(coords_every[0, 0], structure)
        med_every, _ = ca_spacing(ca_every)
        valid_every = is_valid_structure(ca_every)
        print(f"  {'every':>8}  {med_every:>9.2f} Å  {'✅' if valid_every else '❌':>6}")

        valid_swaps = [r for r in results if r["valid"]]
        if valid_swaps:
            earliest = min(r["swap_at"] for r in valid_swaps)
            latest_invalid = max(
                (r["swap_at"] for r in results if not r["valid"]),
                default=-1,
            )
            print(f"\n  Earliest swap producing valid structure: step {earliest}")
            if earliest == 0:
                print("  → Swapping once at the start is enough.")
                print("    The bf16 diffusion can self-correct after a clean start.")
            elif earliest < 50:
                print(f"  → Early steps (0–{earliest}) are where divergence becomes fatal.")
                print("    Focus precision improvements on high-sigma regime.")
            else:
                print(f"  → Must correct trajectory past step {earliest}.")
                print("    Divergence accumulates throughout — per-eval bias is the issue.")
        else:
            if valid_every:
                print("\n  Only full trajectory replacement works.")
                print("  → Every step contributes; need precision everywhere.")
            else:
                print("\n  Even full replacement doesn't help — trunk conditioning is the issue.")

    del model; gc.collect(); mx.clear_cache()
    return {"ref_median": med_ref, "ref_valid": ref_valid, "swap_results": results}


# ── Main ──────────────────────────────────────────────────────────────────


EXPERIMENTS = {
    "fp32w_all": lambda a: run_fp32_weights_experiment(a.weights_dir, "all", a.num_steps, a.seed),
    "fp32w_down": lambda a: run_fp32_weights_experiment(a.weights_dir, "down", a.num_steps, a.seed),
    "ode_thresh": lambda a: run_ode_threshold(a.weights_dir, a.num_steps, a.seed),
    "inject_bisect": lambda a: run_injection_bisection(a.weights_dir, a.num_steps, a.seed),
}


def main():
    parser = argparse.ArgumentParser(
        description="Precision-targeting experiments for Chai-1 MLX diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
experiment choices:
  fp32w_all      FP32 WEIGHTS in ALL diffusion linears (run this first)
  fp32w_down     FP32 WEIGHTS only in down-projection linears
  ode_thresh     Noise injection to find ODE convergence cliff
  inject_bisect  Reference-injection bisection (swap at step k)
  all            Run all experiments in order
""",
    )
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=list(EXPERIMENTS.keys()) + ["all"],
    )
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Precision experiments — {args.experiment}")
    print(f"  weights: {args.weights_dir}")
    print(f"  steps:   {args.num_steps}")
    print(f"  seed:    {args.seed}")

    if args.experiment == "all":
        all_results = {}
        for name in ["fp32w_all", "fp32w_down", "ode_thresh", "inject_bisect"]:
            try:
                result = EXPERIMENTS[name](args)
                all_results[name] = result
            except Exception as e:
                print(f"\n  ⚠️  {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_results[name] = {"error": str(e)}
            gc.collect()
            mx.clear_cache()

        print(f"\n{'=' * 70}")
        print("ALL EXPERIMENTS COMPLETE")
        print(f"{'=' * 70}")
        for name, r in all_results.items():
            if "error" in r:
                print(f"  {name}: FAILED — {r['error']}")
            elif "valid" in r:
                print(f"  {name}: {'✅' if r['valid'] else '❌'}")
            elif "threshold" in r:
                print(f"  {name}: threshold={r.get('threshold', 'N/A')}")
            else:
                print(f"  {name}: done")
    else:
        EXPERIMENTS[args.experiment](args)


if __name__ == "__main__":
    main()
