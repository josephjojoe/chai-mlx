"""Proof of life: does fp64-accurate matmul fix the diffusion ODE?

Replaces every nn.Linear in the diffusion module with a version that
computes matmul in fp64 on CPU via NumPy.  This is 50-500x slower than
Steel but serves as an oracle: if the ODE converges with this, then a
compensated-summation Metal kernel is a viable fix.

Two modes:

  --mode single   Run one denoise step, compare per-step error vs MPS reference.
  --mode ode      Run the full 200-step ODE, measure median Cα spacing.

Usage::

    python scripts/proof_of_life.py \
        --weights-dir weights/ \
        --input-npz /path/to/input_context.npz \
        --reference-npz /path/to/reference_tensors.npz \
        --mode single

    python scripts/proof_of_life.py \
        --weights-dir weights/ \
        --input-npz /path/to/input_context.npz \
        --reference-npz /path/to/reference_tensors.npz \
        --mode ode
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from chai_mlx import ChaiMLX
from chai_mlx.utils import resolve_dtype
from layer_parity import _npz_dict, load_feature_context
from stage_isolation_parity import reconstruct_trunk_outputs


# ---------------------------------------------------------------------------
# Reference matmul: fp64 NumPy oracle
# ---------------------------------------------------------------------------

def reference_matmul(a: mx.array, b: mx.array) -> mx.array:
    out_dtype = a.dtype
    a64 = np.asarray(a.astype(mx.float32)).astype(np.float64)
    b64 = np.asarray(b.astype(mx.float32)).astype(np.float64)
    c64 = a64 @ b64
    return mx.array(c64.astype(np.float32)).astype(out_dtype)


# ---------------------------------------------------------------------------
# Reference SDPA: fp64 NumPy oracle for scaled_dot_product_attention
# ---------------------------------------------------------------------------

_sdpa_call_count = [0]

def reference_sdpa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float = 1.0,
    mask: mx.array | None = None,
) -> mx.array:
    _sdpa_call_count[0] += 1
    out_dtype = q.dtype
    mx.eval(q, k, v)
    if mask is not None:
        mx.eval(mask)

    q64 = np.asarray(q.astype(mx.float32)).astype(np.float64)
    k64 = np.asarray(k.astype(mx.float32)).astype(np.float64)
    v64 = np.asarray(v.astype(mx.float32)).astype(np.float64)

    scores = np.einsum("...qd,...kd->...qk", q64, k64) * scale
    if mask is not None:
        m64 = np.asarray(mask.astype(mx.float32)).astype(np.float64)
        scores = scores + m64

    # Stable softmax in fp64
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

    out = np.einsum("...qk,...kd->...qd", attn, v64)
    return mx.array(out.astype(np.float32)).astype(out_dtype)


def install_reference_sdpa():
    """Monkeypatch mx.fast.scaled_dot_product_attention with fp64 reference."""
    _sdpa_call_count[0] = 0
    mx.fast.scaled_dot_product_attention = reference_sdpa
    return _sdpa_call_count


def uninstall_reference_sdpa(original_sdpa):
    mx.fast.scaled_dot_product_attention = original_sdpa


# ---------------------------------------------------------------------------
# ReferenceLinear: drop-in nn.Linear replacement using fp64 matmul
# ---------------------------------------------------------------------------

class ReferenceLinear(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.weight = base.weight
        self.bias = getattr(base, "bias", None)

    def __call__(self, x: mx.array) -> mx.array:
        mx.eval(x)
        out = reference_matmul(x, self.weight.T)
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# Module-tree surgery: replace nn.Linear → ReferenceLinear
# ---------------------------------------------------------------------------

def _swap_linears(module: nn.Module, path: str = "", count: list[int] | None = None) -> int:
    if count is None:
        count = [0]
    for name, child in module.children().items():
        if isinstance(child, nn.Linear) and not isinstance(child, ReferenceLinear):
            module[name] = ReferenceLinear(child)
            count[0] += 1
        elif isinstance(child, nn.Module):
            _swap_linears(child, path=f"{path}.{name}", count=count)
        elif isinstance(child, list):
            for i, item in enumerate(child):
                if isinstance(item, nn.Linear) and not isinstance(item, ReferenceLinear):
                    child[i] = ReferenceLinear(item)
                    count[0] += 1
                elif isinstance(item, nn.Module):
                    _swap_linears(item, path=f"{path}.{name}[{i}]", count=count)
    return count[0]


# ---------------------------------------------------------------------------
# Step 1: sanity check reference_matmul vs Steel
# ---------------------------------------------------------------------------

def sanity_check():
    print("=" * 70)
    print("STEP 1: SANITY CHECK — reference_matmul vs Steel")
    print("=" * 70)

    shapes = [(384, 384), (384, 768), (768, 384), (256, 256)]
    mx.random.seed(42)

    for M, K in shapes:
        N = K
        a = mx.random.normal((1, M, K)).astype(mx.bfloat16)
        b = mx.random.normal((1, K, N)).astype(mx.bfloat16)
        mx.eval(a, b)

        steel = a @ b
        mx.eval(steel)
        ref = reference_matmul(a, b)
        mx.eval(ref)

        diff = np.abs(
            np.asarray(steel.astype(mx.float32)) -
            np.asarray(ref.astype(mx.float32))
        )
        max_err = float(diff.max())
        mean_err = float(diff.mean())
        print(f"  [{M}×{K}] @ [{K}×{N}]:  max={max_err:.4e}  mean={mean_err:.4e}")

    print()


# ---------------------------------------------------------------------------
# Step 3: single denoise step
# ---------------------------------------------------------------------------

def single_denoise_step(
    model: ChaiMLX,
    ref: dict[str, np.ndarray],
    ctx_extras: tuple,
    *,
    recycles: int,
    swap_cache_too: bool,
    swap_sdpa: bool,
) -> float:
    ctx, extras = ctx_extras
    structure = ctx.structure_inputs
    dtype = resolve_dtype(model.cfg)

    coords = extras.get("coords")
    sigma = extras.get("sigma")
    if coords is None or sigma is None:
        raise ValueError("input NPZ must contain coords and sigma")

    # Use reference trunk outputs so we isolate the denoiser
    print("  Reconstructing reference trunk outputs...")
    ref_trunk = reconstruct_trunk_outputs(ref, structure, dtype=dtype)

    # Swap linears in diffusion module
    dm = model.diffusion_module
    if swap_cache_too:
        n_swapped = _swap_linears(dm)
        target = "entire diffusion module (incl. cache prep)"
    else:
        n_swapped = _swap_linears(dm.diffusion_transformer)
        n_swapped += _swap_linears(dm.atom_attention_encoder)
        n_swapped += _swap_linears(dm.atom_attention_decoder)
        if isinstance(dm.structure_cond_to_token_structure_proj, nn.Linear):
            dm["structure_cond_to_token_structure_proj"] = ReferenceLinear(
                dm.structure_cond_to_token_structure_proj
            )
            n_swapped += 1
        target = "diffusion transformer + atom attention + structure proj"
    print(f"  Swapped {n_swapped} nn.Linear → ReferenceLinear in {target}")

    original_sdpa = mx.fast.scaled_dot_product_attention
    if swap_sdpa:
        sdpa_count = install_reference_sdpa()
        print("  Installed reference SDPA (fp64 NumPy)")

    print("  Building DiffusionCache from reference trunk outputs...")
    cache = model.prepare_diffusion_cache(ref_trunk)
    mx.eval(cache.s_static, cache.z_cond, cache.blocked_pair_base,
            cache.atom_cond, cache.atom_single_cond, *cache.pair_biases)

    print("  Running single denoise step with reference matmul...")
    t0 = time.time()
    denoised = dm.denoise(cache, coords, sigma)
    mx.eval(denoised)
    elapsed = time.time() - t0
    print(f"  Denoise took {elapsed:.1f}s")

    if swap_sdpa:
        print(f"  Reference SDPA called {sdpa_count[0]} times")
        uninstall_reference_sdpa(original_sdpa)

    denoised_np = np.asarray(denoised.astype(mx.float32))

    ref_key = "denoise.output"
    if ref_key in ref:
        ref_out = ref[ref_key]
        if ref_out.shape != denoised_np.shape:
            print(f"  [WARN] shape mismatch: ref={ref_out.shape} mlx={denoised_np.shape}")
            min_s = min(ref_out.shape[1], denoised_np.shape[1])
            ref_out = ref_out[:, :min_s]
            denoised_np = denoised_np[:, :min_s]
        diff = np.abs(ref_out.astype(np.float32) - denoised_np.astype(np.float32))
        max_err = float(diff.max())
        mean_err = float(diff.mean())
        rms = float(np.sqrt(np.mean(diff ** 2)))
        print(f"\n  Per-step denoise error vs MPS reference:")
        print(f"    max:  {max_err:.4e}")
        print(f"    mean: {mean_err:.4e}")
        print(f"    rms:  {rms:.4e}")

        print(f"\n  BASELINE (stock MLX, from docs): max ~4.0")
        if max_err < 0.05:
            print(f"  → EXCELLENT: >80× improvement. Matmul accuracy is clearly sufficient.")
        elif max_err < 0.5:
            print(f"  → GOOD: ~{4.0/max_err:.0f}× improvement. Promising.")
        elif max_err < 2.0:
            print(f"  → PARTIAL: only ~{4.0/max_err:.1f}× improvement. Other ops may contribute.")
        else:
            print(f"  → MINIMAL: still {max_err:.1f}. Matmul alone is NOT sufficient.")
        return max_err
    else:
        print("  [SKIP] No denoise.output in reference NPZ for comparison")
        print(f"  Output stats: rms={float(np.sqrt(np.mean(denoised_np**2))):.4f}")
        return float("nan")


# ---------------------------------------------------------------------------
# Step 4: full 200-step ODE
# ---------------------------------------------------------------------------

def full_ode(
    model: ChaiMLX,
    ref: dict[str, np.ndarray],
    ctx_extras: tuple,
    *,
    recycles: int,
    swap_cache_too: bool,
    use_mlx_trunk: bool,
) -> float:
    ctx, extras = ctx_extras
    structure = ctx.structure_inputs
    dtype = resolve_dtype(model.cfg)

    if use_mlx_trunk:
        print("  Running MLX trunk (stock Steel)...")
        emb = model.embed_inputs(ctx)
        trunk_out = model.trunk(emb, recycles=recycles)
        mx.eval(trunk_out.single_trunk, trunk_out.pair_trunk)
    else:
        print("  Reconstructing reference trunk outputs...")
        trunk_out = reconstruct_trunk_outputs(ref, structure, dtype=dtype)

    dm = model.diffusion_module
    if swap_cache_too:
        n_swapped = _swap_linears(dm)
        target = "entire diffusion module"
    else:
        n_swapped = _swap_linears(dm.diffusion_transformer)
        n_swapped += _swap_linears(dm.atom_attention_encoder)
        n_swapped += _swap_linears(dm.atom_attention_decoder)
        if isinstance(dm.structure_cond_to_token_structure_proj, nn.Linear):
            dm["structure_cond_to_token_structure_proj"] = ReferenceLinear(
                dm.structure_cond_to_token_structure_proj
            )
            n_swapped += 1
        target = "diffusion transformer + atom attention + structure proj"
    print(f"  Swapped {n_swapped} nn.Linear → ReferenceLinear in {target}")

    print("  Building DiffusionCache...")
    cache = model.prepare_diffusion_cache(trunk_out)
    mx.eval(cache.s_static, cache.z_cond, cache.blocked_pair_base,
            cache.atom_cond, cache.atom_single_cond, *cache.pair_biases)

    print("  Initializing noise...")
    mx.random.seed(42)
    coords = dm.init_noise(1, 1, structure)
    mx.eval(coords)

    print("  Running 200-step ODE with reference matmul...")
    print("  (this will be slow — each step goes through CPU fp64)")
    t0 = time.time()
    schedule = list(dm.schedule())
    for step_idx, (sigma_curr, sigma_next, gamma) in enumerate(schedule):
        step_t = time.time()
        coords = dm.diffusion_step(cache, coords, sigma_curr, sigma_next, gamma)
        mx.eval(coords)
        step_elapsed = time.time() - step_t
        if step_idx % 20 == 0 or step_idx == len(schedule) - 1:
            sigma_val = float(sigma_curr.item()) if hasattr(sigma_curr, 'item') else float(sigma_curr)
            print(f"    step {step_idx:3d}/{len(schedule)} "
                  f"σ={sigma_val:.4f} "
                  f"({step_elapsed:.1f}s/step)")

    total = time.time() - t0
    print(f"  ODE complete in {total:.0f}s ({total/60:.1f}min)")

    coords_np = np.asarray(coords.astype(mx.float32))
    atom_mask = np.asarray(structure.atom_exists_mask.astype(mx.float32))[0]
    valid_atoms = atom_mask > 0.5

    if valid_atoms.sum() > 1:
        valid_coords = coords_np[0, 0, valid_atoms]
        ca_dists = np.sqrt(np.sum(np.diff(valid_coords, axis=0) ** 2, axis=-1))
        median_ca = float(np.median(ca_dists))
    else:
        median_ca = float("nan")

    print(f"\n  Median atom spacing: {median_ca:.2f} Å")
    print(f"  BASELINE (stock MLX, from docs): ~25 Å (broken)")
    print(f"  TARGET: ~3.8 Å (MPS reference) or ~1.5 Å (all-atom median)")

    trunk_label = "MLX trunk (stock Steel)" if use_mlx_trunk else "MPS reference trunk"
    if median_ca < 5.0:
        print(f"\n  → OUTCOME A: Valid structure! ({trunk_label})")
        print(f"    Matmul accuracy is sufficient. Proceed to Phase 2 (kernel).")
    elif median_ca < 15.0:
        print(f"\n  → OUTCOME B: Improved but not valid ({trunk_label})")
        print(f"    Matmul is part of the problem. Check SDPA, softmax, other reductions.")
    else:
        print(f"\n  → OUTCOME C: No meaningful improvement ({trunk_label})")
        print(f"    Matmul alone is NOT sufficient. Look elsewhere.")

    return median_ca


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Proof of life: fp64 matmul in denoiser")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--mode", choices=["sanity", "single", "ode", "all"], default="all")
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--swap-cache-too", action="store_true",
                        help="Also swap linears in DiffusionConditioning (cache preparation)")
    parser.add_argument("--swap-sdpa", action="store_true",
                        help="Also replace mx.fast.scaled_dot_product_attention with fp64 reference")
    parser.add_argument("--use-mlx-trunk", action="store_true",
                        help="Use MLX trunk outputs instead of MPS reference "
                             "(tests end-to-end including trunk conditioning)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_sanity = args.mode in ("sanity", "all")
    run_single = args.mode in ("single", "all")
    run_ode = args.mode in ("ode", "all")

    if run_sanity:
        sanity_check()

    if run_single or run_ode:
        print("Loading model...")
        model = ChaiMLX.from_pretrained(args.weights_dir, strict=False)
        ctx, extras = load_feature_context(args.input_npz)
        ref = _npz_dict(args.reference_npz)

    if run_single:
        print("\n" + "=" * 70)
        print("STEP 3: SINGLE DENOISE STEP — per-step error with reference matmul")
        print("=" * 70)
        max_err = single_denoise_step(
            model, ref, (ctx, extras),
            recycles=args.recycles,
            swap_cache_too=args.swap_cache_too,
            swap_sdpa=args.swap_sdpa,
        )

        # Reload model for ODE since we mutated it
        if run_ode:
            print("\n  Reloading model for ODE run...")
            model = ChaiMLX.from_pretrained(args.weights_dir, strict=False)

    if run_ode:
        print("\n" + "=" * 70)
        print("STEP 4: FULL 200-STEP ODE — median Cα spacing")
        print("=" * 70)
        median_ca = full_ode(
            model, ref, (ctx, extras),
            recycles=args.recycles,
            swap_cache_too=args.swap_cache_too,
            use_mlx_trunk=args.use_mlx_trunk,
        )


if __name__ == "__main__":
    main()
