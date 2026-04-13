"""Measure how trunk divergence propagates into diffusion conditioning tensors.

The diffusion module consumes trunk outputs as continuous conditioning via
``prepare_cache``, which produces ``s_static``, ``z_cond``, ``pair_biases``,
``blocked_pair_base``, ``atom_cond``, and ``atom_single_cond``.

This script builds DiffusionCache from both MLX trunk outputs and MPS
reference trunk outputs, then compares the resulting cache tensors.  This
answers whether the trunk's chaotic divergence (~1069 max on pair) corrupts
the specific tensors the diffusion ODE consumes, or whether the cache
projection acts as a bottleneck that attenuates the divergence.

Usage::

    python scripts/cache_conditioning_divergence.py \
        --weights-dir weights/ \
        --input-npz /path/to/input_context.npz \
        --reference-npz /path/to/reference_tensors.npz \
        [--recycles 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from chai_mlx import ChaiMLX
from chai_mlx.utils import resolve_dtype
from layer_parity import _npz_dict, load_feature_context
from stage_isolation_parity import reconstruct_trunk_outputs


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    diff = (a.astype(np.float64) - b.astype(np.float64)).ravel()
    return float(np.sqrt(np.dot(diff, diff)))


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.sqrt(np.mean(diff ** 2)))


def _stats(name: str, ref: np.ndarray, mlx: np.ndarray) -> dict:
    diff = np.abs(ref.astype(np.float32) - mlx.astype(np.float32))
    l2 = _l2(ref, mlx)
    rms = _rms(ref, mlx)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    ref_rms = float(np.sqrt(np.mean(ref.astype(np.float64) ** 2)))
    rel_rms = rms / max(ref_rms, 1e-15)
    numel = int(np.prod(ref.shape))

    print(f"  {name}")
    print(f"    shape:    {ref.shape}")
    print(f"    L2:       {l2:.6e}")
    print(f"    RMS:      {rms:.6e}  (ref RMS: {ref_rms:.4e}, relative: {rel_rms:.4f})")
    print(f"    max:      {max_err:.6e}")
    print(f"    mean:     {mean_err:.6e}")
    print(f"    numel:    {numel}")

    return {
        "name": name,
        "shape": list(ref.shape),
        "l2": l2,
        "rms": rms,
        "max": max_err,
        "mean": mean_err,
        "ref_rms": ref_rms,
        "rel_rms": rel_rms,
        "numel": numel,
    }


def _mx_to_np(x: mx.array) -> np.ndarray:
    v = x.astype(mx.float32) if x.dtype == mx.bfloat16 else x
    mx.eval(v)
    return np.array(v, copy=False)


def measure_cache_divergence(
    model: ChaiMLX,
    ref: dict[str, np.ndarray],
    ctx_extras: tuple,
    *,
    recycles: int,
) -> list[dict]:
    ctx, extras = ctx_extras
    structure = ctx.structure_inputs
    dtype = resolve_dtype(model.cfg)

    # --- MLX trunk → MLX cache ---
    print("Running MLX embed + trunk...")
    emb = model.embed_inputs(ctx)
    trunk_out_mlx = model.trunk(emb, recycles=recycles)
    mx.eval(trunk_out_mlx.single_trunk, trunk_out_mlx.pair_trunk)

    print("Building cache from MLX trunk outputs...")
    cache_mlx = model.prepare_diffusion_cache(trunk_out_mlx)
    mx.eval(cache_mlx.s_static, cache_mlx.z_cond, cache_mlx.blocked_pair_base,
            cache_mlx.atom_cond, cache_mlx.atom_single_cond, *cache_mlx.pair_biases)

    # --- Reference trunk → reference cache ---
    print("Reconstructing reference trunk outputs from NPZ...")
    ref_trunk = reconstruct_trunk_outputs(ref, structure, dtype=dtype)

    print("Building cache from reference trunk outputs...")
    cache_ref = model.prepare_diffusion_cache(ref_trunk)
    mx.eval(cache_ref.s_static, cache_ref.z_cond, cache_ref.blocked_pair_base,
            cache_ref.atom_cond, cache_ref.atom_single_cond, *cache_ref.pair_biases)

    # --- Also compare the raw trunk outputs that feed into cache ---
    print("\n" + "=" * 70)
    print("TRUNK OUTPUT DIVERGENCE (raw inputs to prepare_cache)")
    print("=" * 70)
    trunk_results = []
    for key in ("single_trunk", "pair_trunk", "single_structure", "pair_structure"):
        ref_key = f"trunk.outputs.{key}"
        if ref_key in ref:
            got = _mx_to_np(getattr(trunk_out_mlx, key))
            trunk_results.append(_stats(key, ref[ref_key], got))

    # --- Compare cache tensors ---
    print("\n" + "=" * 70)
    print("DIFFUSION CACHE CONDITIONING DIVERGENCE")
    print("=" * 70)
    print("(same model weights, different trunk outputs)")
    print()

    cache_results = []
    for name in ("s_static", "z_cond", "blocked_pair_base", "atom_cond", "atom_single_cond"):
        ref_arr = _mx_to_np(getattr(cache_ref, name))
        mlx_arr = _mx_to_np(getattr(cache_mlx, name))
        cache_results.append(_stats(f"cache.{name}", ref_arr, mlx_arr))

    for i, (ref_bias, mlx_bias) in enumerate(zip(cache_ref.pair_biases, cache_mlx.pair_biases)):
        cache_results.append(_stats(
            f"cache.pair_biases.{i}",
            _mx_to_np(ref_bias),
            _mx_to_np(mlx_bias),
        ))

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n  {'Tensor':<30} {'L2':>12} {'RMS':>12} {'max':>12} {'rel_rms':>10}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 10}")

    for r in trunk_results:
        print(f"  {r['name']:<30} {r['l2']:>12.4e} {r['rms']:>12.4e} "
              f"{r['max']:>12.4e} {r['rel_rms']:>10.4f}")

    print()
    for r in cache_results:
        print(f"  {r['name']:<30} {r['l2']:>12.4e} {r['rms']:>12.4e} "
              f"{r['max']:>12.4e} {r['rel_rms']:>10.4f}")

    all_results = trunk_results + cache_results

    # Key question: does the projection attenuate or preserve the divergence?
    trunk_pair_rel = next((r["rel_rms"] for r in trunk_results if r["name"] == "pair_trunk"), None)
    z_cond_rel = next((r["rel_rms"] for r in cache_results if r["name"] == "cache.z_cond"), None)
    s_static_rel = next((r["rel_rms"] for r in cache_results if r["name"] == "cache.s_static"), None)

    if trunk_pair_rel is not None and z_cond_rel is not None:
        ratio = z_cond_rel / max(trunk_pair_rel, 1e-15)
        print(f"\n  z_cond relative RMS / pair_trunk relative RMS = {ratio:.4f}")
        if ratio < 0.5:
            print("  → Cache projection ATTENUATES trunk divergence on z_cond")
        elif ratio < 2.0:
            print("  → Cache projection PRESERVES trunk divergence on z_cond (roughly 1:1)")
        else:
            print("  → Cache projection AMPLIFIES trunk divergence on z_cond")

    if s_static_rel is not None:
        print(f"  s_static relative RMS = {s_static_rel:.4f}")
        if s_static_rel < 0.01:
            print("  → s_static is well-preserved (hybrid approach may work for single conditioning)")
        elif s_static_rel < 0.1:
            print("  → s_static has moderate divergence")
        else:
            print("  → s_static has large divergence (hybrid approach likely compromised)")

    bias_rels = [r["rel_rms"] for r in cache_results if r["name"].startswith("cache.pair_biases")]
    if bias_rels:
        print(f"\n  pair_biases relative RMS: min={min(bias_rels):.4f} "
              f"max={max(bias_rels):.4f} mean={np.mean(bias_rels):.4f}")

    return all_results


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Measure trunk divergence propagation into diffusion cache tensors",
    )
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--recycles", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False)
    ctx, extras = load_feature_context(args.input_npz)
    ref = _npz_dict(args.reference_npz)

    measure_cache_divergence(
        model,
        ref,
        (ctx, extras),
        recycles=args.recycles,
    )


if __name__ == "__main__":
    main()
