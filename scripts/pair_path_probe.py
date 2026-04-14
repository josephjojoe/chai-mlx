"""Probe pair path in MSA iteration 0 to isolate the source of pair_iter_0 error.

Feeds exact Torch pair_after_opm_0 into individual MLX pair operations
and reports where the error diverges.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from trunk_block_trace import _load_trunk_model
from chai_mlx.utils import resolve_dtype


def _compare(name: str, ref: np.ndarray, got: np.ndarray, mask: np.ndarray | None = None) -> None:
    diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
    if mask is not None and np.any(mask):
        diff_masked = diff[mask]
        mx_d, mn_d, p99 = float(diff_masked.max()), float(diff_masked.mean()), float(np.quantile(diff_masked, 0.99))
    else:
        mx_d, mn_d, p99 = float(diff.max()), float(diff.mean()), float(np.quantile(diff, 0.99))
    ref_range = float(np.abs(ref.astype(np.float32)).max())
    print(f"  {name:<40} max={mx_d:12.4e}  mean={mn_d:12.4e}  p99={p99:12.4e}  ref_range={ref_range:.3e}")


def _mx_np(x: mx.array) -> np.ndarray:
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    mx.eval(x)
    return np.array(x, copy=False)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--opm-trace", type=Path, required=True)
    parser.add_argument("--prepair-trace", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--compute-dtype", default="bfloat16")
    args = parser.parse_args()

    mx.set_default_device(mx.Device(mx.DeviceType.gpu, 0))

    model = _load_trunk_model(args.weights_dir, compute_dtype=args.compute_dtype, pairformer_block_limit=0)
    msa_module = model.trunk_module.msa_module
    dtype = mx.bfloat16 if args.compute_dtype == "bfloat16" else mx.float32

    with np.load(args.opm_trace, allow_pickle=False) as f:
        pair_after_opm_0_np = f["trunk.recycle_0.pre_pairformer.pair_after_opm_0"]

    with np.load(args.prepair_trace, allow_pickle=False) as f:
        pair_iter_0_ref = f["trunk.recycle_0.pre_pairformer.pair_iter_0"]

    with np.load(args.input_npz, allow_pickle=False) as f:
        token_pair_mask_np = f["structure_inputs.token_pair_mask"]

    pair_mask = mx.array(token_pair_mask_np)
    pair_mask_np = token_pair_mask_np.astype(bool)
    mask_3d = np.broadcast_to(pair_mask_np[..., None], pair_after_opm_0_np.shape)

    pair_after_opm_0 = mx.array(pair_after_opm_0_np).astype(dtype)
    mx.eval(pair_after_opm_0)

    print("\n=== Pair path decomposition: MSA iteration 0 ===\n")

    # Step 1: pair_transition[0]
    print("Step 1: pair_transition[0]")
    pair_transition_out = msa_module.pair_transition[0](pair_after_opm_0)
    mx.eval(pair_transition_out)
    pt_np = _mx_np(pair_transition_out)
    pt_rms = float(np.sqrt(np.mean(pt_np.astype(np.float32)**2)))
    pt_max = float(np.abs(pt_np.astype(np.float32)).max())
    print(f"  pair_transition_out rms={pt_rms:.4f}, max_abs={pt_max:.4f}")

    # Step 2: triangular_multiplication[0]
    print("\nStep 2: triangular_multiplication[0]")
    tri_mult_result = msa_module.triangular_multiplication[0](pair_after_opm_0, pair_mask=pair_mask)
    mx.eval(tri_mult_result)
    tm_np = _mx_np(tri_mult_result)
    tm_delta = tm_np.astype(np.float32) - pair_after_opm_0_np.astype(np.float32)
    tm_delta_rms = float(np.sqrt(np.mean(tm_delta**2)))
    tm_delta_max = float(np.abs(tm_delta).max())
    print(f"  tri_mult delta rms={tm_delta_rms:.4f}, max_abs={tm_delta_max:.4f}")

    # Step 3: Combine tri_mult + pair_transition (matches MSA module code)
    print("\nStep 3: Combined result = tri_mult_result + pair_transition_out")
    pair_after_tri_mult = tri_mult_result + pair_transition_out
    mx.eval(pair_after_tri_mult)
    patm_np = _mx_np(pair_after_tri_mult)
    patm_rms = float(np.sqrt(np.mean(patm_np.astype(np.float32)**2)))
    print(f"  combined rms={patm_rms:.4f}")

    # Step 4: triangular_attention[0]
    print("\nStep 4: triangular_attention[0]")
    pair_after_tri_attn = msa_module.triangular_attention[0](pair_after_tri_mult, pair_mask=pair_mask)
    mx.eval(pair_after_tri_attn)
    pata_np = _mx_np(pair_after_tri_attn)

    # Compare with ref
    print("\n=== Comparison with Torch pair_iter_0 reference ===\n")
    _compare("MLX pair_iter_0 vs Torch pair_iter_0", pair_iter_0_ref, pata_np, mask_3d)

    # Now let's also check individual steps for sanity
    # Check tri_mult result magnitude
    print("\n=== Sub-step magnitudes ===")
    print(f"  pair_after_opm_0 rms: {float(np.sqrt(np.mean(pair_after_opm_0_np.astype(np.float32)**2))):.4f}")
    print(f"  tri_mult_result rms:  {float(np.sqrt(np.mean(tm_np.astype(np.float32)**2))):.4f}")
    print(f"  pair_transition rms:  {pt_rms:.4f}")
    print(f"  combined rms:         {patm_rms:.4f}")
    print(f"  final (after tri_attn) rms: {float(np.sqrt(np.mean(pata_np.astype(np.float32)**2))):.4f}")
    print(f"  ref pair_iter_0 rms:  {float(np.sqrt(np.mean(pair_iter_0_ref.astype(np.float32)**2))):.4f}")

    # Check for extreme values
    print(f"\n  final max_abs: {float(np.abs(pata_np.astype(np.float32)).max()):.4f}")
    print(f"  ref max_abs:   {float(np.abs(pair_iter_0_ref.astype(np.float32)).max()):.4f}")

    # Check if there are NaN/Inf
    print(f"\n  final has_nan: {np.any(np.isnan(pata_np))}")
    print(f"  final has_inf: {np.any(np.isinf(pata_np))}")


if __name__ == "__main__":
    main()
