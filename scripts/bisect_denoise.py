"""Bisect the denoise error by block and test zero-input diagnostic.

Two experiments:
1. Block bisection: run MLX denoise step-by-step, capture activations after
   each of the 16 transformer blocks, and compare the final output against
   TorchScript reference. Also runs TorchScript with hooks to try to get
   per-block reference data.
2. Zero-input: run both MLX and TorchScript with zero noisy coords.  If the
   outputs still disagree, the bug is in the conditioning path (not atom
   processing).

Usage::

    python scripts/bisect_denoise.py --weights-dir weights/ \
        --input-npz /tmp/chai_mlx_input.npz \
        --reference-npz /tmp/chai_mlx_reference.npz

This script is intended to mirror the actual top-level MLX denoise path as
closely as practical. In particular, the token-conditioning projection follows
the TorchScript wrapper and projects the sigma-conditioned single representation
(`s_cond`), not `trunk.single_structure`.
"""

from __future__ import annotations

import argparse
import re
import sys
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


def _mx_to_np(x: mx.array) -> np.ndarray:
    v = x.astype(mx.float32) if x.dtype == mx.bfloat16 else x
    mx.eval(v)
    return np.array(v, copy=False)


def _err_stats(ref: np.ndarray, got: np.ndarray) -> dict:
    diff = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    return {
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "rms": float(np.sqrt(np.mean(diff ** 2))),
    }


# ---------------------------------------------------------------------------
# MLX denoise with per-stage capture
# ---------------------------------------------------------------------------

def mlx_denoise_capture(
    model: ChaiMLX,
    cache,
    coords: mx.array,
    sigma: mx.array,
) -> dict[str, np.ndarray]:
    """Run MLX denoise, capturing coarse top-level intermediates.

    The capture intentionally follows ``DiffusionModule.denoise`` rather than a
    synthetic decomposition so this is a good first-line sanity check when the
    question is "what did the real runtime do?" rather than "which exact leaf
    op diverged?".
    """
    cap: dict[str, np.ndarray] = {}
    dm = model.diffusion_module
    trunk = cache.trunk_outputs
    structure = cache.structure_inputs

    sigma_f = sigma.astype(mx.float32)
    sigma_sq = sigma_f * sigma_f
    sigma_data_sq = dm.cfg.diffusion.sigma_data ** 2
    c_in = (sigma_sq + sigma_data_sq) ** -0.5
    c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
    c_out = sigma_f * dm.cfg.diffusion.sigma_data / mx.sqrt(sigma_sq + sigma_data_sq)

    cap["c_in"] = _mx_to_np(c_in)
    cap["c_skip"] = _mx_to_np(c_skip)
    cap["c_out"] = _mx_to_np(c_out)

    num_samples = coords.shape[1]
    scaled_coords = coords * c_in[:, :, None, None]
    cap["scaled_coords"] = _mx_to_np(scaled_coords)

    s_cond = dm.diffusion_conditioning.with_sigma(cache.s_static, sigma_f)
    mx.eval(s_cond)
    cap["s_cond"] = _mx_to_np(s_cond)

    x = dm.structure_cond_to_token_structure_proj(s_cond)
    mx.eval(x)
    cap["token_structure_proj"] = _mx_to_np(x)

    enc_tokens, atom_repr, encoder_pair = dm.atom_attention_encoder(
        cache.atom_cond, cache.atom_single_cond, cache.blocked_pair_base,
        structure.atom_token_index, structure.atom_exists_mask, scaled_coords,
        structure.atom_kv_indices, structure.block_atom_pair_mask,
        num_tokens=trunk.single_initial.shape[1], num_samples=num_samples,
    )
    mx.eval(enc_tokens, atom_repr)
    cap["enc_tokens"] = _mx_to_np(enc_tokens)
    cap["atom_repr_encoder"] = _mx_to_np(atom_repr)

    x = x + enc_tokens
    mx.eval(x)
    cap["pre_transformer"] = _mx_to_np(x)

    b, ds, n, d = x.shape
    out = x
    for i, (block, pair_bias) in enumerate(
        zip(dm.diffusion_transformer.blocks, cache.pair_biases)
    ):
        bias = mx.broadcast_to(pair_bias[:, None, :, :, :], (b, ds, *pair_bias.shape[1:]))
        out = block(
            out.reshape(b * ds, n, d),
            s_cond.reshape(b * ds, n, s_cond.shape[-1]),
            bias.reshape(b * ds, *pair_bias.shape[1:]),
        ).reshape(b, ds, n, d)
        mx.eval(out)
        cap[f"block_{i:02d}"] = _mx_to_np(out)

    x_post = dm.post_attn_layernorm(out)
    mx.eval(x_post)
    cap["post_ln"] = _mx_to_np(x_post)

    decoder_cond = dm.post_atom_cond_layernorm(
        mx.broadcast_to(
            cache.atom_single_cond[:, None, :, :],
            (coords.shape[0], num_samples, *cache.atom_single_cond.shape[1:]),
        )
    )
    pos_updates = dm.atom_attention_decoder(
        x_post, atom_repr, decoder_cond, encoder_pair,
        structure.atom_token_index, structure.atom_exists_mask,
        structure.atom_kv_indices, structure.block_atom_pair_mask,
    )
    mx.eval(pos_updates)
    cap["pos_updates"] = _mx_to_np(pos_updates)

    output = c_skip[:, :, None, None] * coords + c_out[:, :, None, None] * pos_updates
    mx.eval(output)
    cap["output"] = _mx_to_np(output)

    return cap


# ---------------------------------------------------------------------------
# TorchScript denoise (black box + hooks attempt)
# ---------------------------------------------------------------------------

def _build_static_inputs(ref, structure_np, device):
    """Build static diffusion inputs dict for TorchScript."""
    import torch

    p = "trunk.outputs"
    return dict(
        token_single_initial_repr=torch.from_numpy(ref.get(f"{p}.single_structure", ref.get("embedding.outputs.single_structure"))).float().to(device),
        token_pair_initial_repr=torch.from_numpy(ref.get(f"{p}.pair_structure", ref.get("embedding.outputs.pair_structure"))).float().to(device),
        token_single_trunk_repr=torch.from_numpy(ref[f"{p}.single_trunk"]).float().to(device),
        token_pair_trunk_repr=torch.from_numpy(ref[f"{p}.pair_trunk"]).float().to(device),
        atom_single_input_feats=torch.from_numpy(ref[f"{p}.atom_single_structure_input"]).float().to(device),
        atom_block_pair_input_feats=torch.from_numpy(ref[f"{p}.atom_pair_structure_input"]).float().to(device),
        atom_single_mask=torch.from_numpy(structure_np["atom_exists_mask"]).bool().to(device),
        atom_block_pair_mask=torch.from_numpy(structure_np["block_atom_pair_mask"]).bool().to(device),
        token_single_mask=torch.from_numpy(structure_np["token_exists_mask"]).bool().to(device),
        block_indices_h=torch.from_numpy(structure_np["atom_q_indices"]).squeeze(0).to(device),
        block_indices_w=torch.from_numpy(structure_np["atom_kv_indices"]).squeeze(0).to(device),
        atom_token_indices=torch.from_numpy(structure_np["atom_token_index"]).long().to(device),
    )


def torchscript_denoise(
    coords_np: np.ndarray,
    sigma_np: np.ndarray,
    ref: dict[str, np.ndarray],
    structure_np: dict,
    *,
    try_hooks: bool = True,
    device: str = "mps",
) -> dict[str, np.ndarray]:
    """Run TorchScript diffusion module, optionally capturing per-block hooks."""
    import torch

    DEVICE = torch.device(device)
    LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
    if LOCAL_CHAI_LAB.exists():
        sys.path.insert(0, str(LOCAL_CHAI_LAB))
    from chai_lab.chai1 import _component_moved_to

    def to_np(t):
        return t.detach().cpu().float().numpy()

    static_inputs = _build_static_inputs(ref, structure_np, DEVICE)
    coords_t = torch.from_numpy(coords_np).float().to(DEVICE)
    sigma_t = torch.from_numpy(sigma_np).float().to(DEVICE)
    crop_size = int(static_inputs["token_single_mask"].shape[-1])

    cap: dict[str, np.ndarray] = {}
    hooks = []
    hook_data: dict[str, np.ndarray] = {}

    with torch.no_grad():
        with _component_moved_to("diffusion_module.pt", device=DEVICE) as diffusion_module:
            jit = diffusion_module.jit_module

            if try_hooks:
                block_pattern = re.compile(r"diffusion_transformer\.blocks\.(\d+)$")
                for name, mod in jit.named_modules():
                    m = block_pattern.search(name)
                    if m:
                        idx = int(m.group(1))
                        def make_hook(block_idx, mod_name):
                            def hook_fn(module, inp, output):
                                if isinstance(output, torch.Tensor):
                                    hook_data[f"block_{block_idx:02d}"] = to_np(output)
                                elif isinstance(output, tuple) and len(output) > 0:
                                    hook_data[f"block_{block_idx:02d}"] = to_np(output[0])
                            return hook_fn
                        h = mod.register_forward_hook(make_hook(idx, name))
                        hooks.append(h)
                print(f"  Registered {len(hooks)} TorchScript block hooks")

            output = diffusion_module.forward(
                atom_noised_coords=coords_t,
                noise_sigma=sigma_t,
                crop_size=crop_size,
                **static_inputs,
            )

            for h in hooks:
                h.remove()

    out_np = to_np(output)
    if out_np.ndim == 3:
        out_np = out_np[:, None, :, :]
    cap["output"] = out_np

    if hook_data:
        print(f"  Captured {len(hook_data)} TorchScript block outputs via hooks")
        cap.update(hook_data)
    else:
        print("  No TorchScript block hooks fired (typical for JIT modules)")

    return cap


def _get_structure_np(structure) -> dict:
    """Extract structure inputs as numpy arrays."""
    result = {}
    for name in ["atom_exists_mask", "token_exists_mask", "atom_token_index",
                 "atom_q_indices", "atom_kv_indices", "block_atom_pair_mask"]:
        val = getattr(structure, name, None)
        if val is not None:
            arr = val
            if hasattr(arr, 'dtype') and arr.dtype == mx.bool_:
                arr = arr.astype(mx.int32)
            result[name] = np.array(arr, copy=False)
    return result


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_bisection(
    model: ChaiMLX,
    ref: dict[str, np.ndarray],
    ctx_extras: tuple,
    ts_ref: dict[str, np.ndarray] | None = None,
) -> dict:
    """Block bisection: compare MLX per-block against TorchScript."""
    ctx, extras = ctx_extras
    structure = ctx.structure_inputs
    dtype = resolve_dtype(model.cfg)

    coords = extras["coords"]
    sigma = extras["sigma"]

    # MLX denoise with per-block capture
    print("\n[1] Running MLX denoise with per-block capture...")
    ref_trunk = reconstruct_trunk_outputs(ref, structure, dtype=dtype)
    cache = model.prepare_diffusion_cache(ref_trunk)
    mx.eval(cache.s_static, cache.z_cond, cache.blocked_pair_base,
            cache.atom_cond, cache.atom_single_cond, *cache.pair_biases)
    mlx_cap = mlx_denoise_capture(model, cache, coords, sigma)
    print(f"  Captured {len(mlx_cap)} MLX tensors")

    # TorchScript reference
    if ts_ref is not None:
        ts_cap = ts_ref
        print(f"\n[2] Using pre-generated TorchScript reference ({len(ts_cap)} tensors)")
    else:
        ts_out = ref.get("denoise.output")
        ts_cap = {"output": ts_out} if ts_out is not None else {}
        print(f"\n[2] Using reference NPZ output as TorchScript baseline")

    # Compare
    print("\n" + "=" * 72)
    print("BLOCK BISECTION RESULTS")
    print("=" * 72)

    # Per-block error (against TorchScript per-block if available, else RMS only)
    ts_blocks = {k: v for k, v in ts_cap.items() if k.startswith("block_")}
    has_ts_blocks = len(ts_blocks) > 0

    header = f"  {'stage':<24}"
    if has_ts_blocks:
        header += f" {'max_err':>10} {'mean_err':>10}"
    header += f" {'rms_value':>10}"
    print(header)
    print(f"  {'─' * 24}" + (f" {'─' * 10} {'─' * 10}" if has_ts_blocks else "") + f" {'─' * 10}")

    # Pre-transformer stages
    for key in ["scaled_coords", "s_cond", "token_structure_proj",
                "enc_tokens", "pre_transformer"]:
        if key in mlx_cap:
            arr = mlx_cap[key]
            rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
            line = f"  {key:<24}"
            if has_ts_blocks:
                line += f" {'—':>10} {'—':>10}"
            line += f" {rms:>10.4f}"
            print(line)

    # Per-block
    for i in range(16):
        key = f"block_{i:02d}"
        if key not in mlx_cap:
            continue
        mlx_arr = mlx_cap[key]
        rms = float(np.sqrt(np.mean(mlx_arr.astype(np.float64) ** 2)))
        line = f"  {key:<24}"
        if has_ts_blocks and key in ts_blocks:
            e = _err_stats(ts_blocks[key], mlx_arr)
            line += f" {e['max']:>10.4e} {e['mean']:>10.4e}"
        elif has_ts_blocks:
            line += f" {'—':>10} {'—':>10}"
        line += f" {rms:>10.4f}"
        print(line)

    # Post-transformer stages
    for key in ["post_ln", "pos_updates", "output"]:
        if key in mlx_cap:
            arr = mlx_cap[key]
            rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))
            line = f"  {key:<24}"
            if has_ts_blocks:
                line += f" {'—':>10} {'—':>10}"
            if key == "output" and "output" in ts_cap:
                e = _err_stats(ts_cap["output"], arr)
                line = f"  {key:<24}"
                line += f" {e['max']:>10.4e} {e['mean']:>10.4e}"
            line += f" {rms:>10.4f}"
            print(line)

    # Final output comparison
    if "output" in ts_cap and "output" in mlx_cap:
        e = _err_stats(ts_cap["output"], mlx_cap["output"])
        print(f"\n  FINAL OUTPUT ERROR: max={e['max']:.4e}  mean={e['mean']:.4e}  rms={e['rms']:.4e}")

    return {"mlx": mlx_cap, "ts": ts_cap}


def run_zero_input(
    model: ChaiMLX,
    ref: dict[str, np.ndarray],
    ctx_extras: tuple,
    ts_ref: dict[str, np.ndarray] | None = None,
) -> dict:
    """Zero-input diagnostic: isolate conditioning vs atom processing."""
    ctx, extras = ctx_extras
    structure = ctx.structure_inputs
    dtype = resolve_dtype(model.cfg)

    sigma = extras["sigma"]
    coords = extras["coords"]
    zero_coords = mx.zeros_like(coords)

    print("\n" + "=" * 72)
    print("ZERO-INPUT DIAGNOSTIC")
    print("=" * 72)

    # MLX with zero coords
    print("\n[1] Running MLX denoise with zero coords...")
    ref_trunk = reconstruct_trunk_outputs(ref, structure, dtype=dtype)
    cache = model.prepare_diffusion_cache(ref_trunk)
    mx.eval(cache.s_static, cache.z_cond, cache.blocked_pair_base,
            cache.atom_cond, cache.atom_single_cond, *cache.pair_biases)
    mlx_zero = mlx_denoise_capture(model, cache, zero_coords, sigma)
    print(f"  MLX zero-input output rms: {np.sqrt(np.mean(mlx_zero['output']**2)):.4f}")

    # TorchScript with zero coords
    ts_zero = ts_ref if ts_ref is not None else {}
    if ts_zero:
        print(f"\n[2] Using pre-generated TorchScript zero-input reference")
    else:
        print(f"\n[2] No TorchScript zero-input reference available")

    if "output" in ts_zero:
        ts_out = ts_zero["output"]
        mlx_out = mlx_zero["output"]

        e = _err_stats(ts_out, mlx_out)
        print(f"\n  Zero-input output error: max={e['max']:.4e} mean={e['mean']:.4e}")
        print(f"  MLX zero output rms:  {np.sqrt(np.mean(mlx_out**2)):.4f}")
        print(f"  TS zero output rms:   {np.sqrt(np.mean(ts_out**2)):.4f}")

        real_err = 4.02
        zero_err = e["max"]
        print()
        if zero_err > real_err * 0.8:
            print(f"  → CONCLUSION: Zero-input error ({zero_err:.2f}) ≈ real-input error ({real_err:.2f})")
            print(f"    Bug is in CONDITIONING PATH (doesn't depend on atom coords)")
        elif zero_err < real_err * 0.2:
            print(f"  → CONCLUSION: Zero-input error ({zero_err:.2f}) << real-input error ({real_err:.2f})")
            print(f"    Bug is in ATOM PROCESSING (depends on noisy coords)")
        else:
            print(f"  → CONCLUSION: Zero-input error ({zero_err:.2f}) partially explains real error ({real_err:.2f})")
            print(f"    Bug has components in both conditioning and atom processing")
    else:
        print("  No TorchScript output for comparison")
        print(f"  MLX zero-input output rms: {np.sqrt(np.mean(mlx_zero['output']**2)):.4f}")

    return {"mlx": mlx_zero, "ts": ts_zero}


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bisect denoise error by block")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--ts-real-npz", type=Path, default=None,
                        help="Pre-generated TorchScript output for real input")
    parser.add_argument("--ts-zero-npz", type=Path, default=None,
                        help="Pre-generated TorchScript output for zero input")
    args = parser.parse_args(list(argv) if argv is not None else None)

    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False)
    ctx, extras = load_feature_context(args.input_npz)
    ref = _npz_dict(args.reference_npz)

    # Override TorchScript references if provided
    ts_real_ref = _npz_dict(args.ts_real_npz) if args.ts_real_npz else None
    ts_zero_ref = _npz_dict(args.ts_zero_npz) if args.ts_zero_npz else None

    bisect_results = run_bisection(model, ref, (ctx, extras), ts_ref=ts_real_ref)
    zero_results = run_zero_input(model, ref, (ctx, extras), ts_ref=ts_zero_ref)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if "output" in bisect_results["ts"]:
        e = _err_stats(bisect_results["ts"]["output"], bisect_results["mlx"]["output"])
        print(f"  Real-input final error:  max={e['max']:.4e}")
    if "output" in zero_results.get("ts", {}):
        e = _err_stats(zero_results["ts"]["output"], zero_results["mlx"]["output"])
        print(f"  Zero-input final error:  max={e['max']:.4e}")


if __name__ == "__main__":
    main()
