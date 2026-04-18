"""Trunk-recycle and diffusion-step error-growth analysis: MLX vs CUDA.

This harness consumes a CUDA intermediates NPZ (produced by
``modal run -m cuda_harness.run_intermediates``) and answers two
questions:

1. **Trunk recycle growth.**  How does the MLX-vs-CUDA error on
   ``(single_repr, pair_repr)`` evolve across recycles 1..N?  We feed
   the CUDA embeddings for recycle 0 and then, for each recycle,
   capture MLX's intermediate and compare against the CUDA snapshot.
   Two modes are supported:

   * ``--mode isolated``: at each recycle, reseed MLX with the CUDA
     output of the *previous* recycle.  This measures the intrinsic
     per-recycle error (no cascading).
   * ``--mode cascading``: let MLX recycle from its own previous
     output, mirroring what a real MLX inference does.  This measures
     the cumulative drift.

2. **Diffusion step growth.**  For every snapshotted diffusion step
   in the NPZ (configurable on the CUDA side; default = first, mid,
   last), we feed the CUDA ``atom_pos_hat`` into MLX's denoise and
   compare the resulting ``denoised_pos`` against CUDA's.  We also
   emit the per-sample coordinate residual
   ``‖atom_pos_after - atom_pos_hat‖`` on each side to see whether MLX
   is taking meaningfully different diffusion steps.

Outputs a Markdown-friendly table per-stage and can also write a JSON
summary and CSV for plotting.

Usage
-----

::

    # Use snapshot_steps with more steps for a denser diffusion curve:
    modal run -m cuda_harness.run_intermediates \\
        --seeds 42 \\
        --snapshot-steps 1,25,50,100,150,199,200

    python scripts/cuda_error_accumulation.py \\
        --weights-dir weights \\
        --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \\
        --mode cascading \\
        --csv /tmp/chai_mlx_cuda/error_accumulation.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np

from chai_mlx import ChaiMLX
from chai_mlx.utils import resolve_dtype

from scripts.cuda_parity import (
    _as_mx,
    _load_npz,
    _read_manifest,
    _reconstruct_embedding_outputs,
    _reconstruct_feature_context,
    _reconstruct_trunk_outputs,
    _tensor_to_numpy,
)


@dataclass
class TrunkRow:
    recycle: int
    single_max_abs: float
    single_mean_abs: float
    single_rms: float
    single_ref_range: float
    pair_max_abs: float
    pair_mean_abs: float
    pair_rms: float
    pair_ref_range: float
    mode: str


@dataclass
class DiffusionRow:
    step: int
    sigma_curr: float
    denoised_max_abs: float
    denoised_mean_abs: float
    denoised_rms: float
    denoised_ref_range: float
    mlx_residual_rms: float
    cuda_residual_rms: float


def _diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    d = np.abs(a - b)
    return (
        float(d.max()) if d.size else 0.0,
        float(d.mean()) if d.size else 0.0,
        float(np.sqrt((d**2).mean())) if d.size else 0.0,
        float(np.abs(a).max()) if a.size else 0.0,
    )


def _run_trunk_curve(
    model: ChaiMLX,
    cuda_emb,
    data: dict[str, np.ndarray],
    *,
    num_recycles: int,
    mode: str,
) -> list[TrunkRow]:
    single_init = cuda_emb.single_initial
    pair_init = cuda_emb.pair_initial
    si = cuda_emb.structure_inputs

    if mode == "isolated":
        # At recycle k, feed CUDA recycle-(k-1) output as prev representation.
        prev_pairs: list[tuple[mx.array, mx.array]] = []
        for k in range(num_recycles):
            prev_key_s = f"trunk.recycle_{k - 1}.single" if k > 0 else None
            prev_key_p = f"trunk.recycle_{k - 1}.pair" if k > 0 else None
            if prev_key_s is not None and prev_key_s in data:
                prev_pairs.append(
                    (
                        _as_mx(data[prev_key_s], single_init.dtype),
                        _as_mx(data[prev_key_p], pair_init.dtype),
                    )
                )
            else:
                prev_pairs.append((single_init, pair_init))
    else:
        prev_pairs = [(single_init, pair_init)]

    rows: list[TrunkRow] = []
    prev_single, prev_pair = single_init, pair_init
    for k in range(num_recycles):
        if mode == "isolated":
            prev_single, prev_pair = prev_pairs[k]
        single = single_init + model.trunk_module.token_single_recycle_proj(prev_single)
        pair = pair_init + model.trunk_module.token_pair_recycle_proj(prev_pair)
        mx.eval(single, pair)

        pair = model.trunk_module.template_embedder(
            pair,
            cuda_emb.template_input,
            template_input_masks=si.template_input_masks,
            token_pair_mask=si.token_pair_mask,
        )
        mx.eval(pair)

        pair = model.trunk_module.msa_module(
            single,
            pair,
            cuda_emb.msa_input,
            token_pair_mask=si.token_pair_mask,
            msa_mask=si.msa_mask,
        )
        mx.eval(pair)

        single, pair = model.trunk_module.pairformer_stack(
            single,
            pair,
            pair_mask=si.token_pair_mask,
            single_mask=si.token_exists_mask,
        )
        mx.eval(single, pair)

        mlx_s = _tensor_to_numpy(single)
        mlx_p = _tensor_to_numpy(pair)
        cuda_s = data.get(f"trunk.recycle_{k}.single")
        cuda_p = data.get(f"trunk.recycle_{k}.pair")
        if cuda_s is None or cuda_p is None:
            continue

        max_s, mean_s, rms_s, range_s = _diff_stats(cuda_s, mlx_s)
        max_p, mean_p, rms_p, range_p = _diff_stats(cuda_p, mlx_p)

        rows.append(
            TrunkRow(
                recycle=k,
                single_max_abs=max_s,
                single_mean_abs=mean_s,
                single_rms=rms_s,
                single_ref_range=range_s,
                pair_max_abs=max_p,
                pair_mean_abs=mean_p,
                pair_rms=rms_p,
                pair_ref_range=range_p,
                mode=mode,
            )
        )

        if mode == "cascading":
            prev_single, prev_pair = single, pair

    return rows


def _run_diffusion_curve(
    model: ChaiMLX,
    cuda_emb,
    data: dict[str, np.ndarray],
    *,
    dtype: mx.Dtype,
) -> list[DiffusionRow]:
    trunk_out = _reconstruct_trunk_outputs(data, cuda_emb, dtype=dtype)
    cache = model.prepare_diffusion_cache(trunk_out)
    mx.eval(
        cache.s_static,
        cache.z_cond,
        cache.blocked_pair_base,
        cache.atom_cond,
        cache.atom_single_cond,
        *cache.pair_biases,
    )

    step_keys = sorted(
        {
            int(k.split("_")[1].split(".")[0])
            for k in data
            if k.startswith("diffusion.step_") and k.endswith(".denoised")
        }
    )
    rows: list[DiffusionRow] = []
    for step_idx in step_keys:
        tag = f"diffusion.step_{step_idx:04d}"
        atom_pos_hat = _as_mx(data[f"{tag}.atom_pos_hat"], dtype)
        atom_pos_after = data[f"{tag}.atom_pos_after"].astype(np.float32)
        sigma_curr = float(data[f"{tag}.sigma_curr"])

        if atom_pos_hat.ndim == 3:
            ds = atom_pos_hat.shape[0]
            atom_pos_hat_bds = atom_pos_hat.reshape(1, ds, atom_pos_hat.shape[1], atom_pos_hat.shape[2])
        else:
            ds = atom_pos_hat.shape[1]
            atom_pos_hat_bds = atom_pos_hat
        sigma_mx = mx.full((1, ds), sigma_curr, dtype=mx.float32)

        denoised = model.diffusion_module.denoise(cache, atom_pos_hat_bds, sigma_mx)
        mx.eval(denoised)
        mlx_denoised = _tensor_to_numpy(denoised)

        cuda_denoised = data[f"{tag}.denoised"].astype(np.float32)
        if mlx_denoised.shape != cuda_denoised.shape and mlx_denoised.ndim == cuda_denoised.ndim + 1:
            mlx_denoised = mlx_denoised[0]
        elif mlx_denoised.ndim == cuda_denoised.ndim - 1:
            mlx_denoised = mlx_denoised[None]

        max_d, mean_d, rms_d, range_d = _diff_stats(cuda_denoised, mlx_denoised)

        atom_pos_hat_np = np.array(atom_pos_hat)
        if atom_pos_hat_np.shape != atom_pos_after.shape and atom_pos_hat_np.ndim == atom_pos_after.ndim + 1:
            atom_pos_hat_np = atom_pos_hat_np[0]

        cuda_residual = atom_pos_after - atom_pos_hat_np
        cuda_residual_rms = float(np.sqrt((cuda_residual**2).mean()))

        mlx_residual = mlx_denoised - atom_pos_hat_np
        mlx_residual_rms = float(np.sqrt((mlx_residual**2).mean()))

        rows.append(
            DiffusionRow(
                step=step_idx,
                sigma_curr=sigma_curr,
                denoised_max_abs=max_d,
                denoised_mean_abs=mean_d,
                denoised_rms=rms_d,
                denoised_ref_range=range_d,
                mlx_residual_rms=mlx_residual_rms,
                cuda_residual_rms=cuda_residual_rms,
            )
        )
    return rows


def _print_trunk_rows(rows: list[TrunkRow]) -> None:
    if not rows:
        print("[trunk] no recycles to report")
        return
    print("\nTrunk recycle error (MLX vs CUDA):")
    print(
        f"  {'rec':>3}  {'s_max':>9} {'s_mean':>9} {'s_rms':>9} {'|s|_ref':>9}  "
        f"{'z_max':>9} {'z_mean':>9} {'z_rms':>9} {'|z|_ref':>9}  mode"
    )
    for r in rows:
        print(
            f"  {r.recycle:>3}  "
            f"{r.single_max_abs:>9.3e} {r.single_mean_abs:>9.3e} {r.single_rms:>9.3e} {r.single_ref_range:>9.3e}  "
            f"{r.pair_max_abs:>9.3e} {r.pair_mean_abs:>9.3e} {r.pair_rms:>9.3e} {r.pair_ref_range:>9.3e}  {r.mode}"
        )


def _print_diffusion_rows(rows: list[DiffusionRow]) -> None:
    if not rows:
        print("[diffusion] no snapshots to report")
        return
    print("\nDiffusion per-step error (MLX vs CUDA):")
    print(
        f"  {'step':>4} {'sigma':>10}  {'d_max':>10} {'d_mean':>10} {'d_rms':>10} {'|d|_ref':>10}  "
        f"{'mlx_res_rms':>12} {'cuda_res_rms':>13} {'ratio':>8}"
    )
    for r in rows:
        ratio = r.mlx_residual_rms / max(r.cuda_residual_rms, 1e-12)
        print(
            f"  {r.step:>4} {r.sigma_curr:>10.4f}  "
            f"{r.denoised_max_abs:>10.3e} {r.denoised_mean_abs:>10.3e} {r.denoised_rms:>10.3e} {r.denoised_ref_range:>10.3e}  "
            f"{r.mlx_residual_rms:>12.3e} {r.cuda_residual_rms:>13.3e} {ratio:>8.3f}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--compute-dtype", default=None, choices=["bfloat16", "float32"])
    parser.add_argument("--mode", default="cascading", choices=["cascading", "isolated"])
    parser.add_argument("--skip-trunk", action="store_true")
    parser.add_argument("--skip-diffusion", action="store_true")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    print(f"[load] {args.npz}")
    data = _load_npz(args.npz)
    manifest = _read_manifest(data)
    if manifest:
        print(
            f"  target={manifest.get('target')} seed={manifest.get('seed')} "
            f"n_tokens={manifest.get('n_tokens')} num_recycles={manifest.get('num_recycles')} "
            f"num_steps={manifest.get('num_steps')} gpu={manifest.get('gpu_name')}"
        )

    model = ChaiMLX.from_pretrained(
        args.weights_dir, strict=False, compute_dtype=args.compute_dtype
    )
    dtype = resolve_dtype(model.cfg)
    dtype_name = "float32" if dtype == mx.float32 else "bfloat16"
    print(f"  compute_dtype={dtype_name}")

    ctx = _reconstruct_feature_context(data)
    cuda_emb = _reconstruct_embedding_outputs(data, ctx.structure_inputs, dtype=dtype)

    trunk_rows: list[TrunkRow] = []
    diffusion_rows: list[DiffusionRow] = []
    num_recycles = int(manifest.get("num_recycles", 3))

    if not args.skip_trunk:
        trunk_rows = _run_trunk_curve(
            model, cuda_emb, data, num_recycles=num_recycles, mode=args.mode
        )
        _print_trunk_rows(trunk_rows)

    if not args.skip_diffusion:
        diffusion_rows = _run_diffusion_curve(model, cuda_emb, data, dtype=dtype)
        _print_diffusion_rows(diffusion_rows)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "stage",
                    "recycle_or_step",
                    "sigma_curr",
                    "max_abs",
                    "mean_abs",
                    "rms",
                    "ref_range",
                    "extra_mlx_residual_rms",
                    "extra_cuda_residual_rms",
                ]
            )
            for r in trunk_rows:
                writer.writerow(
                    ["trunk.single", r.recycle, "", r.single_max_abs, r.single_mean_abs, r.single_rms, r.single_ref_range, "", ""]
                )
                writer.writerow(
                    ["trunk.pair", r.recycle, "", r.pair_max_abs, r.pair_mean_abs, r.pair_rms, r.pair_ref_range, "", ""]
                )
            for r in diffusion_rows:
                writer.writerow(
                    [
                        "diffusion.denoised",
                        r.step,
                        r.sigma_curr,
                        r.denoised_max_abs,
                        r.denoised_mean_abs,
                        r.denoised_rms,
                        r.denoised_ref_range,
                        r.mlx_residual_rms,
                        r.cuda_residual_rms,
                    ]
                )
        print(f"[save] csv -> {args.csv}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                {
                    "npz": str(args.npz),
                    "compute_dtype": dtype_name,
                    "mode": args.mode,
                    "manifest": manifest,
                    "trunk": [asdict(r) for r in trunk_rows],
                    "diffusion": [asdict(r) for r in diffusion_rows],
                },
                indent=2,
                default=str,
            )
        )
        print(f"[save] summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
