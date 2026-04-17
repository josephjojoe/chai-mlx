"""Isolation parity tests: feed reference tensors at stage boundaries.

Answers two key questions:

1. Is the MLX trunk individually faithful (given perfect TorchScript embeddings)?
2. Is the MLX diffusion module individually faithful (given perfect trunk outputs)?

By feeding TorchScript reference tensors at each stage boundary, we eliminate
error cascade from upstream stages, isolating each module's own numerical error.

Usage::

    python scripts/stage_isolation_parity.py \
        --weights-dir /path/to/safetensors_dir \
        --input-npz /path/to/input_context.npz \
        --reference-npz /path/to/reference_tensors.npz \
        [--tol 0.1] [--verbose]

The input/reference NPZ files are the same ones produced by
``chai_lab_reference_dump.py``.  If ``granular_reference_dump.py`` was used
instead, per-pairformer-block comparisons are also performed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chai_mlx import ChaiMLX
from chai_mlx.data.types import (
    EmbeddingOutputs,
    StructureInputs,
    TrunkOutputs,
)
from chai_mlx.utils import resolve_dtype
from layer_parity import (
    _mx_array,
    _npz_dict,
    _record,
    capture_cache,
    capture_confidence,
    capture_denoise,
    capture_trunk,
    load_feature_context,
)


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------

def _ref(ref: dict[str, np.ndarray], key: str, dtype: mx.Dtype = mx.float32) -> mx.array:
    arr = mx.array(ref[key])
    return arr.astype(dtype) if arr.dtype != dtype else arr


def reconstruct_embedding_outputs(
    ref: dict[str, np.ndarray],
    structure: StructureInputs,
    dtype: mx.Dtype = mx.float32,
) -> EmbeddingOutputs:
    """Build ``EmbeddingOutputs`` from TorchScript reference tensors."""
    p = "embedding.outputs"
    return EmbeddingOutputs(
        token_single_input=_ref(ref, f"{p}.token_single_input", dtype),
        token_pair_input=_ref(ref, f"{p}.token_pair_input", dtype),
        token_pair_structure_input=_ref(ref, f"{p}.token_pair_structure_input", dtype),
        atom_single_input=_ref(ref, f"{p}.atom_single_input", dtype),
        atom_single_structure_input=_ref(ref, f"{p}.atom_single_structure_input", dtype),
        atom_pair_input=_ref(ref, f"{p}.atom_pair_input", dtype),
        atom_pair_structure_input=_ref(ref, f"{p}.atom_pair_structure_input", dtype),
        msa_input=_ref(ref, f"{p}.msa_input", dtype),
        template_input=_ref(ref, f"{p}.template_input", dtype),
        single_initial=_ref(ref, f"{p}.single_initial", dtype),
        single_structure=_ref(ref, f"{p}.single_structure", dtype),
        pair_initial=_ref(ref, f"{p}.pair_initial", dtype),
        pair_structure=_ref(ref, f"{p}.pair_structure", dtype),
        structure_inputs=structure,
    )


def reconstruct_trunk_outputs(
    ref: dict[str, np.ndarray],
    structure: StructureInputs,
    dtype: mx.Dtype = mx.float32,
) -> TrunkOutputs:
    """Build ``TrunkOutputs`` from TorchScript reference tensors."""
    p = "trunk.outputs"
    return TrunkOutputs(
        single_initial=_ref(ref, f"{p}.single_initial", dtype),
        single_trunk=_ref(ref, f"{p}.single_trunk", dtype),
        single_structure=_ref(ref, f"{p}.single_structure", dtype),
        pair_initial=_ref(ref, f"{p}.pair_initial", dtype),
        pair_trunk=_ref(ref, f"{p}.pair_trunk", dtype),
        pair_structure=_ref(ref, f"{p}.pair_structure", dtype),
        atom_single_structure_input=_ref(ref, f"{p}.atom_single_structure_input", dtype),
        atom_pair_structure_input=_ref(ref, f"{p}.atom_pair_structure_input", dtype),
        msa_input=_ref(ref, f"{p}.msa_input", dtype),
        template_input=_ref(ref, f"{p}.template_input", dtype),
        structure_inputs=structure,
    )


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _compare(name: str, ref_arr: np.ndarray, got_arr: np.ndarray, tol: float) -> bool:
    """Compare two tensors.  Returns True on pass."""
    if ref_arr.shape != got_arr.shape:
        print(f"  [FAIL] {name}: shape mismatch ref={ref_arr.shape} mlx={got_arr.shape}")
        return False
    diff = np.abs(ref_arr.astype(np.float32) - got_arr.astype(np.float32))
    max_d = float(diff.max()) if diff.size else 0.0
    mean_d = float(diff.mean()) if diff.size else 0.0
    ref_max = float(np.abs(ref_arr.astype(np.float32)).max()) if diff.size else 1.0
    rel = max_d / max(ref_max, 1e-8)
    status = "PASS" if max_d <= tol else "FAIL"
    print(f"  [{status}] {name}: max={max_d:.3e} mean={mean_d:.3e} "
          f"rel={rel:.4f} ref_range={ref_max:.3e}")
    return max_d <= tol


def _analyze_block_statistics(
    tensors: dict[str, np.ndarray],
    ref: dict[str, np.ndarray] | None,
    *,
    prefix: str,
    verbose: bool,
) -> None:
    """Per-block output statistics and optional per-block comparison."""
    block_keys = sorted(
        [k for k in tensors if k.startswith(f"{prefix}.block_") and k.endswith(".single")],
        key=lambda k: int(k.split("block_")[1].split(".")[0]),
    )
    if not block_keys:
        print("    (no per-block data captured)")
        return

    has_ref_blocks = ref is not None and any(k in ref for k in block_keys)

    header = f"  {'block':>7} {'s_rms':>10} {'z_rms':>10} {'s_delta':>10} {'z_delta':>10}"
    if has_ref_blocks:
        header += f" {'s_err':>10} {'z_err':>10}"
    print(header)

    prev_s: np.ndarray | None = None
    prev_z: np.ndarray | None = None

    for key in block_keys:
        idx = int(key.split("block_")[1].split(".")[0])
        pair_key = key.replace(".single", ".pair")

        s = tensors[key].astype(np.float32)
        z = tensors[pair_key].astype(np.float32) if pair_key in tensors else None

        s_rms = float(np.sqrt(np.mean(s ** 2)))
        z_rms = float(np.sqrt(np.mean(z ** 2))) if z is not None else 0.0

        s_delta = float(np.sqrt(np.mean((s - prev_s) ** 2))) if prev_s is not None else 0.0
        z_delta = float(np.sqrt(np.mean((z - prev_z) ** 2))) if prev_z is not None and z is not None else 0.0

        show = verbose or idx in (0, 1, 11, 23, 35, 47)
        if show:
            line = f"  {idx:7d} {s_rms:10.4f} {z_rms:10.4f} {s_delta:10.4f} {z_delta:10.4f}"
            if has_ref_blocks and ref is not None:
                if key in ref:
                    s_err = float(np.abs(s - ref[key].astype(np.float32)).max())
                    line += f" {s_err:10.3e}"
                if pair_key in ref:
                    z_err = float(np.abs(z - ref[pair_key].astype(np.float32)).max()) if z is not None else 0.0
                    line += f" {z_err:10.3e}"
            print(line)

        prev_s, prev_z = s, z


# ---------------------------------------------------------------------------
# Isolation tests
# ---------------------------------------------------------------------------

def run_trunk_isolation(
    model: ChaiMLX,
    ref: dict[str, np.ndarray],
    structure: StructureInputs,
    *,
    recycles: int,
    tol: float,
    verbose: bool,
) -> int:
    """Feed TorchScript embedding outputs into the MLX trunk."""
    print("\n" + "=" * 60)
    print("TRUNK ISOLATION TEST")
    print("=" * 60)
    dtype = resolve_dtype(model.cfg)
    print(f"Feeding TorchScript embedding outputs -> MLX trunk (dtype={dtype})")

    ref_emb = reconstruct_embedding_outputs(ref, structure, dtype=dtype)
    trunk_tensors: dict[str, np.ndarray] = {}
    trunk_out = capture_trunk(
        model,
        ref_emb,
        recycles=recycles,
        tensors=trunk_tensors,
        capture_detail="pairformer",
        record_outputs=False,
    )

    failures = 0

    ref_s = ref.get("trunk.outputs.single_trunk")
    ref_z = ref.get("trunk.outputs.pair_trunk")

    if ref_s is not None:
        got_s = np.array(trunk_out.single_trunk, copy=False)
        if not _compare("single_trunk", ref_s, got_s, tol):
            failures += 1
    else:
        print("  [SKIP] trunk.outputs.single_trunk not in reference")

    if ref_z is not None:
        got_z = np.array(trunk_out.pair_trunk, copy=False)
        if not _compare("pair_trunk", ref_z, got_z, tol):
            failures += 1
    else:
        print("  [SKIP] trunk.outputs.pair_trunk not in reference")

    print("\n  Per-block statistics (pairformer, last recycle):")
    _analyze_block_statistics(
        trunk_tensors, ref, prefix=f"trunk.recycle_{recycles - 1}.pairformer",
        verbose=verbose,
    )

    return failures


def run_diffusion_isolation(
    model: ChaiMLX,
    ref: dict[str, np.ndarray],
    structure: StructureInputs,
    coords: mx.array,
    sigma: mx.array,
    *,
    tol: float,
    verbose: bool,
) -> int:
    """Feed TorchScript trunk outputs into the MLX diffusion module."""
    print("\n" + "=" * 60)
    print("DIFFUSION ISOLATION TEST")
    print("=" * 60)
    dtype = resolve_dtype(model.cfg)
    print(f"Feeding TorchScript trunk outputs -> MLX diffusion (dtype={dtype})")

    ref_trunk = reconstruct_trunk_outputs(ref, structure, dtype=dtype)

    cache_tensors: dict[str, np.ndarray] = {}
    cache = capture_cache(model, ref_trunk, tensors=cache_tensors)

    denoise_tensors: dict[str, np.ndarray] = {}
    capture_denoise(
        model, cache, coords, sigma,
        tensors=denoise_tensors,
    )

    failures = 0

    keys_to_check = [
        "denoise.encoder_tokens",
        "denoise.encoder_atom_repr",
        "denoise.token_repr_pre_transformer",
        "denoise.token_repr_post_transformer",
        "denoise.decoder_cond",
        "denoise.pos_updates",
        "denoise.output",
    ]

    for key in keys_to_check:
        if key in ref and key in denoise_tensors:
            if not _compare(key, ref[key], denoise_tensors[key], tol):
                failures += 1
        elif key in denoise_tensors and key not in ref:
            if verbose:
                got = denoise_tensors[key].astype(np.float32)
                print(f"  [INFO] {key}: rms={np.sqrt(np.mean(got**2)):.4f} "
                      f"range=[{got.min():.3e}, {got.max():.3e}]")

    # Per-transformer-block intermediates (only available in granular reference)
    for i in range(20):
        key = f"denoise.transformer.block_{i}.token_repr"
        if key in ref and key in denoise_tensors:
            if not _compare(key, ref[key], denoise_tensors[key], tol):
                failures += 1

    return failures


def run_confidence_isolation(
    model: ChaiMLX,
    ref: dict[str, np.ndarray],
    structure: StructureInputs,
    coords: mx.array,
    *,
    tol: float,
    verbose: bool,
) -> int:
    """Feed TorchScript trunk outputs into the MLX confidence head."""
    print("\n" + "=" * 60)
    print("CONFIDENCE ISOLATION TEST")
    print("=" * 60)
    dtype = resolve_dtype(model.cfg)
    print(f"Feeding TorchScript trunk outputs -> MLX confidence head (dtype={dtype})")

    ref_trunk = reconstruct_trunk_outputs(ref, structure, dtype=dtype)

    conf_tensors: dict[str, np.ndarray] = {}
    capture_confidence(model, ref_trunk, coords, tensors=conf_tensors)

    failures = 0

    for key in ("confidence.outputs.pae_logits",
                "confidence.outputs.pde_logits",
                "confidence.outputs.plddt_logits"):
        if key in ref and key in conf_tensors:
            if not _compare(key, ref[key], conf_tensors[key], tol):
                failures += 1
        elif key not in ref:
            print(f"  [SKIP] {key} not in reference")

    return failures


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Isolation parity: feed reference tensors at stage boundaries",
    )
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=0.1,
                        help="Max absolute error tolerance (default: 0.1)")
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float32"),
        default=None,
        help="Override MLX model compute dtype",
    )
    parser.add_argument(
        "--mlx-device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Run the MLX side on GPU or CPU",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Show all per-block statistics instead of summary")
    args = parser.parse_args(list(argv) if argv is not None else None)

    mx.set_default_device(mx.Device(getattr(mx.DeviceType, args.mlx_device), 0))
    print(f"Using MLX device: {mx.default_device()}")

    model = ChaiMLX.from_pretrained(
        args.weights_dir,
        strict=False,
        compute_dtype=args.compute_dtype,
    )
    ctx, extras = load_feature_context(args.input_npz)
    ref = _npz_dict(args.reference_npz)
    structure = ctx.structure_inputs

    failures = 0

    failures += run_trunk_isolation(
        model, ref, structure,
        recycles=args.recycles, tol=args.tol, verbose=args.verbose,
    )

    coords = extras.get("coords")
    sigma = extras.get("sigma")

    if coords is not None and sigma is not None:
        failures += run_diffusion_isolation(
            model, ref, structure, coords, sigma,
            tol=args.tol, verbose=args.verbose,
        )
    else:
        print("\n[SKIP] Diffusion isolation: no coords/sigma in input NPZ")

    if coords is not None:
        failures += run_confidence_isolation(
            model, ref, structure, coords,
            tol=args.tol, verbose=args.verbose,
        )
    else:
        print("\n[SKIP] Confidence isolation: no coords in input NPZ")

    print("\n" + "=" * 60)
    if failures:
        print(f"RESULT: {failures} failure(s)")
        raise SystemExit(1)
    print("RESULT: all isolation tests passed")


if __name__ == "__main__":
    main()
