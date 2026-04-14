"""Detailed error distribution analysis across all pipeline stages.

Runs each stage in isolation (feeding reference inputs), then produces
per-element, per-residue, and per-atom error distributions — not just
max/mean summaries.

Usage::

    # First generate reference dumps if not already done:
    python scripts/chai_lab_reference_dump.py \
        --input-npz /tmp/chai_parity/input.npz \
        --reference-npz /tmp/chai_parity/reference.npz

    # Then run diagnostics:
    python scripts/error_diagnostics.py \
        --weights-dir weights/ \
        --input-npz /tmp/chai_parity/input.npz \
        --reference-npz /tmp/chai_parity/reference.npz
"""

from __future__ import annotations

import argparse
import gc
import sys
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np

from chai_mlx import ChaiMLX
from chai_mlx.data.types import EmbeddingOutputs, StructureInputs, TrunkOutputs
from chai_mlx.utils import resolve_dtype


def _percentiles(arr: np.ndarray, name: str) -> None:
    flat = np.abs(arr.astype(np.float32)).ravel()
    pcts = [50, 75, 90, 95, 99, 99.9, 100]
    vals = np.percentile(flat, pcts)
    print(f"  {name}:")
    print(f"    shape={arr.shape}  mean_abs={flat.mean():.4e}  std={flat.std():.4e}")
    parts = "  ".join(f"p{p}={v:.4e}" for p, v in zip(pcts, vals))
    print(f"    {parts}")


def _error_percentiles(ref: np.ndarray, got: np.ndarray, name: str) -> np.ndarray:
    diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
    flat = diff.ravel()
    pcts = [50, 75, 90, 95, 99, 99.9, 100]
    vals = np.percentile(flat, pcts)
    ref_abs = np.abs(ref.astype(np.float32))
    print(f"  {name}:")
    print(f"    elements={flat.size:,}  mean={flat.mean():.4e}  std={flat.std():.4e}")
    parts = "  ".join(f"p{p}={v:.4e}" for p, v in zip(pcts, vals))
    print(f"    {parts}")
    print(f"    ref range: [{ref.min():.4e}, {ref.max():.4e}]  ref_rms={np.sqrt((ref.astype(np.float32)**2).mean()):.4e}")
    return diff


def _to_numpy_f32(x: mx.array) -> np.ndarray:
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    return np.array(x, copy=False)


def _load_structure(inp_data) -> StructureInputs:
    si_fields = {}
    for key in inp_data.files:
        if key.startswith("structure_inputs."):
            si_fields[key.split(".", 1)[1]] = mx.array(inp_data[key])
    return StructureInputs(**si_fields)


# ── Trunk diagnostics ────────────────────────────────────────────────────


def run_trunk_diagnostics(
    model: ChaiMLX,
    ref_data,
    structure: StructureInputs,
    dtype: mx.Dtype,
) -> None:
    print("\n" + "=" * 70)
    print("TRUNK ERROR DISTRIBUTION")
    print("=" * 70)

    p = "embedding.outputs"
    emb = EmbeddingOutputs(
        token_single_input=mx.array(ref_data[f"{p}.token_single_input"]).astype(dtype),
        token_pair_input=mx.array(ref_data[f"{p}.token_pair_input"]).astype(dtype),
        token_pair_structure_input=mx.array(ref_data[f"{p}.token_pair_structure_input"]).astype(dtype),
        atom_single_input=mx.array(ref_data[f"{p}.atom_single_input"]).astype(dtype),
        atom_single_structure_input=mx.array(ref_data[f"{p}.atom_single_structure_input"]).astype(dtype),
        atom_pair_input=mx.array(ref_data[f"{p}.atom_pair_input"]).astype(dtype),
        atom_pair_structure_input=mx.array(ref_data[f"{p}.atom_pair_structure_input"]).astype(dtype),
        msa_input=mx.array(ref_data[f"{p}.msa_input"][:, :1]).astype(dtype),
        template_input=mx.array(ref_data[f"{p}.template_input"]).astype(dtype),
        single_initial=mx.array(ref_data[f"{p}.single_initial"]).astype(dtype),
        single_structure=mx.array(ref_data[f"{p}.single_structure"]).astype(dtype),
        pair_initial=mx.array(ref_data[f"{p}.pair_initial"]).astype(dtype),
        pair_structure=mx.array(ref_data[f"{p}.pair_structure"]).astype(dtype),
        structure_inputs=structure,
    )

    trunk_out = model.trunk(emb, recycles=1)
    mx.eval(trunk_out.single_trunk, trunk_out.pair_trunk)

    ref_s = ref_data["trunk.outputs.single_trunk"]
    ref_z = ref_data["trunk.outputs.pair_trunk"]
    got_s = _to_numpy_f32(trunk_out.single_trunk)
    got_z = _to_numpy_f32(trunk_out.pair_trunk)

    print("\n── single_trunk (B, N, 384) ──")
    s_diff = _error_percentiles(ref_s, got_s, "absolute error")

    per_residue_s = s_diff[0].mean(axis=-1)  # (N,) mean over feature dim
    print(f"\n  Per-residue mean error (over feature dim):")
    print(f"    min={per_residue_s.min():.4e}  median={np.median(per_residue_s):.4e}  "
          f"max={per_residue_s.max():.4e}")
    print(f"    worst 5 residues: {np.argsort(per_residue_s)[-5:][::-1]} "
          f"with errors {np.sort(per_residue_s)[-5:][::-1]}")

    rel_s = s_diff / np.maximum(np.abs(ref_s.astype(np.float32)), 1e-8)
    print(f"\n  Relative error (|err|/|ref|):")
    flat_rel = rel_s.ravel()
    for p_val in [50, 90, 99, 100]:
        print(f"    p{p_val}={np.percentile(flat_rel, p_val):.4f}")

    print("\n── pair_trunk (B, N, N, 256) ──")
    z_diff = _error_percentiles(ref_z, got_z, "absolute error")

    per_pair_z = z_diff[0].mean(axis=-1)  # (N, N)
    print(f"\n  Per-pair mean error (over feature dim):")
    print(f"    min={per_pair_z.min():.4e}  median={np.median(per_pair_z):.4e}  "
          f"max={per_pair_z.max():.4e}")
    diag_err = np.diag(per_pair_z)
    offdiag_mask = ~np.eye(per_pair_z.shape[0], dtype=bool)
    offdiag_err = per_pair_z[offdiag_mask]
    print(f"    diagonal (self-pairs):  median={np.median(diag_err):.4e}  max={diag_err.max():.4e}")
    print(f"    off-diagonal:           median={np.median(offdiag_err):.4e}  max={offdiag_err.max():.4e}")

    del trunk_out, emb, s_diff, z_diff
    gc.collect()
    mx.clear_cache()


# ── Diffusion diagnostics ────────────────────────────────────────────────


def run_diffusion_diagnostics(
    model: ChaiMLX,
    ref_data,
    inp_data,
    structure: StructureInputs,
    dtype: mx.Dtype,
) -> None:
    print("\n" + "=" * 70)
    print("DIFFUSION ERROR DISTRIBUTION")
    print("=" * 70)

    p = "trunk.outputs"
    ref_trunk = TrunkOutputs(
        single_initial=mx.array(ref_data[f"{p}.single_initial"]).astype(dtype),
        single_trunk=mx.array(ref_data[f"{p}.single_trunk"]).astype(dtype),
        single_structure=mx.array(ref_data[f"{p}.single_structure"]).astype(dtype),
        pair_initial=mx.array(ref_data[f"{p}.pair_initial"]).astype(dtype),
        pair_trunk=mx.array(ref_data[f"{p}.pair_trunk"]).astype(dtype),
        pair_structure=mx.array(ref_data[f"{p}.pair_structure"]).astype(dtype),
        atom_single_structure_input=mx.array(ref_data[f"{p}.atom_single_structure_input"]).astype(dtype),
        atom_pair_structure_input=mx.array(ref_data[f"{p}.atom_pair_structure_input"]).astype(dtype),
        msa_input=mx.array(ref_data[f"{p}.msa_input"][:, :1]).astype(dtype),
        template_input=mx.array(ref_data[f"{p}.template_input"]).astype(dtype),
        structure_inputs=structure,
    )

    coords = mx.array(inp_data["coords"])
    sigma = mx.array(inp_data["sigma"])

    cache = model.prepare_diffusion_cache(ref_trunk)
    mx.eval(cache.s_static, cache.z_cond, cache.blocked_pair_base,
            cache.atom_cond, cache.atom_single_cond, *cache.pair_biases)

    denoised = model.denoise(cache, coords, sigma)
    mx.eval(denoised)

    ref_den = ref_data["denoise.output"]
    got_den = _to_numpy_f32(denoised)

    print("\n── denoised coordinates (B, S, A, 3) ──")
    _error_percentiles(ref_den, got_den, "absolute error (Å)")

    atom_mask = np.array(inp_data["structure_inputs.atom_exists_mask"][0])  # (A,)
    valid = atom_mask > 0.5

    ref_coords = ref_den[0, 0]   # (A, 3)
    got_coords = got_den[0, 0]   # (A, 3)
    per_atom_err = np.sqrt(((ref_coords - got_coords) ** 2).sum(axis=-1))  # (A,)
    valid_err = per_atom_err[valid]

    print(f"\n  Per-atom Euclidean distance (valid atoms only, n={valid.sum()}):")
    pcts = [10, 25, 50, 75, 90, 95, 99, 100]
    vals = np.percentile(valid_err, pcts)
    parts = "  ".join(f"p{p}={v:.3f}Å" for p, v in zip(pcts, vals))
    print(f"    {parts}")
    print(f"    mean={valid_err.mean():.3f}Å  std={valid_err.std():.3f}Å")

    under_1 = (valid_err < 1.0).sum()
    under_2 = (valid_err < 2.0).sum()
    under_5 = (valid_err < 5.0).sum()
    n = len(valid_err)
    print(f"\n  Fraction of atoms within threshold:")
    print(f"    <1.0Å: {under_1}/{n} ({100*under_1/n:.1f}%)")
    print(f"    <2.0Å: {under_2}/{n} ({100*under_2/n:.1f}%)")
    print(f"    <5.0Å: {under_5}/{n} ({100*under_5/n:.1f}%)")

    token_centre_idx = inp_data.get("structure_inputs.token_centre_atom_index")
    if token_centre_idx is None:
        warnings.warn(
            "structure_inputs.token_centre_atom_index missing; falling back to "
            "token_reference_atom_index for representative-atom error reporting.",
            stacklevel=2,
        )
        token_centre_idx = inp_data["structure_inputs.token_reference_atom_index"]
    token_centre_idx = np.array(token_centre_idx[0])  # (N,)
    token_mask = np.array(inp_data["structure_inputs.token_exists_mask"][0]) > 0.5
    ca_indices = token_centre_idx[token_mask].astype(int)
    ca_err = per_atom_err[ca_indices]
    print(f"\n  Representative atom (Cα-like) error (n={len(ca_err)}):")
    pcts_ca = [10, 25, 50, 75, 90, 95, 100]
    vals_ca = np.percentile(ca_err, pcts_ca)
    parts_ca = "  ".join(f"p{p}={v:.3f}Å" for p, v in zip(pcts_ca, vals_ca))
    print(f"    {parts_ca}")

    del cache, ref_trunk, denoised
    gc.collect()
    mx.clear_cache()


# ── Confidence diagnostics ───────────────────────────────────────────────


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def _tvd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(p - q).sum(axis=-1)


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return (p * np.log(np.maximum(p, eps) / np.maximum(q, eps))).sum(axis=-1)


def run_confidence_diagnostics(
    model: ChaiMLX,
    ref_data,
    structure: StructureInputs,
    dtype: mx.Dtype,
) -> None:
    print("\n" + "=" * 70)
    print("CONFIDENCE HEAD ERROR DISTRIBUTION")
    print("=" * 70)

    p = "trunk.outputs"
    ref_trunk = TrunkOutputs(
        single_initial=mx.array(ref_data[f"{p}.single_initial"]).astype(dtype),
        single_trunk=mx.array(ref_data[f"{p}.single_trunk"]).astype(dtype),
        single_structure=mx.array(ref_data[f"{p}.single_structure"]).astype(dtype),
        pair_initial=mx.array(ref_data[f"{p}.pair_initial"]).astype(dtype),
        pair_trunk=mx.array(ref_data[f"{p}.pair_trunk"]).astype(dtype),
        pair_structure=mx.array(ref_data[f"{p}.pair_structure"]).astype(dtype),
        atom_single_structure_input=mx.array(ref_data[f"{p}.atom_single_structure_input"]).astype(dtype),
        atom_pair_structure_input=mx.array(ref_data[f"{p}.atom_pair_structure_input"]).astype(dtype),
        msa_input=mx.array(ref_data[f"{p}.msa_input"][:, :1]).astype(dtype),
        template_input=mx.array(ref_data[f"{p}.template_input"]).astype(dtype),
        structure_inputs=structure,
    )

    coords = mx.array(ref_data["denoise.output"])

    conf = model.confidence(ref_trunk, coords)
    mx.eval(conf.pae_logits, conf.pde_logits, conf.plddt_logits)

    heads = [
        ("PAE", "pae_logits", conf.pae_logits, "confidence.outputs.pae_logits"),
        ("PDE", "pde_logits", conf.pde_logits, "confidence.outputs.pde_logits"),
        ("pLDDT", "plddt_logits", conf.plddt_logits, "confidence.outputs.plddt_logits"),
    ]

    for label, field, mlx_logits, ref_key in heads:
        print(f"\n── {label} ({ref_data[ref_key].shape}) ──")
        ref_logits = ref_data[ref_key]
        got_logits = _to_numpy_f32(mlx_logits)

        _error_percentiles(ref_logits, got_logits, "logit absolute error")

        ref_probs = _softmax(ref_logits)
        got_probs = _softmax(got_logits)

        ref_argmax = ref_logits.argmax(axis=-1).ravel()
        got_argmax = got_logits.argmax(axis=-1).ravel()
        agree = (ref_argmax == got_argmax).sum()
        total = ref_argmax.size
        print(f"\n  Argmax agreement: {agree}/{total} ({100*agree/total:.2f}%)")

        off_by = np.abs(ref_argmax.astype(int) - got_argmax.astype(int))
        print(f"  Argmax disagreement where wrong:")
        if agree < total:
            wrong = off_by[off_by > 0]
            print(f"    mean off-by={wrong.mean():.2f} bins  max off-by={wrong.max()} bins")
            for d in [1, 2, 3, 5]:
                n = (wrong <= d).sum()
                print(f"    within {d} bin(s): {n}/{len(wrong)} ({100*n/len(wrong):.1f}%)")

        tvd = _tvd(ref_probs, got_probs)
        flat_tvd = tvd.ravel()
        print(f"\n  TVD (Total Variation Distance) distribution:")
        pcts = [10, 25, 50, 75, 90, 95, 99, 100]
        vals = np.percentile(flat_tvd, pcts)
        parts = "  ".join(f"p{p}={v:.4f}" for p, v in zip(pcts, vals))
        print(f"    {parts}")
        print(f"    mean={flat_tvd.mean():.4f}")

        kl_fwd = _kl(ref_probs, got_probs)
        flat_kl = kl_fwd.ravel()
        finite_kl = flat_kl[np.isfinite(flat_kl)]
        print(f"\n  KL divergence (ref || mlx) distribution:")
        vals_kl = np.percentile(finite_kl, pcts)
        parts_kl = "  ".join(f"p{p}={v:.4f}" for p, v in zip(pcts, vals_kl))
        print(f"    {parts_kl}")
        print(f"    mean={finite_kl.mean():.4f}")

        ref_entropy = -(_kl(ref_probs, ref_probs) + (ref_probs * np.log(np.maximum(ref_probs, 1e-10))).sum(axis=-1))
        ref_entropy = -(ref_probs * np.log(np.maximum(ref_probs, 1e-10))).sum(axis=-1)
        got_entropy = -(got_probs * np.log(np.maximum(got_probs, 1e-10))).sum(axis=-1)
        n_bins = ref_logits.shape[-1]
        max_entropy = np.log(n_bins)
        print(f"\n  Entropy (ref): mean={ref_entropy.mean():.3f}  (max possible={max_entropy:.3f})")
        print(f"  Entropy (mlx): mean={got_entropy.mean():.3f}")
        print(f"  → Both {'peaked' if ref_entropy.mean() < max_entropy * 0.5 else 'diffuse'} "
              f"(using {100*ref_entropy.mean()/max_entropy:.0f}% of max entropy)")

    del conf, ref_trunk
    gc.collect()
    mx.clear_cache()


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Detailed error distribution diagnostics")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--stages", type=str, default="trunk,diffusion,confidence",
                        help="Comma-separated stages to run (default: all)")
    args = parser.parse_args()

    stages = set(args.stages.split(","))
    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False)
    dtype = resolve_dtype(model.cfg)
    print(f"Model compute_dtype: {dtype}")

    ref_data = np.load(args.reference_npz)
    inp_data = np.load(args.input_npz)
    structure = _load_structure(inp_data)

    if "trunk" in stages:
        run_trunk_diagnostics(model, ref_data, structure, dtype)

    if "diffusion" in stages:
        run_diffusion_diagnostics(model, ref_data, inp_data, structure, dtype)

    if "confidence" in stages:
        run_confidence_diagnostics(model, ref_data, structure, dtype)

    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
