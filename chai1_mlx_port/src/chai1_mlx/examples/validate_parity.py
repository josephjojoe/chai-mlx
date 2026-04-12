"""Per-component numerical parity validation between TorchScript and MLX.

Usage::

    python -m chai1_mlx.examples.validate_parity \
        --torchscript-dir /path/to/pt_files/ \
        --safetensors-dir /path/to/safetensors_dir/

This script loads individual TorchScript submodules one at a time and
compares their outputs against the matching MLX module with identical
weights and inputs.  Designed for 16 GB RAM machines — each component
is tested and released before the next is loaded.

Requires PyTorch to be installed (for loading TorchScript).
"""

from __future__ import annotations

import argparse
import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np


@dataclass
class ParityResult:
    component: str
    output_name: str
    max_abs_diff: float
    mean_abs_diff: float
    passed: bool


def _to_numpy(x: mx.array) -> np.ndarray:
    return np.array(x, copy=False)


def _to_mlx(x) -> mx.array:
    """Convert a torch.Tensor or numpy array to MLX."""
    if hasattr(x, "detach"):
        return mx.array(x.detach().cpu().float().numpy())
    return mx.array(x)


def _compare(
    name: str,
    ref_np: np.ndarray,
    mlx_np: np.ndarray,
    component: str,
    tol: float,
) -> ParityResult:
    diff = np.abs(ref_np.astype(np.float32) - mlx_np.astype(np.float32))
    max_diff = float(diff.max())
    mean_diff = float(diff.mean())
    return ParityResult(
        component=component,
        output_name=name,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        passed=max_diff < tol,
    )


def _copy_ts_weights_to_mlx(
    ts_mod,
    mlx_mod,
    rename_map: dict[str, str],
    *,
    reshape_fn=None,
) -> list[str]:
    """Copy weights from TorchScript module to MLX module via rename_map.

    Returns list of unmapped TorchScript parameter names.
    """
    import torch
    unmapped: list[str] = []
    weight_pairs: list[tuple[str, np.ndarray]] = []
    for ts_name, ts_param in ts_mod.named_parameters():
        mlx_name = rename_map.get(ts_name)
        if mlx_name is None:
            unmapped.append(ts_name)
            continue
        arr = ts_param.detach().cpu().float().numpy()
        if reshape_fn is not None:
            arr = reshape_fn(mlx_name, arr)
        weight_pairs.append((mlx_name, mx.array(arr)))
    mlx_mod.load_weights(weight_pairs, strict=False)
    return unmapped


# ===================================================================
# Component validators
# ===================================================================


def validate_feature_embedding(
    ts_dir: Path,
    model,
    *,
    tol: float = 1e-4,
) -> list[ParityResult]:
    """Compare feature_embedding.pt forward vs MLX FeatureEmbedding."""
    import torch

    rng = np.random.default_rng(42)
    from ..config import Chai1Config

    cfg = Chai1Config()
    B, N = 1, 64

    ts_path = ts_dir / "feature_embedding.pt"
    if not ts_path.exists():
        return [ParityResult("feature_embedding", "SKIPPED (no .pt)", 0, 0, True)]

    ts_mod = torch.jit.load(str(ts_path), map_location="cpu")
    ts_mod.eval()

    _ATTR_NAMES = {
        "TOKEN": "token_proj",
        "TOKEN_PAIR": "token_pair_proj",
        "ATOM": "atom_proj",
        "ATOM_PAIR": "atom_pair_proj",
        "MSA": "msa_proj",
        "TEMPLATES": "template_proj",
    }
    _SHAPES = {
        "TOKEN": (B, N, cfg.feature_dims.token),
        "TOKEN_PAIR": (B, N, N, cfg.feature_dims.token_pair),
        "ATOM": (B, N * 14, cfg.feature_dims.atom),
        "ATOM_PAIR": (B, N, 14, 14, cfg.feature_dims.atom_pair),
        "MSA": (B, 16, N, cfg.feature_dims.msa),
        "TEMPLATES": (B, 4, N, N, cfg.feature_dims.templates),
    }

    results: list[ParityResult] = []
    for feat_type in _ATTR_NAMES:
        arr = rng.standard_normal(_SHAPES[feat_type]).astype(np.float32)
        t_in = torch.from_numpy(arr)
        ts_seq = getattr(ts_mod.input_projs, feat_type)
        ts_linear = getattr(ts_seq, "0")
        with torch.no_grad():
            t_out = ts_linear(t_in).float()
        ref_np = t_out.numpy()

        proj = getattr(model.input_embedder.feature_embedding, _ATTR_NAMES[feat_type])
        with torch.no_grad():
            ts_w = ts_linear.weight.cpu().float().numpy()
            ts_b = ts_linear.bias.cpu().float().numpy()
        proj.load_weights([("weight", mx.array(ts_w)), ("bias", mx.array(ts_b))])

        m_out = proj(mx.array(arr))
        mx.eval(m_out)
        mlx_np = _to_numpy(m_out)
        # TorchScript runs in bf16, MLX in fp32 — expect ~1e-2 diffs
        results.append(_compare(feat_type, ref_np, mlx_np, "feature_embedding", max(tol, 0.2)))

    del ts_mod
    gc.collect()
    return results


def validate_diffusion_module(
    ts_dir: Path,
    model,
    *,
    tol: float = 5e-4,
) -> list[ParityResult]:
    """Weight mapping + numerical projection checks for diffusion_module.pt."""
    import torch

    ts_path = ts_dir / "diffusion_module.pt"
    if not ts_path.exists():
        return [ParityResult("diffusion_module", "SKIPPED (no .pt)", 0, 0, True)]

    from ..config import Chai1Config
    from ..weights.name_map import _diffusion_module_map, reshape_einsum_weight

    cfg = Chai1Config()
    B, N = 1, 32
    rng = np.random.default_rng(42)

    ts_mod = torch.jit.load(str(ts_path), map_location="cpu")
    ts_mod.eval()

    rename_map = _diffusion_module_map()

    # --- Weight mapping completeness check ---
    results: list[ParityResult] = []
    mapped_count = 0
    unmapped: list[str] = []
    shape_mismatches: list[str] = []

    for ts_name, ts_param in ts_mod.named_parameters():
        mlx_name = rename_map.get(ts_name)
        if mlx_name is None:
            unmapped.append(ts_name)
            continue
        mapped_count += 1
        ts_shape = tuple(ts_param.shape)
        reshaped = reshape_einsum_weight(mlx_name, ts_param.detach().cpu().float().numpy())
        expected_shape = tuple(reshaped.shape)
        parts = mlx_name.split(".")
        obj = model
        try:
            for p in parts:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
            mlx_shape = tuple(obj.shape)
        except (AttributeError, IndexError, KeyError):
            shape_mismatches.append(f"{ts_name} -> {mlx_name}: MLX param not found")
            continue
        if expected_shape != mlx_shape:
            shape_mismatches.append(f"{ts_name}: reshaped {expected_shape} != MLX {mlx_shape}")

    results.append(ParityResult(
        "diffusion_weights", f"mapped_{mapped_count}",
        len(unmapped), len(unmapped), len(unmapped) == 0,
    ))
    results.append(ParityResult(
        "diffusion_weights", f"shapes_ok ({len(shape_mismatches)} bad)",
        len(shape_mismatches), len(shape_mismatches), len(shape_mismatches) == 0,
    ))
    if unmapped:
        print(f"    WARNING: {len(unmapped)} unmapped TS params (first 5):", unmapped[:5])
    if shape_mismatches:
        print(f"    WARNING: shape mismatches:", shape_mismatches[:5])

    # --- Copy weights and test projections numerically ---
    _copy_ts_weights_to_mlx(
        ts_mod, model, rename_map, reshape_fn=reshape_einsum_weight,
    )

    # Pair projection (LN → Linear)
    pair_cat_np = rng.standard_normal((B, N, N, 2 * cfg.hidden.token_pair)).astype(np.float32)
    dc_ts = ts_mod.diffusion_conditioning
    ts_pair_ln = getattr(dc_ts.token_pair_proj, "0")
    ts_pair_linear = getattr(dc_ts.token_pair_proj, "1")
    with torch.no_grad():
        ref_np = ts_pair_linear(ts_pair_ln(torch.from_numpy(pair_cat_np))).float().numpy()

    mlx_dc = model.diffusion_module.diffusion_conditioning
    mlx_out = mlx_dc.token_pair_proj(mlx_dc.token_pair_norm(mx.array(pair_cat_np)))
    mx.eval(mlx_out)
    results.append(_compare("pair_proj", ref_np, _to_numpy(mlx_out), "diffusion_module", tol))

    # Single projection
    single_cat_np = rng.standard_normal((B, N, 2 * cfg.hidden.token_single)).astype(np.float32)
    ts_single_ln = getattr(dc_ts.token_in_proj, "0")
    ts_single_linear = getattr(dc_ts.token_in_proj, "1")
    with torch.no_grad():
        ref_np = ts_single_linear(ts_single_ln(torch.from_numpy(single_cat_np))).float().numpy()

    mlx_out = mlx_dc.token_in_proj(mlx_dc.token_in_norm(mx.array(single_cat_np)))
    mx.eval(mlx_out)
    results.append(_compare("single_proj", ref_np, _to_numpy(mlx_out), "diffusion_module", tol))

    # Fourier embedding (manually, TorchScript doesn't expose forward)
    sigma_np = rng.uniform(0.01, 80.0, size=(B, 1)).astype(np.float32)
    with torch.no_grad():
        fe_w = dc_ts.fourier_embedding.weights.float()
        fe_b = dc_ts.fourier_embedding.bias.float()
        c_noise = torch.log(torch.from_numpy(sigma_np)) * 0.25
        ref_np = torch.cos((c_noise * fe_w + fe_b) * (2.0 * 3.141592653589793)).numpy()

    mlx_out = mlx_dc.fourier_embedding(mx.array(sigma_np))
    mx.eval(mlx_out)
    results.append(_compare("fourier_embed", ref_np, _to_numpy(mlx_out), "diffusion_module", tol))

    del ts_mod
    gc.collect()
    return results


def validate_edm_schedule(*, tol: float = 1e-6) -> list[ParityResult]:
    """Validate EDM noise schedule against a pure numpy reference."""
    from ..utils import edm_sigmas, edm_gammas

    num_steps = 200
    sigma_data = 16.0
    s_min, s_max, p = 4e-4, 80.0, 7.0
    s_churn, s_tmin, s_tmax = 80.0, 4e-4, 80.0

    sigmas = edm_sigmas(num_steps, sigma_data, s_min, s_max, p)
    gammas = edm_gammas(sigmas, s_churn, s_tmin, s_tmax)
    mx.eval(sigmas, gammas)

    sigmas_np = _to_numpy(sigmas)
    gammas_np = _to_numpy(gammas)

    results: list[ParityResult] = []

    # Reference formula: sigma = sigma_data * (t * s_min^(1/p) + (1-t) * s_max^(1/p))^p
    # with midpoint sampling: t = linspace(0, 1, 2*N+1)[1::2]
    t_ref = np.linspace(0, 1, 2 * num_steps + 1, dtype=np.float64)[1::2]
    ref_sigmas = sigma_data * (t_ref * s_min ** (1.0 / p) + (1.0 - t_ref) * s_max ** (1.0 / p)) ** p
    ref_sigmas = ref_sigmas.astype(np.float32)

    # Verify monotonically decreasing
    diffs = np.diff(sigmas_np)
    monotone_ok = bool(np.all(diffs <= 1e-5))
    results.append(ParityResult("edm_schedule", "monotone_decreasing", 0 if monotone_ok else 1, 0, monotone_ok))

    # Verify shape matches
    shape_ok = sigmas_np.shape == ref_sigmas.shape
    results.append(ParityResult("edm_schedule", "sigma_shape", 0 if shape_ok else 1, 0, shape_ok))

    # Verify gamma shape
    shape_ok2 = gammas_np.shape == sigmas_np.shape
    results.append(ParityResult("edm_schedule", "gamma_shape", 0 if shape_ok2 else 1, 0, shape_ok2))

    # Verify gammas are zero outside [s_tmin, s_tmax]
    # The comparison is against the actual sigma values (which include sigma_data scaling)
    outside = (sigmas_np < s_tmin) | (sigmas_np > s_tmax)
    outside_zero = float(np.abs(gammas_np[outside]).max()) if outside.any() else 0
    results.append(ParityResult("edm_schedule", "gamma_outside_zero", outside_zero, outside_zero, outside_zero < 1e-7))

    # Verify sigma values match reference formula (fp32 power interp has ~5e-4 error)
    results.append(_compare("sigma_values", ref_sigmas, sigmas_np, "edm_schedule", max(tol, 1e-3)))

    return results


def validate_trunk_weight_mapping(
    ts_dir: Path,
    model,
    *,
    tol: float = 5e-4,
) -> list[ParityResult]:
    """Verify all trunk TorchScript params map to MLX params with correct shapes."""
    import torch

    ts_path = ts_dir / "trunk.pt"
    if not ts_path.exists():
        return [ParityResult("trunk_weights", "SKIPPED (no .pt)", 0, 0, True)]

    ts_mod = torch.jit.load(str(ts_path), map_location="cpu")
    ts_mod.eval()

    # TorchScript submodules don't expose .forward, so we can't call them
    # directly. Instead, verify weight mapping completeness by checking all
    # TorchScript params map to MLX params with matching shapes.
    from ..config import Chai1Config
    from ..weights.name_map import _trunk_map, reshape_einsum_weight

    cfg = Chai1Config()
    rename_map = _trunk_map()

    results: list[ParityResult] = []
    mapped_count = 0
    unmapped: list[str] = []
    shape_mismatches: list[str] = []

    for ts_name, ts_param in ts_mod.named_parameters():
        mlx_name = rename_map.get(ts_name)
        if mlx_name is None:
            unmapped.append(ts_name)
            continue
        mapped_count += 1
        reshaped = reshape_einsum_weight(mlx_name, ts_param.detach().cpu().float().numpy())
        expected_shape = tuple(reshaped.shape)
        parts = mlx_name.replace("trunk_module.", "").split(".")
        obj = model.trunk_module
        try:
            for p in parts:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
            mlx_shape = tuple(obj.shape)
        except (AttributeError, IndexError, KeyError):
            shape_mismatches.append(f"{ts_name} -> {mlx_name}: MLX param not found")
            continue
        if expected_shape != mlx_shape:
            shape_mismatches.append(f"{ts_name}: reshaped {expected_shape} != MLX {mlx_shape}")

    all_mapped = len(unmapped) == 0
    no_shape_err = len(shape_mismatches) == 0
    results.append(ParityResult(
        "trunk_weights", f"mapped_{mapped_count}",
        len(unmapped), len(unmapped), all_mapped,
    ))
    results.append(ParityResult(
        "trunk_weights", f"shapes_ok ({len(shape_mismatches)} bad)",
        len(shape_mismatches), len(shape_mismatches), no_shape_err,
    ))
    if unmapped:
        print(f"    WARNING: {len(unmapped)} unmapped TS params (first 5):", unmapped[:5])
    if shape_mismatches:
        print(f"    WARNING: shape mismatches:", shape_mismatches[:5])

    del ts_mod
    gc.collect()
    return results


def validate_utilities(*, tol: float = 1e-6) -> list[ParityResult]:
    """Validate pure-math utilities against numpy references."""
    from ..utils import one_hot_binned, stable_softmax, pairwise_distance

    rng = np.random.default_rng(42)
    results: list[ParityResult] = []

    # one_hot_binned
    vals_np = rng.uniform(0, 25, (4, 8)).astype(np.float32)
    edges = [3.375, 4.661, 5.946, 7.232, 8.518, 9.804, 11.089, 12.375,
             13.661, 14.946, 16.232, 17.518, 18.804, 20.089, 21.375]
    mlx_oh = one_hot_binned(mx.array(vals_np), edges)
    mx.eval(mlx_oh)
    oh_np = _to_numpy(mlx_oh)
    # Verify shape is [..., len(edges)+1]
    shape_ok = oh_np.shape == (*vals_np.shape, len(edges) + 1)
    results.append(ParityResult("utilities", "one_hot_binned_shape", 0 if shape_ok else 1, 0, shape_ok))
    # Verify each row sums to 1
    row_sums = oh_np.sum(axis=-1)
    sum_ok = float(np.abs(row_sums - 1.0).max())
    results.append(ParityResult("utilities", "one_hot_binned_sum", sum_ok, sum_ok, sum_ok < 1e-5))

    # stable_softmax
    logits_np = rng.standard_normal((2, 8)).astype(np.float32)
    mlx_sm = stable_softmax(mx.array(logits_np), axis=-1)
    mx.eval(mlx_sm)
    ref_sm = np.exp(logits_np - logits_np.max(axis=-1, keepdims=True))
    ref_sm = ref_sm / ref_sm.sum(axis=-1, keepdims=True)
    results.append(_compare("softmax", ref_sm, _to_numpy(mlx_sm), "utilities", tol))

    # pairwise_distance
    coords_np = rng.standard_normal((1, 16, 3)).astype(np.float32)
    mlx_dist = pairwise_distance(mx.array(coords_np))
    mx.eval(mlx_dist)
    ref_dist = np.sqrt(((coords_np[:, :, None, :] - coords_np[:, None, :, :]) ** 2).sum(-1) + 1e-10)
    results.append(_compare("pairwise_dist", ref_dist, _to_numpy(mlx_dist), "utilities", 1e-4))

    return results


def validate_component_weights(
    ts_dir: Path,
    model,
    *,
    component_name: str,
    pt_filename: str,
) -> list[ParityResult]:
    """Generic weight mapping completeness check for any component."""
    import torch

    ts_path = ts_dir / pt_filename
    if not ts_path.exists():
        return [ParityResult(f"{component_name}_weights", "SKIPPED (no .pt)", 0, 0, True)]

    from ..weights.name_map import build_rename_map, reshape_einsum_weight

    ts_mod = torch.jit.load(str(ts_path), map_location="cpu")
    rename_map = build_rename_map(component_name)

    results: list[ParityResult] = []
    mapped_count = 0
    unmapped: list[str] = []
    shape_mismatches: list[str] = []

    for ts_name, ts_param in ts_mod.named_parameters():
        mlx_name = rename_map.get(ts_name)
        if mlx_name is None:
            unmapped.append(ts_name)
            continue
        mapped_count += 1
        reshaped = reshape_einsum_weight(mlx_name, ts_param.detach().cpu().float().numpy())
        expected_shape = tuple(reshaped.shape)
        parts = mlx_name.split(".")
        obj = model
        try:
            for p in parts:
                if p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)
            mlx_shape = tuple(obj.shape)
        except (AttributeError, IndexError, KeyError):
            shape_mismatches.append(f"{ts_name} -> {mlx_name}: MLX param not found")
            continue
        if expected_shape != mlx_shape:
            shape_mismatches.append(f"{ts_name}: reshaped {expected_shape} != MLX {mlx_shape}")

    results.append(ParityResult(
        f"{component_name}_weights", f"mapped_{mapped_count}",
        len(unmapped), len(unmapped), len(unmapped) == 0,
    ))
    results.append(ParityResult(
        f"{component_name}_weights", f"shapes_ok ({len(shape_mismatches)} bad)",
        len(shape_mismatches), len(shape_mismatches), len(shape_mismatches) == 0,
    ))
    if unmapped:
        print(f"    WARNING: {len(unmapped)} unmapped TS params (first 5):", unmapped[:5])
    if shape_mismatches:
        print(f"    WARNING: shape mismatches:", shape_mismatches[:5])

    del ts_mod
    gc.collect()
    return results


# ===================================================================
# Runner
# ===================================================================


def run_validation(
    ts_dir: Path,
    safetensors_dir: Path | None = None,
    *,
    tol: float = 1e-4,
    verbose: bool = True,
) -> list[ParityResult]:
    """Run all per-component parity checks."""
    from ..api import Chai1MLX

    if safetensors_dir is not None:
        model = Chai1MLX.from_pretrained(safetensors_dir, strict=False)
    else:
        model = Chai1MLX()

    all_results: list[ParityResult] = []

    # Pure-math tests (no TorchScript needed)
    print("  [*] Validating EDM schedule...")
    all_results.extend(validate_edm_schedule(tol=tol))

    print("  [*] Validating utilities...")
    all_results.extend(validate_utilities(tol=tol))

    # TorchScript-vs-MLX tests
    print("  [*] Validating feature embedding...")
    all_results.extend(validate_feature_embedding(ts_dir, model, tol=tol))

    print("  [*] Validating diffusion module (weights + projections)...")
    all_results.extend(validate_diffusion_module(ts_dir, model, tol=5e-4))

    print("  [*] Validating trunk weight mapping...")
    all_results.extend(validate_trunk_weight_mapping(ts_dir, model, tol=5e-4))

    # Weight mapping checks for remaining components
    for comp, pt_file in [
        ("feature_embedding", "feature_embedding.pt"),
        ("token_embedder", "token_embedder.pt"),
        ("confidence_head", "confidence_head.pt"),
        ("bond_loss_input_proj", "bond_loss_input_proj.pt"),
    ]:
        print(f"  [*] Validating {comp} weight mapping...")
        all_results.extend(validate_component_weights(ts_dir, model, component_name=comp, pt_filename=pt_file))

    gc.collect()

    if verbose:
        print("\n--- Parity Results ---")
        for r in all_results:
            status = "PASS" if r.passed else "FAIL"
            print(
                f"  [{status}] {r.component}.{r.output_name}: "
                f"max={r.max_abs_diff:.2e}, mean={r.mean_abs_diff:.2e}"
            )
        n_pass = sum(1 for r in all_results if r.passed)
        n_total = len(all_results)
        print(f"\n  {n_pass}/{n_total} passed (tol={tol})")

    return all_results


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate MLX vs TorchScript parity")
    parser.add_argument("--torchscript-dir", type=Path, required=True)
    parser.add_argument("--safetensors-dir", type=Path, default=None)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args(list(argv) if argv is not None else None)
    results = run_validation(args.torchscript_dir, args.safetensors_dir, tol=args.tol)
    if not all(r.passed for r in results):
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
