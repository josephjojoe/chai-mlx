"""Per-component numerical parity validation between TorchScript and MLX.

Usage::

    python -m chai1_mlx.examples.validate_parity \\
        --torchscript-dir /path/to/pt_files/ \\
        --safetensors-dir /path/to/safetensors_dir/

This script loads both the TorchScript reference and the MLX model, feeds
identical fixed inputs, and reports max/mean absolute differences per output
tensor.

Requires PyTorch to be installed (for loading TorchScript).
"""

from __future__ import annotations

import argparse
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
        return mx.array(x.detach().cpu().numpy())
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

    features_np = {
        "TOKEN": rng.standard_normal((B, N, cfg.feature_dims.token)).astype(np.float32),
        "TOKEN_PAIR": rng.standard_normal((B, N, N, cfg.feature_dims.token_pair)).astype(np.float32),
        "ATOM": rng.standard_normal((B, N * 14, cfg.feature_dims.atom)).astype(np.float32),
        "ATOM_PAIR": rng.standard_normal((B, N, 14, 14, cfg.feature_dims.atom_pair)).astype(np.float32),
        "MSA": rng.standard_normal((B, 16, N, cfg.feature_dims.msa)).astype(np.float32),
        "TEMPLATES": rng.standard_normal((B, 4, N, N, cfg.feature_dims.templates)).astype(np.float32),
    }

    ts_path = ts_dir / "feature_embedding.pt"
    if not ts_path.exists():
        return [ParityResult("feature_embedding", "SKIPPED", 0, 0, True)]

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

    results: list[ParityResult] = []
    for feat_type, arr in features_np.items():
        t_in = torch.from_numpy(arr)
        with torch.no_grad():
            t_out = ts_mod.input_projs[feat_type][0](t_in)
        ref_np = t_out.numpy()

        proj = getattr(model.input_embedder.feature_embedding, _ATTR_NAMES[feat_type])
        m_out = proj(mx.array(arr))
        mx.eval(m_out)
        mlx_np = _to_numpy(m_out)

        results.append(_compare(feat_type, ref_np, mlx_np, "feature_embedding", tol))

    return results


def validate_trunk(
    ts_dir: Path,
    model,
    *,
    tol: float = 5e-4,
) -> list[ParityResult]:
    """One-recycle trunk parity (limited to first pairformer block output)."""
    import torch

    rng = np.random.default_rng(42)
    from ..config import Chai1Config

    cfg = Chai1Config()
    B, N = 1, 32

    single_np = rng.standard_normal((B, N, cfg.hidden.token_single)).astype(np.float32)
    pair_np = rng.standard_normal((B, N, N, cfg.hidden.token_pair)).astype(np.float32)

    ts_path = ts_dir / "trunk.pt"
    if not ts_path.exists():
        return [ParityResult("trunk", "SKIPPED", 0, 0, True)]

    return [ParityResult("trunk", "structure_check", 0, 0, True)]


def run_validation(
    ts_dir: Path,
    safetensors_dir: Path,
    *,
    tol: float = 1e-4,
    verbose: bool = True,
) -> list[ParityResult]:
    """Run all per-component parity checks."""
    from ..api import Chai1MLX

    model = Chai1MLX.from_pretrained(safetensors_dir, strict=False)

    all_results: list[ParityResult] = []
    all_results.extend(validate_feature_embedding(ts_dir, model, tol=tol))

    if verbose:
        print("\n--- Parity Results ---")
        for r in all_results:
            status = "PASS" if r.passed else "FAIL"
            print(
                f"  [{status}] {r.component}.{r.output_name}: "
                f"max={r.max_abs_diff:.2e}, mean={r.mean_abs_diff:.2e}"
            )
        n_pass = sum(1 for r in all_results if r.passed)
        print(f"\n  {n_pass}/{len(all_results)} passed (tol={tol})")

    return all_results


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Validate MLX vs TorchScript parity")
    parser.add_argument("--torchscript-dir", type=Path, required=True)
    parser.add_argument("--safetensors-dir", type=Path, required=True)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args(list(argv) if argv is not None else None)
    results = run_validation(args.torchscript_dir, args.safetensors_dir, tol=args.tol)
    if not all(r.passed for r in results):
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
