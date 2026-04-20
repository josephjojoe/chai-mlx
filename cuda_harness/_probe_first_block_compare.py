"""Locally compare MLX block-0 vs TorchScript / eager PyTorch block-0 runs.

Consumes the NPZs written by ``_probe_first_block_mlx.py`` and
``_probe_first_block_ts_cuda.py`` and prints per-tensor / per-sub-op
error summaries for every (mlx, reference) pair.

Usage::

    python3 cuda_harness/_probe_first_block_compare.py

The pairs compared (higher-truth reference first, then progressively
more realistic implementations):

    mlx_fp32 vs eager_fp64     — "is MLX's summation order correct?"
    mlx_bf16 vs eager_fp64     — "plus: how much does MLX bf16 cost us?"
    eager_fp32 vs eager_fp64   — "plus: baseline eager bf16 vs fp64 floor"
    eager_bf16 vs eager_fp64   — "plus: baseline eager bf16 vs fp64 floor"
    ts_bf16    vs eager_fp64   — "CUDA scripted vs fp64 oracle"
    mlx_bf16   vs ts_bf16      — "the headline MLX vs CUDA gap"
    mlx_fp32   vs ts_bf16      — "MLX fp32 vs CUDA bf16 (diagnostic)"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_DIR = Path("/tmp/chai_mlx_cuda/first_block_probe")


# Ordered sub-ops inside the block we want to report on.
_KEYS = [
    "pair_transition_out",
    "z_after_tri_mult",
    "z_after_residual",
    "z_after_tri_attn",
    "attn_delta",
    "s_after_attn",
    "s_after_transition",
    "z_final",
    "s_final",
]


@dataclass
class ErrorStats:
    max_abs: float
    mean_abs: float
    p99_abs: float
    ref_range: float
    rel: float
    ref_norm: float
    cand_norm: float


def _stats(cand: np.ndarray, ref: np.ndarray) -> ErrorStats:
    cand = cand.astype(np.float64)
    ref = ref.astype(np.float64)
    diff = np.abs(cand - ref)
    ref_range = float(ref.max() - ref.min())
    rel = float(diff.max() / ref_range) if ref_range > 0 else float("inf")
    return ErrorStats(
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        p99_abs=float(np.percentile(diff, 99)),
        ref_range=ref_range,
        rel=rel,
        ref_norm=float(np.linalg.norm(ref)),
        cand_norm=float(np.linalg.norm(cand)),
    )


def _load(path: Path) -> dict[str, np.ndarray] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        data = np.load(path)
    except (EOFError, OSError, ValueError):
        return None
    return {k: data[k] for k in data.files}


def _compare_two(cand_name: str, cand: dict[str, np.ndarray], ref_name: str, ref: dict[str, np.ndarray]) -> dict:
    report: dict[str, dict] = {}
    print(f"\n=== {cand_name} vs {ref_name} ===")
    print(
        f"  {'key':26s} {'max_abs':>10s} {'mean':>10s} {'p99':>10s}"
        f" {'ref_range':>10s} {'rel':>10s} {'|cand|':>10s} {'|ref|':>10s}"
    )
    for k in _KEYS:
        if k not in cand or k not in ref:
            continue
        s = _stats(cand[k], ref[k])
        print(
            f"  {k:26s} {s.max_abs:10.4e} {s.mean_abs:10.4e} {s.p99_abs:10.4e}"
            f" {s.ref_range:10.4f} {s.rel:10.4e} {s.cand_norm:10.4f} {s.ref_norm:10.4f}"
        )
        report[k] = {
            "max_abs": s.max_abs,
            "mean_abs": s.mean_abs,
            "p99_abs": s.p99_abs,
            "ref_range": s.ref_range,
            "rel": s.rel,
            "cand_norm": s.cand_norm,
            "ref_norm": s.ref_norm,
        }
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--json", type=Path, default=None, help="optional path to write merged summary")
    args = p.parse_args()

    payloads: dict[str, dict[str, np.ndarray] | None] = {
        "mlx_fp32": _load(args.dir / "mlx_out_fp32.npz"),
        "mlx_bf16": _load(args.dir / "mlx_out_bf16.npz"),
        "eager_fp64": _load(args.dir / "cuda_out_eager_fp64.npz"),
        "eager_fp32": _load(args.dir / "cuda_out_eager_fp32.npz"),
        "eager_bf16": _load(args.dir / "cuda_out_eager_bf16.npz"),
        "ts_fp32": _load(args.dir / "cuda_out_ts_fp32.npz"),
        "ts_bf16": _load(args.dir / "cuda_out_ts_bf16.npz"),
    }

    print("Files loaded:")
    for k, v in payloads.items():
        status = "ok" if v is not None else "MISSING"
        print(f"  {k:15s} {status}")

    pairs = [
        ("mlx_fp32", "eager_fp64"),
        ("mlx_bf16", "eager_fp64"),
        ("eager_fp32", "eager_fp64"),
        ("eager_bf16", "eager_fp64"),
        ("ts_fp32", "eager_fp64"),
        ("ts_bf16", "eager_fp64"),
        ("mlx_bf16", "ts_bf16"),
        ("mlx_fp32", "ts_fp32"),
        ("mlx_bf16", "eager_bf16"),
        ("mlx_fp32", "eager_fp32"),
    ]

    summary: dict[str, dict] = {}
    for cand_name, ref_name in pairs:
        cand = payloads.get(cand_name)
        ref = payloads.get(ref_name)
        if cand is None or ref is None:
            print(f"\n=== {cand_name} vs {ref_name} ===  SKIP (missing side)")
            continue
        summary[f"{cand_name}_vs_{ref_name}"] = _compare_two(cand_name, cand, ref_name, ref)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
