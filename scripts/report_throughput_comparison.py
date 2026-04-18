"""Combine CUDA and MLX per-module throughput into one side-by-side report.

Reads the JSON output of :mod:`scripts.mlx_throughput` and the
``<target>_throughput.json`` files produced by
:mod:`cuda_harness.bench_throughput`, then prints a Markdown table and
(optionally) writes a CSV + JSON summary.

Usage
-----

::

    python scripts/report_throughput_comparison.py \\
        --mlx-json /tmp/chai_mlx_cuda/throughput/mlx.json \\
        --cuda-dir /tmp/chai_mlx_cuda/throughput \\
        --csv /tmp/chai_mlx_cuda/throughput/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Row:
    target: str
    n_tokens: int
    gpu_name: str
    mlx_dtype: str
    feature_embedding_cuda_ms: float | None
    feature_embedding_mlx_ms: float | None
    trunk_recycle_cuda_ms: float | None
    trunk_recycle_mlx_ms: float | None
    diffusion_step_cuda_ms: float | None
    diffusion_step_mlx_ms: float | None
    diffusion_total_cuda_ms: float | None
    diffusion_total_mlx_ms: float | None
    confidence_cuda_ms: float | None
    confidence_mlx_ms: float | None
    end_to_end_cuda_ms: float | None
    end_to_end_mlx_ms: float | None

    def speedup(self, cuda: float | None, mlx: float | None) -> float | None:
        if cuda is None or mlx is None or cuda == 0:
            return None
        return mlx / cuda


def _load_mlx(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text())
    return {r["target"]: r for r in raw}


def _load_cuda(folder: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(folder.glob("*_throughput.json")):
        payload = json.loads(path.read_text())
        out[payload["target"]] = payload
    return out


def _mean(x: dict, key: str) -> float | None:
    return x["summary"][key]["mean"] if x else None


def _mlx_mean(x: dict, key: str) -> float | None:
    return x[key]["mean_ms"] if x else None


def _build_rows(mlx_map: dict[str, dict], cuda_map: dict[str, dict]) -> list[Row]:
    rows: list[Row] = []
    targets = sorted(set(mlx_map) | set(cuda_map))
    for target in targets:
        mlx = mlx_map.get(target)
        cuda = cuda_map.get(target)
        if mlx is None and cuda is None:
            continue
        gpu = cuda["gpu_name"] if cuda else "—"
        n_tokens = (mlx or cuda)["n_tokens"]
        dtype = mlx["dtype"] if mlx else "—"
        rows.append(
            Row(
                target=target,
                n_tokens=n_tokens,
                gpu_name=gpu,
                mlx_dtype=dtype,
                feature_embedding_cuda_ms=_mean(cuda, "feature_embedding_ms") if cuda else None,
                feature_embedding_mlx_ms=_mlx_mean(mlx, "feature_embedding_ms") if mlx else None,
                trunk_recycle_cuda_ms=_mean(cuda, "trunk_recycle_ms") if cuda else None,
                trunk_recycle_mlx_ms=_mlx_mean(mlx, "trunk_recycle_ms") if mlx else None,
                diffusion_step_cuda_ms=_mean(cuda, "diffusion_step_ms") if cuda else None,
                diffusion_step_mlx_ms=_mlx_mean(mlx, "diffusion_step_ms") if mlx else None,
                diffusion_total_cuda_ms=_mean(cuda, "diffusion_total_ms") if cuda else None,
                diffusion_total_mlx_ms=_mlx_mean(mlx, "diffusion_total_ms") if mlx else None,
                confidence_cuda_ms=_mean(cuda, "confidence_ms") if cuda else None,
                confidence_mlx_ms=_mlx_mean(mlx, "confidence_ms") if mlx else None,
                end_to_end_cuda_ms=_mean(cuda, "end_to_end_ms") if cuda else None,
                end_to_end_mlx_ms=_mlx_mean(mlx, "end_to_end_ms") if mlx else None,
            )
        )
    return rows


def _fmt_ms(x: float | None) -> str:
    return f"{x:8.1f}" if x is not None else "     —  "


def _fmt_ratio(mlx: float | None, cuda: float | None) -> str:
    if mlx is None or cuda is None or cuda == 0:
        return "    —"
    return f"{mlx / cuda:5.2f}×"


def _print_table(rows: list[Row]) -> None:
    print("\n" + "=" * 145)
    print("Per-module throughput (ms, mean across repeats)")
    print("=" * 145)
    header = (
        f"  {'target':<6} {'N':>4} {'gpu':<14}  "
        f"{'feat C/M':>18}  {'trunk/rec C/M':>20}  "
        f"{'diff/step C/M':>20}  {'diff_tot C/M':>20}  "
        f"{'conf C/M':>18}  {'e2e C/M':>20}"
    )
    print(header)
    print("  " + "-" * 141)
    for r in rows:
        line = (
            f"  {r.target:<6} {r.n_tokens:>4} {r.gpu_name[:14]:<14}  "
            f"{_fmt_ms(r.feature_embedding_cuda_ms)}/{_fmt_ms(r.feature_embedding_mlx_ms)}  "
            f"{_fmt_ms(r.trunk_recycle_cuda_ms)}/{_fmt_ms(r.trunk_recycle_mlx_ms)}  "
            f"{_fmt_ms(r.diffusion_step_cuda_ms)}/{_fmt_ms(r.diffusion_step_mlx_ms)}  "
            f"{_fmt_ms(r.diffusion_total_cuda_ms)}/{_fmt_ms(r.diffusion_total_mlx_ms)}  "
            f"{_fmt_ms(r.confidence_cuda_ms)}/{_fmt_ms(r.confidence_mlx_ms)}  "
            f"{_fmt_ms(r.end_to_end_cuda_ms)}/{_fmt_ms(r.end_to_end_mlx_ms)}"
        )
        print(line)

    print("\nMLX/CUDA ratio (>1 means MLX slower):")
    for r in rows:
        print(
            f"  {r.target:<6} N={r.n_tokens:<4}  "
            f"feat={_fmt_ratio(r.feature_embedding_mlx_ms, r.feature_embedding_cuda_ms)}  "
            f"trunk/rec={_fmt_ratio(r.trunk_recycle_mlx_ms, r.trunk_recycle_cuda_ms)}  "
            f"diff/step={_fmt_ratio(r.diffusion_step_mlx_ms, r.diffusion_step_cuda_ms)}  "
            f"diff_tot={_fmt_ratio(r.diffusion_total_mlx_ms, r.diffusion_total_cuda_ms)}  "
            f"conf={_fmt_ratio(r.confidence_mlx_ms, r.confidence_cuda_ms)}  "
            f"e2e={_fmt_ratio(r.end_to_end_mlx_ms, r.end_to_end_cuda_ms)}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mlx-json", type=Path, required=True)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    mlx_map = _load_mlx(args.mlx_json)
    cuda_map = _load_cuda(args.cuda_dir)
    rows = _build_rows(mlx_map, cuda_map)
    _print_table(rows)

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[f.name for f in Row.__dataclass_fields__.values()]
                if rows
                else [],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(
                    {k: v for k, v in r.__dict__.items()}
                )
        print(f"[save] csv -> {args.csv}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps([r.__dict__ for r in rows], indent=2, default=str)
        )
        print(f"[save] summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
