"""Per-op microbenchmark for the fused Metal kernels.

Measures AdaLayerNorm / SwiGLU / ConditionedTransition (which combines AdaLN +
SwiGLU + gated residual) at the exact shapes used by the Chai-1 diffusion
transformer and atom transformer, comparing ``use_kernel=False`` vs
``use_kernel=True``.

Run::

    python3 scripts/kernel_microbench.py
    python3 scripts/kernel_microbench.py --sizes 256 512 1024 --num-samples 1
    python3 scripts/kernel_microbench.py --dtype float32 --iters 200

The output is a JSON file (``/tmp/kernel_microbench.json`` by default)
containing per-op timings and the speedup ratio, plus a human-readable table.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from chai_mlx.config import ChaiConfig
from chai_mlx.nn.layers.common import AdaLayerNorm, ConditionedTransition, SwiGLU


DTYPES = {"bfloat16": mx.bfloat16, "float32": mx.float32}


@dataclass
class BenchResult:
    name: str
    shape: tuple[int, ...]
    cond_shape: tuple[int, ...] | None
    dtype: str
    iters: int
    ms_ref: float
    ms_kern: float

    @property
    def speedup(self) -> float:
        return self.ms_ref / self.ms_kern if self.ms_kern > 0 else float("inf")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "cond_shape": None if self.cond_shape is None else list(self.cond_shape),
            "dtype": self.dtype,
            "iters": self.iters,
            "ms_ref": round(self.ms_ref, 4),
            "ms_kernel": round(self.ms_kern, 4),
            "speedup": round(self.speedup, 3),
        }


def _time_op(fn, *, warmup: int, iters: int) -> float:
    """Return milliseconds per iteration (median of `iters` runs after warmup)."""
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn()
        mx.eval(out)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    mid = len(samples) // 2
    return samples[mid]


def bench_adaln(
    B: int, N: int, D: int, CD: int, *, dtype: mx.Dtype, iters: int, warmup: int
) -> BenchResult:
    mx.random.seed(0)
    x = mx.random.normal((B, N, D)).astype(dtype)
    cond = mx.random.normal((B, N, CD)).astype(dtype)
    layer = AdaLayerNorm(D, CD)
    _ = layer(x, cond)
    mx.eval(layer.parameters())

    ms_ref = _time_op(lambda: layer(x, cond, use_kernel=False), warmup=warmup, iters=iters)
    ms_kern = _time_op(lambda: layer(x, cond, use_kernel=True), warmup=warmup, iters=iters)
    return BenchResult(
        name="AdaLayerNorm",
        shape=(B, N, D),
        cond_shape=(B, N, CD),
        dtype=str(dtype),
        iters=iters,
        ms_ref=ms_ref,
        ms_kern=ms_kern,
    )


def bench_swiglu(
    B: int, N: int, inner: int, *, dtype: mx.Dtype, iters: int, warmup: int
) -> BenchResult:
    """SwiGLU takes [..., 2*inner] -> [..., inner]."""
    mx.random.seed(0)
    u = mx.random.normal((B, N, 2 * inner)).astype(dtype)
    layer = SwiGLU()

    ms_ref = _time_op(lambda: layer(u, use_kernel=False), warmup=warmup, iters=iters)
    ms_kern = _time_op(lambda: layer(u, use_kernel=True), warmup=warmup, iters=iters)
    return BenchResult(
        name="SwiGLU",
        shape=(B, N, 2 * inner),
        cond_shape=None,
        dtype=str(dtype),
        iters=iters,
        ms_ref=ms_ref,
        ms_kern=ms_kern,
    )


def bench_conditioned_transition(
    B: int, N: int, D: int, CD: int, expansion: int, *, dtype: mx.Dtype, iters: int, warmup: int
) -> BenchResult:
    """Fused block: AdaLN + up + SwiGLU + down + gated residual."""
    mx.random.seed(0)
    x = mx.random.normal((B, N, D)).astype(dtype)
    cond = mx.random.normal((B, N, CD)).astype(dtype)
    layer = ConditionedTransition(D, CD, expansion=expansion)
    _ = layer(x, cond)
    mx.eval(layer.parameters())

    ms_ref = _time_op(lambda: layer(x, cond, use_kernel=False), warmup=warmup, iters=iters)
    ms_kern = _time_op(lambda: layer(x, cond, use_kernel=True), warmup=warmup, iters=iters)
    return BenchResult(
        name=f"ConditionedTransition(exp={expansion})",
        shape=(B, N, D),
        cond_shape=(B, N, CD),
        dtype=str(dtype),
        iters=iters,
        ms_ref=ms_ref,
        ms_kern=ms_kern,
    )


def run_suite(n_tokens: int, num_samples: int, dtype: mx.Dtype, iters: int, warmup: int) -> list[BenchResult]:
    cfg = ChaiConfig()
    atom_mult = cfg.atom_blocks.atom_multiplier  # 23
    n_atoms = atom_mult * n_tokens
    # Diffusion uses a flattened (B * num_samples) first dim.
    B_flat = num_samples

    D_tok = cfg.hidden.diffusion      # 768
    CD_tok = cfg.hidden.token_single  # 384
    D_atom = cfg.hidden.atom_single   # 128
    CD_atom = cfg.hidden.atom_single  # 128

    print(
        f"\n[n_tokens={n_tokens}, n_atoms={n_atoms}, num_samples={num_samples}, "
        f"dtype={dtype}, iters={iters}]"
    )

    results: list[BenchResult] = []

    # === Diffusion transformer (token-level) ===
    results.append(bench_adaln(
        B_flat, n_tokens, D_tok, CD_tok, dtype=dtype, iters=iters, warmup=warmup,
    ))
    # Transition uses expansion=2 -> inner = 2 * 768 = 1536
    results.append(bench_swiglu(
        B_flat, n_tokens, 2 * D_tok, dtype=dtype, iters=iters, warmup=warmup,
    ))
    results.append(bench_conditioned_transition(
        B_flat, n_tokens, D_tok, CD_tok, 2,
        dtype=dtype, iters=iters, warmup=warmup,
    ))

    # === Atom transformer (atom-level) ===
    results.append(bench_adaln(
        B_flat, n_atoms, D_atom, CD_atom, dtype=dtype, iters=iters, warmup=warmup,
    ))
    results.append(bench_swiglu(
        B_flat, n_atoms, 2 * D_atom, dtype=dtype, iters=iters, warmup=warmup,
    ))
    results.append(bench_conditioned_transition(
        B_flat, n_atoms, D_atom, CD_atom, 2,
        dtype=dtype, iters=iters, warmup=warmup,
    ))

    # Per-call counts in one diffusion step (from reading the code):
    #   Diffusion transformer: 16 blocks, each has AdaLN x2 + SwiGLU x1 + gated residual x1
    #   Atom encoder: 3 blocks, same pattern. Atom decoder: 3 blocks, same.
    # => 16 "ConditionedTransition" at token shape + 16 "AdaLN (attn)" at token shape
    #    +  6 "ConditionedTransition" at atom shape + 6 "AdaLN (attn)" at atom shape
    # Each denoise call is invoked either 1x or 2x per diffusion step (second_order).
    return results


def pretty_print(results: list[BenchResult]) -> None:
    print(f"\n  {'op':<32} {'shape':>22} {'ref(ms)':>10} {'kern(ms)':>10} {'speedup':>9}")
    for r in results:
        shape_str = "x".join(str(s) for s in r.shape)
        print(
            f"  {r.name:<32} {shape_str:>22} "
            f"{r.ms_ref:>10.3f} {r.ms_kern:>10.3f} {r.speedup:>8.2f}x"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Fused-kernel microbench")
    p.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--num-samples", type=int, default=1,
                   help="Diffusion num_samples (flattened into batch)")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPES.keys()))
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", type=Path, default=Path("/tmp/kernel_microbench.json"))
    args = p.parse_args()

    dtype = DTYPES[args.dtype]
    all_results: list[dict] = []
    for n in args.sizes:
        results = run_suite(
            n_tokens=n,
            num_samples=args.num_samples,
            dtype=dtype,
            iters=args.iters,
            warmup=args.warmup,
        )
        pretty_print(results)
        for r in results:
            d = r.to_dict()
            d["n_tokens"] = n
            d["num_samples"] = args.num_samples
            all_results.append(d)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\n  saved -> {args.out}")


if __name__ == "__main__":
    main()
