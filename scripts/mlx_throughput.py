"""Per-module MLX throughput benchmark on Apple Silicon.

This is the local MLX equivalent of
:mod:`cuda_harness.bench_throughput`, designed to produce directly
comparable numbers.  Each module is timed with proper warmup, and we
call ``mx.eval`` after each phase so the printed numbers are real
end-of-compute wall clock on the Metal backend.

Reported fields (per target):

* ``feature_embedding_ms`` — MLX ``InputEmbedder`` forward, including
  bond projection fusion.
* ``trunk_recycle_ms`` — mean per-recycle wall time for the trunk.
* ``diffusion_cache_ms`` — one-off cost of
  ``DiffusionModule.prepare_cache``.
* ``diffusion_step_ms`` — mean per-step wall time over the full
  ``num_steps`` roll-out, covering both the first-order and second-order
  denoise inside ``diffusion_step``.
* ``diffusion_total_ms`` — full roll-out wall time.
* ``confidence_ms`` — ``ConfidenceHead`` forward on the final coords.
* ``end_to_end_ms`` — sum of all the above.

The benchmark is deliberately single-process and uses the public
``ChaiMLX`` module API — the same one :class:`ChaiMLX.run_inference`
uses internally — so the numbers here reflect the production path.

Usage
-----

::

    python scripts/mlx_throughput.py \\
        --weights-dir weights \\
        --targets 1L2Y,1VII,1CRN,1UBQ \\
        --num-steps 200 \\
        --num-recycles 3 \\
        --num-repeats 3 \\
        --warmup 1 \\
        --dtype bfloat16 \\
        --output-json /tmp/chai_mlx_cuda/throughput/mlx.json

Then combine the Modal CUDA output and the MLX output with
:mod:`scripts.report_throughput_comparison`.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists():
    sys.path.insert(0, str(LOCAL_CHAI_LAB))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx

from chai_mlx import ChaiMLX
from chai_mlx.data.featurize import featurize_fasta

from cuda_harness.modal_common import DEFAULT_TARGETS


def _sync(*arrays: mx.array) -> None:
    """Force evaluation so timings capture real work, not lazy dispatch."""
    for arr in arrays:
        mx.eval(arr)


@dataclass
class ModuleTiming:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n: int


@dataclass
class TargetResult:
    target: str
    sequence: str
    n_tokens: int
    dtype: str
    num_recycles: int
    num_steps: int
    num_repeats: int
    warmup: int
    mlx_version: str
    feature_embedding_ms: ModuleTiming
    trunk_recycle_ms: ModuleTiming
    diffusion_cache_ms: ModuleTiming
    diffusion_step_ms: ModuleTiming
    diffusion_total_ms: ModuleTiming
    confidence_ms: ModuleTiming
    end_to_end_ms: ModuleTiming


def _summarize(xs: list[float]) -> ModuleTiming:
    if not xs:
        return ModuleTiming(mean_ms=0.0, std_ms=0.0, min_ms=0.0, max_ms=0.0, n=0)
    return ModuleTiming(
        mean_ms=float(statistics.fmean(xs)),
        std_ms=float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
        min_ms=float(min(xs)),
        max_ms=float(max(xs)),
        n=len(xs),
    )


def _bench_target(
    weights_dir: Path,
    target: str,
    sequence: str,
    *,
    num_recycles: int,
    num_steps: int,
    num_repeats: int,
    warmup: int,
    dtype: str,
    feature_dir: Path,
) -> TargetResult:
    feature_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = feature_dir / f"{target}.fasta"
    fasta_path.write_text(f">protein|name={target}\n{sequence}\n")

    print(f"[mlx] loading weights (dtype={dtype}) for {target}")
    model = ChaiMLX.from_pretrained(weights_dir, strict=False, compute_dtype=dtype)

    print(f"[mlx] featurizing {target} ({len(sequence)} residues)")
    ctx = featurize_fasta(
        fasta_path,
        output_dir=feature_dir / "mlx_features",
        use_esm_embeddings=False,
        use_msa_server=False,
        use_templates_server=False,
    )

    feat_times: list[float] = []
    trunk_times: list[float] = []
    cache_times: list[float] = []
    step_times: list[float] = []
    diff_total_times: list[float] = []
    confidence_times: list[float] = []
    e2e_times: list[float] = []

    total = num_repeats + warmup
    for trial in range(total):
        is_warmup = trial < warmup
        label = "warmup" if is_warmup else "repeat"
        print(f"[mlx] {target} trial {trial + 1}/{total} ({label})")
        mx.random.seed(42 + trial)
        mx.clear_cache()
        gc.collect()

        trial_t0 = time.perf_counter()
        t0 = time.perf_counter()
        emb = model.embed_inputs(ctx)
        _sync(
            emb.single_initial,
            emb.pair_initial,
            emb.token_pair_input,
            emb.token_pair_structure_input,
            emb.atom_single_input,
            emb.atom_single_structure_input,
            emb.atom_pair_input,
            emb.atom_pair_structure_input,
            emb.msa_input,
            emb.template_input,
            emb.single_structure,
        )
        feat_ms = (time.perf_counter() - t0) * 1000

        si = emb.structure_inputs
        single_init = emb.single_initial
        pair_init = emb.pair_initial
        prev_single = single_init
        prev_pair = pair_init
        per_recycle_ms: list[float] = []
        for _ in range(num_recycles):
            t0 = time.perf_counter()
            single = single_init + model.trunk_module.token_single_recycle_proj(prev_single)
            pair = pair_init + model.trunk_module.token_pair_recycle_proj(prev_pair)
            mx.eval(single, pair)
            pair = model.trunk_module.template_embedder(
                pair,
                emb.template_input,
                template_input_masks=si.template_input_masks,
                token_pair_mask=si.token_pair_mask,
            )
            mx.eval(pair)
            pair = model.trunk_module.msa_module(
                single,
                pair,
                emb.msa_input,
                token_pair_mask=si.token_pair_mask,
                msa_mask=si.msa_mask,
            )
            mx.eval(pair)
            single, pair = model.trunk_module.pairformer_stack(
                single, pair,
                pair_mask=si.token_pair_mask,
                single_mask=si.token_exists_mask,
            )
            mx.eval(single, pair)
            per_recycle_ms.append((time.perf_counter() - t0) * 1000)
            prev_single, prev_pair = single, pair

        from chai_mlx.data.types import TrunkOutputs
        trunk_out = TrunkOutputs(
            single_initial=single_init,
            single_trunk=single,
            single_structure=emb.single_structure,
            pair_initial=pair_init,
            pair_trunk=pair,
            pair_structure=emb.pair_structure,
            atom_single_structure_input=emb.atom_single_structure_input,
            atom_pair_structure_input=emb.atom_pair_structure_input,
            msa_input=emb.msa_input,
            template_input=emb.template_input,
            structure_inputs=si,
        )

        t0 = time.perf_counter()
        cache = model.prepare_diffusion_cache(trunk_out)
        _sync(
            cache.s_static, cache.z_cond, cache.blocked_pair_base,
            cache.atom_cond, cache.atom_single_cond, *cache.pair_biases,
        )
        cache_ms = (time.perf_counter() - t0) * 1000

        batch_size = emb.token_single_input.shape[0]
        coords = model.init_noise(batch_size, 5, emb.structure_inputs)
        mx.eval(coords)
        this_step_times: list[float] = []
        t_diff0 = time.perf_counter()
        for step_idx, (sigma_curr, sigma_next, gamma) in enumerate(
            model.schedule(num_steps=num_steps), start=1
        ):
            t0 = time.perf_counter()
            coords = model.diffusion_step(cache, coords, sigma_curr, sigma_next, gamma)
            mx.eval(coords)
            this_step_times.append((time.perf_counter() - t0) * 1000)
            if step_idx % 32 == 0:
                mx.clear_cache()
        diff_total_ms = (time.perf_counter() - t_diff0) * 1000

        t0 = time.perf_counter()
        conf = model.confidence(trunk_out, coords)
        _sync(conf.pae_logits, conf.pde_logits, conf.plddt_logits)
        conf_ms = (time.perf_counter() - t0) * 1000

        e2e_ms = (time.perf_counter() - trial_t0) * 1000

        if is_warmup:
            continue
        feat_times.append(feat_ms)
        trunk_times.append(statistics.mean(per_recycle_ms))
        cache_times.append(cache_ms)
        step_times.append(statistics.mean(this_step_times))
        diff_total_times.append(diff_total_ms)
        confidence_times.append(conf_ms)
        e2e_times.append(e2e_ms)

    return TargetResult(
        target=target,
        sequence=sequence,
        n_tokens=len(sequence),
        dtype=dtype,
        num_recycles=num_recycles,
        num_steps=num_steps,
        num_repeats=num_repeats,
        warmup=warmup,
        mlx_version=mx.__version__ if hasattr(mx, "__version__") else "unknown",
        feature_embedding_ms=_summarize(feat_times),
        trunk_recycle_ms=_summarize(trunk_times),
        diffusion_cache_ms=_summarize(cache_times),
        diffusion_step_ms=_summarize(step_times),
        diffusion_total_ms=_summarize(diff_total_times),
        confidence_ms=_summarize(confidence_times),
        end_to_end_ms=_summarize(e2e_times),
    )


def _print_row(result: TargetResult) -> None:
    def _f(t: ModuleTiming) -> str:
        if t.n == 0:
            return "   --  "
        return f"{t.mean_ms:6.1f}±{t.std_ms:4.1f}"

    print(
        f"  {result.target:<6} {result.n_tokens:>4}  "
        f"feat {_f(result.feature_embedding_ms)}  "
        f"trunk/rec {_f(result.trunk_recycle_ms)}  "
        f"diff/step {_f(result.diffusion_step_ms)}  "
        f"diff_total {_f(result.diffusion_total_ms)}  "
        f"conf {_f(result.confidence_ms)}  "
        f"e2e {_f(result.end_to_end_ms)}"
    )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--targets", default="1L2Y,1VII,1CRN,1UBQ")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--num-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--feature-dir", type=Path, default=Path("/tmp/chai_mlx_cuda/mlx_bench_features"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    results: list[TargetResult] = []
    print("=" * 100)
    print(f"MLX throughput: dtype={args.dtype} recycles={args.num_recycles} steps={args.num_steps}")
    print("=" * 100)
    for target in targets:
        if target not in DEFAULT_TARGETS:
            raise KeyError(f"Unknown target {target!r}. Known: {sorted(DEFAULT_TARGETS)}")
        result = _bench_target(
            args.weights_dir,
            target,
            DEFAULT_TARGETS[target],
            num_recycles=args.num_recycles,
            num_steps=args.num_steps,
            num_repeats=args.num_repeats,
            warmup=args.warmup,
            dtype=args.dtype,
            feature_dir=args.feature_dir / target,
        )
        _print_row(result)
        results.append(result)
        gc.collect()
        mx.clear_cache()

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps([asdict(r) for r in results], indent=2, default=str)
        )
        print(f"[save] json -> {args.output_json}")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "target",
                    "n_tokens",
                    "dtype",
                    "num_recycles",
                    "num_steps",
                    "feature_embedding_ms_mean",
                    "trunk_recycle_ms_mean",
                    "diffusion_cache_ms_mean",
                    "diffusion_step_ms_mean",
                    "diffusion_total_ms_mean",
                    "confidence_ms_mean",
                    "end_to_end_ms_mean",
                ]
            )
            for r in results:
                writer.writerow(
                    [
                        r.target,
                        r.n_tokens,
                        r.dtype,
                        r.num_recycles,
                        r.num_steps,
                        r.feature_embedding_ms.mean_ms,
                        r.trunk_recycle_ms.mean_ms,
                        r.diffusion_cache_ms.mean_ms,
                        r.diffusion_step_ms.mean_ms,
                        r.diffusion_total_ms.mean_ms,
                        r.confidence_ms.mean_ms,
                        r.end_to_end_ms.mean_ms,
                    ]
                )
        print(f"[save] csv -> {args.output_csv}")


if __name__ == "__main__":
    main()
