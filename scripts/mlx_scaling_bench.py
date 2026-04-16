"""Benchmark MLX E2E inference at different token counts.

Runs full fold (embed → trunk → diffusion → confidence → ranking) with dummy
inputs at each requested size and logs per-stage timings.  Results are saved
incrementally to a JSON file under /tmp so you can interrupt early.

The stage ordering and tensor-lifetime behavior are intentionally aligned with
``ChaiMLX.run_inference`` (production path), so memory pressure is closer to
real inference than the older debug-style benchmark flow.

Usage::

    python scripts/mlx_scaling_bench.py --weights-dir weights
    python scripts/mlx_scaling_bench.py --weights-dir weights --sizes 128 256
    python scripts/mlx_scaling_bench.py --weights-dir weights --dtype float32
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx

from chai_mlx import ChaiMLX, featurize


def build_dummy_inputs(n_tokens: int, msa_depth: int = 4, n_templates: int = 2):
    B = 1
    n_atoms = 23 * n_tokens
    num_blocks = n_atoms // 32
    token_exists = mx.ones((B, n_tokens), dtype=mx.bool_)
    atom_exists = mx.ones((B, n_atoms), dtype=mx.bool_)
    atom_token_index = mx.arange(n_atoms)[None, :] // 23
    atom_within_token_index = mx.arange(n_atoms)[None, :] % 37
    token_ref_atom_index = mx.arange(n_tokens)[None, :] * 23
    q_idx = mx.arange(n_atoms).reshape(1, num_blocks, 32)
    kv_idx = mx.clip(
        q_idx[:, :, :1] + mx.arange(128)[None, None, :] - 48,
        0,
        n_atoms - 1,
    )
    block_mask = mx.ones((B, num_blocks, 32, 128), dtype=mx.bool_)

    return {
        "token_features": mx.random.normal((B, n_tokens, 2638)),
        "token_pair_features": mx.random.normal((B, n_tokens, n_tokens, 163)),
        "atom_features": mx.random.normal((B, n_atoms, 395)),
        "atom_pair_features": mx.random.normal((B, num_blocks, 32, 128, 14)),
        "msa_features": mx.random.normal((B, msa_depth, n_tokens, 42)),
        "template_features": mx.random.normal((B, n_templates, n_tokens, n_tokens, 76)),
        "bond_adjacency": mx.zeros((B, n_tokens, n_tokens, 1)),
        "structure_inputs": {
            "atom_exists_mask": atom_exists,
            "token_exists_mask": token_exists,
            "token_pair_mask": token_exists[:, :, None] & token_exists[:, None, :],
            "atom_token_index": atom_token_index,
            "atom_within_token_index": atom_within_token_index,
            "token_reference_atom_index": token_ref_atom_index,
            "token_asym_id": mx.zeros((B, n_tokens), dtype=mx.int32),
            "token_entity_id": mx.zeros((B, n_tokens), dtype=mx.int32),
            "token_chain_id": mx.zeros((B, n_tokens), dtype=mx.int32),
            "token_is_polymer": mx.ones((B, n_tokens), dtype=mx.bool_),
            "bond_adjacency": mx.zeros((B, n_tokens, n_tokens, 1)),
            "atom_q_indices": q_idx,
            "atom_kv_indices": kv_idx,
            "block_atom_pair_mask": block_mask,
            "msa_mask": mx.ones((B, msa_depth, n_tokens), dtype=mx.float32),
            "template_input_masks": mx.ones(
                (B, n_templates, n_tokens, n_tokens), dtype=mx.float32
            ),
        },
    }


def bench_size(
    model: ChaiMLX,
    n_tokens: int,
    recycles: int,
    num_steps: int,
    num_samples: int,
) -> dict:
    from tqdm import tqdm

    print(f"\n{'='*60}")
    print(f"  n_tokens = {n_tokens}  (pair tensor: {n_tokens}x{n_tokens})")
    print(f"{'='*60}")

    mx.random.seed(42)
    raw = build_dummy_inputs(n_tokens)
    ctx = featurize(raw)
    print("  featurize done")

    timings: dict[str, float] = {"n_tokens": n_tokens}

    # --- embed ---
    t0 = time.perf_counter()
    emb = model.embed_inputs(ctx)
    mx.eval(emb.single_initial, emb.pair_initial, emb.token_single_input)
    t_embed = time.perf_counter() - t0
    timings["embed_s"] = round(t_embed, 2)
    print(f"  embed:     {t_embed:7.2f}s")

    # Match production memory behavior: keep only structure metadata after embed.
    structure = ctx.structure_inputs
    batch_size = emb.token_single_input.shape[0]
    del raw, ctx
    mx.clear_cache()

    # --- trunk ---
    t0 = time.perf_counter()
    trunk_out = model.trunk(emb, recycles=recycles)
    mx.eval(trunk_out.single_trunk, trunk_out.pair_trunk)
    t_trunk = time.perf_counter() - t0
    timings["trunk_s"] = round(t_trunk, 2)
    timings["trunk_recycles"] = recycles
    print(f"  trunk:     {t_trunk:7.2f}s  ({recycles} recycles, production path)")

    # --- cache ---
    t0 = time.perf_counter()
    cache = model.prepare_diffusion_cache(trunk_out)
    mx.eval(cache.s_static, cache.z_cond, cache.blocked_pair_base,
            cache.atom_cond, cache.atom_single_cond, *cache.pair_biases)
    t_cache = time.perf_counter() - t0
    timings["cache_s"] = round(t_cache, 2)
    print(f"  cache:     {t_cache:7.2f}s")

    # Production path drops embeddings before diffusion.
    del emb
    mx.clear_cache()

    # --- diffusion ---
    coords = model.init_noise(
        batch_size=batch_size,
        num_samples=num_samples,
        structure=structure,
    )
    schedule = list(model.schedule(num_steps=num_steps))

    t0 = time.perf_counter()
    for sigma_curr, sigma_next, gamma in tqdm(
        schedule, desc=f"  diffusion (n={n_tokens})", unit="step", leave=True
    ):
        coords = model.diffusion_step(cache, coords, sigma_curr, sigma_next, gamma)
        mx.eval(coords)
    t_diff = time.perf_counter() - t0
    timings["diffusion_s"] = round(t_diff, 2)
    timings["diffusion_steps"] = num_steps
    timings["diffusion_step_avg_s"] = round(t_diff / len(schedule), 3)
    print(f"  diffusion: {t_diff:7.2f}s  ({num_steps} steps, {t_diff/len(schedule):.3f}s/step)")

    # --- confidence + ranking ---
    t0 = time.perf_counter()
    conf = model.confidence(trunk_out, coords)
    rank = model.rank_outputs(conf, coords, structure)
    mx.eval(conf.pae_logits, conf.pde_logits, conf.plddt_logits, rank.aggregate_score)
    t_conf = time.perf_counter() - t0
    timings["confidence_ranking_s"] = round(t_conf, 2)
    print(f"  conf+rank: {t_conf:7.2f}s")

    total = t_embed + t_trunk + t_cache + t_diff + t_conf
    timings["total_s"] = round(total, 2)
    print(f"  TOTAL:     {total:7.2f}s")

    del trunk_out, cache, conf, rank, coords, structure
    mx.clear_cache()

    return timings


def main():
    parser = argparse.ArgumentParser(description="MLX E2E scaling benchmark")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[128, 256, 512, 1024],
        help="Token counts to benchmark (default: 128 256 512 1024)",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--out", type=Path, default=Path("/tmp/mlx_scaling_bench.json"),
        help="Output JSON path (default: /tmp/mlx_scaling_bench.json)",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.weights_dir} (dtype={args.dtype}) ...")
    model = ChaiMLX.from_pretrained(
        args.weights_dir, strict=False, compute_dtype=args.dtype
    )
    print("Model loaded.\n")

    all_results: list[dict] = []
    out_path = args.out

    for n in args.sizes:
        result = bench_size(
            model, n,
            recycles=args.recycles,
            num_steps=args.num_steps,
            num_samples=args.num_samples,
        )
        result["dtype"] = args.dtype
        all_results.append(result)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2) + "\n")
        print(f"  -> saved to {out_path}")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'n':>6}  {'embed':>7}  {'trunk':>7}  {'cache':>7}  {'diff':>7}  {'conf':>7}  {'TOTAL':>7}")
    for r in all_results:
        print(
            f"  {r['n_tokens']:>6}  "
            f"{r['embed_s']:>7.2f}  "
            f"{r['trunk_s']:>7.2f}  "
            f"{r['cache_s']:>7.2f}  "
            f"{r['diffusion_s']:>7.2f}  "
            f"{r['confidence_ranking_s']:>7.2f}  "
            f"{r['total_s']:>7.2f}"
        )


if __name__ == "__main__":
    main()
