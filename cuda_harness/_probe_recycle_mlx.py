"""MLX-side boundary probe inside a single trunk recycle.

Takes a chai-lab intermediates NPZ (produced by
``cuda_harness/run_intermediates.py``), reconstructs the CUDA embedding
outputs, and runs the MLX ``trunk_module`` manually step-by-step for
recycle 0, saving ``(single, pair)`` after each sub-stage:

  a) post-recycle-projection (identity on recycle 0 because the recycle
     cache starts at zero, but we capture anyway for consistency)
  b) post-template_embedder
  c) post-msa_module
  d) post-pairformer_stack (== ``trunk.recycle_0.single/pair``)

We then report the max / mean / p99 / std of each tensor on the MLX side
at fp32.  The CUDA reference only has (d); we'll add (a/b/c) CUDA dumps
in a follow-up Modal run if this probe suggests it's worth it.

The goal is to see where the MLX chain accumulates magnitude / drift
most aggressively, and to produce numbered checkpoint dumps we can
later diff against CUDA intermediates once we capture those.

Usage::

    python3 cuda_harness/_probe_recycle_mlx.py \\
        --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \\
        --weights-dir weights

Writes::

    /tmp/chai_mlx_cuda/recycle_probe/mlx_recycle_0_fp32.npz
        post_proj.{single,pair}
        post_template.{single,pair}     # single unchanged
        post_msa.{single,pair}          # single unchanged
        post_pairformer.{single,pair}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np

from chai_mlx import ChaiMLX

import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.cuda_parity import (  # noqa: E402
    _load_npz,
    _reconstruct_embedding_outputs,
    _reconstruct_feature_context,
)


OUT_DIR = Path("/tmp/chai_mlx_cuda/recycle_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _stats(name: str, arr: np.ndarray) -> None:
    print(
        f"  {name:24s} shape={tuple(arr.shape)}  "
        f"max_abs={np.abs(arr).max():.3f}  mean={arr.mean():+.3e}  "
        f"std={arr.std():.3e}  p99_abs={np.percentile(np.abs(arr), 99):.3f}"
    )


def _np(x: mx.array) -> np.ndarray:
    return np.asarray(x.astype(mx.float32))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    args = parser.parse_args()

    print(f"[load] {args.npz}")
    data = _load_npz(args.npz)

    print(f"[load] model (fp32)")
    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False, compute_dtype="float32")

    print("[reconstruct] feature context + CUDA embedding outputs")
    ctx = _reconstruct_feature_context(data)
    cuda_emb = _reconstruct_embedding_outputs(data, ctx.structure_inputs, dtype=mx.float32)

    single_init = cuda_emb.single_initial
    pair_init = cuda_emb.pair_initial
    si = cuda_emb.structure_inputs

    # Recycle 0: cached prev_single / prev_pair start as zero (first recycle).
    # We still step through the recycle projections to exercise that code path.
    prev_single = single_init
    prev_pair = pair_init

    print("\n[recycle 0] stepping sub-stages")

    single = single_init + model.trunk_module.token_single_recycle_proj(prev_single)
    pair = pair_init + model.trunk_module.token_pair_recycle_proj(prev_pair)
    mx.eval(single, pair)
    post_proj = (_np(single), _np(pair))
    _stats("post_proj.single", post_proj[0])
    _stats("post_proj.pair", post_proj[1])

    pair = model.trunk_module.template_embedder(
        pair,
        cuda_emb.template_input,
        template_input_masks=si.template_input_masks,
        token_pair_mask=si.token_pair_mask,
    )
    mx.eval(pair)
    post_template = (post_proj[0], _np(pair))
    _stats("post_template.single", post_template[0])
    _stats("post_template.pair", post_template[1])

    pair = model.trunk_module.msa_module(
        single,
        pair,
        cuda_emb.msa_input,
        token_pair_mask=si.token_pair_mask,
        msa_mask=si.msa_mask,
    )
    mx.eval(pair)
    post_msa = (post_proj[0], _np(pair))
    _stats("post_msa.single", post_msa[0])
    _stats("post_msa.pair", post_msa[1])

    single, pair = model.trunk_module.pairformer_stack(
        single,
        pair,
        pair_mask=si.token_pair_mask,
        single_mask=si.token_exists_mask,
    )
    mx.eval(single, pair)
    post_pairformer = (_np(single), _np(pair))
    _stats("post_pairformer.single", post_pairformer[0])
    _stats("post_pairformer.pair", post_pairformer[1])

    # Also compute the CUDA reference drift for recycle 0:
    cuda_s = data["trunk.recycle_0.single"].astype(np.float32)
    cuda_z = data["trunk.recycle_0.pair"].astype(np.float32)
    s_diff = np.abs(post_pairformer[0] - cuda_s)
    z_diff = np.abs(post_pairformer[1] - cuda_z)
    print("\n[vs CUDA recycle_0]")
    print(
        f"  single  max_abs={s_diff.max():.3f}  mean_abs={s_diff.mean():.3e}  "
        f"rel={s_diff.max() / np.abs(cuda_s).max():.4f}"
    )
    print(
        f"  pair    max_abs={z_diff.max():.3f}  mean_abs={z_diff.mean():.3e}  "
        f"rel={z_diff.max() / np.abs(cuda_z).max():.4f}"
    )

    # Dump everything for later CUDA-side comparison.
    out_path = OUT_DIR / "mlx_recycle_0_fp32.npz"
    np.savez(
        out_path,
        post_proj_single=post_proj[0],
        post_proj_pair=post_proj[1],
        post_template_pair=post_template[1],
        post_msa_pair=post_msa[1],
        post_pairformer_single=post_pairformer[0],
        post_pairformer_pair=post_pairformer[1],
    )
    print(f"\n[save] {out_path}")


if __name__ == "__main__":
    main()
