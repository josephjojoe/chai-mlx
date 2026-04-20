"""Take MLX's post-msa (s, z) from recycle 0 of the 1L2Y intermediates run
and feed it through MLX's pairformer_stack at fp32.  Writes both the
inputs and the per-block outputs for side-by-side comparison against an
eager-PyTorch run on the same inputs.

This probe is about input distribution.  The synthetic-input stack
probe showed MLX ≈ eager-PyTorch to 1e-4 rel err on randn-scale inputs.
We now switch to the real post-msa tensors (which have a very different
statistical profile: max_abs ~500, std ~35) to check whether the
pairformer stack is numerically sensitive to input scale.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from chai_mlx import ChaiMLX
from scripts.cuda_parity import (  # noqa: E402
    _load_npz,
    _reconstruct_embedding_outputs,
    _reconstruct_feature_context,
)


OUT_DIR = Path("/tmp/chai_mlx_cuda/real_inputs_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    npz_path = Path("/tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz")
    weights_dir = Path("weights")

    data = _load_npz(npz_path)
    model = ChaiMLX.from_pretrained(weights_dir, strict=False, compute_dtype="float32")

    ctx = _reconstruct_feature_context(data)
    cuda_emb = _reconstruct_embedding_outputs(data, ctx.structure_inputs, dtype=mx.float32)

    si = cuda_emb.structure_inputs

    # Rebuild MLX's post-msa (s, z) for recycle 0 exactly as in
    # _probe_recycle_mlx.
    single = cuda_emb.single_initial + model.trunk_module.token_single_recycle_proj(
        cuda_emb.single_initial
    )
    pair = cuda_emb.pair_initial + model.trunk_module.token_pair_recycle_proj(
        cuda_emb.pair_initial
    )
    pair = model.trunk_module.template_embedder(
        pair,
        cuda_emb.template_input,
        template_input_masks=si.template_input_masks,
        token_pair_mask=si.token_pair_mask,
    )
    pair = model.trunk_module.msa_module(
        single,
        pair,
        cuda_emb.msa_input,
        token_pair_mask=si.token_pair_mask,
        msa_mask=si.msa_mask,
    )
    mx.eval(single, pair)

    pair_mask = si.token_pair_mask
    single_mask = si.token_exists_mask

    s_in = np.asarray(single.astype(mx.float32))
    z_in = np.asarray(pair.astype(mx.float32))
    pm_in = np.asarray(pair_mask.astype(mx.float32))
    sm_in = np.asarray(single_mask.astype(mx.float32))
    print(
        f"[input] s shape={s_in.shape} max_abs={np.abs(s_in).max():.3f} std={s_in.std():.3e}"
    )
    print(
        f"[input] z shape={z_in.shape} max_abs={np.abs(z_in).max():.3f} std={z_in.std():.3e}"
    )
    np.savez(
        OUT_DIR / "inputs.npz",
        single=s_in,
        pair=z_in,
        pair_mask=pm_in,
        single_mask=sm_in,
    )
    print(f"[save] shared inputs -> {OUT_DIR / 'inputs.npz'}")

    stack = model.trunk_module.pairformer_stack
    dump: dict[str, np.ndarray] = {}
    s, z = single, pair
    for i, block in enumerate(stack.blocks):
        z, s = block(z, s, pair_mask=pair_mask, single_mask=single_mask)
        mx.eval(s, z)
        dump[f"s_block_{i:02d}"] = np.asarray(s.astype(mx.float32))
        dump[f"z_block_{i:02d}"] = np.asarray(z.astype(mx.float32))

    dump["s_final"] = dump[f"s_block_{len(stack.blocks) - 1:02d}"]
    dump["z_final"] = dump[f"z_block_{len(stack.blocks) - 1:02d}"]
    np.savez(OUT_DIR / "mlx_out_fp32.npz", **dump)
    print(f"[save] MLX stack outputs -> {OUT_DIR / 'mlx_out_fp32.npz'}")

    cuda_s = data["trunk.recycle_0.single"].astype(np.float32)
    cuda_z = data["trunk.recycle_0.pair"].astype(np.float32)
    s_final = dump[f"s_block_{len(stack.blocks) - 1:02d}"]
    z_final = dump[f"z_block_{len(stack.blocks) - 1:02d}"]
    s_diff = np.abs(s_final - cuda_s)
    z_diff = np.abs(z_final - cuda_z)
    print(
        f"\n[vs CUDA recycle_0] MLX(real inputs) vs CUDA(real trunk)\n"
        f"  single  max_abs={s_diff.max():.3f}  mean_abs={s_diff.mean():.3e}  "
        f"rel={s_diff.max() / np.abs(cuda_s).max():.4f}\n"
        f"  pair    max_abs={z_diff.max():.3f}  mean_abs={z_diff.mean():.3e}  "
        f"rel={z_diff.max() / np.abs(cuda_z).max():.4f}"
    )


if __name__ == "__main__":
    main()
