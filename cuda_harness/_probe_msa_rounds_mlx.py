"""MLX-side: dump the pair / msa tensor after EACH sub-stage of msa_module's
4 rounds, matching the ``_probe_msa_rounds_cuda.py`` eager reimplementation
step-for-step.

Writes per-round intermediates to
``/tmp/chai_mlx_cuda/msa_module_probe/mlx_rounds_fp32.npz`` with keys:

    post_proj_pair                         # == pair_init on recycle 0
    post_template_pair
    msa_init
    after_linear_s2m_msa                   # msa after linear_s2m add
    round_0.opm_delta_pair                 # pair delta from outer_product_mean
    round_0.pair_after_opm                 # = pair + opm_delta
    round_0.msa_after_transition
    round_0.msa_after_pw                   # msa += pair_weighted_avg
    round_0.pair_trans_out                 # output of pair_transition
    round_0.pair_after_tri_mult            # = triangular_mult_out + pair_trans
    round_0.pair_after_tri_attn            # post-round
    ...
    round_3.opm_delta_pair
    round_3.pair_after_opm
    round_3.pair_trans_out
    round_3.pair_after_tri_mult
    round_3.pair_after_tri_attn            # final post-MSA pair

Usage::

    python3 cuda_harness/_probe_msa_rounds_mlx.py \\
        --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \\
        --weights-dir weights
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from chai_mlx import ChaiMLX


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.cuda_parity import (  # noqa: E402
    _load_npz,
    _reconstruct_embedding_outputs,
    _reconstruct_feature_context,
)


OUT_DIR = Path("/tmp/chai_mlx_cuda/msa_module_probe")


def _np(x: mx.array) -> np.ndarray:
    return np.asarray(x.astype(mx.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = ap.parse_args()

    compute_dtype_str = "float32" if args.dtype == "fp32" else "reference"
    mx_dtype = mx.float32 if args.dtype == "fp32" else mx.bfloat16

    print(f"[load] {args.npz}")
    data = _load_npz(args.npz)

    print(f"[load] model (compute_dtype={compute_dtype_str})")
    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False, compute_dtype=compute_dtype_str)

    print("[reconstruct] feature context + CUDA embedding outputs")
    ctx = _reconstruct_feature_context(data)
    cuda_emb = _reconstruct_embedding_outputs(data, ctx.structure_inputs, dtype=mx_dtype)

    single_init = cuda_emb.single_initial
    pair_init = cuda_emb.pair_initial
    si = cuda_emb.structure_inputs
    msa_input = cuda_emb.msa_input
    msa_mask = si.msa_mask

    # Recycle 0: prev caches zero → single/pair = init.
    single = single_init
    pair = pair_init

    dump: dict[str, np.ndarray] = {
        "post_proj_pair": _np(pair),
        "post_template_pair": _np(pair),    # 1L2Y has no templates
        "msa_init": _np(msa_input),
    }

    msa_module = model.trunk_module.msa_module
    token_pair_mask = si.token_pair_mask
    msa = msa_input
    if msa.shape[1] > 0:
        msa = msa + msa_module.linear_s2m(single)[:, None, :, :]
    mx.eval(msa)
    dump["after_linear_s2m_msa"] = _np(msa)

    for i in range(4):
        opm_delta = msa_module.outer_product_mean[i](msa, msa_mask=msa_mask)
        mx.eval(opm_delta)
        dump[f"round_{i}.opm_delta_pair"] = _np(opm_delta)

        pair = pair + opm_delta
        mx.eval(pair)
        dump[f"round_{i}.pair_after_opm"] = _np(pair)

        if i < 3:
            msa = msa + msa_module.msa_transition[i](msa)
            mx.eval(msa)
            dump[f"round_{i}.msa_after_transition"] = _np(msa)

            msa = msa + msa_module.msa_pair_weighted_averaging[i](
                msa, pair, token_pair_mask=token_pair_mask, msa_mask=msa_mask,
            )
            mx.eval(msa)
            dump[f"round_{i}.msa_after_pw"] = _np(msa)

        pair_trans_out = msa_module.pair_transition[i](pair)
        mx.eval(pair_trans_out)
        dump[f"round_{i}.pair_trans_out"] = _np(pair_trans_out)

        pair = msa_module.triangular_multiplication[i](pair, pair_mask=token_pair_mask) + pair_trans_out
        mx.eval(pair)
        dump[f"round_{i}.pair_after_tri_mult"] = _np(pair)

        pair = msa_module.triangular_attention[i](pair, pair_mask=token_pair_mask)
        mx.eval(pair)
        dump[f"round_{i}.pair_after_tri_attn"] = _np(pair)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"mlx_rounds_{args.dtype}.npz"
    np.savez_compressed(out_path, **dump)
    print(f"wrote {out_path} ({out_path.stat().st_size / (1 << 20):.2f} MB)")

    # Summary.
    for k, v in dump.items():
        print(f"  {k:45s} shape={v.shape}  max_abs={float(np.abs(v).max()):.3f}  mean={v.mean():+.3e}")


if __name__ == "__main__":
    main()
