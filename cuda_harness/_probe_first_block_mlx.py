"""Throwaway: run MLX's PairformerBlock (block 0 of the trunk) on a
deterministic seeded (single, pair, masks) input and dump inputs + outputs.

Rationale
---------
Previous single-op probes (``/tmp/chai_mlx_cuda/op_probe``) showed that all
primitive MLX ops (matmul, softmax, layernorm, exp, rsqrt, sum) match their
CUDA counterparts to 1-2 ULPs of fp32 rounding on representative pairformer
shapes, and the 48-deep residual stack diverges ~linearly in absolute terms
(sub-linearly in rel err thanks to random-walk noise).  Yet the full chai-1
trunk on the same input produces ~35% rel err between MLX and CUDA outputs.
That gap cannot be explained by op-level drift alone.

This probe closes the gap by comparing **MLX's implementation of the
pairformer block** against a **line-for-line eager PyTorch reimplementation
of the same block** (built in ``_probe_first_block_cuda.py``) using the same
upstream chai-1 weights.  Both sides consume bit-identical inputs.

If MLX-block-0 ≈ eager-PyTorch-block-0 (to fp32 ULPs), the port is correct and
the 35% gap lives elsewhere: most plausibly inside CUDA's fused TorchScript
kernels that aggregate all 48 blocks into one graph with custom scheduling.

If MLX-block-0 ≠ eager-PyTorch-block-0 at fp32, there is a port-level bug
inside a specific sub-op (triangle-mult, triangle-attn, attention-pair-bias,
transition).  We'll see which one by inspecting which sub-tensor diverges
first inside the block.

Usage
-----
Run this locally (one-shot, ~1s on M-series):
    python3 cuda_harness/_probe_first_block_mlx.py

Writes
    /tmp/chai_mlx_cuda/first_block_probe/inputs.npz     (shared inputs)
    /tmp/chai_mlx_cuda/first_block_probe/mlx_out_fp32.npz
    /tmp/chai_mlx_cuda/first_block_probe/mlx_out_bf16.npz
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from chai_mlx.model.core import ChaiMLX


OUT_DIR = Path("/tmp/chai_mlx_cuda/first_block_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Representative 1L2Y shapes (match cuda_harness/run_intermediates.py @ 1L2Y).
N = 256
D_SINGLE = 384
D_PAIR = 256


def _make_inputs(seed: int = 42) -> dict[str, np.ndarray]:
    """Deterministic block-0 inputs.

    Reality check: the actual block-0 input distribution is the output of
    the trunk's ``msa_module`` (on recycle 0), which is bounded, real-valued,
    roughly zero-mean.  ``randn`` is a reasonable proxy; it's the cleanest
    choice for a deterministic probe and avoids pulling upstream-drift into
    the comparison.

    Masks: fully unmasked (token_exists_mask = True), mimicking a 256-token
    padded 1L2Y crop where tokens 0..19 are live and the rest are padding.
    We use an all-ones mask here because we want to stress block 0 across
    its full input space, not test padding behaviour.  If we find divergence
    with the all-ones mask, it's not a masking issue.
    """
    rng = np.random.default_rng(seed)
    return {
        "single": rng.standard_normal((1, N, D_SINGLE), dtype=np.float32),
        "pair": rng.standard_normal((1, N, N, D_PAIR), dtype=np.float32),
        "pair_mask": np.ones((1, N, N), dtype=bool),
        "single_mask": np.ones((1, N), dtype=bool),
    }


def _run_block_0(
    model: ChaiMLX,
    inputs_np: dict[str, np.ndarray],
    compute_dtype: mx.Dtype,
) -> dict[str, np.ndarray]:
    """Run exactly ``pairformer_stack.blocks[0]`` on the given inputs.

    We also dump the two intermediate outputs of the triangle_mult and
    triangle_attention sub-ops so that if the overall block diverges, we can
    see which sub-op is responsible.
    """
    s = mx.array(inputs_np["single"]).astype(compute_dtype)
    z = mx.array(inputs_np["pair"]).astype(compute_dtype)
    pair_mask = mx.array(inputs_np["pair_mask"])
    single_mask = mx.array(inputs_np["single_mask"])

    block0 = model.trunk_module.pairformer_stack.blocks[0]

    # Run the block in the same order the block's __call__ uses, but expose
    # the intermediate tensors so we can localise a divergence.
    pair_transition_out = block0.transition_pair(z)
    mx.eval(pair_transition_out)

    z_after_tri_mult = block0.triangle_multiplication(z, pair_mask=pair_mask)
    mx.eval(z_after_tri_mult)

    z_after_residual = z_after_tri_mult + pair_transition_out
    mx.eval(z_after_residual)

    z_after_tri_attn = block0.triangle_attention(z_after_residual, pair_mask=pair_mask)
    mx.eval(z_after_tri_attn)

    attn_delta = block0.attention_pair_bias(
        s, z_after_tri_attn, pair_mask=pair_mask,
    )
    attn_delta = attn_delta * single_mask.astype(attn_delta.dtype)[..., None]
    mx.eval(attn_delta)

    s_after_attn = s + attn_delta
    mx.eval(s_after_attn)

    s_after_transition = s_after_attn + block0.transition_single(s_after_attn)
    mx.eval(s_after_transition)

    # Final outputs: (z_after_tri_attn, s_after_transition) matches what
    # PairformerBlock.__call__ returns.
    return {
        "pair_transition_out": np.asarray(pair_transition_out.astype(mx.float32)),
        "z_after_tri_mult": np.asarray(z_after_tri_mult.astype(mx.float32)),
        "z_after_residual": np.asarray(z_after_residual.astype(mx.float32)),
        "z_after_tri_attn": np.asarray(z_after_tri_attn.astype(mx.float32)),
        "attn_delta": np.asarray(attn_delta.astype(mx.float32)),
        "s_after_attn": np.asarray(s_after_attn.astype(mx.float32)),
        "s_after_transition": np.asarray(s_after_transition.astype(mx.float32)),
        "z_final": np.asarray(z_after_tri_attn.astype(mx.float32)),
        "s_final": np.asarray(s_after_transition.astype(mx.float32)),
    }


def main() -> None:
    inputs = _make_inputs(seed=42)
    np.savez(OUT_DIR / "inputs.npz", **inputs)
    print(f"saved shared inputs → {OUT_DIR / 'inputs.npz'}")

    for dtype_name, dtype in (("fp32", mx.float32), ("bf16", mx.bfloat16)):
        print(f"\n=== {dtype_name} ===")
        model = ChaiMLX.from_pretrained(
            "weights",
            compute_dtype="float32" if dtype_name == "fp32" else "bfloat16",
        )
        outputs = _run_block_0(model, inputs, dtype)
        out_path = OUT_DIR / f"mlx_out_{dtype_name}.npz"
        np.savez(out_path, **outputs)
        # Print a lightweight digest.
        for k, v in outputs.items():
            print(f"  {k:26s} shape={v.shape}  max_abs={float(np.abs(v).max()):.4f}  mean={float(v.mean()):+.4e}")
        print(f"  → saved {out_path}")


if __name__ == "__main__":
    main()
