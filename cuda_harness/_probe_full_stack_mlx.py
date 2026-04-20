"""MLX side of the 48-block pairformer-stack probe.

Takes the same deterministic synthetic inputs as ``_probe_first_block_mlx.py``
and runs the full ``pairformer_stack`` (all 48 blocks with their per-block
weights) at fp32.  Captures intermediate ``(s, z)`` after every block so we
can see whether the fp32 stack accumulates drift super-linearly or stays at
the ULP floor, block by block.

This complements the block-0 probe:

* Block-0 probe (§3.6): MLX vs eager-PyTorch on one block.  fp32 match at
  1-4e-6 rel err.  Rules out sub-op port bugs.
* Full-stack probe (this file + ``_probe_full_stack_cuda.py``): MLX vs
  eager-PyTorch on the entire 48-block stack.  Tells us whether any
  *later* block diverges, or whether accumulation behaves like the toy
  op-probe stack (sub-linear rel err with depth).

We run only at fp32: this is the configuration chai-lab uses in production
and therefore the one that bears on closing the cross-framework parity gap.

Outputs:
    /tmp/chai_mlx_cuda/full_stack_probe/inputs.npz      (shared inputs)
    /tmp/chai_mlx_cuda/full_stack_probe/mlx_out_fp32.npz
        - ``s_block_{i}`` (i=0..47): single rep after block i
        - ``z_block_{i}`` (i=0..47): pair rep after block i
        - ``s_final``, ``z_final``: == block 47 outputs
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from chai_mlx.model.core import ChaiMLX


OUT_DIR = Path("/tmp/chai_mlx_cuda/full_stack_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N = 256
D_SINGLE = 384
D_PAIR = 256


def _make_inputs(seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "single": rng.standard_normal((1, N, D_SINGLE), dtype=np.float32),
        "pair": rng.standard_normal((1, N, N, D_PAIR), dtype=np.float32),
        "pair_mask": np.ones((1, N, N), dtype=bool),
        "single_mask": np.ones((1, N), dtype=bool),
    }


def _run_stack_fp32(model: ChaiMLX, inputs_np: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Iterate the pairformer blocks manually so we can dump per-block outputs."""
    s = mx.array(inputs_np["single"]).astype(mx.float32)
    z = mx.array(inputs_np["pair"]).astype(mx.float32)
    pair_mask = mx.array(inputs_np["pair_mask"])
    single_mask = mx.array(inputs_np["single_mask"])

    stack = model.trunk_module.pairformer_stack
    out: dict[str, np.ndarray] = {}
    for i, block in enumerate(stack.blocks):
        z, s = block(z, s, pair_mask=pair_mask, single_mask=single_mask)
        mx.eval(s, z)
        out[f"s_block_{i:02d}"] = np.asarray(s.astype(mx.float32))
        out[f"z_block_{i:02d}"] = np.asarray(z.astype(mx.float32))
    out["s_final"] = out[f"s_block_{len(stack.blocks) - 1:02d}"]
    out["z_final"] = out[f"z_block_{len(stack.blocks) - 1:02d}"]
    return out


def main() -> None:
    inputs = _make_inputs(seed=42)
    np.savez(OUT_DIR / "inputs.npz", **inputs)
    print(f"saved shared inputs -> {OUT_DIR / 'inputs.npz'}")

    model = ChaiMLX.from_pretrained("weights", compute_dtype="float32")
    out = _run_stack_fp32(model, inputs)
    np.savez(OUT_DIR / "mlx_out_fp32.npz", **out)
    print(f"saved MLX fp32 stack outputs -> {OUT_DIR / 'mlx_out_fp32.npz'}")
    for i in (0, 11, 23, 35, 47):
        s = out[f"s_block_{i:02d}"]
        z = out[f"z_block_{i:02d}"]
        print(
            f"  block {i:2d}: s max_abs={np.abs(s).max():.3f} mean={s.mean():+.3e}  "
            f"z max_abs={np.abs(z).max():.3f} mean={z.mean():+.3e}"
        )


if __name__ == "__main__":
    main()
