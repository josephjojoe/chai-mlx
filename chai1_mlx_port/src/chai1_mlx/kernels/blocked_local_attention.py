from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .sources import BLOCKED_LOCAL_ATTENTION_SOURCE


@lru_cache(maxsize=1)
def _blocked_local_attention_kernel():
    return mx.fast.metal_kernel(
        name="chai1_blocked_local_attention",
        input_names=["q", "k", "v", "additive_bias", "scale"],
        output_names=["out"],
        source=BLOCKED_LOCAL_ATTENTION_SOURCE,
    )


def blocked_local_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    additive_bias: mx.array,
    scale: float,
) -> mx.array:
    """Experimental custom kernel.

    Shapes:
      q: [M, H, Q, D]
      k: [M, H, K, D]
      v: [M, H, K, D]
      additive_bias: [M, H, Q, K]
    """
    out_shape = q.shape
    return _blocked_local_attention_kernel()(
        inputs=[q, k, v, additive_bias, mx.array(scale, dtype=q.dtype)],
        template=[("T", q.dtype)],
        grid=(int(mx.prod(mx.array(out_shape)).item()), 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[q.dtype],
    )[0]
