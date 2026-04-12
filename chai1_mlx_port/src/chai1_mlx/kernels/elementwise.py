from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .sources import ADALN_APPLY_SOURCE, GATED_RESIDUAL_SOURCE, SWIGLU_SOURCE


@lru_cache(maxsize=1)
def _swiglu_kernel():
    return mx.fast.metal_kernel(
        name="chai1_swiglu",
        input_names=["u"],
        output_names=["out"],
        source=SWIGLU_SOURCE,
    )


@lru_cache(maxsize=1)
def _gated_residual_kernel():
    return mx.fast.metal_kernel(
        name="chai1_gated_residual",
        input_names=["x", "sub", "gate"],
        output_names=["out"],
        source=GATED_RESIDUAL_SOURCE,
    )


@lru_cache(maxsize=1)
def _adaln_kernel():
    return mx.fast.metal_kernel(
        name="chai1_adaln_apply",
        input_names=["x_norm", "scale", "shift"],
        output_names=["out"],
        source=ADALN_APPLY_SOURCE,
    )


def fused_swiglu_activation(u: mx.array) -> mx.array:
    out_shape = u.shape[:-1] + (u.shape[-1] // 2,)
    return _swiglu_kernel()(
        inputs=[u],
        template=[("T", u.dtype)],
        grid=(int(mx.prod(mx.array(out_shape)).item()), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[out_shape],
        output_dtypes=[u.dtype],
    )[0]


def fused_gated_residual(x: mx.array, sub: mx.array, gate: mx.array) -> mx.array:
    return _gated_residual_kernel()(
        inputs=[x, sub, gate],
        template=[("T", x.dtype)],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def fused_adaln(x_norm: mx.array, scale: mx.array, shift: mx.array) -> mx.array:
    return _adaln_kernel()(
        inputs=[x_norm, scale, shift],
        template=[("T", x_norm.dtype)],
        grid=(x_norm.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[x_norm.shape],
        output_dtypes=[x_norm.dtype],
    )[0]
