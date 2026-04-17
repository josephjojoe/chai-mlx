from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

from .sources import (
    ADALN_APPLY_SOURCE,
    FUSED_ADALN_NOAFFINE_SOURCE,
    FUSED_ADALN_SOURCE,
    GATED_RESIDUAL_SOURCE,
    SWIGLU_SOURCE,
)


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


@lru_cache(maxsize=1)
def _fused_adaln_kernel():
    return mx.fast.metal_kernel(
        name="chai1_fused_adaln",
        input_names=["x", "ln_w", "ln_b", "scale", "shift", "eps"],
        output_names=["y"],
        source=FUSED_ADALN_SOURCE,
    )


@lru_cache(maxsize=1)
def _fused_adaln_noaffine_kernel():
    return mx.fast.metal_kernel(
        name="chai1_fused_adaln_noaffine",
        input_names=["x", "scale", "shift", "eps"],
        output_names=["y"],
        source=FUSED_ADALN_NOAFFINE_SOURCE,
    )


def fused_adaln_full(
    x: mx.array,
    ln_weight: mx.array | None,
    ln_bias: mx.array | None,
    scale: mx.array,
    shift: mx.array,
    eps: float = 1e-5,
) -> mx.array:
    """Fused LayerNorm + AdaLN affine in a single Metal kernel.

    When ``ln_weight`` and ``ln_bias`` are ``None`` (as in chai's ``AdaLayerNorm``
    which uses ``affine=False``), a no-affine variant of the kernel is used so
    the path runs without needing dummy identity tensors.
    """
    D = x.shape[-1]
    num_rows = x.size // D
    tg_size = min(D, 1024)
    # Round up to multiple of 32 (SIMD width) for correct warp reductions.
    tg_size = ((tg_size + 31) // 32) * 32
    eps_arr = mx.array(eps, dtype=mx.float32)

    if ln_weight is None and ln_bias is None:
        return _fused_adaln_noaffine_kernel()(
            inputs=[x, scale, shift, eps_arr],
            template=[("T", x.dtype)],
            grid=(1, num_rows, 1),
            threadgroup=(tg_size, 1, 1),
            output_shapes=[x.shape],
            output_dtypes=[x.dtype],
        )[0]

    if ln_weight is None or ln_bias is None:
        raise ValueError(
            "fused_adaln_full requires both ln_weight and ln_bias, or neither"
        )
    return _fused_adaln_kernel()(
        inputs=[x, ln_weight, ln_bias, scale, shift, eps_arr],
        template=[("T", x.dtype)],
        grid=(1, num_rows, 1),
        threadgroup=(tg_size, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]
