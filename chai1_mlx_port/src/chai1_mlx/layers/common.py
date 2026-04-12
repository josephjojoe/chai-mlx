from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..config import Chai1Config
from ..kernels.elementwise import fused_adaln, fused_gated_residual, fused_swiglu_activation
from ..utils import chunk_last, sigmoid, silu


class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim, bias=False)

    def __call__(self, x: mx.array, cond: mx.array, *, use_kernel: bool = False) -> mx.array:
        scale, shift = chunk_last(self.to_scale_shift(cond), 2)
        if use_kernel:
            return fused_adaln(self.norm(x), scale, shift)
        return self.norm(x) * (1.0 + scale) + shift


class SwiGLU(nn.Module):
    def __call__(self, x: mx.array, *, use_kernel: bool = False) -> mx.array:
        if use_kernel:
            return fused_swiglu_activation(x)
        a, b = chunk_last(x, 2)
        return silu(a) * b


class Transition(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: int,
        *,
        bias_out: bool = True,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.up = nn.Linear(dim, 2 * expansion * dim, bias=False)
        self.swiglu = SwiGLU()
        self.down = nn.Linear(expansion * dim, dim, bias=bias_out)

    def __call__(self, x: mx.array, *, use_kernel: bool = False) -> mx.array:
        return self.down(self.swiglu(self.up(self.norm(x)), use_kernel=use_kernel))


class ConditionedTransition(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        expansion: int,
        *,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.adaln = AdaLayerNorm(dim, cond_dim, eps=eps)
        self.up = nn.Linear(dim, 2 * expansion * dim, bias=False)
        self.swiglu = SwiGLU()
        self.down = nn.Linear(expansion * dim, dim, bias=False)
        self.gate = nn.Linear(cond_dim, dim, bias=True)

    def __call__(self, x: mx.array, cond: mx.array, *, use_kernel: bool = False) -> mx.array:
        y = self.adaln(x, cond, use_kernel=use_kernel)
        y = self.down(self.swiglu(self.up(y), use_kernel=use_kernel))
        gate = self.gate(cond)
        if use_kernel:
            return fused_gated_residual(x, y, gate)
        return x + sigmoid(gate) * y


class ResidualMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.fc2(nn.relu(self.fc1(x)))


class OutputGate(nn.Module):
    def __init__(self, dim: int, cond_dim: int | None = None, bias: bool = True) -> None:
        super().__init__()
        self.cond_dim = cond_dim
        self.linear = nn.Linear(cond_dim if cond_dim is not None else dim, dim, bias=bias)

    def __call__(self, x: mx.array, cond: mx.array | None = None, *, use_kernel: bool = False) -> mx.array:
        gate_src = x if cond is None else cond
        gate = self.linear(gate_src)
        if use_kernel:
            return fused_gated_residual(mx.zeros_like(x), x, gate)
        return sigmoid(gate) * x


@dataclass(frozen=True)
class TriangleDims:
    pair_dim: int
    heads: int
    head_dim: int


def default_triangle_dims(cfg: Chai1Config) -> TriangleDims:
    return TriangleDims(
        pair_dim=cfg.hidden.token_pair,
        heads=cfg.pairformer.triangle_heads,
        head_dim=cfg.pairformer.triangle_head_dim,
    )
