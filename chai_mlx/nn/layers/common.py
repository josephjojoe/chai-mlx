from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from chai_mlx.config import ChaiConfig
from chai_mlx.utils import chunk_last, sigmoid, silu


class FP32LayerNorm(nn.LayerNorm):
    """LayerNorm that always computes reductions in float32.

    Subclasses ``nn.LayerNorm`` so parameter names (``weight``, ``bias``)
    are unchanged, preserving safetensors compatibility.  When the model
    runs in float32 the casts are no-ops.
    """

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        y = super().__call__(x.astype(mx.float32))
        return y.astype(orig_dtype)


class AdaLayerNorm(nn.Module):
    _ADALN_EPS: float = 0.1

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = FP32LayerNorm(dim, eps=self._ADALN_EPS, affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim, bias=False)

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        scale, shift = chunk_last(self.to_scale_shift(cond), 2)
        return self.norm(x) * (1.0 + scale) + shift


class SwiGLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        a, b = chunk_last(x, 2)
        return silu(a) * b


class Transition(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion: int,
        *,
        bias_out: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm = FP32LayerNorm(dim, eps=eps)
        self.up = nn.Linear(dim, 2 * expansion * dim, bias=False)
        self.swiglu = SwiGLU()
        self.down = nn.Linear(expansion * dim, dim, bias=bias_out)
        self.chunk_budget = 1 << 30

    def _n_chunks(self, x: mx.array) -> int:
        out_features = int(self.up.weight.shape[0])
        in_features = max(int(self.up.weight.shape[1]), 1)
        ratio = max(out_features // in_features, 1)
        est = math.prod(int(dim) for dim in x.shape) * ratio
        return max(1, est // self.chunk_budget)

    def __call__(self, x: mx.array) -> mx.array:
        n_chunks = min(self._n_chunks(x), max(int(x.shape[-2]), 1))
        if n_chunks <= 1:
            return self.down(self.swiglu(self.up(self.norm(x))))

        chunk_size = math.ceil(int(x.shape[-2]) / n_chunks)
        out_chunks: list[mx.array] = []
        for start in range(0, int(x.shape[-2]), chunk_size):
            x_chunk = x[..., start : start + chunk_size, :]
            out_chunks.append(
                self.down(self.swiglu(self.up(self.norm(x_chunk))))
            )
        return mx.concatenate(out_chunks, axis=-2)


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

    def delta(self, x: mx.array, cond: mx.array) -> mx.array:
        """Gated transition delta (no residual)."""
        y = self.adaln(x, cond)
        y = self.down(self.swiglu(self.up(y)))
        return sigmoid(self.gate(cond)) * y

    def __call__(self, x: mx.array, cond: mx.array) -> mx.array:
        y = self.adaln(x, cond)
        y = self.down(self.swiglu(self.up(y)))
        gate = self.gate(cond)
        return x + sigmoid(gate) * y


class ResidualMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.fc2(nn.relu(self.fc1(x)))


@dataclass(frozen=True)
class TriangleDims:
    pair_dim: int
    heads: int
    head_dim: int


def default_triangle_dims(cfg: ChaiConfig) -> TriangleDims:
    return TriangleDims(
        pair_dim=cfg.hidden.token_pair,
        heads=cfg.pairformer.triangle_heads,
        head_dim=cfg.pairformer.triangle_head_dim,
    )
