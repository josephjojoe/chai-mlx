from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from ..layers.common import AdaLayerNorm
from ..utils import chunk_last, make_additive_mask, merge_heads, sigmoid, split_heads


class AttentionPairBias(nn.Module):
    def __init__(
        self,
        single_dim: int,
        pair_dim: int,
        num_heads: int,
        head_dim: int,
        *,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.single_norm = nn.LayerNorm(single_dim, eps=eps)
        self.pair_norm = nn.LayerNorm(pair_dim, eps=eps)
        self.pair_linear = nn.Linear(pair_dim, num_heads, bias=False)
        self.input2qkvg = nn.Linear(single_dim, 4 * num_heads * head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * head_dim, single_dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.query_bias = mx.zeros((num_heads, head_dim), dtype=mx.float32)

    def _bias(self, pair: mx.array, pair_mask: mx.array | None) -> mx.array:
        bias = self.pair_linear(self.pair_norm(pair)).transpose(0, 3, 1, 2)
        if pair_mask is not None:
            bias = bias + make_additive_mask(pair_mask)[:, None, :, :]
        return bias

    def __call__(
        self,
        x: mx.array,
        pair: mx.array,
        *,
        pair_mask: mx.array | None = None,
        precomputed_bias: mx.array | None = None,
    ) -> mx.array:
        qkvg = self.input2qkvg(self.single_norm(x))
        q, k, v, g = chunk_last(qkvg, 4)
        q = split_heads(q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = split_heads(k, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = split_heads(v, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        g = split_heads(g, self.num_heads, self.head_dim)
        q = q + self.query_bias[None, :, None, :]
        bias = precomputed_bias if precomputed_bias is not None else self._bias(pair, pair_mask)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5, mask=bias
        )
        out = out.transpose(0, 2, 1, 3)
        out = out * sigmoid(g)
        return self.output_proj(merge_heads(out))


class DiffusionSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        pair_dim: int,
        num_heads: int,
        head_dim: int,
        *,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.adaln = AdaLayerNorm(dim, cond_dim, eps=eps)
        self.pair_norm = nn.LayerNorm(pair_dim, eps=eps)
        self.pair_linear = nn.Linear(pair_dim, num_heads, bias=False)
        self.to_qkv = nn.Linear(dim, 3 * num_heads * head_dim, bias=False)
        self.to_out = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.gate_proj = nn.Linear(cond_dim, dim, bias=True)
        self.query_bias = mx.zeros((num_heads, head_dim), dtype=mx.float32)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def pair_bias(self, z_cond: mx.array, pair_mask: mx.array | None = None) -> mx.array:
        bias = self.pair_linear(self.pair_norm(z_cond)).transpose(0, 3, 1, 2)
        if pair_mask is not None:
            bias = bias + make_additive_mask(pair_mask)[:, None, :, :]
        return bias

    def __call__(
        self,
        x: mx.array,
        s_cond: mx.array,
        *,
        pair_bias: mx.array,
        use_kernel: bool = False,
    ) -> mx.array:
        x_norm = self.adaln(x, s_cond, use_kernel=use_kernel)
        q, k, v = chunk_last(self.to_qkv(x_norm), 3)
        q = split_heads(q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = split_heads(k, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = split_heads(v, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = q + self.query_bias[None, :, None, :]
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim ** -0.5, mask=pair_bias
        )
        out = self.to_out(merge_heads(out.transpose(0, 2, 1, 3)))
        return x + sigmoid(self.gate_proj(s_cond)) * out


class MSAPairWeightedAveraging(nn.Module):
    def __init__(self, msa_dim: int, pair_dim: int, num_heads: int = 8, value_dim: int = 32, eps: float = 1e-5) -> None:
        super().__init__()
        self.layernorm_msa = nn.LayerNorm(msa_dim, eps=eps)
        self.linear_msa2vg = nn.Linear(msa_dim, num_heads * 2 * value_dim, bias=False)
        self.layernorm_pair = nn.LayerNorm(pair_dim, eps=eps)
        self.linear_pair = nn.Linear(pair_dim, num_heads, bias=False)
        self.linear_out_no_bias = nn.Linear(num_heads * value_dim, msa_dim, bias=False)
        self.num_heads = num_heads
        self.value_dim = value_dim

    def __call__(self, msa: mx.array, pair: mx.array, token_pair_mask: mx.array | None = None) -> mx.array:
        pair_logits = self.linear_pair(self.layernorm_pair(pair)).transpose(0, 3, 1, 2)
        if token_pair_mask is not None:
            pair_logits = pair_logits + make_additive_mask(token_pair_mask)[:, None, :, :]
        weights = mx.softmax(pair_logits.astype(mx.float32), axis=-1)

        vg = self.linear_msa2vg(self.layernorm_msa(msa))
        v, g = chunk_last(vg, 2)
        v = split_heads(v, self.num_heads, self.value_dim).transpose(0, 1, 3, 2, 4)
        g = split_heads(g, self.num_heads, self.value_dim).transpose(0, 1, 3, 2, 4)
        out = mx.einsum("bhij,bmhjd->bmhid", weights, v)
        out = out * sigmoid(g)
        out = out.transpose(0, 1, 3, 2, 4)
        return self.linear_out_no_bias(merge_heads(out))
