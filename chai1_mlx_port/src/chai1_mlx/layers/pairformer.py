from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import AttentionPairBias
from .common import Transition
from .triangle import ConfidenceTriangleAttention, TriangleAttention, TriangleMultiplication


class PairformerBlock(nn.Module):
    def __init__(
        self,
        pair_dim: int,
        *,
        single_dim: int | None = None,
        single_heads: int = 16,
        single_head_dim: int = 24,
        triangle_heads: int = 8,
        triangle_head_dim: int = 32,
        use_fused_triangle_attention: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.transition_pair = Transition(pair_dim, expansion=2, eps=eps)
        self.triangle_multiplication = TriangleMultiplication(pair_dim, eps=eps)
        if use_fused_triangle_attention:
            self.triangle_attention = ConfidenceTriangleAttention(
                pair_dim,
                num_heads=triangle_heads,
                head_dim=triangle_head_dim,
                eps=eps,
            )
        else:
            self.triangle_attention = TriangleAttention(
                pair_dim,
                num_heads=triangle_heads,
                head_dim=triangle_head_dim,
                eps=eps,
            )
        self.single_dim = single_dim
        if single_dim is not None:
            self.attention_pair_bias = AttentionPairBias(
                single_dim,
                pair_dim,
                num_heads=single_heads,
                head_dim=single_head_dim,
                eps=eps,
            )
            self.transition_single = Transition(single_dim, expansion=2, eps=eps)

    def __call__(
        self,
        z: mx.array,
        s: mx.array | None = None,
        *,
        pair_mask: mx.array | None = None,
        single_mask: mx.array | None = None,
        precomputed_single_pair_bias: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        # TorchScript: transition_pair reads from the ORIGINAL z (same input
        # as triangle_mult), not from the post-tri_attn pair.
        pair_transition_out = self.transition_pair(z)
        z = self.triangle_multiplication(z, pair_mask=pair_mask)
        z = z + pair_transition_out
        z = self.triangle_attention(z, pair_mask=pair_mask)
        if self.single_dim is not None and s is not None:
            attn_delta = self.attention_pair_bias(
                s,
                z,
                pair_mask=pair_mask,
                precomputed_bias=precomputed_single_pair_bias,
            )
            if single_mask is not None:
                attn_delta = attn_delta * single_mask[..., None]
            s = s + attn_delta
            s = s + self.transition_single(s)
        return z, s


class PairformerStack(nn.Module):
    def __init__(self, blocks: list[PairformerBlock]) -> None:
        super().__init__()
        self.blocks = blocks

    def __call__(
        self,
        s: mx.array,
        z: mx.array,
        *,
        pair_mask: mx.array | None = None,
        single_mask: mx.array | None = None,
        precomputed_single_pair_biases: tuple[mx.array, ...] | None = None,
    ) -> tuple[mx.array, mx.array]:
        for i, block in enumerate(self.blocks):
            bias = None if precomputed_single_pair_biases is None else precomputed_single_pair_biases[i]
            z, s = block(z, s, pair_mask=pair_mask, single_mask=single_mask, precomputed_single_pair_bias=bias)
            mx.eval(s, z)
        assert s is not None
        return s, z
