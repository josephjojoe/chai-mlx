from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..utils import chunk_last, make_additive_mask, merge_heads, sigmoid, split_heads


class TriangleMultiplication(nn.Module):
    def __init__(self, pair_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.layernorm_z_in = nn.LayerNorm(pair_dim, eps=eps)
        self.merged_linear_p = nn.Linear(pair_dim, 4 * pair_dim, bias=False)
        self.merged_linear_g = nn.Linear(pair_dim, 5 * pair_dim, bias=False)
        self.layernorm_out = nn.LayerNorm(pair_dim, eps=eps)
        self.layernorm_in = nn.LayerNorm(pair_dim, eps=eps)
        self.linear_z_out = nn.Linear(pair_dim, pair_dim, bias=False)

    def __call__(self, z: mx.array, pair_mask: mx.array | None = None) -> mx.array:
        z_normed = self.layernorm_z_in(z)
        p = self.merged_linear_p(z_normed)
        g = sigmoid(self.merged_linear_g(z_normed))

        ab = p * g[..., : 4 * z.shape[-1]]
        ab_left, ab_right = chunk_last(ab, 2)
        a1, b1 = chunk_last(ab_left, 2)
        a2, b2 = chunk_last(ab_right, 2)

        if pair_mask is not None:
            row_mask = pair_mask[..., None].astype(z.dtype)
            col_mask = pair_mask.transpose(0, 2, 1)[..., None].astype(z.dtype)
            a1, b1 = a1 * row_mask, b1 * row_mask
            a2, b2 = a2 * col_mask, b2 * col_mask

        x_out = mx.einsum("bikd,bjkd->bijd", a1, b1)
        x_in = mx.einsum("bkid,bkjd->bijd", a2, b2)
        out = self.linear_z_out(self.layernorm_out(x_out) + self.layernorm_in(x_in))
        out = out * g[..., 4 * z.shape[-1] :]
        return z + out


class TriangleAttention(nn.Module):
    def __init__(self, pair_dim: int, num_heads: int, head_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.pair_norm = nn.LayerNorm(pair_dim, eps=eps)
        self.pair2b = nn.Linear(pair_dim, num_heads, bias=False)
        self.pair2qkvg1 = nn.Linear(pair_dim, 4 * num_heads * head_dim, bias=False)
        self.pair2qkvg2 = nn.Linear(pair_dim, 4 * num_heads * head_dim, bias=False)
        self.linear_out = nn.Linear(2 * num_heads * head_dim, pair_dim, bias=False)
        self.out_scalers = mx.ones((pair_dim,), dtype=mx.float32)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def _starting_mask(self, pair_mask: mx.array) -> mx.array:
        # [b, i, j] and [b, i, k] -> [b, i, j, k]
        return pair_mask[:, :, :, None] & pair_mask[:, :, None, :]

    def _ending_mask(self, pair_mask: mx.array) -> mx.array:
        # operate over columns instead of rows -> [b, j, i, k]
        col = pair_mask.transpose(0, 2, 1)
        return col[:, :, :, None] & col[:, :, None, :]

    def __call__(self, z: mx.array, pair_mask: mx.array | None = None) -> mx.array:
        b, n, _, _ = z.shape
        z_ln = self.pair_norm(z)
        bias = self.pair2b(z_ln).transpose(0, 3, 1, 2)

        q1, k1, v1, g1 = chunk_last(self.pair2qkvg1(z_ln), 4)
        q2, k2, v2, g2 = chunk_last(self.pair2qkvg2(z_ln), 4)

        q1 = split_heads(q1, self.num_heads, self.head_dim)
        k1 = split_heads(k1, self.num_heads, self.head_dim)
        v1 = split_heads(v1, self.num_heads, self.head_dim)
        g1 = split_heads(g1, self.num_heads, self.head_dim)

        q2 = split_heads(q2, self.num_heads, self.head_dim)
        k2 = split_heads(k2, self.num_heads, self.head_dim)
        v2 = split_heads(v2, self.num_heads, self.head_dim)
        g2 = split_heads(g2, self.num_heads, self.head_dim)

        # Starting-node attention: fix i, attend j -> k, with bias from (j, k).
        q_start = q1.transpose(0, 1, 3, 2, 4).reshape(b * n, self.num_heads, n, self.head_dim)
        k_start = k1.transpose(0, 1, 3, 2, 4).reshape(b * n, self.num_heads, n, self.head_dim)
        v_start = v1.transpose(0, 1, 3, 2, 4).reshape(b * n, self.num_heads, n, self.head_dim)
        bias_start = mx.broadcast_to(bias[:, None, :, :, :], (b, n, self.num_heads, n, n)).reshape(
            b * n, self.num_heads, n, n
        )
        if pair_mask is not None:
            start_mask = self._starting_mask(pair_mask).reshape(b * n, 1, n, n)
            bias_start = bias_start + make_additive_mask(start_mask)
        out_start = mx.fast.scaled_dot_product_attention(
            q_start,
            k_start,
            v_start,
            scale=self.head_dim ** -0.5,
            mask=bias_start,
        )
        out_start = out_start.reshape(b, n, self.num_heads, n, self.head_dim).transpose(0, 1, 3, 2, 4)
        out_start = out_start * sigmoid(g1)

        # Ending-node attention: fix j, attend i -> k, with bias from (i, k).
        q_end = q2.transpose(0, 2, 3, 1, 4).reshape(b * n, self.num_heads, n, self.head_dim)
        k_end = k2.transpose(0, 2, 3, 1, 4).reshape(b * n, self.num_heads, n, self.head_dim)
        v_end = v2.transpose(0, 2, 3, 1, 4).reshape(b * n, self.num_heads, n, self.head_dim)
        bias_end = mx.broadcast_to(bias[:, None, :, :, :], (b, n, self.num_heads, n, n)).reshape(
            b * n, self.num_heads, n, n
        )
        if pair_mask is not None:
            end_mask = self._ending_mask(pair_mask).reshape(b * n, 1, n, n)
            bias_end = bias_end + make_additive_mask(end_mask)
        out_end = mx.fast.scaled_dot_product_attention(
            q_end,
            k_end,
            v_end,
            scale=self.head_dim ** -0.5,
            mask=bias_end,
        )
        out_end = out_end.reshape(b, n, self.num_heads, n, self.head_dim).transpose(0, 3, 1, 2, 4)
        out_end = out_end * sigmoid(g2)

        out = mx.concatenate([out_start, out_end], axis=-1)
        out = self.linear_out(merge_heads(out)) * self.out_scalers
        return z + out


class ConfidenceTriangleAttention(nn.Module):
    """Fused single-projection triangle attention used in the confidence head.

    Verified from TorchScript ``confidence_head.pt`` (class
    ``TriangleAttentionUpdate_v1``).  Uses **4 heads at 64 head_dim**.

    Weight layout of ``pair2qkvgb: Linear(256, 2056, no bias)``::

        2056 = 2048 (qkvg, both directions) + 8 (bias, both directions)
        2048 = 2 dirs × 4 components(q,k,v,g) × 4 heads × 64 dim
           8 = 2 dirs × 4 heads

    ``linear_out: Linear(512, 512, no bias)``  — NOT (512→256).
    Output 512 is split into two 256-dim halves (both added to residual).
    No ``out_scalers``.
    """

    def __init__(self, pair_dim: int, num_heads: int, head_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.pair_norm = nn.LayerNorm(pair_dim, eps=eps)
        qkvg_dim = 2 * 4 * num_heads * head_dim
        bias_dim = 2 * num_heads
        self.pair2qkvgb = nn.Linear(pair_dim, qkvg_dim + bias_dim, bias=False)
        out_dim = 2 * num_heads * head_dim
        self.linear_out = nn.Linear(out_dim, out_dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self._qkvg_dim = qkvg_dim

    def __call__(self, z: mx.array, pair_mask: mx.array | None = None) -> mx.array:
        b, n, _, _ = z.shape
        H, D = self.num_heads, self.head_dim

        z_ln = self.pair_norm(z)
        proj = self.pair2qkvgb(z_ln)  # [B, N, N, 2056]

        qkvg_all = proj[..., : self._qkvg_dim]  # [B, N, N, 2048]
        bias_all = proj[..., self._qkvg_dim :]   # [B, N, N, 8]

        half = self._qkvg_dim // 2  # 1024
        qkvg_start = qkvg_all[..., :half]        # [B, N, N, 1024]
        qkvg_end_raw = qkvg_all[..., half:]      # [B, N, N, 1024]
        qkvg_end = qkvg_end_raw.transpose(0, 2, 1, 3)  # swap pair dims

        # [B, N, N, 4*H*D] -> [B, N, N, 4, H, D]
        qkvg_start = qkvg_start.reshape(b, n, n, 4, H, D)
        qkvg_end = qkvg_end.reshape(b, n, n, 4, H, D)

        # Bias: [B, N, N, 2*H] -> [B, 2, H, N, N]
        bias = bias_all.reshape(b, n, n, 2, H).transpose(0, 3, 4, 1, 2)
        if pair_mask is not None:
            pm = pair_mask.reshape(b, 1, 1, n, n)
            bias = mx.where(pm, bias, -10000.0)
        bias_s = bias[:, 0]  # [B, H, N, N]
        bias_e = bias[:, 1]  # [B, H, N, N]

        def _run_direction(qkvg_dir: mx.array, bias_dir: mx.array) -> mx.array:
            """Run SDPA for one direction (starting or ending)."""
            q = qkvg_dir[..., 0, :, :]  # [B, N, N, H, D]
            k = qkvg_dir[..., 1, :, :]
            v = qkvg_dir[..., 2, :, :]
            g = qkvg_dir[..., 3, :, :]
            # Reshape for SDPA: [B, N(batch), N(seq), H, D] -> [B*N, H, N, D]
            q = q.transpose(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            k = k.transpose(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            v = v.transpose(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            # Bias: [B, H, N, N] broadcast over batched rows -> [B*N, H, N, N]
            mask = mx.broadcast_to(
                bias_dir[:, None, :, :, :], (b, n, H, n, n)
            ).reshape(b * n, H, n, n)
            attn = mx.fast.scaled_dot_product_attention(
                q, k, v, scale=D ** -0.5, mask=mask,
            )
            # [B*N, H, N, D] -> [B, N, N, H, D]
            attn = attn.reshape(b, n, H, n, D).transpose(0, 1, 3, 2, 4)
            return attn * sigmoid(g)

        out_s = _run_direction(qkvg_start, bias_s)  # [B, N, N, H, D]
        out_e = _run_direction(qkvg_end, bias_e)    # [B, N, N, H, D]

        # Merge heads and concatenate directions: [B, N, N, 2*H*D=512]
        out_s_flat = out_s.reshape(b, n, n, H * D)
        out_e_flat = out_e.reshape(b, n, n, H * D)
        combined = mx.concatenate([out_s_flat, out_e_flat], axis=-1)

        # linear_out: 512 -> 512, then sum both 256-dim halves
        projected = self.linear_out(combined)
        pair_dim = z.shape[-1]
        return z + projected[..., :pair_dim] + projected[..., pair_dim:]
