from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..utils import chunk_last, make_additive_mask, merge_heads, sigmoid


class TriangleMultiplication(nn.Module):
    def __init__(self, pair_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.layernorm_z_in = nn.LayerNorm(pair_dim, eps=eps)
        self.merged_linear_p = nn.Linear(pair_dim, 4 * pair_dim, bias=False)
        self.merged_linear_g = nn.Linear(pair_dim, 5 * pair_dim, bias=False)
        self.layernorm_out = nn.LayerNorm(pair_dim, eps=eps, affine=False)
        self.layernorm_in = nn.LayerNorm(pair_dim, eps=eps, affine=False)
        self.linear_z_out = nn.Linear(pair_dim, pair_dim, bias=False)

    _CHUNK_SIZE: int = 32

    def __call__(self, z: mx.array, pair_mask: mx.array | None = None) -> mx.array:
        return self._forward_chunked(z, pair_mask, self._CHUNK_SIZE)

    def _forward_unchunked(self, z: mx.array, pair_mask: mx.array | None) -> mx.array:
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

    def _forward_chunked(self, z: mx.array, pair_mask: mx.array | None, chunk_size: int) -> mx.array:
        """Memory-efficient triangle multiplication by chunking over the feature dimension.

        Instead of materializing full [b, n, n, 4*d] projections, we process
        ``chunk_size`` channels at a time.  The einsum ``bikd,bjkd->bijd``
        treats d as a free index (not contracted), so each chunk produces an
        independent slice of the output that must be *concatenated*, not summed.
        Peak intermediate memory drops from ~11× to ~3-4× the pair tensor.
        """
        d = z.shape[-1]
        z_normed = self.layernorm_z_in(z)

        w_p = self.merged_linear_p.weight  # [4d, d]
        w_g = self.merged_linear_g.weight  # [5d, d]

        if pair_mask is not None:
            row_mask = pair_mask[..., None].astype(z.dtype)
            col_mask = pair_mask.transpose(0, 2, 1)[..., None].astype(z.dtype)
        else:
            row_mask = col_mask = None

        out_chunks: list[mx.array] = []
        in_chunks: list[mx.array] = []

        for c in range(0, d, chunk_size):
            c_end = min(c + chunk_size, d)

            a1 = (z_normed @ w_p[c:c_end].T) * sigmoid(z_normed @ w_g[c:c_end].T)
            b1 = (z_normed @ w_p[d + c:d + c_end].T) * sigmoid(z_normed @ w_g[d + c:d + c_end].T)
            a2 = (z_normed @ w_p[2 * d + c:2 * d + c_end].T) * sigmoid(z_normed @ w_g[2 * d + c:2 * d + c_end].T)
            b2 = (z_normed @ w_p[3 * d + c:3 * d + c_end].T) * sigmoid(z_normed @ w_g[3 * d + c:3 * d + c_end].T)

            if row_mask is not None:
                a1 = a1 * row_mask
                b1 = b1 * row_mask
                a2 = a2 * col_mask
                b2 = b2 * col_mask

            out_chunks.append(mx.einsum("bikd,bjkd->bijd", a1, b1))
            in_chunks.append(mx.einsum("bkid,bkjd->bijd", a2, b2))
            mx.eval(out_chunks[-1], in_chunks[-1])

        x_out = mx.concatenate(out_chunks, axis=-1)
        x_in = mx.concatenate(in_chunks, axis=-1)

        g_out = sigmoid(z_normed @ w_g[4 * d:].T)
        out = self.linear_z_out(self.layernorm_out(x_out) + self.layernorm_in(x_in))
        out = out * g_out
        return z + out


class TriangleAttention(nn.Module):
    """Triangle attention v2a (trunk / MSA module).

    pair2b outputs 2*num_heads values: first num_heads for starting-node,
    last num_heads for ending-node direction (ARCHITECTURE §6.3.7).

    The row-batch dimension (N rows, each attending over N columns) is
    chunked to avoid materializing an [N, H, N, N] bias tensor.  Peak
    memory drops from O(N³) to O(chunk_size × N²).
    """

    _ROW_CHUNK: int = 32

    def __init__(self, pair_dim: int, num_heads: int, head_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.pair_norm = nn.LayerNorm(pair_dim, eps=eps, affine=False)
        self.pair2b = nn.Linear(pair_dim, 2 * num_heads, bias=False)
        self.pair2qkvg1 = nn.Linear(pair_dim, 4 * num_heads * head_dim, bias=False)
        self.pair2qkvg2 = nn.Linear(pair_dim, 4 * num_heads * head_dim, bias=False)
        self.linear_out = nn.Linear(2 * num_heads * head_dim, pair_dim, bias=False)
        self.out_scalers = mx.ones((pair_dim,), dtype=mx.float32)
        self.num_heads = num_heads
        self.head_dim = head_dim

    def _sdpa_lazy(
        self,
        z_ln: mx.array,
        proj_linear: nn.Linear,
        bias_raw: mx.array,
        row_mask_bool: mx.array | None,
        *,
        transpose_pair: bool = False,
    ) -> mx.array:
        """SDPA with fused row-chunked projection to avoid materializing
        the full [b, n, n, 4*H*D] projection tensor.

        For each chunk of rows, the linear projection is computed only for
        those rows, keeping peak memory at O(chunk × N × H × D) instead of
        O(N² × H × D).
        """
        b, n, _, _ = z_ln.shape
        H, D = self.num_heads, self.head_dim
        chunk = self._ROW_CHUNK

        gate_chunks: list[mx.array] = []
        out_chunks: list[mx.array] = []
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            c = end - start
            if transpose_pair:
                z_rows = z_ln[:, :, start:end].transpose(0, 2, 1, 3)
            else:
                z_rows = z_ln[:, start:end]  # [b, c, n, pair_dim]
            proj_c = proj_linear(z_rows).reshape(b, c, n, 4, H, D)
            q_c = proj_c[:, :, :, 0].transpose(0, 1, 3, 2, 4).reshape(b * c, H, n, D)
            k_c = proj_c[:, :, :, 1].transpose(0, 1, 3, 2, 4).reshape(b * c, H, n, D)
            v_c = proj_c[:, :, :, 2].transpose(0, 1, 3, 2, 4).reshape(b * c, H, n, D)
            gate_chunks.append(proj_c[:, :, :, 3])  # [b, c, n, H, D]
            mask_c = mx.broadcast_to(
                bias_raw[:, None, :, :, :], (b, c, H, n, n)
            ).reshape(b * c, H, n, n)
            if row_mask_bool is not None:
                pm_c = row_mask_bool[:, start:end]
                attn_mask = (pm_c[:, :, :, None] & pm_c[:, :, None, :])
                mask_c = mask_c + make_additive_mask(attn_mask.reshape(b * c, 1, n, n))
            attn_c = mx.fast.scaled_dot_product_attention(
                q_c, k_c, v_c, scale=D ** -0.5, mask=mask_c,
            )
            out_chunks.append(attn_c.reshape(b, c, H, n, D))
            mx.eval(out_chunks[-1], gate_chunks[-1])

        # [b, n, H, n, D] → [b, n, n, H, D]
        out = mx.concatenate(out_chunks, axis=1).transpose(0, 1, 3, 2, 4)
        g = mx.concatenate(gate_chunks, axis=1)
        result = out * sigmoid(g)
        if transpose_pair:
            result = result.transpose(0, 2, 1, 3, 4)
        return result

    def __call__(self, z: mx.array, pair_mask: mx.array | None = None) -> mx.array:
        b, n, _, _ = z.shape
        H, D = self.num_heads, self.head_dim
        z_ln = self.pair_norm(z)

        bias_all = self.pair2b(z_ln)  # [b, n, n, 2*H]
        bias_start = bias_all[..., :H].transpose(0, 3, 1, 2)
        bias_end = bias_all[..., H:].transpose(0, 3, 1, 2)
        mx.eval(bias_start, bias_end)

        if pair_mask is not None:
            pm_bool = pair_mask.astype(mx.bool_)
            col_mask_bool = pm_bool.transpose(0, 2, 1)
        else:
            pm_bool = col_mask_bool = None

        out_s = self._sdpa_lazy(z_ln, self.pair2qkvg1, bias_start, pm_bool)
        out_e = self._sdpa_lazy(z_ln, self.pair2qkvg2, bias_end, col_mask_bool, transpose_pair=True)

        combined = mx.concatenate([
            merge_heads(out_s), merge_heads(out_e),
        ], axis=-1)
        out = self.linear_out(combined) * self.out_scalers
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

    Row-batch dimension is chunked identically to :class:`TriangleAttention`.
    """

    _ROW_CHUNK: int = 32

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
        half = self._qkvg_dim // 2
        chunk = self._ROW_CHUNK

        z_ln = self.pair_norm(z)

        # Only project the bias portion upfront (small: 2*H per element)
        bias_proj = z_ln @ self.pair2qkvgb.weight[self._qkvg_dim:].T  # [b,n,n,2H]
        bias = bias_proj.reshape(b, n, n, 2, H).transpose(0, 3, 4, 1, 2)
        if pair_mask is not None:
            pm = pair_mask.reshape(b, 1, 1, n, n)
            bias = mx.where(pm, bias, -10000.0)
        bias_s = bias[:, 0]
        bias_e = bias[:, 1]
        mx.eval(bias_s, bias_e)

        w_qkvg = self.pair2qkvgb.weight[:self._qkvg_dim]  # [qkvg_dim, pair_dim]
        w_start = w_qkvg[:half]  # [half, pair_dim]
        w_end = w_qkvg[half:]    # [half, pair_dim]

        def _run_direction(w: mx.array, bias_dir: mx.array, *, transpose: bool) -> mx.array:
            gate_chunks: list[mx.array] = []
            out_chunks: list[mx.array] = []
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                c = end - start
                if transpose:
                    z_rows = z_ln[:, :, start:end].transpose(0, 2, 1, 3)
                else:
                    z_rows = z_ln[:, start:end]
                proj_c = (z_rows @ w.T).reshape(b, c, n, 4, H, D)
                q_c = proj_c[:, :, :, 0].transpose(0, 1, 3, 2, 4).reshape(b * c, H, n, D)
                k_c = proj_c[:, :, :, 1].transpose(0, 1, 3, 2, 4).reshape(b * c, H, n, D)
                v_c = proj_c[:, :, :, 2].transpose(0, 1, 3, 2, 4).reshape(b * c, H, n, D)
                gate_chunks.append(proj_c[:, :, :, 3])
                mask_c = mx.broadcast_to(
                    bias_dir[:, None, :, :, :], (b, c, H, n, n)
                ).reshape(b * c, H, n, n)
                attn_c = mx.fast.scaled_dot_product_attention(
                    q_c, k_c, v_c, scale=D ** -0.5, mask=mask_c,
                )
                out_chunks.append(attn_c.reshape(b, c, H, n, D))
                mx.eval(out_chunks[-1], gate_chunks[-1])

            out = mx.concatenate(out_chunks, axis=1).transpose(0, 1, 3, 2, 4)
            g = mx.concatenate(gate_chunks, axis=1)
            result = out * sigmoid(g)
            if transpose:
                result = result.transpose(0, 2, 1, 3, 4)
            return result.reshape(b, n, n, H * D)

        out_s = _run_direction(w_start, bias_s, transpose=False)
        out_e = _run_direction(w_end, bias_e, transpose=True)

        combined = mx.concatenate([out_s, out_e], axis=-1)
        projected = self.linear_out(combined)
        pair_dim = z.shape[-1]
        return z + projected[..., :pair_dim] + projected[..., pair_dim:]
