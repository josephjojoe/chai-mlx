"""Modal probe: run an eager-PyTorch port of PairformerBlock (block 0) using
the weights extracted from the upstream TorchScript trunk.pt.

This mirrors ``chai_mlx.nn.layers.pairformer.PairformerBlock`` line-for-line
in eager PyTorch so we can compare MLX-block-0 against PyTorch-block-0 on
bit-identical inputs. The inputs come from
``/tmp/chai_mlx_cuda/first_block_probe/inputs.npz`` (written by
``_probe_first_block_mlx.py``).

If MLX-block-0 ≈ eager-PyTorch-block-0 (fp32 ULPs), the chai-mlx port is
numerically correct at the block level and the ~35% trunk-level gap lives
in fused TorchScript kernels that compile 48 blocks into one graph.

If MLX-block-0 ≠ eager-PyTorch-block-0, there is a port-level bug inside
one of the block's sub-ops.  The per-sub-op intermediates
(``pair_transition_out``, ``z_after_tri_mult``, ``z_after_tri_attn``,
``attn_delta``, ``s_after_attn``, ``s_after_transition``) let us bisect
exactly where.

Usage::

    modal run -m cuda_harness._probe_first_block_cuda
"""

from __future__ import annotations

from pathlib import Path

import modal

from cuda_harness.modal_common import (
    MINUTES,
    MODELS_DIR,
    app,
    chai_model_volume,
    image,
)


@app.function(
    timeout=10 * MINUTES,
    gpu="H100",
    volumes={MODELS_DIR: chai_model_volume},
    image=image,
)
def cuda_block_0_probe(inputs_npz: bytes) -> dict[str, bytes]:
    """Run eager PyTorch block-0 on fp32 and bf16 and return NPZ bytes."""
    import io

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")

    # ------------------------------------------------------------------
    # 1) Rebuild a PyTorch eager PairformerBlock that mirrors the MLX
    #    implementation ``chai_mlx.nn.layers.pairformer.PairformerBlock``.
    #    All hyperparams match the trunk's block-0 weight shapes:
    #       pair_dim=256, single_dim=384, triangle_heads=4,
    #       triangle_head_dim=64, single_heads=16, single_head_dim=24.
    # ------------------------------------------------------------------

    def _fp32_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, eps: float) -> torch.Tensor:
        """LayerNorm that always does the mean/variance reduction in fp32
        (mirroring ``chai_mlx.nn.layers.common.FP32LayerNorm``)."""
        orig_dtype = x.dtype
        x_f = x.float()
        if weight is not None:
            w_f = weight.float()
        else:
            w_f = None
        if bias is not None:
            b_f = bias.float()
        else:
            b_f = None
        y = F.layer_norm(x_f, (x_f.shape[-1],), weight=w_f, bias=b_f, eps=eps)
        return y.to(orig_dtype)

    def _lin(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """``F.linear`` that mirrors MLX's promotion rule.

        In MLX, ``x @ W.T`` with ``x`` in fp32 and ``W`` in bf16 returns fp32:
        the lower-precision operand is promoted to the higher-precision one
        before the matmul. PyTorch's ``F.linear`` instead requires both to
        have identical dtypes, so we promote both to the common type.
        """
        if x.dtype != w.dtype:
            promoted = torch.promote_types(x.dtype, w.dtype)
            if x.dtype != promoted:
                x = x.to(promoted)
            if w.dtype != promoted:
                w = w.to(promoted)
        return F.linear(x, w)

    def _add_promote(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Addition that mirrors MLX's promotion rule for mixed dtypes."""
        if a.dtype != b.dtype:
            promoted = torch.promote_types(a.dtype, b.dtype)
            return a.to(promoted) + b.to(promoted)
        return a + b

    def _mul_promote(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiplication that mirrors MLX's promotion rule for mixed dtypes."""
        if a.dtype != b.dtype:
            promoted = torch.promote_types(a.dtype, b.dtype)
            return a.to(promoted) * b.to(promoted)
        return a * b

    class Transition(nn.Module):
        def __init__(self, dim: int, expansion: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.norm_w = nn.Parameter(torch.zeros(dim))
            self.norm_b = nn.Parameter(torch.zeros(dim))
            self.up_w = nn.Parameter(torch.zeros(2 * expansion * dim, dim))
            self.down_w = nn.Parameter(torch.zeros(dim, expansion * dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            normed = _fp32_layernorm(x, self.norm_w, self.norm_b, self.eps)
            up = _lin(normed, self.up_w)
            a, b = up.chunk(2, dim=-1)
            gated = F.silu(a) * b
            return _lin(gated, self.down_w)

    class TriangleMultiplication(nn.Module):
        """Mirrors ``chai_mlx.nn.layers.triangle.TriangleMultiplication`` but
        without the chunking (we only run block 0 once; memory is fine)."""

        def __init__(self, pair_dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.ln_in_w = nn.Parameter(torch.zeros(pair_dim))
            self.ln_in_b = nn.Parameter(torch.zeros(pair_dim))
            self.merged_p = nn.Parameter(torch.zeros(4 * pair_dim, pair_dim))
            self.merged_g = nn.Parameter(torch.zeros(5 * pair_dim, pair_dim))
            self.lin_out = nn.Parameter(torch.zeros(pair_dim, pair_dim))
            self.eps = eps
            self.pair_dim = pair_dim

        def forward(self, z: torch.Tensor, pair_mask: torch.Tensor | None) -> torch.Tensor:
            d = self.pair_dim
            z_ln = _fp32_layernorm(z, self.ln_in_w, self.ln_in_b, self.eps)

            # Four "directions" from merged_p: rows 0..d, d..2d, 2d..3d, 3d..4d.
            # Paired with sigmoid(merged_g[same rows]).
            p = _lin(z_ln, self.merged_p)  # (b,n,n, 4d)
            g4 = torch.sigmoid(_lin(z_ln, self.merged_g[: 4 * d]))  # (b,n,n, 4d)
            # Direction outputs:
            a1, b1, a2, b2 = (p * g4).chunk(4, dim=-1)

            if pair_mask is not None:
                pm = pair_mask.unsqueeze(-1).to(z.dtype)
                pm_T = pair_mask.transpose(-1, -2).unsqueeze(-1).to(z.dtype)
                a1 = a1 * pm
                b1 = b1 * pm
                a2 = a2 * pm_T
                b2 = b2 * pm_T

            # Outgoing direction: sum over k of a1[...,i,k,:] * b1[...,j,k,:]
            x_out = torch.einsum("bikd,bjkd->bijd", a1, b1)
            # Incoming direction: sum over k of a2[...,k,i,:] * b2[...,k,j,:]
            x_in = torch.einsum("bkid,bkjd->bijd", a2, b2)

            # ``layernorm_out`` and ``layernorm_in`` are affine=False LNs.
            x_out_ln = F.layer_norm(x_out.float(), (d,), eps=self.eps).to(x_out.dtype)
            x_in_ln = F.layer_norm(x_in.float(), (d,), eps=self.eps).to(x_in.dtype)

            g_out = torch.sigmoid(_lin(z_ln, self.merged_g[4 * d:]))
            out = _lin(x_out_ln + x_in_ln, self.lin_out) * g_out
            return z + out

    class TriangleAttention(nn.Module):
        """Mirrors ``chai_mlx.nn.layers.triangle.TriangleAttention`` (v2a)."""

        def __init__(self, pair_dim: int, num_heads: int, head_dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.pair_dim = pair_dim
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.eps = eps
            # pair_norm is affine=False — no learnable params.
            self.pair2b = nn.Parameter(torch.zeros(2 * num_heads, pair_dim))
            self.pair2qkvg1 = nn.Parameter(torch.zeros(4 * num_heads * head_dim, pair_dim))
            self.pair2qkvg2 = nn.Parameter(torch.zeros(4 * num_heads * head_dim, pair_dim))
            self.linear_out = nn.Parameter(torch.zeros(pair_dim, 2 * num_heads * head_dim))
            self.out_scalers = nn.Parameter(torch.ones(pair_dim))

        def _run_direction(
            self, z_ln: torch.Tensor, proj_w: torch.Tensor, bias_dir: torch.Tensor,
            pair_mask_2d: torch.Tensor | None, *, transpose_pair: bool,
        ) -> torch.Tensor:
            b, n, _, _ = z_ln.shape
            H, D = self.num_heads, self.head_dim
            if transpose_pair:
                z_rows = z_ln.transpose(1, 2)  # (b,n,n,d) with roles swapped
            else:
                z_rows = z_ln
            proj = _lin(z_rows, proj_w).reshape(b, n, n, H, 4, D)
            q = proj[..., 0, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            k = proj[..., 1, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            v = proj[..., 2, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            g = proj[..., 3, :]  # (b, n, n, H, D)
            # bias: (b, H, n, n) → broadcast to (b, n, H, n, n) → reshape (b*n, H, n, n)
            mask_full = bias_dir.unsqueeze(1).expand(b, n, H, n, n).reshape(b * n, H, n, n)
            if pair_mask_2d is not None:
                pm = pair_mask_2d.to(torch.bool)  # (b, n, n)
                attn_mask = pm.unsqueeze(-1) & pm.unsqueeze(-2)  # (b, n, n, n)
                add_mask = torch.where(
                    attn_mask.reshape(b * n, 1, n, n),
                    torch.zeros((), dtype=mask_full.dtype, device=mask_full.device),
                    torch.full((), -1e4, dtype=mask_full.dtype, device=mask_full.device),
                )
                mask_full = mask_full + add_mask
            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask_full, scale=D ** -0.5,
            )
            # (b*n, H, n, D) → (b, n, H, n, D) → (b, n, n, H, D) so gate shape aligns.
            attn = attn.reshape(b, n, H, n, D).permute(0, 1, 3, 2, 4)
            result = attn * torch.sigmoid(g)
            return result  # (b, n, n, H, D)

        def forward(self, z: torch.Tensor, pair_mask: torch.Tensor | None) -> torch.Tensor:
            b, n, _, _ = z.shape
            H = self.num_heads
            z_ln = F.layer_norm(z.float(), (self.pair_dim,), eps=self.eps).to(z.dtype)

            bias_all = _lin(z_ln, self.pair2b)  # (b, n, n, 2H)
            bias_start = bias_all[..., :H].permute(0, 3, 1, 2)
            bias_end = bias_all[..., H:].permute(0, 3, 1, 2)

            out_s = self._run_direction(
                z_ln, self.pair2qkvg1, bias_start, pair_mask, transpose_pair=False,
            )  # (b, n, n, H, D)
            if pair_mask is not None:
                col_mask = pair_mask.transpose(-1, -2)
            else:
                col_mask = None
            out_e = self._run_direction(
                z_ln, self.pair2qkvg2, bias_end, col_mask, transpose_pair=True,
            )

            # (b, n, n, H*D) each
            out_s_f = out_s.reshape(b, n, n, H * self.head_dim)
            out_e_f = out_e.reshape(b, n, n, H * self.head_dim)
            combined = torch.cat([out_s_f, out_e_f], dim=-1)
            lin_out = _lin(combined, self.linear_out)
            if lin_out.dtype != self.out_scalers.dtype:
                promoted = torch.promote_types(lin_out.dtype, self.out_scalers.dtype)
                lin_out = lin_out.to(promoted)
                scalers = self.out_scalers.to(promoted)
            else:
                scalers = self.out_scalers
            out = lin_out * scalers
            if z.dtype != out.dtype:
                promoted = torch.promote_types(z.dtype, out.dtype)
                return z.to(promoted) + out.to(promoted)
            return z + out

    class AttentionPairBias(nn.Module):
        """Mirrors ``chai_mlx.nn.layers.attention.AttentionPairBias``."""

        def __init__(
            self, single_dim: int, pair_dim: int, num_heads: int, head_dim: int, eps: float = 1e-5,
        ) -> None:
            super().__init__()
            self.single_dim = single_dim
            self.pair_dim = pair_dim
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.eps = eps
            self.single_norm_w = nn.Parameter(torch.zeros(single_dim))
            self.single_norm_b = nn.Parameter(torch.zeros(single_dim))
            self.pair_norm_w = nn.Parameter(torch.zeros(pair_dim))
            self.pair_norm_b = nn.Parameter(torch.zeros(pair_dim))
            self.pair_linear = nn.Parameter(torch.zeros(num_heads, pair_dim))
            self.input2qkvg = nn.Parameter(torch.zeros(4 * num_heads * head_dim, single_dim))
            self.output_proj = nn.Parameter(torch.zeros(single_dim, num_heads * head_dim))
            self.query_bias = nn.Parameter(torch.zeros(num_heads, head_dim))

        def forward(self, x: torch.Tensor, pair: torch.Tensor, pair_mask: torch.Tensor | None) -> torch.Tensor:
            b, n, _ = x.shape
            H, D = self.num_heads, self.head_dim

            # Pair bias: LN(pair) → pair_linear (H heads) → (b, H, n, n) + additive mask
            pair_ln = _fp32_layernorm(pair, self.pair_norm_w, self.pair_norm_b, self.eps)
            bias = _lin(pair_ln, self.pair_linear).permute(0, 3, 1, 2)  # (b, H, n, n)
            if pair_mask is not None:
                pm = pair_mask.to(torch.bool)  # (b, n, n)
                add = torch.where(
                    pm,
                    torch.zeros((), dtype=bias.dtype, device=bias.device),
                    torch.full((), -1e4, dtype=bias.dtype, device=bias.device),
                )
                bias = bias + add.unsqueeze(1)

            # Single stream: LN → qkvg projection → split
            x_ln = _fp32_layernorm(x, self.single_norm_w, self.single_norm_b, self.eps)
            qkvg = _lin(x_ln, self.input2qkvg)  # (b, n, 4*H*D)
            q, k, v, g = qkvg.chunk(4, dim=-1)
            q = q.reshape(b, n, H, D).permute(0, 2, 1, 3)
            k = k.reshape(b, n, H, D).permute(0, 2, 1, 3)
            v = v.reshape(b, n, H, D).permute(0, 2, 1, 3)
            g = g.reshape(b, n, H, D)  # keep (b, n, H, D)
            qb = self.query_bias.unsqueeze(0).unsqueeze(-2)
            if q.dtype != qb.dtype:
                promoted = torch.promote_types(q.dtype, qb.dtype)
                q = q.to(promoted)
                qb = qb.to(promoted)
            q = q + qb  # (b, H, n, D)

            sdpa_dtype = q.dtype
            for t in (k, v, bias):
                if t is not None:
                    sdpa_dtype = torch.promote_types(sdpa_dtype, t.dtype)
            q = q.to(sdpa_dtype)
            k = k.to(sdpa_dtype)
            v = v.to(sdpa_dtype)
            bias_sdpa = bias.to(sdpa_dtype) if bias is not None else None

            attn = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias_sdpa, scale=D ** -0.5,
            )  # (b, H, n, D)
            attn = attn.permute(0, 2, 1, 3)  # (b, n, H, D)
            g_sig = torch.sigmoid(g)
            if attn.dtype != g_sig.dtype:
                promoted = torch.promote_types(attn.dtype, g_sig.dtype)
                attn = attn.to(promoted)
                g_sig = g_sig.to(promoted)
            attn = attn * g_sig  # (b, n, H, D)
            attn_flat = attn.reshape(b, n, H * D)
            return _lin(attn_flat, self.output_proj)

    class PairformerBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transition_pair = Transition(256, expansion=2)
            self.triangle_multiplication = TriangleMultiplication(256)
            self.triangle_attention = TriangleAttention(256, num_heads=4, head_dim=64)
            self.attention_pair_bias = AttentionPairBias(384, 256, num_heads=16, head_dim=24)
            self.transition_single = Transition(384, expansion=2)

        def forward(
            self,
            z: torch.Tensor,
            s: torch.Tensor,
            pair_mask: torch.Tensor | None,
            single_mask: torch.Tensor | None,
        ) -> dict[str, torch.Tensor]:
            pair_transition_out = self.transition_pair(z)
            z_after_tri_mult = self.triangle_multiplication(z, pair_mask)
            z_after_residual = _add_promote(z_after_tri_mult, pair_transition_out)
            z_after_tri_attn = self.triangle_attention(z_after_residual, pair_mask)
            attn_delta = self.attention_pair_bias(s, z_after_tri_attn, pair_mask)
            if single_mask is not None:
                mask_f = single_mask.to(attn_delta.dtype).unsqueeze(-1)
                attn_delta = _mul_promote(attn_delta, mask_f)
            s_after_attn = _add_promote(s, attn_delta)
            s_after_transition = _add_promote(s_after_attn, self.transition_single(s_after_attn))
            return {
                "pair_transition_out": pair_transition_out,
                "z_after_tri_mult": z_after_tri_mult,
                "z_after_residual": z_after_residual,
                "z_after_tri_attn": z_after_tri_attn,
                "attn_delta": attn_delta,
                "s_after_attn": s_after_attn,
                "s_after_transition": s_after_transition,
                "z_final": z_after_tri_attn,
                "s_final": s_after_transition,
            }

    # ------------------------------------------------------------------
    # 2) Load block 0 weights from upstream trunk.pt.
    # ------------------------------------------------------------------
    trunk_path = MODELS_DIR / "models_v2" / "trunk.pt"
    print(f"Loading {trunk_path}...")
    trunk = torch.jit.load(str(trunk_path), map_location="cpu")
    trunk.eval()

    def _sub(root, dotted: str):
        obj = root
        for part in dotted.split("."):
            obj = getattr(obj, str(part))
        return obj

    block_0 = _sub(trunk, "pairformer_stack.blocks.0")

    ts_params = {name: p.detach().clone() for name, p in block_0.named_parameters()}
    for k, v in ts_params.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    # ------------------------------------------------------------------
    # 3) Build eager PyTorch block and assign weights.
    # ------------------------------------------------------------------
    block = PairformerBlock()

    def _assign(param: torch.nn.Parameter, value: torch.Tensor) -> None:
        if tuple(param.shape) != tuple(value.shape):
            raise ValueError(
                f"shape mismatch: param {tuple(param.shape)} vs value {tuple(value.shape)}"
            )
        param.data.copy_(value)

    # Transition (pair): chai-lab stores ``linear_no_bias_ab`` (1024, 256) and
    # ``linear_out`` (256, 512).  Our eager Transition uses ``up_w`` (1024, 256)
    # and ``down_w`` (256, 512) with identical semantics.
    _assign(block.transition_pair.norm_w, ts_params["transition_pair.layer_norm.weight"])
    _assign(block.transition_pair.norm_b, ts_params["transition_pair.layer_norm.bias"])
    _assign(block.transition_pair.up_w, ts_params["transition_pair.linear_no_bias_ab.weight"])
    _assign(block.transition_pair.down_w, ts_params["transition_pair.linear_out.weight"])

    # Triangle multiplication.
    _assign(block.triangle_multiplication.ln_in_w, ts_params["triangle_multiplication.layernorm_z_in.weight"])
    _assign(block.triangle_multiplication.ln_in_b, ts_params["triangle_multiplication.layernorm_z_in.bias"])
    _assign(block.triangle_multiplication.merged_p, ts_params["triangle_multiplication.merged_linear_p.weight"])
    _assign(block.triangle_multiplication.merged_g, ts_params["triangle_multiplication.merged_linear_g.weight"])
    _assign(block.triangle_multiplication.lin_out, ts_params["triangle_multiplication.linear_z_out.weight"])

    # Triangle attention.
    _assign(block.triangle_attention.pair2b, ts_params["triangle_attention.pair2b.weight"])
    _assign(block.triangle_attention.pair2qkvg1, ts_params["triangle_attention.pair2qkvg1.weight"])
    _assign(block.triangle_attention.pair2qkvg2, ts_params["triangle_attention.pair2qkvg2.weight"])
    _assign(block.triangle_attention.linear_out, ts_params["triangle_attention.linear_out.weight"])
    _assign(block.triangle_attention.out_scalers, ts_params["triangle_attention.out_scalers"])

    # Attention pair bias.
    _assign(block.attention_pair_bias.single_norm_w, ts_params["attention_pair_bias.single_layer_norm.weight"])
    _assign(block.attention_pair_bias.single_norm_b, ts_params["attention_pair_bias.single_layer_norm.bias"])
    _assign(block.attention_pair_bias.pair_norm_w, ts_params["attention_pair_bias.pair_layer_norm.weight"])
    _assign(block.attention_pair_bias.pair_norm_b, ts_params["attention_pair_bias.pair_layer_norm.bias"])
    _assign(block.attention_pair_bias.pair_linear, ts_params["attention_pair_bias.pair_linear.weight"])
    _assign(block.attention_pair_bias.query_bias, ts_params["attention_pair_bias.attention.query_bias"])
    # input2qkvg: TS shape (384, 4, 16, 24) → (1536, 384). See
    # ``chai_mlx.io.weights.name_map.reshape_einsum_weight`` for the exact
    # contraction; this is ``.reshape(384, -1).T``.
    w_qkvg = ts_params["attention_pair_bias.attention.input2qkvg.weight"]
    in_dim = w_qkvg.shape[0]
    _assign(block.attention_pair_bias.input2qkvg, w_qkvg.reshape(in_dim, -1).T.contiguous())
    # output_proj: TS shape (16, 24, 384) → (384, 384).  ``.reshape(-1, out).T``.
    w_out = ts_params["attention_pair_bias.attention.output_proj.weight"]
    out_dim = w_out.shape[-1]
    _assign(block.attention_pair_bias.output_proj, w_out.reshape(-1, out_dim).T.contiguous())

    # Transition (single).
    _assign(block.transition_single.norm_w, ts_params["transition_single.layer_norm.weight"])
    _assign(block.transition_single.norm_b, ts_params["transition_single.layer_norm.bias"])
    _assign(block.transition_single.up_w, ts_params["transition_single.linear_no_bias_ab.weight"])
    _assign(block.transition_single.down_w, ts_params["transition_single.linear_out.weight"])

    block = block.to(device).eval()

    # ------------------------------------------------------------------
    # 4) Load shared inputs & run for fp32 and bf16.
    # ------------------------------------------------------------------
    import numpy as np

    inputs = np.load(io.BytesIO(inputs_npz))
    single_np = inputs["single"]
    pair_np = inputs["pair"]
    pair_mask_np = inputs["pair_mask"]
    single_mask_np = inputs["single_mask"]

    result: dict[str, bytes] = {}

    for dtype_name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        print(f"\n=== {dtype_name} ===")
        s = torch.from_numpy(single_np).to(device).to(dtype)
        z = torch.from_numpy(pair_np).to(device).to(dtype)
        pair_mask = torch.from_numpy(pair_mask_np).to(device)
        single_mask = torch.from_numpy(single_mask_np).to(device)

        # Cast the block's parameters to the target dtype (except the
        # LayerNorm weights which our ``_fp32_layernorm`` promotes to fp32
        # automatically). To simplify, we keep all params in fp32 and let the
        # forward do its own casts — this matches chai-mlx's
        # ``_preserve_fp32_param_keys`` semantics for LN weights, and the
        # sub-op code upcasts to the x.dtype where needed via F.linear.
        # We explicitly cast the non-LN params we care about.
        # To mirror chai-mlx exactly, we cast the eager block's linear
        # weights to the compute dtype and keep layernorm weights fp32.
        def _cast_params(mod: torch.nn.Module, dtype: torch.dtype) -> None:
            for name, p in mod.named_parameters():
                if any(tag in name for tag in (
                    "norm_w", "norm_b", "ln_in_w", "ln_in_b", "out_scalers", "query_bias",
                )):
                    continue
                p.data = p.data.to(dtype)

        # Reset to fp32 before casting (on fp32 pass we want fp32 params).
        block_fresh = PairformerBlock()
        for name, p_target in block_fresh.named_parameters():
            p_target.data.copy_(dict(block.named_parameters())[name].data.float())
        block_fresh = block_fresh.to(device).eval()
        _cast_params(block_fresh, dtype)

        outs = block_fresh(z, s, pair_mask, single_mask)

        dump = {k: v.detach().float().cpu().numpy() for k, v in outs.items()}
        for k, v in dump.items():
            print(f"  {k:26s} shape={v.shape}  max_abs={float(np.abs(v).max()):.4f}  mean={float(v.mean()):+.4e}")
        buf = io.BytesIO()
        np.savez(buf, **dump)
        result[f"cuda_out_{dtype_name}"] = buf.getvalue()

    return result


@app.local_entrypoint()
def main() -> None:
    inputs_path = Path("/tmp/chai_mlx_cuda/first_block_probe/inputs.npz")
    inputs_npz = inputs_path.read_bytes()
    result = cuda_block_0_probe.remote(inputs_npz)
    out_dir = inputs_path.parent
    for name, data in result.items():
        out_path = out_dir / f"{name}.npz"
        out_path.write_bytes(data)
        print(f"saved {out_path} ({len(data) / 1024**2:.1f} MB)")
