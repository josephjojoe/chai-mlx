"""Modal probe: eager-PyTorch 48-block pairformer stack at fp32.

Companion to ``_probe_full_stack_mlx.py``.  Consumes the same synthetic
``inputs.npz`` and walks all 48 pairformer blocks in eager PyTorch on H100
with TF32 off, dumping ``(s, z)`` after each block.  Weights for every
block are lifted straight from ``models_v2/trunk.pt``.

This extends the block-0 probe (``_probe_first_block_cuda.py``) to the
full stack so we can ask:

  Q: Starting from identical synthetic inputs, does MLX's 48-block stack
     match eager-PyTorch's 48-block stack at fp32?

If yes (rel err stays at 1e-5 to 1e-4 territory across all 48 blocks),
then the 35% observed in ``parity_1L2Y_s42_fp32.json`` is NOT coming
from the pairformer stack itself on synthetic inputs -- it's either
driven by something specific to the real trunk-0 input distribution
(e.g. a numerical pathology only certain element magnitudes hit), or
by an upstream drift we haven't isolated yet (MLX's template_embedder
/ msa_module / recycle projections running on fp32 CUDA embeddings).

Usage::

    modal run -m cuda_harness._probe_full_stack_cuda

Writes locally:
    /tmp/chai_mlx_cuda/full_stack_probe/cuda_out_fp32.npz
        ``s_block_{i}`` and ``z_block_{i}`` for i=0..47, plus
        ``s_final``, ``z_final``.
"""
from __future__ import annotations

from pathlib import Path

from cuda_harness.modal_common import (
    MINUTES,
    MODELS_DIR,
    app,
    chai_model_volume,
    image,
)


@app.function(
    timeout=15 * MINUTES,
    gpu="H100",
    volumes={MODELS_DIR: chai_model_volume},
    image=image,
)
def cuda_full_stack_probe(inputs_npz: bytes) -> dict[str, bytes]:
    """Run the eager-PyTorch 48-block pairformer stack at fp32."""
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
    # 1) Eager PyTorch port of PairformerBlock (fp32 only, no bf16 casts).
    #    Identical to ``_probe_first_block_cuda.py`` module-for-module;
    #    we duplicate it here so this file is standalone.
    # ------------------------------------------------------------------

    def _fp32_layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, eps: float) -> torch.Tensor:
        orig = x.dtype
        y = F.layer_norm(x.float(), (x.shape[-1],), weight=weight.float(), bias=bias.float() if bias is not None else None, eps=eps)
        return y.to(orig)

    class Transition(nn.Module):
        def __init__(self, dim: int, expansion: int = 2, eps: float = 1e-5) -> None:
            super().__init__()
            self.norm_w = nn.Parameter(torch.zeros(dim))
            self.norm_b = nn.Parameter(torch.zeros(dim))
            self.up_w = nn.Parameter(torch.zeros(2 * expansion * dim, dim))
            self.down_w = nn.Parameter(torch.zeros(dim, expansion * dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            normed = _fp32_layernorm(x, self.norm_w, self.norm_b, self.eps)
            up = F.linear(normed, self.up_w)
            a, b = up.chunk(2, dim=-1)
            return F.linear(F.silu(a) * b, self.down_w)

    class TriangleMultiplication(nn.Module):
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
            p = F.linear(z_ln, self.merged_p)
            g4 = torch.sigmoid(F.linear(z_ln, self.merged_g[: 4 * d]))
            a1, b1, a2, b2 = (p * g4).chunk(4, dim=-1)
            if pair_mask is not None:
                pm = pair_mask.unsqueeze(-1).to(z.dtype)
                pm_T = pair_mask.transpose(-1, -2).unsqueeze(-1).to(z.dtype)
                a1 = a1 * pm
                b1 = b1 * pm
                a2 = a2 * pm_T
                b2 = b2 * pm_T
            x_out = torch.einsum("bikd,bjkd->bijd", a1, b1)
            x_in = torch.einsum("bkid,bkjd->bijd", a2, b2)
            x_out_ln = F.layer_norm(x_out.float(), (d,), eps=self.eps).to(x_out.dtype)
            x_in_ln = F.layer_norm(x_in.float(), (d,), eps=self.eps).to(x_in.dtype)
            g_out = torch.sigmoid(F.linear(z_ln, self.merged_g[4 * d:]))
            out = F.linear(x_out_ln + x_in_ln, self.lin_out) * g_out
            return z + out

    class TriangleAttention(nn.Module):
        def __init__(self, pair_dim: int, num_heads: int, head_dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.pair_dim = pair_dim
            self.num_heads = num_heads
            self.head_dim = head_dim
            H, D = num_heads, head_dim
            self.pair2b = nn.Parameter(torch.zeros(2 * H, pair_dim))
            self.pair2qkvg1 = nn.Parameter(torch.zeros(H * 4 * D, pair_dim))
            self.pair2qkvg2 = nn.Parameter(torch.zeros(H * 4 * D, pair_dim))
            self.linear_out = nn.Parameter(torch.zeros(pair_dim, 2 * H * D))
            self.out_scalers = nn.Parameter(torch.zeros(pair_dim))
            self.eps = eps

        def _run_direction(self, z_ln, proj_w, bias_dir, pair_mask_2d, *, transpose_pair: bool):
            b, n, _, _ = z_ln.shape
            H, D = self.num_heads, self.head_dim
            z_rows = z_ln.transpose(1, 2) if transpose_pair else z_ln
            proj = F.linear(z_rows, proj_w).reshape(b, n, n, H, 4, D)
            q = proj[..., 0, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            k = proj[..., 1, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            v = proj[..., 2, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            g = proj[..., 3, :]
            mask_full = bias_dir.unsqueeze(1).expand(b, n, H, n, n).reshape(b * n, H, n, n)
            if pair_mask_2d is not None:
                pm = pair_mask_2d.to(torch.bool)
                attn_mask = pm.unsqueeze(-1) & pm.unsqueeze(-2)
                add_mask = torch.where(
                    attn_mask.reshape(b * n, 1, n, n),
                    torch.zeros((), dtype=mask_full.dtype, device=mask_full.device),
                    torch.full((), -1e4, dtype=mask_full.dtype, device=mask_full.device),
                )
                mask_full = mask_full + add_mask
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_full, scale=D ** -0.5)
            attn = attn.reshape(b, n, H, n, D).permute(0, 1, 3, 2, 4)
            return attn * torch.sigmoid(g)

        def forward(self, z: torch.Tensor, pair_mask: torch.Tensor | None) -> torch.Tensor:
            b, n, _, _ = z.shape
            H = self.num_heads
            z_ln = F.layer_norm(z.float(), (self.pair_dim,), eps=self.eps).to(z.dtype)
            bias_all = F.linear(z_ln, self.pair2b)
            bias_start = bias_all[..., :H].permute(0, 3, 1, 2)
            bias_end = bias_all[..., H:].permute(0, 3, 1, 2)
            out_s = self._run_direction(z_ln, self.pair2qkvg1, bias_start, pair_mask, transpose_pair=False)
            col_mask = pair_mask.transpose(-1, -2) if pair_mask is not None else None
            out_e = self._run_direction(z_ln, self.pair2qkvg2, bias_end, col_mask, transpose_pair=True)
            out_s_f = out_s.reshape(b, n, n, H * self.head_dim)
            out_e_f = out_e.reshape(b, n, n, H * self.head_dim)
            combined = torch.cat([out_s_f, out_e_f], dim=-1)
            return z + F.linear(combined, self.linear_out) * self.out_scalers

    class AttentionPairBias(nn.Module):
        def __init__(self, single_dim: int, pair_dim: int, num_heads: int, head_dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            H, D = num_heads, head_dim
            self.single_dim = single_dim
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.single_norm_w = nn.Parameter(torch.zeros(single_dim))
            self.single_norm_b = nn.Parameter(torch.zeros(single_dim))
            self.pair_norm_w = nn.Parameter(torch.zeros(pair_dim))
            self.pair_norm_b = nn.Parameter(torch.zeros(pair_dim))
            self.pair_linear = nn.Parameter(torch.zeros(H, pair_dim))
            self.input2qkvg = nn.Parameter(torch.zeros(4 * H * D, single_dim))
            self.output_proj = nn.Parameter(torch.zeros(single_dim, H * D))
            self.query_bias = nn.Parameter(torch.zeros(H, D))
            self.eps = eps

        def forward(self, x: torch.Tensor, pair: torch.Tensor, pair_mask: torch.Tensor | None) -> torch.Tensor:
            b, n, _ = x.shape
            H, D = self.num_heads, self.head_dim

            pair_ln = _fp32_layernorm(pair, self.pair_norm_w, self.pair_norm_b, self.eps)
            bias = F.linear(pair_ln, self.pair_linear).permute(0, 3, 1, 2)
            if pair_mask is not None:
                pm = pair_mask.to(torch.bool)
                add = torch.where(
                    pm,
                    torch.zeros((), dtype=bias.dtype, device=bias.device),
                    torch.full((), -1e4, dtype=bias.dtype, device=bias.device),
                )
                bias = bias + add.unsqueeze(1)

            x_ln = _fp32_layernorm(x, self.single_norm_w, self.single_norm_b, self.eps)
            qkvg = F.linear(x_ln, self.input2qkvg)
            q, k, v, g = qkvg.chunk(4, dim=-1)
            q = q.reshape(b, n, H, D).permute(0, 2, 1, 3)
            k = k.reshape(b, n, H, D).permute(0, 2, 1, 3)
            v = v.reshape(b, n, H, D).permute(0, 2, 1, 3)
            g = g.reshape(b, n, H, D)
            q = q + self.query_bias.unsqueeze(0).unsqueeze(-2)

            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=D ** -0.5)
            attn = attn.permute(0, 2, 1, 3)
            attn = attn * torch.sigmoid(g)
            return F.linear(attn.reshape(b, n, H * D), self.output_proj)

    class PairformerBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transition_pair = Transition(256, expansion=2)
            self.triangle_multiplication = TriangleMultiplication(256)
            self.triangle_attention = TriangleAttention(256, num_heads=4, head_dim=64)
            self.attention_pair_bias = AttentionPairBias(384, 256, num_heads=16, head_dim=24)
            self.transition_single = Transition(384, expansion=2)

        def forward(self, z: torch.Tensor, s: torch.Tensor, pair_mask, single_mask) -> tuple:
            pair_transition_out = self.transition_pair(z)
            z = self.triangle_multiplication(z, pair_mask)
            z = z + pair_transition_out
            z = self.triangle_attention(z, pair_mask)
            attn_delta = self.attention_pair_bias(s, z, pair_mask)
            if single_mask is not None:
                attn_delta = attn_delta * single_mask.to(attn_delta.dtype).unsqueeze(-1)
            s = s + attn_delta
            s = s + self.transition_single(s)
            return z, s

    # ------------------------------------------------------------------
    # 2) Load trunk.pt and build all 48 eager blocks.
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

    def _assign(param, value):
        if tuple(param.shape) != tuple(value.shape):
            raise ValueError(f"shape mismatch: {tuple(param.shape)} vs {tuple(value.shape)}")
        param.data.copy_(value)

    def _load_block_i(i: int) -> PairformerBlock:
        ts_block = _sub(trunk, f"pairformer_stack.blocks.{i}")
        ts_params = {name: p.detach().clone() for name, p in ts_block.named_parameters()}
        blk = PairformerBlock()
        _assign(blk.transition_pair.norm_w, ts_params["transition_pair.layer_norm.weight"])
        _assign(blk.transition_pair.norm_b, ts_params["transition_pair.layer_norm.bias"])
        _assign(blk.transition_pair.up_w, ts_params["transition_pair.linear_no_bias_ab.weight"])
        _assign(blk.transition_pair.down_w, ts_params["transition_pair.linear_out.weight"])
        _assign(blk.triangle_multiplication.ln_in_w, ts_params["triangle_multiplication.layernorm_z_in.weight"])
        _assign(blk.triangle_multiplication.ln_in_b, ts_params["triangle_multiplication.layernorm_z_in.bias"])
        _assign(blk.triangle_multiplication.merged_p, ts_params["triangle_multiplication.merged_linear_p.weight"])
        _assign(blk.triangle_multiplication.merged_g, ts_params["triangle_multiplication.merged_linear_g.weight"])
        _assign(blk.triangle_multiplication.lin_out, ts_params["triangle_multiplication.linear_z_out.weight"])
        _assign(blk.triangle_attention.pair2b, ts_params["triangle_attention.pair2b.weight"])
        _assign(blk.triangle_attention.pair2qkvg1, ts_params["triangle_attention.pair2qkvg1.weight"])
        _assign(blk.triangle_attention.pair2qkvg2, ts_params["triangle_attention.pair2qkvg2.weight"])
        _assign(blk.triangle_attention.linear_out, ts_params["triangle_attention.linear_out.weight"])
        _assign(blk.triangle_attention.out_scalers, ts_params["triangle_attention.out_scalers"])
        _assign(blk.attention_pair_bias.single_norm_w, ts_params["attention_pair_bias.single_layer_norm.weight"])
        _assign(blk.attention_pair_bias.single_norm_b, ts_params["attention_pair_bias.single_layer_norm.bias"])
        _assign(blk.attention_pair_bias.pair_norm_w, ts_params["attention_pair_bias.pair_layer_norm.weight"])
        _assign(blk.attention_pair_bias.pair_norm_b, ts_params["attention_pair_bias.pair_layer_norm.bias"])
        _assign(blk.attention_pair_bias.pair_linear, ts_params["attention_pair_bias.pair_linear.weight"])
        _assign(blk.attention_pair_bias.query_bias, ts_params["attention_pair_bias.attention.query_bias"])
        w_qkvg = ts_params["attention_pair_bias.attention.input2qkvg.weight"]
        _assign(blk.attention_pair_bias.input2qkvg, w_qkvg.reshape(w_qkvg.shape[0], -1).T.contiguous())
        w_out = ts_params["attention_pair_bias.attention.output_proj.weight"]
        _assign(blk.attention_pair_bias.output_proj, w_out.reshape(-1, w_out.shape[-1]).T.contiguous())
        _assign(blk.transition_single.norm_w, ts_params["transition_single.layer_norm.weight"])
        _assign(blk.transition_single.norm_b, ts_params["transition_single.layer_norm.bias"])
        _assign(blk.transition_single.up_w, ts_params["transition_single.linear_no_bias_ab.weight"])
        _assign(blk.transition_single.down_w, ts_params["transition_single.linear_out.weight"])
        return blk.to(device).eval()

    print("Building 48 eager PyTorch blocks...")
    blocks = []
    for i in range(48):
        blocks.append(_load_block_i(i))
        if i % 12 == 0 or i == 47:
            print(f"  block {i:2d} loaded")

    # Drop the TorchScript trunk from memory now we have all weights.
    del trunk
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 3) Run the stack at fp32.
    # ------------------------------------------------------------------
    inputs = np.load(io.BytesIO(inputs_npz))
    s = torch.from_numpy(inputs["single"]).to(device).to(torch.float32)
    z = torch.from_numpy(inputs["pair"]).to(device).to(torch.float32)
    pair_mask = torch.from_numpy(inputs["pair_mask"]).to(device)
    single_mask = torch.from_numpy(inputs["single_mask"]).to(device)

    dump: dict[str, np.ndarray] = {}
    for i, blk in enumerate(blocks):
        z, s = blk(z, s, pair_mask, single_mask)
        dump[f"s_block_{i:02d}"] = s.detach().float().cpu().numpy()
        dump[f"z_block_{i:02d}"] = z.detach().float().cpu().numpy()
        if i % 12 == 0 or i == 47:
            print(
                f"  block {i:2d}: s max_abs={float(s.abs().max()):.3f} "
                f"z max_abs={float(z.abs().max()):.3f}"
            )
    dump["s_final"] = dump[f"s_block_{len(blocks) - 1:02d}"]
    dump["z_final"] = dump[f"z_block_{len(blocks) - 1:02d}"]

    buf = io.BytesIO()
    np.savez(buf, **dump)
    return {"cuda_out_fp32": buf.getvalue()}


@app.local_entrypoint()
def main() -> None:
    inputs_path = Path("/tmp/chai_mlx_cuda/full_stack_probe/inputs.npz")
    out_dir = inputs_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_npz = inputs_path.read_bytes()
    result = cuda_full_stack_probe.remote(inputs_npz)
    for name, data in result.items():
        out_path = out_dir / f"{name}.npz"
        out_path.write_bytes(data)
        print(f"saved {out_path} ({len(data) / 1024**2:.1f} MB)")
