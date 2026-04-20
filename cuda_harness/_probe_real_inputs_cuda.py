"""Run the eager-PyTorch 48-block pairformer stack at fp32 on the real
post-msa inputs that MLX produced at recycle 0 of 1L2Y.

Companion to ``_probe_real_inputs_mlx.py``.  The inputs are the MLX-side
``(s, z, pair_mask, single_mask)`` captured just after ``msa_module``; we
feed them unchanged to the eager-PyTorch stack.  If MLX and eager-PyTorch
produce outputs that match to ULP precision here, then:

  - the pairformer stack is faithfully ported (consistent with the
    synthetic-input probe);
  - MLX's upstream ``msa_module`` / ``template_embedder`` /
    ``recycle_proj`` is what drifts vs CUDA on the real 1L2Y run,
    because that's what produces inputs that differ from what CUDA's
    pairformer sees.

If they *don't* match (rel err grows past ~1e-4), then the pairformer
itself is sensitive to input distribution in ways our randn probe
didn't expose, i.e. there is a port-level issue only real data hits.
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
def cuda_real_inputs_probe(inputs_npz: bytes) -> dict[str, bytes]:
    import io

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")

    # --- Eager PairformerBlock (same layout as _probe_full_stack_cuda) ---
    def _ln(x, w, b, eps):
        orig = x.dtype
        y = F.layer_norm(x.float(), (x.shape[-1],), weight=w.float(), bias=b.float() if b is not None else None, eps=eps)
        return y.to(orig)

    class Transition(nn.Module):
        def __init__(self, dim, expansion=2, eps=1e-5):
            super().__init__()
            self.norm_w = nn.Parameter(torch.zeros(dim))
            self.norm_b = nn.Parameter(torch.zeros(dim))
            self.up_w = nn.Parameter(torch.zeros(2 * expansion * dim, dim))
            self.down_w = nn.Parameter(torch.zeros(dim, expansion * dim))
            self.eps = eps

        def forward(self, x):
            normed = _ln(x, self.norm_w, self.norm_b, self.eps)
            up = F.linear(normed, self.up_w)
            a, b = up.chunk(2, dim=-1)
            return F.linear(F.silu(a) * b, self.down_w)

    class TriangleMultiplication(nn.Module):
        def __init__(self, pair_dim, eps=1e-5):
            super().__init__()
            self.ln_in_w = nn.Parameter(torch.zeros(pair_dim))
            self.ln_in_b = nn.Parameter(torch.zeros(pair_dim))
            self.merged_p = nn.Parameter(torch.zeros(4 * pair_dim, pair_dim))
            self.merged_g = nn.Parameter(torch.zeros(5 * pair_dim, pair_dim))
            self.lin_out = nn.Parameter(torch.zeros(pair_dim, pair_dim))
            self.eps = eps
            self.pair_dim = pair_dim

        def forward(self, z, pair_mask):
            d = self.pair_dim
            z_ln = _ln(z, self.ln_in_w, self.ln_in_b, self.eps)
            p = F.linear(z_ln, self.merged_p)
            g4 = torch.sigmoid(F.linear(z_ln, self.merged_g[: 4 * d]))
            a1, b1, a2, b2 = (p * g4).chunk(4, dim=-1)
            if pair_mask is not None:
                pm = pair_mask.unsqueeze(-1).to(z.dtype)
                pm_T = pair_mask.transpose(-1, -2).unsqueeze(-1).to(z.dtype)
                a1, b1, a2, b2 = a1 * pm, b1 * pm, a2 * pm_T, b2 * pm_T
            x_out = torch.einsum("bikd,bjkd->bijd", a1, b1)
            x_in = torch.einsum("bkid,bkjd->bijd", a2, b2)
            x_out_ln = F.layer_norm(x_out.float(), (d,), eps=self.eps).to(x_out.dtype)
            x_in_ln = F.layer_norm(x_in.float(), (d,), eps=self.eps).to(x_in.dtype)
            g_out = torch.sigmoid(F.linear(z_ln, self.merged_g[4 * d:]))
            return z + F.linear(x_out_ln + x_in_ln, self.lin_out) * g_out

    class TriangleAttention(nn.Module):
        def __init__(self, pair_dim, num_heads, head_dim, eps=1e-5):
            super().__init__()
            self.pair_dim, self.num_heads, self.head_dim, self.eps = pair_dim, num_heads, head_dim, eps
            H, D = num_heads, head_dim
            self.pair2b = nn.Parameter(torch.zeros(2 * H, pair_dim))
            self.pair2qkvg1 = nn.Parameter(torch.zeros(H * 4 * D, pair_dim))
            self.pair2qkvg2 = nn.Parameter(torch.zeros(H * 4 * D, pair_dim))
            self.linear_out = nn.Parameter(torch.zeros(pair_dim, 2 * H * D))
            self.out_scalers = nn.Parameter(torch.ones(pair_dim))

        def _dir(self, z_ln, W, bias_dir, pm2d, *, transpose):
            b, n, _, _ = z_ln.shape
            H, D = self.num_heads, self.head_dim
            zr = z_ln.transpose(1, 2) if transpose else z_ln
            proj = F.linear(zr, W).reshape(b, n, n, H, 4, D)
            q = proj[..., 0, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            k = proj[..., 1, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            v = proj[..., 2, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            g = proj[..., 3, :]
            mask_full = bias_dir.unsqueeze(1).expand(b, n, H, n, n).reshape(b * n, H, n, n)
            if pm2d is not None:
                pm = pm2d.to(torch.bool)
                am = pm.unsqueeze(-1) & pm.unsqueeze(-2)
                add = torch.where(
                    am.reshape(b * n, 1, n, n),
                    torch.zeros((), dtype=mask_full.dtype, device=mask_full.device),
                    torch.full((), -1e4, dtype=mask_full.dtype, device=mask_full.device),
                )
                mask_full = mask_full + add
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_full, scale=D ** -0.5)
            attn = attn.reshape(b, n, H, n, D).permute(0, 1, 3, 2, 4)
            return attn * torch.sigmoid(g)

        def forward(self, z, pair_mask):
            b, n, _, _ = z.shape
            H = self.num_heads
            z_ln = F.layer_norm(z.float(), (self.pair_dim,), eps=self.eps).to(z.dtype)
            bias_all = F.linear(z_ln, self.pair2b)
            bias_s = bias_all[..., :H].permute(0, 3, 1, 2)
            bias_e = bias_all[..., H:].permute(0, 3, 1, 2)
            out_s = self._dir(z_ln, self.pair2qkvg1, bias_s, pair_mask, transpose=False)
            cm = pair_mask.transpose(-1, -2) if pair_mask is not None else None
            out_e = self._dir(z_ln, self.pair2qkvg2, bias_e, cm, transpose=True)
            out_s_f = out_s.reshape(b, n, n, H * self.head_dim)
            out_e_f = out_e.reshape(b, n, n, H * self.head_dim)
            combined = torch.cat([out_s_f, out_e_f], dim=-1)
            return z + F.linear(combined, self.linear_out) * self.out_scalers

    class AttentionPairBias(nn.Module):
        def __init__(self, s_dim, p_dim, H, D, eps=1e-5):
            super().__init__()
            self.num_heads, self.head_dim, self.eps = H, D, eps
            self.single_norm_w = nn.Parameter(torch.zeros(s_dim))
            self.single_norm_b = nn.Parameter(torch.zeros(s_dim))
            self.pair_norm_w = nn.Parameter(torch.zeros(p_dim))
            self.pair_norm_b = nn.Parameter(torch.zeros(p_dim))
            self.pair_linear = nn.Parameter(torch.zeros(H, p_dim))
            self.input2qkvg = nn.Parameter(torch.zeros(4 * H * D, s_dim))
            self.output_proj = nn.Parameter(torch.zeros(s_dim, H * D))
            self.query_bias = nn.Parameter(torch.zeros(H, D))

        def forward(self, x, pair, pair_mask):
            b, n, _ = x.shape
            H, D = self.num_heads, self.head_dim
            pair_ln = _ln(pair, self.pair_norm_w, self.pair_norm_b, self.eps)
            bias = F.linear(pair_ln, self.pair_linear).permute(0, 3, 1, 2)
            if pair_mask is not None:
                pm = pair_mask.to(torch.bool)
                add = torch.where(
                    pm,
                    torch.zeros((), dtype=bias.dtype, device=bias.device),
                    torch.full((), -1e4, dtype=bias.dtype, device=bias.device),
                )
                bias = bias + add.unsqueeze(1)
            x_ln = _ln(x, self.single_norm_w, self.single_norm_b, self.eps)
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
        def __init__(self):
            super().__init__()
            self.transition_pair = Transition(256, expansion=2)
            self.triangle_multiplication = TriangleMultiplication(256)
            self.triangle_attention = TriangleAttention(256, num_heads=4, head_dim=64)
            self.attention_pair_bias = AttentionPairBias(384, 256, H=16, D=24)
            self.transition_single = Transition(384, expansion=2)

        def forward(self, z, s, pair_mask, single_mask):
            pto = self.transition_pair(z)
            z = self.triangle_multiplication(z, pair_mask)
            z = z + pto
            z = self.triangle_attention(z, pair_mask)
            ad = self.attention_pair_bias(s, z, pair_mask)
            if single_mask is not None:
                ad = ad * single_mask.to(ad.dtype).unsqueeze(-1)
            s = s + ad
            s = s + self.transition_single(s)
            return z, s

    # --- Weights ---
    trunk_path = MODELS_DIR / "models_v2" / "trunk.pt"
    print(f"Loading {trunk_path}...")
    trunk = torch.jit.load(str(trunk_path), map_location="cpu").eval()

    def _sub(root, dotted):
        obj = root
        for p in dotted.split("."):
            obj = getattr(obj, str(p))
        return obj

    def _assign(p, v):
        if tuple(p.shape) != tuple(v.shape):
            raise ValueError(f"shape {tuple(p.shape)} vs {tuple(v.shape)}")
        p.data.copy_(v)

    def _load_i(i):
        tsp = {n: p.detach().clone() for n, p in _sub(trunk, f"pairformer_stack.blocks.{i}").named_parameters()}
        b = PairformerBlock()
        _assign(b.transition_pair.norm_w, tsp["transition_pair.layer_norm.weight"])
        _assign(b.transition_pair.norm_b, tsp["transition_pair.layer_norm.bias"])
        _assign(b.transition_pair.up_w, tsp["transition_pair.linear_no_bias_ab.weight"])
        _assign(b.transition_pair.down_w, tsp["transition_pair.linear_out.weight"])
        _assign(b.triangle_multiplication.ln_in_w, tsp["triangle_multiplication.layernorm_z_in.weight"])
        _assign(b.triangle_multiplication.ln_in_b, tsp["triangle_multiplication.layernorm_z_in.bias"])
        _assign(b.triangle_multiplication.merged_p, tsp["triangle_multiplication.merged_linear_p.weight"])
        _assign(b.triangle_multiplication.merged_g, tsp["triangle_multiplication.merged_linear_g.weight"])
        _assign(b.triangle_multiplication.lin_out, tsp["triangle_multiplication.linear_z_out.weight"])
        _assign(b.triangle_attention.pair2b, tsp["triangle_attention.pair2b.weight"])
        _assign(b.triangle_attention.pair2qkvg1, tsp["triangle_attention.pair2qkvg1.weight"])
        _assign(b.triangle_attention.pair2qkvg2, tsp["triangle_attention.pair2qkvg2.weight"])
        _assign(b.triangle_attention.linear_out, tsp["triangle_attention.linear_out.weight"])
        _assign(b.triangle_attention.out_scalers, tsp["triangle_attention.out_scalers"])
        _assign(b.attention_pair_bias.single_norm_w, tsp["attention_pair_bias.single_layer_norm.weight"])
        _assign(b.attention_pair_bias.single_norm_b, tsp["attention_pair_bias.single_layer_norm.bias"])
        _assign(b.attention_pair_bias.pair_norm_w, tsp["attention_pair_bias.pair_layer_norm.weight"])
        _assign(b.attention_pair_bias.pair_norm_b, tsp["attention_pair_bias.pair_layer_norm.bias"])
        _assign(b.attention_pair_bias.pair_linear, tsp["attention_pair_bias.pair_linear.weight"])
        _assign(b.attention_pair_bias.query_bias, tsp["attention_pair_bias.attention.query_bias"])
        w_qkvg = tsp["attention_pair_bias.attention.input2qkvg.weight"]
        _assign(b.attention_pair_bias.input2qkvg, w_qkvg.reshape(w_qkvg.shape[0], -1).T.contiguous())
        w_out = tsp["attention_pair_bias.attention.output_proj.weight"]
        _assign(b.attention_pair_bias.output_proj, w_out.reshape(-1, w_out.shape[-1]).T.contiguous())
        _assign(b.transition_single.norm_w, tsp["transition_single.layer_norm.weight"])
        _assign(b.transition_single.norm_b, tsp["transition_single.layer_norm.bias"])
        _assign(b.transition_single.up_w, tsp["transition_single.linear_no_bias_ab.weight"])
        _assign(b.transition_single.down_w, tsp["transition_single.linear_out.weight"])
        return b.to(device).eval()

    print("Building 48 blocks...")
    blocks = [_load_i(i) for i in range(48)]
    del trunk
    torch.cuda.empty_cache()

    # --- Run on real post-msa inputs ---
    inputs = np.load(io.BytesIO(inputs_npz))
    s = torch.from_numpy(inputs["single"]).to(device).to(torch.float32)
    z = torch.from_numpy(inputs["pair"]).to(device).to(torch.float32)
    pair_mask = torch.from_numpy(inputs["pair_mask"]).to(device)
    single_mask = torch.from_numpy(inputs["single_mask"]).to(device)
    print(f"s: shape={list(s.shape)} max_abs={float(s.abs().max()):.3f}")
    print(f"z: shape={list(z.shape)} max_abs={float(z.abs().max()):.3f}")

    dump = {}
    for i, b in enumerate(blocks):
        z, s = b(z, s, pair_mask, single_mask)
        dump[f"s_block_{i:02d}"] = s.detach().float().cpu().numpy()
        dump[f"z_block_{i:02d}"] = z.detach().float().cpu().numpy()
        if i % 12 == 0 or i == 47:
            print(f"  block {i:2d}: s max_abs={float(s.abs().max()):.3f} z max_abs={float(z.abs().max()):.3f}")
    dump["s_final"] = dump[f"s_block_{len(blocks) - 1:02d}"]
    dump["z_final"] = dump[f"z_block_{len(blocks) - 1:02d}"]

    buf = io.BytesIO()
    np.savez(buf, **dump)
    return {"cuda_out_fp32": buf.getvalue()}


@app.local_entrypoint()
def main() -> None:
    inp = Path("/tmp/chai_mlx_cuda/real_inputs_probe/inputs.npz")
    data = inp.read_bytes()
    res = cuda_real_inputs_probe.remote(data)
    for name, buf in res.items():
        p = inp.parent / f"{name}.npz"
        p.write_bytes(buf)
        print(f"saved {p} ({len(buf) / 1024**2:.1f} MB)")
