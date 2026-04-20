"""Modal probe: full eager-PyTorch trunk (msa_module + pairformer_stack) at
bf16 or fp32, fed the 1L2Y embedding outputs captured in the intermediates
NPZ. This produces an 'eager CUDA' reference at a specified precision so we
can attribute the scripted-vs-eager drift.

Writes ``cuda_full_trunk_eager_{dtype}.npz`` into
``/tmp/chai_mlx_cuda/full_trunk_eager/`` with keys
``post_msa_pair``, ``post_pairformer_single``, ``post_pairformer_pair``,
plus per-block ``pair`` and ``single`` dumps through the pairformer stack.
"""
from __future__ import annotations

import io
from pathlib import Path

from cuda_harness.modal_common import (
    MINUTES,
    MODELS_DIR,
    app,
    chai_model_volume,
    image,
)


OUT_DIR = Path("/tmp/chai_mlx_cuda/full_trunk_eager")


@app.function(
    timeout=20 * MINUTES,
    gpu="H100",
    volumes={MODELS_DIR: chai_model_volume},
    image=image,
)
def cuda_full_trunk_eager(intermediates_npz: bytes, dtype: str = "bf16") -> dict[str, bytes]:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")

    compute_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]

    # Inline the eager MSAModule and PairformerBlock from the existing probes.
    # For brevity we import the same shapes from trunk.pt and wire per-round
    # directly.
    trunk_path = MODELS_DIR / "models_v2" / "trunk.pt"
    print(f"Loading {trunk_path}")
    trunk = torch.jit.load(str(trunk_path), map_location="cpu")
    trunk.eval()

    # ------------------------------------------------------------------
    # Eager submodules (copied verbatim from _probe_msa_module_cuda.py and
    # _probe_first_block_cuda.py, merged here as a single coherent stack).
    # ------------------------------------------------------------------

    def _ln(x, w, b, eps, reduction_dtype=torch.float32):
        orig = x.dtype
        y = F.layer_norm(
            x.to(reduction_dtype), (x.shape[-1],),
            weight=w.to(reduction_dtype) if w is not None else None,
            bias=b.to(reduction_dtype) if b is not None else None,
            eps=eps,
        )
        return y.to(orig)

    def _ln_na(x, eps, reduction_dtype=torch.float32):
        return F.layer_norm(x.to(reduction_dtype), (x.shape[-1],), eps=eps).to(x.dtype)

    # ---------- MSAModule components ----------
    class OPM(nn.Module):
        def __init__(self, msa_dim, pair_dim, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.weight_ab = nn.Parameter(torch.zeros(2, 8, 8, msa_dim))
            self.ln_out_w = nn.Parameter(torch.ones(512))
            self.ln_out_b = nn.Parameter(torch.zeros(512))
            self.linear_out_w = nn.Parameter(torch.zeros(pair_dim, 512))
            self.linear_out_b = nn.Parameter(torch.zeros(pair_dim))
            self.chunk_size = 4096

        def forward(self, msa, msa_mask):
            x = _ln_na(msa, self.eps)
            op = None
            for start in range(0, int(x.shape[1]), self.chunk_size):
                xc = x[:, start: start + self.chunk_size]
                if msa_mask is not None:
                    mm = msa_mask[:, start: start + self.chunk_size].to(x.dtype)[..., None]
                    xc = xc * mm
                proj = torch.einsum("bmnc,defc->bmndef", xc, self.weight_ab.to(xc.dtype))
                a_proj = proj[..., 0, :, :]
                b_proj = proj[..., 1, :, :]
                opc = torch.einsum("bmige,bmjgf->bijgef", a_proj, b_proj)
                op = opc if op is None else op + opc
            op = op.reshape(op.shape[0], op.shape[1], op.shape[2], 512)
            op_ln = F.layer_norm(op.float(), (512,), weight=self.ln_out_w.float(), bias=self.ln_out_b.float(), eps=0.1).to(op.dtype)
            return F.linear(op_ln, self.linear_out_w.to(op_ln.dtype), self.linear_out_b.to(op_ln.dtype))

    class Transition(nn.Module):
        def __init__(self, dim, expansion=4, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.norm_w = nn.Parameter(torch.ones(dim))
            self.norm_b = nn.Parameter(torch.zeros(dim))
            self.up_w = nn.Parameter(torch.zeros(2 * expansion * dim, dim))
            self.down_w = nn.Parameter(torch.zeros(dim, expansion * dim))

        def forward(self, x):
            n = _ln(x, self.norm_w, self.norm_b, self.eps)
            up = F.linear(n, self.up_w.to(n.dtype))
            a, b = up.chunk(2, dim=-1)
            return F.linear(F.silu(a) * b, self.down_w.to(n.dtype))

    class PWAverage(nn.Module):
        def __init__(self, msa_dim, pair_dim, num_heads=8, value_dim=32, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.num_heads = num_heads
            self.value_dim = value_dim
            self.ln_msa_w = nn.Parameter(torch.ones(msa_dim))
            self.ln_msa_b = nn.Parameter(torch.zeros(msa_dim))
            self.ln_pair_w = nn.Parameter(torch.ones(pair_dim))
            self.ln_pair_b = nn.Parameter(torch.zeros(pair_dim))
            self.msa2vg_w = nn.Parameter(torch.zeros(num_heads * 2 * value_dim, msa_dim))
            self.pair_w = nn.Parameter(torch.zeros(num_heads, pair_dim))
            self.out_w = nn.Parameter(torch.zeros(msa_dim, num_heads * value_dim))
            self.chunk_size = 8192

        def forward(self, msa, pair, token_pair_mask, msa_mask):
            pn = _ln(pair, self.ln_pair_w, self.ln_pair_b, self.eps)
            logits = F.linear(pn, self.pair_w.to(pn.dtype)).permute(0, 3, 1, 2)
            if token_pair_mask is not None:
                add = torch.where(token_pair_mask.to(torch.bool),
                                  torch.zeros((), dtype=logits.dtype, device=logits.device),
                                  torch.full((), -1e4, dtype=logits.dtype, device=logits.device))[:, None, :, :]
                logits = logits + add
            weights = F.softmax(logits.float(), dim=-1).to(logits.dtype)
            out_chunks = []
            H, D = self.num_heads, self.value_dim
            for s in range(0, int(msa.shape[1]), self.chunk_size):
                mc = msa[:, s: s + self.chunk_size]
                mn = _ln(mc, self.ln_msa_w, self.ln_msa_b, self.eps)
                vg = F.linear(mn, self.msa2vg_w.to(mn.dtype))
                v, g = vg.chunk(2, dim=-1)
                v = v.reshape(*v.shape[:-1], H, D).permute(0, 1, 3, 2, 4)
                g = g.reshape(*g.shape[:-1], H, D).permute(0, 1, 3, 2, 4)
                if msa_mask is not None:
                    mm = msa_mask[:, s: s + self.chunk_size].to(v.dtype)[:, :, None, :, None]
                    v = v * mm
                out = torch.einsum("bhij,bmhjd->bmhid", weights, v) * torch.sigmoid(g)
                out = out.permute(0, 1, 3, 2, 4).reshape(*out.shape[:-3], -1, H * D)
                out_chunks.append(F.linear(out, self.out_w.to(out.dtype)))
            return torch.cat(out_chunks, dim=1)

    class TriMult(nn.Module):
        def __init__(self, pair_dim, eps=1e-5):
            super().__init__()
            self.pair_dim = pair_dim
            self.eps = eps
            self.ln_w = nn.Parameter(torch.ones(pair_dim))
            self.ln_b = nn.Parameter(torch.zeros(pair_dim))
            self.mp = nn.Parameter(torch.zeros(4 * pair_dim, pair_dim))
            self.mg = nn.Parameter(torch.zeros(5 * pair_dim, pair_dim))
            self.lo = nn.Parameter(torch.zeros(pair_dim, pair_dim))

        def forward(self, z, pair_mask):
            d = self.pair_dim
            zln = _ln(z, self.ln_w, self.ln_b, self.eps)
            p = F.linear(zln, self.mp.to(zln.dtype))
            g4 = torch.sigmoid(F.linear(zln, self.mg[:4 * d].to(zln.dtype)))
            a1, b1, a2, b2 = (p * g4).chunk(4, dim=-1)
            if pair_mask is not None:
                pm = pair_mask[..., None].to(z.dtype)
                pm_T = pair_mask.transpose(-1, -2)[..., None].to(z.dtype)
                a1 = a1 * pm; b1 = b1 * pm
                a2 = a2 * pm_T; b2 = b2 * pm_T
            x_out = torch.einsum("bikd,bjkd->bijd", a1, b1)
            x_in = torch.einsum("bkid,bkjd->bijd", a2, b2)
            x_out_ln = _ln_na(x_out, self.eps)
            x_in_ln = _ln_na(x_in, self.eps)
            g_out = torch.sigmoid(F.linear(zln, self.mg[4 * d:].to(zln.dtype)))
            return z + F.linear(x_out_ln + x_in_ln, self.lo.to(x_out_ln.dtype)) * g_out

    class TriAttn(nn.Module):
        def __init__(self, pair_dim, num_heads, head_dim, eps=1e-5):
            super().__init__()
            self.pair_dim = pair_dim
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.eps = eps
            self.pair2b = nn.Parameter(torch.zeros(2 * num_heads, pair_dim))
            self.pair2qkvg1 = nn.Parameter(torch.zeros(4 * num_heads * head_dim, pair_dim))
            self.pair2qkvg2 = nn.Parameter(torch.zeros(4 * num_heads * head_dim, pair_dim))
            self.lin_out = nn.Parameter(torch.zeros(pair_dim, 2 * num_heads * head_dim))
            self.out_scalers = nn.Parameter(torch.ones(pair_dim))

        def _run(self, zln, proj_w, bias, pair_mask_2d, transpose):
            b, n, _, _ = zln.shape
            H, D = self.num_heads, self.head_dim
            zrows = zln.transpose(1, 2) if transpose else zln
            proj = F.linear(zrows, proj_w.to(zrows.dtype)).reshape(b, n, n, H, 4, D)
            q = proj[..., 0, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            k = proj[..., 1, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            v = proj[..., 2, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            g = proj[..., 3, :]
            mf = bias.unsqueeze(1).expand(b, n, H, n, n).reshape(b * n, H, n, n)
            if pair_mask_2d is not None:
                pm = pair_mask_2d.to(torch.bool)
                am = pm.unsqueeze(-1) & pm.unsqueeze(-2)
                add = torch.where(am.reshape(b * n, 1, n, n),
                                  torch.zeros((), dtype=mf.dtype, device=mf.device),
                                  torch.full((), -1e4, dtype=mf.dtype, device=mf.device))
                mf = mf + add
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mf, scale=D ** -0.5)
            attn = attn.reshape(b, n, H, n, D).permute(0, 1, 3, 2, 4)
            return attn * torch.sigmoid(g)

        def forward(self, z, pair_mask):
            b, n, _, _ = z.shape
            H = self.num_heads
            zln = _ln_na(z, self.eps)
            ba = F.linear(zln, self.pair2b.to(zln.dtype))
            bs = ba[..., :H].permute(0, 3, 1, 2)
            be = ba[..., H:].permute(0, 3, 1, 2)
            out_s = self._run(zln, self.pair2qkvg1, bs, pair_mask, transpose=False)
            col_m = pair_mask.transpose(-1, -2) if pair_mask is not None else None
            out_e = self._run(zln, self.pair2qkvg2, be, col_m, transpose=True)
            b_, n_, _, _, _ = out_s.shape
            combined = torch.cat([
                out_s.reshape(b_, n_, n_, H * self.head_dim),
                out_e.reshape(b_, n_, n_, H * self.head_dim),
            ], dim=-1)
            return z + F.linear(combined, self.lin_out.to(combined.dtype)) * self.out_scalers.to(combined.dtype)

    class AttentionPairBias(nn.Module):
        def __init__(self, single_dim, pair_dim, num_heads, head_dim, eps=1e-5):
            super().__init__()
            self.single_dim = single_dim
            self.pair_dim = pair_dim
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.eps = eps
            self.single_norm_w = nn.Parameter(torch.ones(single_dim))
            self.single_norm_b = nn.Parameter(torch.zeros(single_dim))
            self.pair_norm_w = nn.Parameter(torch.ones(pair_dim))
            self.pair_norm_b = nn.Parameter(torch.zeros(pair_dim))
            self.pair_linear = nn.Parameter(torch.zeros(num_heads, pair_dim))
            self.input2qkvg = nn.Parameter(torch.zeros(4 * num_heads * head_dim, single_dim))
            self.output_proj = nn.Parameter(torch.zeros(single_dim, num_heads * head_dim))
            self.query_bias = nn.Parameter(torch.zeros(num_heads, head_dim))

        def forward(self, x, pair, pair_mask):
            b, n, _ = x.shape
            H, D = self.num_heads, self.head_dim
            pln = _ln(pair, self.pair_norm_w, self.pair_norm_b, self.eps)
            bias = F.linear(pln, self.pair_linear.to(pln.dtype)).permute(0, 3, 1, 2)
            if pair_mask is not None:
                pm = pair_mask.to(torch.bool)
                add = torch.where(pm, torch.zeros((), dtype=bias.dtype, device=bias.device),
                                   torch.full((), -1e4, dtype=bias.dtype, device=bias.device))
                bias = bias + add.unsqueeze(1)
            xln = _ln(x, self.single_norm_w, self.single_norm_b, self.eps)
            qkvg = F.linear(xln, self.input2qkvg.to(xln.dtype))
            q, k, v, g = qkvg.chunk(4, dim=-1)
            q = q.reshape(b, n, H, D).permute(0, 2, 1, 3)
            k = k.reshape(b, n, H, D).permute(0, 2, 1, 3)
            v = v.reshape(b, n, H, D).permute(0, 2, 1, 3)
            g = g.reshape(b, n, H, D)
            qb = self.query_bias.to(q.dtype).unsqueeze(0).unsqueeze(-2)
            q = q + qb
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype), scale=D ** -0.5)
            attn = (attn.permute(0, 2, 1, 3) * torch.sigmoid(g)).reshape(b, n, H * D)
            return F.linear(attn, self.output_proj.to(attn.dtype))

    class PairformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.transition_pair = Transition(256, expansion=2)
            self.triangle_multiplication = TriMult(256)
            self.triangle_attention = TriAttn(256, num_heads=4, head_dim=64)
            self.attention_pair_bias = AttentionPairBias(384, 256, num_heads=16, head_dim=24)
            self.transition_single = Transition(384, expansion=2)

        def forward(self, z, s, pair_mask, single_mask):
            pto = self.transition_pair(z)
            z = self.triangle_multiplication(z, pair_mask)
            z = z + pto
            z = self.triangle_attention(z, pair_mask)
            delta = self.attention_pair_bias(s, z, pair_mask)
            if single_mask is not None:
                delta = delta * single_mask.to(delta.dtype).unsqueeze(-1)
            s = s + delta
            s = s + self.transition_single(s)
            return z, s

    # ---------- assemble MSAModule ----------
    class MSAModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_s2m = nn.Parameter(torch.zeros(64, 384))
            self.opm = nn.ModuleList([OPM(64, 256) for _ in range(4)])
            self.pwavg = nn.ModuleList([PWAverage(64, 256) for _ in range(3)])
            self.msa_tr = nn.ModuleList([Transition(64) for _ in range(3)])
            self.pair_tr = nn.ModuleList([Transition(256) for _ in range(4)])
            self.trimul = nn.ModuleList([TriMult(256) for _ in range(4)])
            self.triattn = nn.ModuleList([TriAttn(256, 4, 64) for _ in range(4)])

        def forward(self, single, pair, msa, token_pair_mask, msa_mask):
            if msa.shape[1] > 0:
                msa = msa + F.linear(single, self.linear_s2m.to(single.dtype))[:, None, :, :]
            for i in range(4):
                pair = pair + self.opm[i](msa, msa_mask)
                if i < 3:
                    msa = msa + self.msa_tr[i](msa)
                    msa = msa + self.pwavg[i](msa, pair, token_pair_mask, msa_mask)
                pto = self.pair_tr[i](pair)
                pair = self.trimul[i](pair, token_pair_mask) + pto
                pair = self.triattn[i](pair, token_pair_mask)
            return pair

    msa_mod = MSAModule()

    # ------- wire MSA params -------
    mm_params = {n: p.detach().clone() for n, p in trunk.msa_module.named_parameters()}

    def _try_copy(dst, names):
        for n in names:
            if n in mm_params:
                v = mm_params[n]
                if tuple(dst.shape) != tuple(v.shape):
                    v = v.reshape(dst.shape)
                dst.data.copy_(v.float())
                return
        raise KeyError(f"no match for {names[0]}")

    _try_copy(msa_mod.linear_s2m, ["linear_s2m.weight"])
    for i in range(4):
        _try_copy(msa_mod.opm[i].weight_ab, [f"outer_product_mean.{i}.weight_ab"])
        _try_copy(msa_mod.opm[i].ln_out_w, [f"outer_product_mean.{i}.ln_out.weight"])
        _try_copy(msa_mod.opm[i].ln_out_b, [f"outer_product_mean.{i}.ln_out.bias"])
        _try_copy(msa_mod.opm[i].linear_out_w, [f"outer_product_mean.{i}.linear_out.weight"])
        _try_copy(msa_mod.opm[i].linear_out_b, [f"outer_product_mean.{i}.linear_out.bias"])
        if i < 3:
            _try_copy(msa_mod.msa_tr[i].norm_w, [f"msa_transition.{i}.layer_norm.weight"])
            _try_copy(msa_mod.msa_tr[i].norm_b, [f"msa_transition.{i}.layer_norm.bias"])
            _try_copy(msa_mod.msa_tr[i].up_w, [f"msa_transition.{i}.linear_no_bias_ab.weight"])
            _try_copy(msa_mod.msa_tr[i].down_w, [f"msa_transition.{i}.linear_out.weight"])
            _try_copy(msa_mod.pwavg[i].ln_msa_w, [f"msa_pair_weighted_averaging.{i}.layernorm_msa.weight"])
            _try_copy(msa_mod.pwavg[i].ln_msa_b, [f"msa_pair_weighted_averaging.{i}.layernorm_msa.bias"])
            _try_copy(msa_mod.pwavg[i].ln_pair_w, [f"msa_pair_weighted_averaging.{i}.layernorm_pair.weight"])
            _try_copy(msa_mod.pwavg[i].ln_pair_b, [f"msa_pair_weighted_averaging.{i}.layernorm_pair.bias"])
            _try_copy(msa_mod.pwavg[i].msa2vg_w, [f"msa_pair_weighted_averaging.{i}.linear_msa2vg.weight"])
            _try_copy(msa_mod.pwavg[i].pair_w, [f"msa_pair_weighted_averaging.{i}.linear_pair.weight"])
            _try_copy(msa_mod.pwavg[i].out_w, [f"msa_pair_weighted_averaging.{i}.linear_out_no_bias.weight"])
        _try_copy(msa_mod.pair_tr[i].norm_w, [f"pair_transition.{i}.layer_norm.weight"])
        _try_copy(msa_mod.pair_tr[i].norm_b, [f"pair_transition.{i}.layer_norm.bias"])
        _try_copy(msa_mod.pair_tr[i].up_w, [f"pair_transition.{i}.linear_no_bias_ab.weight"])
        _try_copy(msa_mod.pair_tr[i].down_w, [f"pair_transition.{i}.linear_out.weight"])
        _try_copy(msa_mod.trimul[i].ln_w, [f"triangular_multiplication.{i}.layernorm_z_in.weight"])
        _try_copy(msa_mod.trimul[i].ln_b, [f"triangular_multiplication.{i}.layernorm_z_in.bias"])
        _try_copy(msa_mod.trimul[i].mp, [f"triangular_multiplication.{i}.merged_linear_p.weight"])
        _try_copy(msa_mod.trimul[i].mg, [f"triangular_multiplication.{i}.merged_linear_g.weight"])
        _try_copy(msa_mod.trimul[i].lo, [f"triangular_multiplication.{i}.linear_z_out.weight"])
        _try_copy(msa_mod.triattn[i].pair2b, [f"triangular_attention.{i}.pair2b.weight"])
        _try_copy(msa_mod.triattn[i].pair2qkvg1, [f"triangular_attention.{i}.pair2qkvg1.weight"])
        _try_copy(msa_mod.triattn[i].pair2qkvg2, [f"triangular_attention.{i}.pair2qkvg2.weight"])
        _try_copy(msa_mod.triattn[i].lin_out, [f"triangular_attention.{i}.linear_out.weight"])
        _try_copy(msa_mod.triattn[i].out_scalers, [f"triangular_attention.{i}.out_scalers"])

    # ---------- Pairformer stack ----------
    ps_params = {n: p.detach().clone() for n, p in trunk.pairformer_stack.named_parameters()}
    pf_blocks = nn.ModuleList([PairformerBlock() for _ in range(48)])
    for bi in range(48):
        pref = f"blocks.{bi}"
        b = pf_blocks[bi]
        b.transition_pair.norm_w.data.copy_(ps_params[f"{pref}.transition_pair.layer_norm.weight"].float())
        b.transition_pair.norm_b.data.copy_(ps_params[f"{pref}.transition_pair.layer_norm.bias"].float())
        b.transition_pair.up_w.data.copy_(ps_params[f"{pref}.transition_pair.linear_no_bias_ab.weight"].float())
        b.transition_pair.down_w.data.copy_(ps_params[f"{pref}.transition_pair.linear_out.weight"].float())
        b.triangle_multiplication.ln_w.data.copy_(ps_params[f"{pref}.triangle_multiplication.layernorm_z_in.weight"].float())
        b.triangle_multiplication.ln_b.data.copy_(ps_params[f"{pref}.triangle_multiplication.layernorm_z_in.bias"].float())
        b.triangle_multiplication.mp.data.copy_(ps_params[f"{pref}.triangle_multiplication.merged_linear_p.weight"].float())
        b.triangle_multiplication.mg.data.copy_(ps_params[f"{pref}.triangle_multiplication.merged_linear_g.weight"].float())
        b.triangle_multiplication.lo.data.copy_(ps_params[f"{pref}.triangle_multiplication.linear_z_out.weight"].float())
        b.triangle_attention.pair2b.data.copy_(ps_params[f"{pref}.triangle_attention.pair2b.weight"].float())
        b.triangle_attention.pair2qkvg1.data.copy_(ps_params[f"{pref}.triangle_attention.pair2qkvg1.weight"].float())
        b.triangle_attention.pair2qkvg2.data.copy_(ps_params[f"{pref}.triangle_attention.pair2qkvg2.weight"].float())
        b.triangle_attention.lin_out.data.copy_(ps_params[f"{pref}.triangle_attention.linear_out.weight"].float())
        b.triangle_attention.out_scalers.data.copy_(ps_params[f"{pref}.triangle_attention.out_scalers"].float())
        b.attention_pair_bias.single_norm_w.data.copy_(ps_params[f"{pref}.attention_pair_bias.single_layer_norm.weight"].float())
        b.attention_pair_bias.single_norm_b.data.copy_(ps_params[f"{pref}.attention_pair_bias.single_layer_norm.bias"].float())
        b.attention_pair_bias.pair_norm_w.data.copy_(ps_params[f"{pref}.attention_pair_bias.pair_layer_norm.weight"].float())
        b.attention_pair_bias.pair_norm_b.data.copy_(ps_params[f"{pref}.attention_pair_bias.pair_layer_norm.bias"].float())
        b.attention_pair_bias.pair_linear.data.copy_(ps_params[f"{pref}.attention_pair_bias.pair_linear.weight"].float())
        b.attention_pair_bias.query_bias.data.copy_(ps_params[f"{pref}.attention_pair_bias.attention.query_bias"].float())
        w_qkvg = ps_params[f"{pref}.attention_pair_bias.attention.input2qkvg.weight"]
        b.attention_pair_bias.input2qkvg.data.copy_(w_qkvg.reshape(w_qkvg.shape[0], -1).T.contiguous().float())
        w_out = ps_params[f"{pref}.attention_pair_bias.attention.output_proj.weight"]
        b.attention_pair_bias.output_proj.data.copy_(w_out.reshape(-1, w_out.shape[-1]).T.contiguous().float())
        b.transition_single.norm_w.data.copy_(ps_params[f"{pref}.transition_single.layer_norm.weight"].float())
        b.transition_single.norm_b.data.copy_(ps_params[f"{pref}.transition_single.layer_norm.bias"].float())
        b.transition_single.up_w.data.copy_(ps_params[f"{pref}.transition_single.linear_no_bias_ab.weight"].float())
        b.transition_single.down_w.data.copy_(ps_params[f"{pref}.transition_single.linear_out.weight"].float())

    del trunk
    torch.cuda.empty_cache()

    # Cast params to compute_dtype (norms/scalers kept fp32 inside forward via _ln).
    full_mod = nn.Module()
    full_mod.msa = msa_mod
    full_mod.blocks = pf_blocks
    full_mod.to(device).eval()
    for name, p in full_mod.named_parameters():
        if any(tag in name for tag in ("norm", "ln_", "out_scalers", "query_bias")):
            continue
        p.data = p.data.to(compute_dtype)

    # Load inputs.
    data = np.load(io.BytesIO(intermediates_npz))
    single_init = torch.from_numpy(data["embedding.token_single_initial"]).to(device, dtype=compute_dtype)
    pair_init = torch.from_numpy(data["embedding.token_pair_initial"]).to(device, dtype=compute_dtype)
    msa_input = torch.from_numpy(data["embedding.msa"]).to(device, dtype=compute_dtype)
    token_exists = torch.from_numpy(data["inputs.batch.token_exists_mask"]).to(device)
    msa_mask = torch.from_numpy(data["inputs.batch.msa_mask"]).to(device)
    token_pair_mask = token_exists[..., :, None] & token_exists[..., None, :]

    # Recycle 0: single/pair = init.
    single = single_init
    pair = pair_init

    print(f"[eager trunk @ {dtype}] msa_module ...")
    pair = full_mod.msa(single, pair, msa_input, token_pair_mask, msa_mask)
    post_msa_np = pair.detach().float().cpu().numpy()
    print(f"  post_msa_pair max_abs={np.abs(post_msa_np).max():.3f}")

    print(f"[eager trunk @ {dtype}] pairformer_stack ...")
    # Avoid per-block dumps to stay under Modal's return-value size limit;
    # we just need the final (single, pair) to compare against
    # ``trunk.recycle_0.{single,pair}`` from the scripted CUDA intermediates.
    s, z = single, pair
    for i in range(48):
        z, s = full_mod.blocks[i](z, s, token_pair_mask, token_exists)
        torch.cuda.synchronize()
        if i % 12 == 0 or i == 47:
            print(f"  block {i:2d}: s max_abs={s.abs().max().item():.3f}  z max_abs={z.abs().max().item():.3f}")
    dump = {
        "post_msa_pair": post_msa_np,
        "post_pairformer_single": s.detach().float().cpu().numpy(),
        "post_pairformer_pair": z.detach().float().cpu().numpy(),
    }

    buf = io.BytesIO()
    np.savez_compressed(buf, **dump)
    return {f"cuda_full_trunk_eager_{dtype}.npz": buf.getvalue()}


@app.local_entrypoint()
def main(
    intermediates_npz: str = "/tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz",
    dtype: str = "bf16",
) -> None:
    src = Path(intermediates_npz)
    if not src.is_file():
        raise FileNotFoundError(src)
    print(f"Sending {src.stat().st_size / (1 << 20):.1f} MB to Modal (dtype={dtype})")
    result = cuda_full_trunk_eager.remote(src.read_bytes(), dtype)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, blob in result.items():
        dst = OUT_DIR / name
        dst.write_bytes(blob)
        print(f"wrote {dst} ({len(blob) / (1 << 20):.2f} MB)")
