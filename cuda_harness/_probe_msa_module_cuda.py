"""Modal probe: eager-PyTorch reimplementation of chai-1's ``msa_module`` run
at fp32 on CUDA from the weights stashed inside ``trunk.pt``.

Why this exists
---------------

The scripted ``trunk.pt`` inlines every submodule into a single
``forward_256`` graph, so we cannot call ``trunk.msa_module(...)`` directly
(``RecursiveScriptModule`` has no exposed ``forward``). The earlier
``_probe_msa_boundary_cuda.py`` hit this wall.

This probe reimplements the MSA module in eager PyTorch, lifts every weight
it needs from ``trunk.pt.msa_module.*``, and runs it on CUDA at fp32 on the
exact inputs captured in a 1L2Y intermediates NPZ (``embedding.msa``,
``embedding.token_single_initial``, ``embedding.token_pair_initial`` +
masks). The output goes into
``/tmp/chai_mlx_cuda/msa_module_probe/cuda_post_msa_fp32.npz`` with key
``post_msa_pair``, matching the layout of
``_probe_recycle_mlx.py``'s MLX-side dump so the two can be diffed directly.

Implementation notes
--------------------

We mirror ``chai_mlx/model/trunk.py::MSAModule.__call__`` step-for-step:

    for i in 0..3:
        pair  += outer_product_mean[i](msa, msa_mask=...)
        if i < 3:
            msa  += msa_transition[i](msa)
            msa  += msa_pair_weighted_averaging[i](msa, pair, ...)
        pair_trans_out   = pair_transition[i](pair)
        pair             = triangular_multiplication[i](pair, ...) + pair_trans_out
        pair             = triangular_attention[i](pair, ...)

LayerNorms use fp32 reductions (matching the TorchScript ``aten::to(dtype=6)``
bracketing). Softmax is computed in fp32. Every matmul input is kept in fp32
here (the CUDA scripted reference casts to bf16; the delta between bf16- and
fp32-matmul accumulation is small compared to the ~34% gap we're chasing).

Usage::

    # 1. produce MLX-side post-msa dump first (cached on disk already):
    python3 cuda_harness/_probe_recycle_mlx.py \\
        --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \\
        --weights-dir weights

    # 2. run this probe:
    modal run -m cuda_harness._probe_msa_module_cuda \\
        --intermediates-npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz
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


OUT_DIR = Path("/tmp/chai_mlx_cuda/msa_module_probe")


@app.function(
    timeout=20 * MINUTES,
    gpu="H100",
    volumes={MODELS_DIR: chai_model_volume},
    image=image,
)
def cuda_msa_module_probe(intermediates_npz: bytes, dtype: str = "fp32") -> dict[str, bytes]:
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")

    # ------------------------------------------------------------------
    # Eager PyTorch MSA module, mirroring ``chai_mlx/model/trunk.py``.
    # ------------------------------------------------------------------

    def _fp32_ln(x, w, b, eps):
        orig = x.dtype
        y = F.layer_norm(x.float(), (x.shape[-1],), weight=w.float() if w is not None else None,
                          bias=b.float() if b is not None else None, eps=eps)
        return y.to(orig)

    def _fp32_ln_noaffine(x, eps):
        orig = x.dtype
        return F.layer_norm(x.float(), (x.shape[-1],), eps=eps).to(orig)

    class OuterProductMean(nn.Module):
        def __init__(self, msa_dim: int, pair_dim: int, eps: float = 1e-5):
            super().__init__()
            self.msa_dim = msa_dim
            self.pair_dim = pair_dim
            self.eps = eps
            self.weight_ab = nn.Parameter(torch.zeros(2, 8, 8, msa_dim))
            self.ln_out_w = nn.Parameter(torch.ones(512))
            self.ln_out_b = nn.Parameter(torch.zeros(512))
            self.linear_out_w = nn.Parameter(torch.zeros(pair_dim, 512))
            self.linear_out_b = nn.Parameter(torch.zeros(pair_dim))
            self.chunk_size = 4096

        def forward(self, msa: torch.Tensor, msa_mask: torch.Tensor | None):
            x = _fp32_ln_noaffine(msa, self.eps)
            op = None
            for start in range(0, int(x.shape[1]), self.chunk_size):
                x_chunk = x[:, start: start + self.chunk_size]
                if msa_mask is not None:
                    mm_chunk = msa_mask[:, start: start + self.chunk_size].to(x.dtype)[..., None]
                    x_chunk = x_chunk * mm_chunk
                proj = torch.einsum("bmnc,defc->bmndef", x_chunk, self.weight_ab.to(x_chunk.dtype))
                a_proj = proj[..., 0, :, :]
                b_proj = proj[..., 1, :, :]
                op_chunk = torch.einsum("bmige,bmjgf->bijgef", a_proj, b_proj)
                op = op_chunk if op is None else op + op_chunk
            op = op.reshape(op.shape[0], op.shape[1], op.shape[2], 512)
            # layer_norm with eps=0.1 (matches MLX code)
            op_ln = F.layer_norm(op.float(), (512,), weight=self.ln_out_w.float(), bias=self.ln_out_b.float(), eps=0.1).to(op.dtype)
            return F.linear(op_ln, self.linear_out_w.to(op_ln.dtype), self.linear_out_b.to(op_ln.dtype))

    class Transition(nn.Module):
        """Chai transition (expansion=4 for msa, =4 for pair here)."""
        def __init__(self, dim: int, expansion: int = 4, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.norm_w = nn.Parameter(torch.ones(dim))
            self.norm_b = nn.Parameter(torch.zeros(dim))
            self.up_w = nn.Parameter(torch.zeros(2 * expansion * dim, dim))
            self.down_w = nn.Parameter(torch.zeros(dim, expansion * dim))

        def forward(self, x):
            n = _fp32_ln(x, self.norm_w, self.norm_b, self.eps)
            up = F.linear(n, self.up_w.to(n.dtype))
            a, b = up.chunk(2, dim=-1)
            return F.linear(F.silu(a) * b, self.down_w.to(n.dtype))

    class MSAPairWeightedAveraging(nn.Module):
        def __init__(self, msa_dim: int, pair_dim: int, num_heads=8, value_dim=32, eps=1e-5):
            super().__init__()
            self.eps = eps
            self.msa_dim = msa_dim
            self.pair_dim = pair_dim
            self.num_heads = num_heads
            self.value_dim = value_dim
            self.ln_msa_w = nn.Parameter(torch.ones(msa_dim))
            self.ln_msa_b = nn.Parameter(torch.zeros(msa_dim))
            self.ln_pair_w = nn.Parameter(torch.ones(pair_dim))
            self.ln_pair_b = nn.Parameter(torch.zeros(pair_dim))
            self.linear_msa2vg_w = nn.Parameter(torch.zeros(num_heads * 2 * value_dim, msa_dim))
            self.linear_pair_w = nn.Parameter(torch.zeros(num_heads, pair_dim))
            self.linear_out_w = nn.Parameter(torch.zeros(msa_dim, num_heads * value_dim))
            self.chunk_size = 8192

        def forward(self, msa, pair, token_pair_mask=None, msa_mask=None):
            pair_n = _fp32_ln(pair, self.ln_pair_w, self.ln_pair_b, self.eps)
            pair_logits = F.linear(pair_n, self.linear_pair_w.to(pair_n.dtype)).permute(0, 3, 1, 2)
            if token_pair_mask is not None:
                additive = torch.where(
                    token_pair_mask.to(torch.bool),
                    torch.zeros((), dtype=pair_logits.dtype, device=pair_logits.device),
                    torch.full((), -1e4, dtype=pair_logits.dtype, device=pair_logits.device),
                )[:, None, :, :]
                pair_logits = pair_logits + additive
            weights = F.softmax(pair_logits.float(), dim=-1).to(pair_logits.dtype)

            out_chunks = []
            for start in range(0, int(msa.shape[1]), self.chunk_size):
                msa_c = msa[:, start: start + self.chunk_size]
                msa_n = _fp32_ln(msa_c, self.ln_msa_w, self.ln_msa_b, self.eps)
                vg = F.linear(msa_n, self.linear_msa2vg_w.to(msa_n.dtype))
                v, g = vg.chunk(2, dim=-1)
                H, D = self.num_heads, self.value_dim
                v = v.reshape(*v.shape[:-1], H, D).permute(0, 1, 3, 2, 4)  # (b, m, H, n, D)
                g = g.reshape(*g.shape[:-1], H, D).permute(0, 1, 3, 2, 4)
                if msa_mask is not None:
                    mm = msa_mask[:, start: start + self.chunk_size].to(v.dtype)[:, :, None, :, None]
                    v = v * mm
                out = torch.einsum("bhij,bmhjd->bmhid", weights, v)
                out = out * torch.sigmoid(g)
                out = out.permute(0, 1, 3, 2, 4).reshape(*out.shape[:-3], -1, H * D)
                out_chunks.append(F.linear(out, self.linear_out_w.to(out.dtype)))
            return torch.cat(out_chunks, dim=1)

    class TriangleMultiplication(nn.Module):
        def __init__(self, pair_dim: int, eps=1e-5):
            super().__init__()
            self.pair_dim = pair_dim
            self.eps = eps
            self.ln_in_w = nn.Parameter(torch.ones(pair_dim))
            self.ln_in_b = nn.Parameter(torch.zeros(pair_dim))
            self.merged_p = nn.Parameter(torch.zeros(4 * pair_dim, pair_dim))
            self.merged_g = nn.Parameter(torch.zeros(5 * pair_dim, pair_dim))
            self.lin_out = nn.Parameter(torch.zeros(pair_dim, pair_dim))

        def forward(self, z, pair_mask=None):
            d = self.pair_dim
            z_ln = _fp32_ln(z, self.ln_in_w, self.ln_in_b, self.eps)
            p = F.linear(z_ln, self.merged_p.to(z_ln.dtype))
            g4 = torch.sigmoid(F.linear(z_ln, self.merged_g[:4 * d].to(z_ln.dtype)))
            a1, b1, a2, b2 = (p * g4).chunk(4, dim=-1)
            if pair_mask is not None:
                pm = pair_mask[..., None].to(z.dtype)
                pm_T = pair_mask.transpose(-1, -2)[..., None].to(z.dtype)
                a1 = a1 * pm; b1 = b1 * pm
                a2 = a2 * pm_T; b2 = b2 * pm_T
            x_out = torch.einsum("bikd,bjkd->bijd", a1, b1)
            x_in = torch.einsum("bkid,bkjd->bijd", a2, b2)
            x_out_ln = _fp32_ln_noaffine(x_out, self.eps)
            x_in_ln = _fp32_ln_noaffine(x_in, self.eps)
            g_out = torch.sigmoid(F.linear(z_ln, self.merged_g[4 * d:].to(z_ln.dtype)))
            out = F.linear(x_out_ln + x_in_ln, self.lin_out.to(x_out_ln.dtype)) * g_out
            return z + out

    class TriangleAttention(nn.Module):
        def __init__(self, pair_dim, num_heads, head_dim, eps=1e-5):
            super().__init__()
            self.pair_dim = pair_dim
            self.num_heads = num_heads
            self.head_dim = head_dim
            self.eps = eps
            self.pair2b = nn.Parameter(torch.zeros(2 * num_heads, pair_dim))
            self.pair2qkvg1 = nn.Parameter(torch.zeros(4 * num_heads * head_dim, pair_dim))
            self.pair2qkvg2 = nn.Parameter(torch.zeros(4 * num_heads * head_dim, pair_dim))
            self.linear_out = nn.Parameter(torch.zeros(pair_dim, 2 * num_heads * head_dim))
            self.out_scalers = nn.Parameter(torch.ones(pair_dim))

        def _run(self, z_ln, proj_w, bias, pair_mask_2d, transpose_pair):
            b, n, _, _ = z_ln.shape
            H, D = self.num_heads, self.head_dim
            z_rows = z_ln.transpose(1, 2) if transpose_pair else z_ln
            proj = F.linear(z_rows, proj_w.to(z_rows.dtype)).reshape(b, n, n, H, 4, D)
            q = proj[..., 0, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            k = proj[..., 1, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            v = proj[..., 2, :].permute(0, 1, 3, 2, 4).reshape(b * n, H, n, D)
            g = proj[..., 3, :]
            mask_full = bias.unsqueeze(1).expand(b, n, H, n, n).reshape(b * n, H, n, n)
            if pair_mask_2d is not None:
                pm = pair_mask_2d.to(torch.bool)
                attn_m = pm.unsqueeze(-1) & pm.unsqueeze(-2)
                add = torch.where(
                    attn_m.reshape(b * n, 1, n, n),
                    torch.zeros((), dtype=mask_full.dtype, device=mask_full.device),
                    torch.full((), -1e4, dtype=mask_full.dtype, device=mask_full.device),
                )
                mask_full = mask_full + add
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_full, scale=D ** -0.5)
            attn = attn.reshape(b, n, H, n, D).permute(0, 1, 3, 2, 4)
            return attn * torch.sigmoid(g)

        def forward(self, z, pair_mask=None):
            b, n, _, _ = z.shape
            H = self.num_heads
            z_ln = _fp32_ln_noaffine(z, self.eps)
            bias_all = F.linear(z_ln, self.pair2b.to(z_ln.dtype))
            bias_s = bias_all[..., :H].permute(0, 3, 1, 2)
            bias_e = bias_all[..., H:].permute(0, 3, 1, 2)
            out_s = self._run(z_ln, self.pair2qkvg1, bias_s, pair_mask, transpose_pair=False)
            col_m = pair_mask.transpose(-1, -2) if pair_mask is not None else None
            out_e = self._run(z_ln, self.pair2qkvg2, bias_e, col_m, transpose_pair=True)
            b_, n_, _, _, _ = out_s.shape
            out_s_f = out_s.reshape(b_, n_, n_, H * self.head_dim)
            out_e_f = out_e.reshape(b_, n_, n_, H * self.head_dim)
            combined = torch.cat([out_s_f, out_e_f], dim=-1)
            lin = F.linear(combined, self.linear_out.to(combined.dtype))
            return z + lin * self.out_scalers.to(lin.dtype)

    class MSAModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_s2m = nn.Parameter(torch.zeros(64, 384))
            self.outer_product_mean = nn.ModuleList([OuterProductMean(64, 256) for _ in range(4)])
            self.msa_pair_weighted_averaging = nn.ModuleList([MSAPairWeightedAveraging(64, 256) for _ in range(3)])
            self.msa_transition = nn.ModuleList([Transition(64, expansion=4) for _ in range(3)])
            self.pair_transition = nn.ModuleList([Transition(256, expansion=4) for _ in range(4)])
            self.triangular_multiplication = nn.ModuleList([TriangleMultiplication(256) for _ in range(4)])
            self.triangular_attention = nn.ModuleList([TriangleAttention(256, num_heads=4, head_dim=64) for _ in range(4)])

        def forward(self, single, pair, msa_input, token_pair_mask=None, msa_mask=None,
                    dump: dict[str, torch.Tensor] | None = None):
            def _dump(key, t):
                if dump is not None:
                    dump[key] = t.detach().float().cpu()

            msa = msa_input
            if msa.shape[1] > 0:
                msa = msa + F.linear(single, self.linear_s2m.to(single.dtype))[:, None, :, :]
            _dump("after_linear_s2m_msa", msa)
            for i in range(4):
                opm_delta = self.outer_product_mean[i](msa, msa_mask=msa_mask)
                _dump(f"round_{i}.opm_delta_pair", opm_delta)
                pair = pair + opm_delta
                _dump(f"round_{i}.pair_after_opm", pair)
                if i < 3:
                    msa = msa + self.msa_transition[i](msa)
                    _dump(f"round_{i}.msa_after_transition", msa)
                    msa = msa + self.msa_pair_weighted_averaging[i](msa, pair, token_pair_mask=token_pair_mask, msa_mask=msa_mask)
                    _dump(f"round_{i}.msa_after_pw", msa)
                pair_trans_out = self.pair_transition[i](pair)
                _dump(f"round_{i}.pair_trans_out", pair_trans_out)
                pair = self.triangular_multiplication[i](pair, pair_mask=token_pair_mask) + pair_trans_out
                _dump(f"round_{i}.pair_after_tri_mult", pair)
                pair = self.triangular_attention[i](pair, pair_mask=token_pair_mask)
                _dump(f"round_{i}.pair_after_tri_attn", pair)
            return pair

    # ------------------------------------------------------------------
    # Load weights from trunk.pt.
    # ------------------------------------------------------------------
    trunk_path = MODELS_DIR / "models_v2" / "trunk.pt"
    print(f"Loading {trunk_path}")
    trunk = torch.jit.load(str(trunk_path), map_location="cpu")
    trunk.eval()
    mm_script = trunk.msa_module

    ts_params = {n: p.detach().clone() for n, p in mm_script.named_parameters()}
    print(f"Found {len(ts_params)} named params. Examples:")
    for i, (k, v) in enumerate(list(ts_params.items())[:15]):
        print(f"  {k}  shape={tuple(v.shape)}  dtype={v.dtype}")
    print(f"  ... ({len(ts_params) - 15} more)")

    compute_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]

    msa_mod = MSAModule()

    # Helper: try a list of candidate names and return the first one present.
    def try_get(candidates: list[str]) -> tuple[str, "torch.Tensor"]:
        for c in candidates:
            if c in ts_params:
                return c, ts_params[c]
        all_keys = list(ts_params.keys())
        raise KeyError(f"None of {candidates} found. Sample keys: {all_keys[:30]}")

    def _assign_from_candidates(param, candidates):
        name, val = try_get(candidates)
        if param.shape != val.shape:
            try:
                val = val.reshape(param.shape)
            except Exception as exc:
                raise ValueError(
                    f"cannot reshape {name} {tuple(val.shape)} -> {tuple(param.shape)}: {exc}"
                )
        param.data.copy_(val.float())

    _assign_from_candidates(msa_mod.linear_s2m, ["linear_s2m.weight"])

    for i in range(4):
        op = msa_mod.outer_product_mean[i]
        _assign_from_candidates(op.weight_ab, [f"outer_product_mean.{i}.weight_ab"])
        _assign_from_candidates(op.ln_out_w, [
            f"outer_product_mean.{i}.ln_out.weight",
            f"outer_product_mean.{i}.layer_norm_out.weight",
        ])
        _assign_from_candidates(op.ln_out_b, [
            f"outer_product_mean.{i}.ln_out.bias",
            f"outer_product_mean.{i}.layer_norm_out.bias",
        ])
        _assign_from_candidates(op.linear_out_w, [f"outer_product_mean.{i}.linear_out.weight"])
        _assign_from_candidates(op.linear_out_b, [f"outer_product_mean.{i}.linear_out.bias"])

        if i < 3:
            mt = msa_mod.msa_transition[i]
            _assign_from_candidates(mt.norm_w, [
                f"msa_transition.{i}.layer_norm.weight",
                f"msa_transition.{i}.norm.weight",
            ])
            _assign_from_candidates(mt.norm_b, [
                f"msa_transition.{i}.layer_norm.bias",
                f"msa_transition.{i}.norm.bias",
            ])
            _assign_from_candidates(mt.up_w, [
                f"msa_transition.{i}.linear_no_bias_ab.weight",
                f"msa_transition.{i}.up.weight",
            ])
            _assign_from_candidates(mt.down_w, [
                f"msa_transition.{i}.linear_out.weight",
                f"msa_transition.{i}.down.weight",
            ])

            pw = msa_mod.msa_pair_weighted_averaging[i]
            _assign_from_candidates(pw.ln_msa_w, [
                f"msa_pair_weighted_averaging.{i}.layernorm_msa.weight",
                f"msa_pair_weighted_averaging.{i}.layer_norm_msa.weight",
                f"msa_pair_weighted_averaging.{i}.ln_msa.weight",
            ])
            _assign_from_candidates(pw.ln_msa_b, [
                f"msa_pair_weighted_averaging.{i}.layernorm_msa.bias",
                f"msa_pair_weighted_averaging.{i}.layer_norm_msa.bias",
                f"msa_pair_weighted_averaging.{i}.ln_msa.bias",
            ])
            _assign_from_candidates(pw.ln_pair_w, [
                f"msa_pair_weighted_averaging.{i}.layernorm_pair.weight",
                f"msa_pair_weighted_averaging.{i}.layer_norm_pair.weight",
                f"msa_pair_weighted_averaging.{i}.ln_pair.weight",
            ])
            _assign_from_candidates(pw.ln_pair_b, [
                f"msa_pair_weighted_averaging.{i}.layernorm_pair.bias",
                f"msa_pair_weighted_averaging.{i}.layer_norm_pair.bias",
                f"msa_pair_weighted_averaging.{i}.ln_pair.bias",
            ])
            _assign_from_candidates(pw.linear_msa2vg_w, [f"msa_pair_weighted_averaging.{i}.linear_msa2vg.weight"])
            _assign_from_candidates(pw.linear_pair_w, [f"msa_pair_weighted_averaging.{i}.linear_pair.weight"])
            _assign_from_candidates(pw.linear_out_w, [
                f"msa_pair_weighted_averaging.{i}.linear_out_no_bias.weight",
                f"msa_pair_weighted_averaging.{i}.linear_out.weight",
            ])

        pt = msa_mod.pair_transition[i]
        _assign_from_candidates(pt.norm_w, [
            f"pair_transition.{i}.layer_norm.weight",
            f"pair_transition.{i}.norm.weight",
        ])
        _assign_from_candidates(pt.norm_b, [
            f"pair_transition.{i}.layer_norm.bias",
            f"pair_transition.{i}.norm.bias",
        ])
        _assign_from_candidates(pt.up_w, [
            f"pair_transition.{i}.linear_no_bias_ab.weight",
            f"pair_transition.{i}.up.weight",
        ])
        _assign_from_candidates(pt.down_w, [
            f"pair_transition.{i}.linear_out.weight",
            f"pair_transition.{i}.down.weight",
        ])

        tm = msa_mod.triangular_multiplication[i]
        _assign_from_candidates(tm.ln_in_w, [
            f"triangular_multiplication.{i}.layernorm_z_in.weight",
            f"triangular_multiplication.{i}.ln_in.weight",
        ])
        _assign_from_candidates(tm.ln_in_b, [
            f"triangular_multiplication.{i}.layernorm_z_in.bias",
            f"triangular_multiplication.{i}.ln_in.bias",
        ])
        _assign_from_candidates(tm.merged_p, [
            f"triangular_multiplication.{i}.merged_linear_p.weight",
            f"triangular_multiplication.{i}.merged_p.weight",
        ])
        _assign_from_candidates(tm.merged_g, [
            f"triangular_multiplication.{i}.merged_linear_g.weight",
            f"triangular_multiplication.{i}.merged_g.weight",
        ])
        _assign_from_candidates(tm.lin_out, [
            f"triangular_multiplication.{i}.linear_z_out.weight",
            f"triangular_multiplication.{i}.lin_out.weight",
        ])

        ta = msa_mod.triangular_attention[i]
        ta_prefix = f"triangular_attention.{i}"
        _assign_from_candidates(ta.pair2b, [f"{ta_prefix}.pair2b.weight"])
        _assign_from_candidates(ta.pair2qkvg1, [f"{ta_prefix}.pair2qkvg1.weight"])
        _assign_from_candidates(ta.pair2qkvg2, [f"{ta_prefix}.pair2qkvg2.weight"])
        _assign_from_candidates(ta.linear_out, [f"{ta_prefix}.linear_out.weight"])
        _assign_from_candidates(ta.out_scalers, [f"{ta_prefix}.out_scalers"])

    msa_mod = msa_mod.to(device).eval()
    # Cast non-LayerNorm / non-out_scalers params to compute dtype, matching
    # chai-lab's autocast policy where linears run in bf16 and norms keep fp32
    # weights. Our _fp32_ln helpers upcast to fp32 per-call regardless of
    # stored param dtype, so we can safely cast norm weights here too (it's a
    # no-op for correctness).
    for name, p in msa_mod.named_parameters():
        keep_fp32 = any(tag in name for tag in ("norm", "ln_", "out_scalers"))
        if keep_fp32:
            continue
        p.data = p.data.to(compute_dtype)  # noqa: F821
    print(f"eager MSAModule assembled; running forward (dtype={dtype})")

    # ------------------------------------------------------------------
    # Load captured inputs.
    # ------------------------------------------------------------------
    data = np.load(io.BytesIO(intermediates_npz), allow_pickle=False)
    single_init = torch.from_numpy(data["embedding.token_single_initial"]).to(device, dtype=compute_dtype)
    pair_init = torch.from_numpy(data["embedding.token_pair_initial"]).to(device, dtype=compute_dtype)
    msa_input = torch.from_numpy(data["embedding.msa"]).to(device, dtype=compute_dtype)

    token_exists = torch.from_numpy(data["inputs.batch.token_exists_mask"]).to(device)
    msa_mask = torch.from_numpy(data["inputs.batch.msa_mask"]).to(device)
    token_pair_mask = token_exists[..., :, None] & token_exists[..., None, :]

    print(f"  single  {tuple(single_init.shape)}  max_abs={single_init.abs().max().item():.3f}")
    print(f"  pair    {tuple(pair_init.shape)}    max_abs={pair_init.abs().max().item():.3f}")
    print(f"  msa     {tuple(msa_input.shape)}    max_abs={msa_input.abs().max().item():.3f}")

    # Recycle 0: prev caches are zero, so single/pair after recycle_proj are just
    # single_init and pair_init.
    single = single_init
    pair = pair_init

    # Template_embedder on 1L2Y is a no-op (no templates): we'd need to also
    # replay it. Inspection of the 1L2Y intermediates shows template_mask is
    # all zero, so template_embedder returns pair unchanged. Confirm:
    template_mask = torch.from_numpy(data["inputs.batch.template_mask"]).to(device)
    num_active_templates = int(template_mask.any(dim=(-1,)).sum().item())
    print(f"  active templates: {num_active_templates} (expected 0 for 1L2Y)")

    dump_intermediates: dict[str, torch.Tensor] = {}
    pair_after_msa = msa_mod(
        single, pair, msa_input,
        token_pair_mask=token_pair_mask, msa_mask=msa_mask,
        dump=dump_intermediates,
    )
    torch.cuda.synchronize()

    pair_np = pair_after_msa.detach().float().cpu().numpy()
    print(f"  post_msa_pair  shape={pair_np.shape}  max_abs={np.abs(pair_np).max():.3f}  mean={pair_np.mean():+.4e}")

    payload = {"post_msa_pair": pair_np}
    for k, v in dump_intermediates.items():
        payload[k] = v.numpy()
        print(f"  {k:40s} max_abs={float(v.abs().max()):.3f}  mean={float(v.mean()):+.4e}")

    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    return {"cuda_post_msa_fp32.npz": buf.getvalue()}


@app.local_entrypoint()
def main(
    intermediates_npz: str = "/tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz",
    dtype: str = "fp32",
) -> None:
    src = Path(intermediates_npz)
    if not src.is_file():
        raise FileNotFoundError(src)
    print(f"Sending {src.stat().st_size / (1 << 20):.1f} MB to Modal (dtype={dtype})")
    result = cuda_msa_module_probe.remote(src.read_bytes(), dtype)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, blob in result.items():
        # When bf16 is requested, rename to cuda_post_msa_bf16.npz to avoid
        # overwriting the fp32 run.
        base_name = name.replace("_fp32", f"_{dtype}") if dtype != "fp32" else name
        dst = OUT_DIR / base_name
        dst.write_bytes(blob)
        print(f"wrote {dst} ({len(blob) / (1 << 20):.2f} MB)")
