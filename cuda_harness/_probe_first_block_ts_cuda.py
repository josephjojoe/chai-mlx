"""Modal probe: call **the actual TorchScript** ``pairformer_stack.blocks[0]``
from ``trunk.pt`` on synthetic inputs, alongside a line-for-line eager
reimplementation at fp32 / bf16 / fp64.

This is the real MLX-vs-CUDA ground-truth probe for the drift investigation.
``_probe_first_block_cuda.py`` (the older probe) only runs eager PyTorch; it
can tell us whether the MLX port matches an eager reimplementation, but not
whether MLX matches the scripted module that chai-lab actually executes.
The scripted module's ordering and tiling of reductions are the real targets
for the drift comparison.

Ground truths produced (each saved as ``cuda_out_<tag>.npz`` + merged into
``block_0_ts.npz``):

* ``ts_bf16``   — scripted block-0, casts ingested in the graph (bf16 linears,
                 fp32 layer_norms / softmax), matches chai-lab's real runtime.
* ``ts_fp32``   — scripted block-0 with inputs forced to fp32 **and** model
                 cast to fp32. The scripted graph still has per-op bf16 casts,
                 so this is "fp32 where possible" rather than "pure fp32".
* ``eager_fp64`` — eager PyTorch at fp64 on CPU (no scaled_dot_product_attention
                 fusion, no tiling). This is the closest we can cheaply get to
                 an "infinite-precision" reference; MLX vs eager_fp64 is the
                 raw MLX summation-order drift, independent of CUDA tiling.
* ``eager_fp32`` — eager PyTorch at fp32 on GPU (TF32 off). Comparable against
                 ``ts_fp32``; any delta is CUDA kernel fusion / tiling.
* ``eager_bf16`` — eager PyTorch at bf16 on GPU. Comparable against ``ts_bf16``.

Each payload contains the 7 sub-op tensors (``pair_transition_out``,
``z_after_tri_mult``, ``z_after_residual``, ``z_after_tri_attn``,
``attn_delta``, ``s_after_attn``, ``s_after_transition``) plus the block's
final ``(s_final, z_final)``.

Usage::

    modal run -m cuda_harness._probe_first_block_ts_cuda
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
def cuda_block_0_ts_probe(inputs_npz: bytes) -> dict[str, bytes]:
    """Run scripted block-0 and eager block-0 at several precisions."""
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
    # 1) Load trunk.pt; grab block 0 (a ScriptModule) and its parameters.
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

    # ------------------------------------------------------------------
    # 2) Eager PyTorch reimplementation (same as _probe_first_block_cuda.py,
    #    with an fp64 knob for the "oracle" reference run).
    # ------------------------------------------------------------------
    def _fp32_layernorm(
        x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, eps: float,
        *, reduction_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.to(reduction_dtype)
        w_f = weight.to(reduction_dtype) if weight is not None else None
        b_f = bias.to(reduction_dtype) if bias is not None else None
        y = F.layer_norm(x_f, (x_f.shape[-1],), weight=w_f, bias=b_f, eps=eps)
        return y.to(orig_dtype)

    def _lin(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        if x.dtype != w.dtype:
            promoted = torch.promote_types(x.dtype, w.dtype)
            x = x.to(promoted)
            w = w.to(promoted)
        return F.linear(x, w)

    class Transition(nn.Module):
        def __init__(self, dim: int, expansion: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.norm_w = nn.Parameter(torch.zeros(dim))
            self.norm_b = nn.Parameter(torch.zeros(dim))
            self.up_w = nn.Parameter(torch.zeros(2 * expansion * dim, dim))
            self.down_w = nn.Parameter(torch.zeros(dim, expansion * dim))
            self.eps = eps

        def forward(self, x: torch.Tensor, *, reduction_dtype: torch.dtype = torch.float32) -> torch.Tensor:
            normed = _fp32_layernorm(x, self.norm_w, self.norm_b, self.eps, reduction_dtype=reduction_dtype)
            up = _lin(normed, self.up_w)
            a, b = up.chunk(2, dim=-1)
            gated = F.silu(a) * b
            return _lin(gated, self.down_w)

    class TriangleMultiplication(nn.Module):
        def __init__(self, pair_dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.pair_dim = pair_dim
            self.ln_in_w = nn.Parameter(torch.zeros(pair_dim))
            self.ln_in_b = nn.Parameter(torch.zeros(pair_dim))
            self.merged_p = nn.Parameter(torch.zeros(4 * pair_dim, pair_dim))
            self.merged_g = nn.Parameter(torch.zeros(5 * pair_dim, pair_dim))
            self.lin_out = nn.Parameter(torch.zeros(pair_dim, pair_dim))
            self.eps = eps

        def forward(
            self, z: torch.Tensor, pair_mask: torch.Tensor | None,
            *, reduction_dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            d = self.pair_dim
            z_ln = _fp32_layernorm(z, self.ln_in_w, self.ln_in_b, self.eps, reduction_dtype=reduction_dtype)
            p = _lin(z_ln, self.merged_p)
            g4 = torch.sigmoid(_lin(z_ln, self.merged_g[: 4 * d]))
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
            x_out_ln = F.layer_norm(x_out.to(reduction_dtype), (d,), eps=self.eps).to(x_out.dtype)
            x_in_ln = F.layer_norm(x_in.to(reduction_dtype), (d,), eps=self.eps).to(x_in.dtype)
            g_out = torch.sigmoid(_lin(z_ln, self.merged_g[4 * d:]))
            out = _lin(x_out_ln + x_in_ln, self.lin_out) * g_out
            return z + out

    class TriangleAttention(nn.Module):
        def __init__(self, pair_dim: int, num_heads: int, head_dim: int, eps: float = 1e-5) -> None:
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

        def _run_direction(
            self, z_ln: torch.Tensor, proj_w: torch.Tensor, bias_dir: torch.Tensor,
            pair_mask_2d: torch.Tensor | None, *, transpose_pair: bool,
        ) -> torch.Tensor:
            b, n, _, _ = z_ln.shape
            H, D = self.num_heads, self.head_dim
            if transpose_pair:
                z_rows = z_ln.transpose(1, 2)
            else:
                z_rows = z_ln
            proj = _lin(z_rows, proj_w).reshape(b, n, n, H, 4, D)
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

        def forward(
            self, z: torch.Tensor, pair_mask: torch.Tensor | None,
            *, reduction_dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            b, n, _, _ = z.shape
            H = self.num_heads
            z_ln = F.layer_norm(z.to(reduction_dtype), (self.pair_dim,), eps=self.eps).to(z.dtype)
            bias_all = _lin(z_ln, self.pair2b)
            bias_start = bias_all[..., :H].permute(0, 3, 1, 2)
            bias_end = bias_all[..., H:].permute(0, 3, 1, 2)
            out_s = self._run_direction(z_ln, self.pair2qkvg1, bias_start, pair_mask, transpose_pair=False)
            col_mask = pair_mask.transpose(-1, -2) if pair_mask is not None else None
            out_e = self._run_direction(z_ln, self.pair2qkvg2, bias_end, col_mask, transpose_pair=True)
            out_s_f = out_s.reshape(b, n, n, H * self.head_dim)
            out_e_f = out_e.reshape(b, n, n, H * self.head_dim)
            combined = torch.cat([out_s_f, out_e_f], dim=-1)
            lin_out = _lin(combined, self.linear_out)
            scalers = self.out_scalers
            if lin_out.dtype != scalers.dtype:
                promoted = torch.promote_types(lin_out.dtype, scalers.dtype)
                lin_out = lin_out.to(promoted)
                scalers = scalers.to(promoted)
            out = lin_out * scalers
            if z.dtype != out.dtype:
                promoted = torch.promote_types(z.dtype, out.dtype)
                return z.to(promoted) + out.to(promoted)
            return z + out

    class AttentionPairBias(nn.Module):
        def __init__(self, single_dim: int, pair_dim: int, num_heads: int, head_dim: int, eps: float = 1e-5) -> None:
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

        def forward(
            self, x: torch.Tensor, pair: torch.Tensor, pair_mask: torch.Tensor | None,
            *, reduction_dtype: torch.dtype = torch.float32,
        ) -> torch.Tensor:
            b, n, _ = x.shape
            H, D = self.num_heads, self.head_dim
            pair_ln = _fp32_layernorm(pair, self.pair_norm_w, self.pair_norm_b, self.eps, reduction_dtype=reduction_dtype)
            bias = _lin(pair_ln, self.pair_linear).permute(0, 3, 1, 2)
            if pair_mask is not None:
                pm = pair_mask.to(torch.bool)
                add = torch.where(
                    pm,
                    torch.zeros((), dtype=bias.dtype, device=bias.device),
                    torch.full((), -1e4, dtype=bias.dtype, device=bias.device),
                )
                bias = bias + add.unsqueeze(1)
            x_ln = _fp32_layernorm(x, self.single_norm_w, self.single_norm_b, self.eps, reduction_dtype=reduction_dtype)
            qkvg = _lin(x_ln, self.input2qkvg)
            q, k, v, g = qkvg.chunk(4, dim=-1)
            q = q.reshape(b, n, H, D).permute(0, 2, 1, 3)
            k = k.reshape(b, n, H, D).permute(0, 2, 1, 3)
            v = v.reshape(b, n, H, D).permute(0, 2, 1, 3)
            g = g.reshape(b, n, H, D)
            qb = self.query_bias.unsqueeze(0).unsqueeze(-2)
            if q.dtype != qb.dtype:
                promoted = torch.promote_types(q.dtype, qb.dtype)
                q = q.to(promoted)
                qb = qb.to(promoted)
            q = q + qb
            sdpa_dtype = q.dtype
            for t in (k, v, bias):
                if t is not None:
                    sdpa_dtype = torch.promote_types(sdpa_dtype, t.dtype)
            q = q.to(sdpa_dtype)
            k = k.to(sdpa_dtype)
            v = v.to(sdpa_dtype)
            bias_sdpa = bias.to(sdpa_dtype)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias_sdpa, scale=D ** -0.5)
            attn = attn.permute(0, 2, 1, 3)
            g_sig = torch.sigmoid(g)
            if attn.dtype != g_sig.dtype:
                promoted = torch.promote_types(attn.dtype, g_sig.dtype)
                attn = attn.to(promoted)
                g_sig = g_sig.to(promoted)
            attn = attn * g_sig
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
            self, z: torch.Tensor, s: torch.Tensor,
            pair_mask: torch.Tensor | None, single_mask: torch.Tensor | None,
            *, reduction_dtype: torch.dtype = torch.float32,
        ) -> dict[str, torch.Tensor]:
            pair_transition_out = self.transition_pair(z, reduction_dtype=reduction_dtype)
            z_after_tri_mult = self.triangle_multiplication(z, pair_mask, reduction_dtype=reduction_dtype)
            z_after_residual = z_after_tri_mult + pair_transition_out
            z_after_tri_attn = self.triangle_attention(z_after_residual, pair_mask, reduction_dtype=reduction_dtype)
            attn_delta = self.attention_pair_bias(s, z_after_tri_attn, pair_mask, reduction_dtype=reduction_dtype)
            if single_mask is not None:
                mask_f = single_mask.to(attn_delta.dtype).unsqueeze(-1)
                attn_delta = attn_delta * mask_f
            s_after_attn = s + attn_delta
            s_after_transition = s_after_attn + self.transition_single(s_after_attn, reduction_dtype=reduction_dtype)
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
    # 3) Wire weights from trunk.pt into the eager module.
    # ------------------------------------------------------------------
    block = PairformerBlock()

    def _assign(param: torch.nn.Parameter, value: torch.Tensor) -> None:
        if tuple(param.shape) != tuple(value.shape):
            raise ValueError(f"shape mismatch: param {tuple(param.shape)} vs value {tuple(value.shape)}")
        param.data.copy_(value.float())

    _assign(block.transition_pair.norm_w, ts_params["transition_pair.layer_norm.weight"])
    _assign(block.transition_pair.norm_b, ts_params["transition_pair.layer_norm.bias"])
    _assign(block.transition_pair.up_w, ts_params["transition_pair.linear_no_bias_ab.weight"])
    _assign(block.transition_pair.down_w, ts_params["transition_pair.linear_out.weight"])

    _assign(block.triangle_multiplication.ln_in_w, ts_params["triangle_multiplication.layernorm_z_in.weight"])
    _assign(block.triangle_multiplication.ln_in_b, ts_params["triangle_multiplication.layernorm_z_in.bias"])
    _assign(block.triangle_multiplication.merged_p, ts_params["triangle_multiplication.merged_linear_p.weight"])
    _assign(block.triangle_multiplication.merged_g, ts_params["triangle_multiplication.merged_linear_g.weight"])
    _assign(block.triangle_multiplication.lin_out, ts_params["triangle_multiplication.linear_z_out.weight"])

    _assign(block.triangle_attention.pair2b, ts_params["triangle_attention.pair2b.weight"])
    _assign(block.triangle_attention.pair2qkvg1, ts_params["triangle_attention.pair2qkvg1.weight"])
    _assign(block.triangle_attention.pair2qkvg2, ts_params["triangle_attention.pair2qkvg2.weight"])
    _assign(block.triangle_attention.linear_out, ts_params["triangle_attention.linear_out.weight"])
    _assign(block.triangle_attention.out_scalers, ts_params["triangle_attention.out_scalers"])

    _assign(block.attention_pair_bias.single_norm_w, ts_params["attention_pair_bias.single_layer_norm.weight"])
    _assign(block.attention_pair_bias.single_norm_b, ts_params["attention_pair_bias.single_layer_norm.bias"])
    _assign(block.attention_pair_bias.pair_norm_w, ts_params["attention_pair_bias.pair_layer_norm.weight"])
    _assign(block.attention_pair_bias.pair_norm_b, ts_params["attention_pair_bias.pair_layer_norm.bias"])
    _assign(block.attention_pair_bias.pair_linear, ts_params["attention_pair_bias.pair_linear.weight"])
    _assign(block.attention_pair_bias.query_bias, ts_params["attention_pair_bias.attention.query_bias"])
    w_qkvg = ts_params["attention_pair_bias.attention.input2qkvg.weight"]
    in_dim = w_qkvg.shape[0]
    _assign(block.attention_pair_bias.input2qkvg, w_qkvg.reshape(in_dim, -1).T.contiguous())
    w_out = ts_params["attention_pair_bias.attention.output_proj.weight"]
    out_dim = w_out.shape[-1]
    _assign(block.attention_pair_bias.output_proj, w_out.reshape(-1, out_dim).T.contiguous())

    _assign(block.transition_single.norm_w, ts_params["transition_single.layer_norm.weight"])
    _assign(block.transition_single.norm_b, ts_params["transition_single.layer_norm.bias"])
    _assign(block.transition_single.up_w, ts_params["transition_single.linear_no_bias_ab.weight"])
    _assign(block.transition_single.down_w, ts_params["transition_single.linear_out.weight"])

    # ------------------------------------------------------------------
    # 4) Shared inputs.
    # ------------------------------------------------------------------
    inputs = np.load(io.BytesIO(inputs_npz))
    single_np = inputs["single"]
    pair_np = inputs["pair"]
    pair_mask_np = inputs["pair_mask"]
    single_mask_np = inputs["single_mask"]

    result: dict[str, bytes] = {}

    # ------------------------------------------------------------------
    # 5) Scripted-block call. We try ``block_0.forward(...)`` at bf16 and
    #    fp32; the scripted graph does its own per-op casts so we only
    #    control the *input* dtype and the *param* dtype policy.
    # ------------------------------------------------------------------
    block_0_gpu = block_0.to(device).eval()

    # Introspect block_0 to find a callable method.
    print("\n=== block_0 introspection ===")
    try:
        # Scripted methods are on the _c side.
        methods = []
        try:
            for m in block_0._c._get_methods():
                methods.append(m.name)
        except Exception:
            pass
        print(f"  scripted methods: {methods}")
        print(f"  type: {type(block_0).__name__}")
        print(f"  dir (sample): {[a for a in dir(block_0) if not a.startswith('_')][:30]}")
        # Try common names we might see.
        for candidate in ("forward", "forward_256", "__call__", "call", "invoke"):
            has = hasattr(block_0, candidate)
            print(f"  has {candidate}: {has}")
    except Exception as exc:
        print(f"  introspection failed: {exc!r}")

    def _run_scripted(input_dtype: torch.dtype) -> dict[str, torch.Tensor]:
        s = torch.from_numpy(single_np).to(device).to(input_dtype)
        z = torch.from_numpy(pair_np).to(device).to(input_dtype)
        pair_mask = torch.from_numpy(pair_mask_np).to(device)
        single_mask = torch.from_numpy(single_mask_np).to(device)

        # block_0 is a RecursiveScriptModule inside a ModuleList. TorchScript
        # strips ``forward`` when the submodule is only ever called inlined
        # by a parent graph (as is the case here — ``trunk.forward_256`` inlines
        # every block). We therefore cannot invoke block_0 directly.
        # Try multiple paths; if all fail, return an error dict.
        errors: list[str] = []
        for call_sig in (
            ("positional (z, s, pair_mask, single_mask)", lambda: block_0_gpu(z, s, pair_mask, single_mask)),
            ("positional (s, z, pair_mask, single_mask)", lambda: block_0_gpu(s, z, pair_mask, single_mask)),
            ("forward_256 (z, s, pair_mask, single_mask)", lambda: getattr(block_0_gpu, "forward_256")(z, s, pair_mask, single_mask)),
            ("positional (z, s, pair_mask)",              lambda: block_0_gpu(z, s, pair_mask)),
        ):
            name, fn = call_sig
            try:
                out = fn()
                if isinstance(out, tuple) and len(out) == 2:
                    out_s, out_z = out
                else:
                    out_s, out_z = None, out  # best-effort
                return {"s_final": out_s if out_s is not None else torch.zeros(1), "z_final": out_z}
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
        raise RuntimeError("block_0 is not directly invocable; tried:\n  " + "\n  ".join(errors))

    for tag, in_dtype in (("ts_bf16", torch.bfloat16), ("ts_fp32", torch.float32)):
        print(f"\n=== scripted block_0 :: {tag} ===")
        try:
            outs = _run_scripted(in_dtype)
            dump = {k: v.detach().float().cpu().numpy() for k, v in outs.items()}
            for k, v in dump.items():
                print(f"  {k:26s} shape={v.shape}  max_abs={float(np.abs(v).max()):.4f}  mean={float(v.mean()):+.4e}")
            buf = io.BytesIO()
            np.savez(buf, **dump)
            result[f"cuda_out_{tag}"] = buf.getvalue()
        except Exception as exc:
            print(f"  FAILED ({exc!r})")
            result[f"cuda_out_{tag}"] = np.array([], dtype=np.uint8).tobytes()

    # ------------------------------------------------------------------
    # 6) Eager reference runs at fp64 (CPU, oracle), fp32 (GPU), bf16 (GPU).
    # ------------------------------------------------------------------
    def _run_eager(tag: str, device_: torch.device, model_dtype: torch.dtype, input_dtype: torch.dtype) -> None:
        print(f"\n=== eager :: {tag} (device={device_}, model_dtype={model_dtype}, input_dtype={input_dtype}) ===")
        # Fresh copy of eager block with params cast appropriately.
        block_fresh = PairformerBlock()
        for name, p_target in block_fresh.named_parameters():
            p_target.data.copy_(dict(block.named_parameters())[name].data.float())
        block_fresh = block_fresh.to(device_).eval()
        for name, p in block_fresh.named_parameters():
            if any(tag_ in name for tag_ in ("norm_w", "norm_b", "ln_in_w", "ln_in_b", "out_scalers", "query_bias")):
                p.data = p.data.to(torch.float64 if tag == "eager_fp64" else torch.float32)
            else:
                p.data = p.data.to(model_dtype)

        s = torch.from_numpy(single_np).to(device_).to(input_dtype)
        z = torch.from_numpy(pair_np).to(device_).to(input_dtype)
        pair_mask = torch.from_numpy(pair_mask_np).to(device_)
        single_mask = torch.from_numpy(single_mask_np).to(device_)

        reduction_dtype = torch.float64 if tag == "eager_fp64" else torch.float32
        outs = block_fresh(z, s, pair_mask, single_mask, reduction_dtype=reduction_dtype)
        dump = {k: v.detach().float().cpu().numpy() for k, v in outs.items()}
        for k, v in dump.items():
            print(f"  {k:26s} shape={v.shape}  max_abs={float(np.abs(v).max()):.4f}  mean={float(v.mean()):+.4e}")
        buf = io.BytesIO()
        np.savez(buf, **dump)
        result[f"cuda_out_{tag}"] = buf.getvalue()

    _run_eager("eager_fp64", torch.device("cpu"), torch.float64, torch.float64)
    _run_eager("eager_fp32", device, torch.float32, torch.float32)
    _run_eager("eager_bf16", device, torch.bfloat16, torch.bfloat16)

    return result


@app.local_entrypoint()
def main() -> None:
    inputs_path = Path("/tmp/chai_mlx_cuda/first_block_probe/inputs.npz")
    if not inputs_path.exists():
        raise FileNotFoundError(
            f"Expected shared inputs at {inputs_path}; run "
            "`python3 cuda_harness/_probe_first_block_mlx.py` first to produce them."
        )
    inputs_npz = inputs_path.read_bytes()
    print(f"Sending {len(inputs_npz) / (1 << 20):.1f} MB of inputs to Modal")
    result = cuda_block_0_ts_probe.remote(inputs_npz)
    out_dir = inputs_path.parent
    for name, data in result.items():
        out_path = out_dir / f"{name}.npz"
        out_path.write_bytes(data)
        print(f"saved {out_path} ({len(data) / 1024**2:.2f} MB)")
