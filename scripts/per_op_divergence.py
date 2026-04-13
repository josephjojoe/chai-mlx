"""Per-operation divergence analysis for a single pairformer block.

Shares identical weights and inputs between MLX (Metal) and PyTorch (MPS),
then runs each sub-operation and measures where divergence enters.

Usage::

    python scripts/per_op_divergence.py --weights-dir weights/

Requires both mlx and torch installed.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ── Helpers ──────────────────────────────────────────────────────────────


def _mx_to_torch(x: mx.array, dtype=None) -> torch.Tensor:
    """MLX array → PyTorch tensor on DEVICE."""
    np_arr = np.array(x.astype(mx.float32), copy=False)
    t = torch.from_numpy(np_arr).to(DEVICE)
    if dtype is not None:
        t = t.to(dtype)
    return t


def _torch_to_mx(t: torch.Tensor, dtype=None) -> mx.array:
    """PyTorch tensor → MLX array."""
    np_arr = t.detach().cpu().float().numpy()
    arr = mx.array(np_arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _compare(name: str, mlx_out: mx.array, torch_out: torch.Tensor) -> dict:
    """Compare MLX and PyTorch outputs, print stats, return error dict."""
    a = np.array(mlx_out.astype(mx.float32), copy=False)
    b = torch_out.detach().cpu().float().numpy()
    if a.shape != b.shape:
        print(f"  {name}: SHAPE MISMATCH mlx={a.shape} torch={b.shape}")
        return {"name": name, "max": float("inf"), "mean": float("inf")}
    diff = np.abs(a - b)
    mx_val = float(diff.max())
    mn_val = float(diff.mean())
    ref_rms = float(np.sqrt((b ** 2).mean()))
    pcts = np.percentile(diff.ravel(), [50, 90, 99, 100])
    print(f"  {name}:")
    print(f"    max={mx_val:.4e}  mean={mn_val:.4e}  ref_rms={ref_rms:.4e}")
    print(f"    p50={pcts[0]:.4e}  p90={pcts[1]:.4e}  p99={pcts[2]:.4e}  p100={pcts[3]:.4e}")
    return {"name": name, "max": mx_val, "mean": mn_val}


# ── Elementary operations ────────────────────────────────────────────────


def test_elementary_ops(pair_dim: int = 256, n: int = 64) -> list[dict]:
    """Test basic operations with random bf16 data and shared fp32 weights."""
    print("\n" + "=" * 70)
    print(f"ELEMENTARY OPERATIONS (pair_dim={pair_dim}, n={n})")
    print("=" * 70)

    results = []

    # Random bf16 input
    x_np = np.random.randn(1, n, pair_dim).astype(np.float32)
    # Round to bf16 to ensure identical inputs
    x_mx = mx.array(x_np).astype(mx.bfloat16)
    mx.eval(x_mx)
    x_np = np.array(x_mx.astype(mx.float32), copy=False).copy()
    x_torch = torch.from_numpy(x_np).to(torch.bfloat16).to(DEVICE)

    w_np = np.random.randn(pair_dim, pair_dim).astype(np.float32) * 0.02
    w_mx = mx.array(w_np).astype(mx.bfloat16)
    w_torch = torch.from_numpy(w_np).to(torch.bfloat16).to(DEVICE)

    # 1. Matmul (bf16 x bf16)
    print("\n── matmul (bf16 × bf16) ──")
    mlx_out = x_mx @ w_mx.T
    mx.eval(mlx_out)
    torch_out = x_torch @ w_torch.T
    results.append(_compare("matmul_bf16", mlx_out, torch_out))

    # 2. Matmul (bf16 input, fp32 weight — type promotion)
    print("\n── matmul (bf16 input × fp32 weight) ──")
    w_mx_fp32 = mx.array(w_np)
    w_torch_fp32 = torch.from_numpy(w_np).to(DEVICE)
    mlx_out2 = x_mx @ w_mx_fp32.T
    mx.eval(mlx_out2)
    torch_out2 = x_torch.float() @ w_torch_fp32.T  # PyTorch promotes
    results.append(_compare("matmul_mixed", mlx_out2, torch_out2))

    # 3. LayerNorm
    print("\n── LayerNorm ──")
    ln_mx = nn.LayerNorm(pair_dim, eps=1e-5)
    mx.eval(ln_mx.weight, ln_mx.bias)
    ln_w = np.array(ln_mx.weight, copy=False).copy()
    ln_b = np.array(ln_mx.bias, copy=False).copy()

    x_mx_f32 = x_mx.astype(mx.float32)
    x_torch_f32 = x_torch.float()
    mlx_ln = ln_mx(x_mx_f32)
    mx.eval(mlx_ln)
    torch_ln = F.layer_norm(
        x_torch_f32,
        (pair_dim,),
        weight=torch.from_numpy(ln_w).to(DEVICE),
        bias=torch.from_numpy(ln_b).to(DEVICE),
        eps=1e-5,
    )
    results.append(_compare("layernorm", mlx_ln, torch_ln))

    # 4. Softmax
    print("\n── softmax ──")
    logits_np = np.random.randn(1, 8, n, n).astype(np.float32)
    logits_mx = mx.array(logits_np)
    logits_torch = torch.from_numpy(logits_np).to(DEVICE)
    mlx_sm = mx.softmax(logits_mx, axis=-1)
    mx.eval(mlx_sm)
    torch_sm = F.softmax(logits_torch, dim=-1)
    results.append(_compare("softmax", mlx_sm, torch_sm))

    # 5. Sigmoid
    print("\n── sigmoid ──")
    mlx_sig = 1.0 / (1.0 + mx.exp(-x_mx_f32))
    mx.eval(mlx_sig)
    torch_sig = torch.sigmoid(x_torch_f32)
    results.append(_compare("sigmoid", mlx_sig, torch_sig))

    # 6. SiLU (x * sigmoid(x))
    print("\n── SiLU ──")
    mlx_silu = x_mx_f32 * (1.0 / (1.0 + mx.exp(-x_mx_f32)))
    mx.eval(mlx_silu)
    torch_silu = F.silu(x_torch_f32)
    results.append(_compare("silu", mlx_silu, torch_silu))

    # 7. SDPA
    print("\n── scaled_dot_product_attention ──")
    nH, dH = 8, 32
    q_np = np.random.randn(1, nH, n, dH).astype(np.float32)
    k_np = np.random.randn(1, nH, n, dH).astype(np.float32)
    v_np = np.random.randn(1, nH, n, dH).astype(np.float32)
    q_mx, k_mx, v_mx = mx.array(q_np), mx.array(k_np), mx.array(v_np)
    q_t = torch.from_numpy(q_np).to(DEVICE)
    k_t = torch.from_numpy(k_np).to(DEVICE)
    v_t = torch.from_numpy(v_np).to(DEVICE)
    mlx_sdpa = mx.fast.scaled_dot_product_attention(q_mx, k_mx, v_mx, scale=dH ** -0.5)
    mx.eval(mlx_sdpa)
    torch_sdpa = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=dH ** -0.5)
    results.append(_compare("sdpa", mlx_sdpa, torch_sdpa))

    # 8. Einsum — the triangle multiplication contraction
    print("\n── einsum (bikd,bjkd->bijd) — triangle mult outer ──")
    a_np = np.random.randn(1, n, n, 32).astype(np.float32)
    b_np = np.random.randn(1, n, n, 32).astype(np.float32)
    # bf16
    a_mx = mx.array(a_np).astype(mx.bfloat16)
    b_mx = mx.array(b_np).astype(mx.bfloat16)
    mx.eval(a_mx, b_mx)
    a_np_r = np.array(a_mx.astype(mx.float32), copy=False).copy()
    b_np_r = np.array(b_mx.astype(mx.float32), copy=False).copy()
    a_t = torch.from_numpy(a_np_r).to(torch.bfloat16).to(DEVICE)
    b_t = torch.from_numpy(b_np_r).to(torch.bfloat16).to(DEVICE)

    mlx_ein = mx.einsum("bikd,bjkd->bijd", a_mx, b_mx)
    mx.eval(mlx_ein)
    torch_ein = torch.einsum("bikd,bjkd->bijd", a_t, b_t)
    results.append(_compare("einsum_tri_outer_bf16", mlx_ein, torch_ein))

    # Same einsum in fp32
    print("\n── einsum (bikd,bjkd->bijd) — fp32 ──")
    a_mx32 = mx.array(a_np_r)
    b_mx32 = mx.array(b_np_r)
    a_t32 = torch.from_numpy(a_np_r).to(DEVICE)
    b_t32 = torch.from_numpy(b_np_r).to(DEVICE)
    mlx_ein32 = mx.einsum("bikd,bjkd->bijd", a_mx32, b_mx32)
    mx.eval(mlx_ein32)
    torch_ein32 = torch.einsum("bikd,bjkd->bijd", a_t32, b_t32)
    results.append(_compare("einsum_tri_outer_fp32", mlx_ein32, torch_ein32))

    return results


# ── Single pairformer block ─────────────────────────────────────────────


def test_pairformer_block(model_path: Path) -> list[dict]:
    """Run one pairformer block step-by-step in both MLX and PyTorch.

    Uses real model weights and synthetic inputs rounded to bf16 for
    bitwise-identical starting values.
    """
    from chai_mlx import ChaiMLX

    print("\n" + "=" * 70)
    print("SINGLE PAIRFORMER BLOCK — PER-OPERATION TRACE")
    print("=" * 70)

    model = ChaiMLX.from_pretrained(model_path, strict=False)
    block = model.trunk_module.pairformer_stack.blocks[0]
    pair_dim = 256
    single_dim = 384
    n = 64

    # Synthetic inputs (rounded to bf16 for identical starting values)
    z_np = (np.random.randn(1, n, n, pair_dim) * 5.0).astype(np.float32)
    s_np = (np.random.randn(1, n, single_dim) * 5.0).astype(np.float32)
    z_mx = mx.array(z_np).astype(mx.bfloat16)
    s_mx = mx.array(s_np).astype(mx.bfloat16)
    mx.eval(z_mx, s_mx)
    z_np = np.array(z_mx.astype(mx.float32), copy=False).copy()
    s_np = np.array(s_mx.astype(mx.float32), copy=False).copy()
    z_torch = torch.from_numpy(z_np).to(torch.bfloat16).to(DEVICE)
    s_torch = torch.from_numpy(s_np).to(torch.bfloat16).to(DEVICE)

    results = []

    # Helper: extract MLX weights as numpy
    def _w(module, attr="weight"):
        arr = getattr(module, attr)
        return np.array(arr.astype(mx.float32), copy=False).copy()

    def _wb(module):
        w = _w(module, "weight")
        b = _w(module, "bias") if hasattr(module, "bias") and module.bias is not None else None
        return w, b

    def _torch_layernorm(x, ln_module, affine=True):
        dim = x.shape[-1]
        x_f32 = x.float()
        w = torch.from_numpy(_w(ln_module)).to(DEVICE) if affine and hasattr(ln_module, "weight") and ln_module.weight is not None else None
        b = None
        if affine and hasattr(ln_module, "bias") and ln_module.bias is not None:
            b = torch.from_numpy(np.array(ln_module.bias.astype(mx.float32), copy=False).copy()).to(DEVICE)
        eps = getattr(ln_module, "eps", 1e-5)
        return F.layer_norm(x_f32, (dim,), weight=w, bias=b, eps=eps).to(x.dtype)

    def _torch_linear(x, linear_module):
        w, b = _wb(linear_module)
        w_t = torch.from_numpy(w).to(x.dtype).to(DEVICE)
        out = x @ w_t.T
        if b is not None:
            b_t = torch.from_numpy(b).to(x.dtype).to(DEVICE)
            out = out + b_t
        return out

    # ── 1. Transition pair ───────────────────────────────────────────
    print("\n── transition_pair ──")
    trans = block.transition_pair

    # Norm
    mlx_normed = trans.norm(z_mx)
    mx.eval(mlx_normed)
    torch_normed = _torch_layernorm(z_torch, trans.norm)
    results.append(_compare("trans_pair.norm", mlx_normed, torch_normed))

    # Up projection
    mlx_up = trans.up(mlx_normed)
    mx.eval(mlx_up)
    torch_up = _torch_linear(torch_normed, trans.up)
    results.append(_compare("trans_pair.up", mlx_up, torch_up))

    # SwiGLU
    mlx_swiglu = trans.swiglu(mlx_up)
    mx.eval(mlx_swiglu)
    a_t, b_t_val = torch_up.chunk(2, dim=-1)
    torch_swiglu = F.silu(a_t) * b_t_val
    results.append(_compare("trans_pair.swiglu", mlx_swiglu, torch_swiglu))

    # Down projection
    mlx_down = trans.down(mlx_swiglu)
    mx.eval(mlx_down)
    torch_down = _torch_linear(torch_swiglu, trans.down)
    results.append(_compare("trans_pair.down", mlx_down, torch_down))
    pair_trans_mlx = mlx_down
    pair_trans_torch = torch_down

    # ── 2. Triangle multiplication ───────────────────────────────────
    print("\n── triangle_multiplication ──")
    tri = block.triangle_multiplication

    mlx_tri_norm = tri.layernorm_z_in(z_mx)
    mx.eval(mlx_tri_norm)
    torch_tri_norm = _torch_layernorm(z_torch, tri.layernorm_z_in)
    results.append(_compare("tri_mult.norm", mlx_tri_norm, torch_tri_norm))

    # Full triangle mult (too complex to decompose further portably)
    mlx_tri_out = tri(z_mx)
    mx.eval(mlx_tri_out)
    # For PyTorch, we run the full operation using MLX's algorithm with PyTorch ops
    d = pair_dim
    w_p = torch.from_numpy(_w(tri.merged_linear_p)).to(torch.bfloat16).to(DEVICE)
    w_g = torch.from_numpy(_w(tri.merged_linear_g)).to(torch.bfloat16).to(DEVICE)
    z_n_t = torch_tri_norm

    out_chunks_t = []
    in_chunks_t = []
    chunk_size = 32
    for c in range(0, d, chunk_size):
        c_end = min(c + chunk_size, d)
        a1 = (z_n_t @ w_p[c:c_end].T) * torch.sigmoid(z_n_t @ w_g[c:c_end].T)
        b1 = (z_n_t @ w_p[d+c:d+c_end].T) * torch.sigmoid(z_n_t @ w_g[d+c:d+c_end].T)
        a2 = (z_n_t @ w_p[2*d+c:2*d+c_end].T) * torch.sigmoid(z_n_t @ w_g[2*d+c:2*d+c_end].T)
        b2 = (z_n_t @ w_p[3*d+c:3*d+c_end].T) * torch.sigmoid(z_n_t @ w_g[3*d+c:3*d+c_end].T)
        out_chunks_t.append(torch.einsum("bikd,bjkd->bijd", a1, b1))
        in_chunks_t.append(torch.einsum("bkid,bkjd->bijd", a2, b2))

    x_out_t = torch.cat(out_chunks_t, dim=-1)
    x_in_t = torch.cat(in_chunks_t, dim=-1)

    # LayerNorm out + in (both affine=False)
    x_out_t_ln = F.layer_norm(x_out_t.float(), (d,), eps=1e-5).to(torch.bfloat16)
    x_in_t_ln = F.layer_norm(x_in_t.float(), (d,), eps=1e-5).to(torch.bfloat16)

    w_z_out = torch.from_numpy(_w(tri.linear_z_out)).to(torch.bfloat16).to(DEVICE)
    combined_t = (x_out_t_ln + x_in_t_ln)
    proj_t = combined_t @ w_z_out.T

    g_out_t = torch.sigmoid(z_n_t @ w_g[4*d:].T)
    tri_result_t = z_torch + proj_t * g_out_t
    results.append(_compare("tri_mult.full", mlx_tri_out, tri_result_t))

    # After triangle mult + transition sum
    z_mlx_post = mlx_tri_out + pair_trans_mlx
    z_torch_post = tri_result_t + pair_trans_torch
    mx.eval(z_mlx_post)
    results.append(_compare("after_tri+trans", z_mlx_post, z_torch_post))

    # ── 3. Triangle attention ────────────────────────────────────────
    print("\n── triangle_attention ──")
    tri_attn = block.triangle_attention

    mlx_ta_norm = tri_attn.pair_norm(z_mlx_post)
    mx.eval(mlx_ta_norm)
    torch_ta_norm = _torch_layernorm(z_torch_post, tri_attn.pair_norm, affine=False)
    results.append(_compare("tri_attn.norm", mlx_ta_norm, torch_ta_norm))

    # Run full triangle attention — compare by loading TorchScript trunk if available
    # For now, measure the bias projection which is the main linear op
    bias_all_mx = tri_attn.pair2b(mlx_ta_norm)
    mx.eval(bias_all_mx)
    bias_all_torch = _torch_linear(torch_ta_norm, tri_attn.pair2b)
    results.append(_compare("tri_attn.pair2b", bias_all_mx, bias_all_torch))

    # SDPA inside triangle attention — construct matching q/k/v
    H, D = tri_attn.num_heads, tri_attn.head_dim
    proj1_mx = tri_attn.pair2qkvg1(mlx_ta_norm[:, :1])  # single row chunk
    mx.eval(proj1_mx)
    proj1_torch = _torch_linear(torch_ta_norm[:, :1], tri_attn.pair2qkvg1)
    results.append(_compare("tri_attn.qkvg_proj", proj1_mx, proj1_torch))

    # Full block pair output (compound — includes all operations)
    print("\n── full block (compound) ──")
    z_mx_full, s_mx_full = block(z_mx, s_mx)
    mx.eval(z_mx_full, s_mx_full)

    # ── 4. Attention pair bias ───────────────────────────────────────
    print("\n── attention_pair_bias ──")
    apb = block.attention_pair_bias

    mlx_s_norm = apb.single_norm(s_mx)
    mx.eval(mlx_s_norm)
    torch_s_norm = _torch_layernorm(s_torch, apb.single_norm)
    results.append(_compare("attn.single_norm", mlx_s_norm, torch_s_norm))

    mlx_qkvg = apb.input2qkvg(mlx_s_norm)
    mx.eval(mlx_qkvg)
    torch_qkvg = _torch_linear(torch_s_norm, apb.input2qkvg)
    results.append(_compare("attn.input2qkvg", mlx_qkvg, torch_qkvg))

    mlx_z_norm = apb.pair_norm(z_mx)  # using original z for pair bias
    mx.eval(mlx_z_norm)
    torch_z_norm = _torch_layernorm(z_torch, apb.pair_norm)
    results.append(_compare("attn.pair_norm", mlx_z_norm, torch_z_norm))

    mlx_pair_proj = apb.pair_linear(mlx_z_norm)
    mx.eval(mlx_pair_proj)
    torch_pair_proj = _torch_linear(torch_z_norm, apb.pair_linear)
    results.append(_compare("attn.pair_linear", mlx_pair_proj, torch_pair_proj))

    # Full SDPA inside attention_pair_bias (re-synced inputs)
    print("\n  [re-synced SDPA test — identical q/k/v/bias to both backends]")
    nH, dH = apb.num_heads, apb.head_dim
    qkvg_np = np.array(mlx_qkvg.astype(mx.float32), copy=False).copy()
    qkvg_mx_rs = mx.array(qkvg_np).astype(mx.bfloat16)
    qkvg_torch_rs = torch.from_numpy(qkvg_np).to(torch.bfloat16).to(DEVICE)
    q_sz = nH * dH
    q_mx_rs = qkvg_mx_rs[..., :q_sz].reshape(1, n, nH, dH).transpose(0, 2, 1, 3)
    k_mx_rs = qkvg_mx_rs[..., q_sz:2*q_sz].reshape(1, n, nH, dH).transpose(0, 2, 1, 3)
    v_mx_rs = qkvg_mx_rs[..., 2*q_sz:3*q_sz].reshape(1, n, nH, dH).transpose(0, 2, 1, 3)
    q_bias_mx = apb.query_bias[None, :, None, :].astype(q_mx_rs.dtype)
    q_mx_rs = q_mx_rs + q_bias_mx
    mx.eval(q_mx_rs, k_mx_rs, v_mx_rs)

    q_torch_rs = qkvg_torch_rs[..., :q_sz].reshape(1, n, nH, dH).permute(0, 2, 1, 3)
    k_torch_rs = qkvg_torch_rs[..., q_sz:2*q_sz].reshape(1, n, nH, dH).permute(0, 2, 1, 3)
    v_torch_rs = qkvg_torch_rs[..., 2*q_sz:3*q_sz].reshape(1, n, nH, dH).permute(0, 2, 1, 3)
    qb_np = np.array(apb.query_bias.astype(mx.float32), copy=False).copy()
    q_torch_rs = q_torch_rs + torch.from_numpy(qb_np).to(torch.bfloat16).to(DEVICE)[None, :, None, :]

    bias_np = np.array(mlx_pair_proj.astype(mx.float32), copy=False).copy()
    bias_mx_rs = mx.array(bias_np).astype(mx.bfloat16).transpose(0, 3, 1, 2)[:, :nH, :, :]
    bias_torch_rs = torch.from_numpy(bias_np).to(torch.bfloat16).to(DEVICE).permute(0, 3, 1, 2)[:, :nH, :, :]

    sdpa_mx_rs = mx.fast.scaled_dot_product_attention(
        q_mx_rs, k_mx_rs, v_mx_rs, scale=dH ** -0.5, mask=bias_mx_rs
    )
    mx.eval(sdpa_mx_rs)
    sdpa_torch_rs = F.scaled_dot_product_attention(
        q_torch_rs, k_torch_rs, v_torch_rs, scale=dH ** -0.5,
        attn_mask=bias_torch_rs,
    )
    results.append(_compare("attn.sdpa_resynced", sdpa_mx_rs, sdpa_torch_rs))

    # ── 5. Transition single ─────────────────────────────────────────
    print("\n── transition_single ──")
    trans_s = block.transition_single

    mlx_s_tn = trans_s.norm(s_mx)
    mx.eval(mlx_s_tn)
    torch_s_tn = _torch_layernorm(s_torch, trans_s.norm)
    results.append(_compare("trans_single.norm", mlx_s_tn, torch_s_tn))

    mlx_s_up = trans_s.up(mlx_s_tn)
    mx.eval(mlx_s_up)
    torch_s_up = _torch_linear(torch_s_tn, trans_s.up)
    results.append(_compare("trans_single.up", mlx_s_up, torch_s_up))

    # ── 6. Error accumulation test ───────────────────────────────────
    # Re-synced block: feed IDENTICAL inputs and run full block in both
    # This isolates per-block error (no prior accumulation)
    print("\n── re-synced full block ──")
    print("  (identical bf16 inputs to both backends)")

    # Run full MLX block
    z_mx_rs, s_mx_rs = block(z_mx, s_mx)
    mx.eval(z_mx_rs, s_mx_rs)

    # Run PyTorch block manually (transition_pair + triangle_mult + tri_attn + attn + trans_s)
    # Since full PyTorch replication is complex, we measure by running MLX's full block
    # vs running the same block twice on slightly perturbed inputs to measure sensitivity
    z_perturbed = z_mx + mx.random.normal(z_mx.shape).astype(mx.bfloat16) * 1e-3
    mx.eval(z_perturbed)
    z_mx_perturbed, s_mx_perturbed = block(z_perturbed, s_mx)
    mx.eval(z_mx_perturbed, s_mx_perturbed)

    # This tells us the Lyapunov sensitivity: how much does 1e-3 perturbation grow?
    pert_input = np.abs(np.array((z_perturbed - z_mx).astype(mx.float32), copy=False))
    pert_output = np.abs(np.array((z_mx_perturbed - z_mx_rs).astype(mx.float32), copy=False))
    amp_factor = float(pert_output.mean()) / max(float(pert_input.mean()), 1e-12)
    print(f"  Perturbation test (1e-3 input noise):")
    print(f"    Input perturbation:  mean={pert_input.mean():.4e}  max={pert_input.max():.4e}")
    print(f"    Output perturbation: mean={pert_output.mean():.4e}  max={pert_output.max():.4e}")
    print(f"    Amplification factor: {amp_factor:.2f}x")
    results.append({"name": "lyapunov_1block", "max": float(pert_output.max()),
                     "mean": amp_factor})

    del model
    gc.collect()
    mx.clear_cache()

    return results


# ── Summary ──────────────────────────────────────────────────────────────


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("SUMMARY — sorted by max error")
    print("=" * 70)
    valid = [r for r in results if np.isfinite(r["max"])]
    valid.sort(key=lambda r: r["max"], reverse=True)
    print(f"  {'Operation':<40} {'max':>12} {'mean':>12}")
    print(f"  {'─' * 40} {'─' * 12} {'─' * 12}")
    for r in valid:
        print(f"  {r['name']:<40} {r['max']:>12.4e} {r['mean']:>12.4e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-operation divergence analysis")
    parser.add_argument("--weights-dir", type=Path, required=True)
    args = parser.parse_args()

    results = test_elementary_ops()
    results += test_pairformer_block(args.weights_dir)
    print_summary(results)


if __name__ == "__main__":
    main()
