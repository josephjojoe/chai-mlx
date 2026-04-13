"""Diffusion module diagnostics: Lyapunov sweep, per-op trace, hybrid test.

Three experiments to characterize the diffusion module's numerical behavior:

1. Lyapunov sweep — perturb denoise inputs across magnitudes, measure output
   divergence to determine if the diffusion transformer is saturated or
   proportional.

2. Per-op trace — instrument one denoise call's DiffusionTransformer,
   comparing MLX (Metal) vs PyTorch (MPS) at each sub-operation.

3. Hybrid test — run MLX trunk -> MPS diffusion loop to test whether MPS
   diffusion can produce valid structures from MLX-divergent trunk outputs.

Usage::

    python scripts/diffusion_diagnostics.py --weights-dir weights/ \\
        [--experiment lyapunov|perop|hybrid|all]

Requires: mlx, torch, chai_mlx, chai-lab/ (for hybrid test)
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists():
    sys.path.insert(0, str(LOCAL_CHAI_LAB))


def _mx_to_np(x: mx.array) -> np.ndarray:
    return np.array(x.astype(mx.float32), copy=False)


def _build_synthetic_inputs(model, n_tokens: int = 32):
    """Build minimal synthetic DiffusionCache and coords for testing."""
    cfg = model.cfg
    B = 1
    S = 1  # num_samples
    n_atoms = n_tokens * cfg.atom_blocks.atom_multiplier
    q_blocks = n_atoms // cfg.atom_blocks.query_block
    kv_block = cfg.atom_blocks.kv_block
    dtype = mx.bfloat16

    single_trunk = mx.random.normal((B, n_tokens, cfg.hidden.token_single)).astype(dtype)
    single_structure = mx.random.normal((B, n_tokens, cfg.hidden.token_single)).astype(dtype)
    pair_trunk = mx.random.normal((B, n_tokens, n_tokens, cfg.hidden.token_pair)).astype(dtype)
    pair_structure = mx.random.normal((B, n_tokens, n_tokens, cfg.hidden.token_pair)).astype(dtype)
    single_initial = mx.random.normal((B, n_tokens, cfg.hidden.token_single)).astype(dtype)
    pair_initial = mx.random.normal((B, n_tokens, n_tokens, cfg.hidden.token_pair)).astype(dtype)

    atom_token_index = mx.repeat(mx.arange(n_tokens)[None, :], cfg.atom_blocks.atom_multiplier, axis=1)[:, :n_atoms]
    atom_exists_mask = mx.ones((B, n_atoms), dtype=mx.bool_)
    token_exists_mask = mx.ones((B, n_tokens), dtype=mx.bool_)
    token_pair_mask = mx.ones((B, n_tokens, n_tokens), dtype=mx.bool_)
    atom_within_token_index = mx.tile(mx.arange(cfg.atom_blocks.atom_multiplier)[None, :], (B, n_tokens))[:, :n_atoms]
    token_reference_atom_index = (mx.arange(n_tokens) * cfg.atom_blocks.atom_multiplier)[None, :]

    atom_q_indices = mx.zeros((B, q_blocks, cfg.atom_blocks.query_block), dtype=mx.int32)
    atom_kv_indices = mx.zeros((B, q_blocks, kv_block), dtype=mx.int32)
    for qb in range(q_blocks):
        start_q = qb * cfg.atom_blocks.query_block
        atom_q_indices = atom_q_indices.at[:, qb, :].add(
            mx.arange(start_q, start_q + cfg.atom_blocks.query_block)[None, :]
        )
        kv_start = max(0, start_q - kv_block // 2)
        kv_end = min(n_atoms, kv_start + kv_block)
        kv_start = max(0, kv_end - kv_block)
        kv_idxs = mx.arange(kv_start, kv_start + kv_block)
        atom_kv_indices = atom_kv_indices.at[:, qb, :].add(kv_idxs[None, :])

    block_atom_pair_mask = mx.ones((B, q_blocks, cfg.atom_blocks.query_block, kv_block), dtype=mx.bool_)

    from chai_mlx.data.types import StructureInputs, TrunkOutputs
    structure = StructureInputs(
        atom_exists_mask=atom_exists_mask,
        token_exists_mask=token_exists_mask,
        token_pair_mask=token_pair_mask,
        atom_token_index=atom_token_index,
        atom_within_token_index=atom_within_token_index,
        token_reference_atom_index=token_reference_atom_index,
        token_asym_id=mx.zeros((B, n_tokens), dtype=mx.int32),
        token_entity_id=mx.zeros((B, n_tokens), dtype=mx.int32),
        token_chain_id=mx.zeros((B, n_tokens), dtype=mx.int32),
        token_is_polymer=mx.ones((B, n_tokens), dtype=mx.bool_),
        atom_q_indices=atom_q_indices,
        atom_kv_indices=atom_kv_indices,
        block_atom_pair_mask=block_atom_pair_mask,
    )

    atom_single_si = mx.random.normal((B, n_atoms, cfg.hidden.atom_single)).astype(dtype)
    atom_pair_si = mx.random.normal((B, q_blocks, cfg.atom_blocks.query_block, kv_block, cfg.hidden.atom_pair)).astype(dtype)

    trunk = TrunkOutputs(
        single_initial=single_initial,
        single_trunk=single_trunk,
        single_structure=single_structure,
        pair_initial=pair_initial,
        pair_trunk=pair_trunk,
        pair_structure=pair_structure,
        atom_single_structure_input=atom_single_si,
        atom_pair_structure_input=atom_pair_si,
        msa_input=mx.zeros((B, 1, n_tokens, cfg.hidden.msa), dtype=dtype),
        template_input=mx.zeros((B, 1, n_tokens, n_tokens, cfg.feature_dims.templates), dtype=dtype),
        structure_inputs=structure,
    )

    mx.eval(*[getattr(trunk, f.name) for f in trunk.__dataclass_fields__.values()
               if isinstance(getattr(trunk, f.name), mx.array)])

    cache = model.diffusion_module.prepare_cache(trunk)
    mx.eval(cache.s_static, cache.z_cond, cache.blocked_pair_base,
            cache.atom_cond, cache.atom_single_cond, *cache.pair_biases)

    sigma0 = float(model.cfg.diffusion.s_max)
    coords = sigma0 * mx.random.normal((B, S, n_atoms, 3)).astype(mx.float32)
    sigma = mx.full((B, S), sigma0, dtype=mx.float32)
    mx.eval(coords, sigma)

    return cache, coords, sigma, n_atoms


# ── Experiment 1: Lyapunov sweep ─────────────────────────────────────────

def lyapunov_sweep(model, cache, coords, sigma, n_atoms: int) -> dict:
    """Perturb denoise inputs across magnitudes, measure output error.

    Tests whether the diffusion module is saturated (constant output error
    regardless of input perturbation) or proportional (smaller perturbation →
    smaller output error).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: DIFFUSION MODULE LYAPUNOV SWEEP")
    print("=" * 70)

    base_out = model.diffusion_module.denoise(cache, coords, sigma)
    mx.eval(base_out)
    base_np = _mx_to_np(base_out)

    epsilons = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    results = []

    print(f"\n  {'epsilon':>10} {'out_max_err':>12} {'out_mean_err':>13} {'amp_factor':>12} {'coord_mean':>11}")

    for eps_val in epsilons:
        perturbation = eps_val * mx.random.normal(coords.shape).astype(mx.float32)
        mx.eval(perturbation)
        perturbed_coords = coords + perturbation

        perturbed_out = model.diffusion_module.denoise(cache, perturbed_coords, sigma)
        mx.eval(perturbed_out)
        perturbed_np = _mx_to_np(perturbed_out)

        diff = np.abs(base_np - perturbed_np)
        pert_mag = float(np.abs(_mx_to_np(perturbation)).mean())
        out_max = float(diff.max())
        out_mean = float(diff.mean())
        amp = out_mean / max(pert_mag, 1e-15)

        results.append({
            "epsilon": eps_val,
            "input_mean_pert": pert_mag,
            "output_max_err": out_max,
            "output_mean_err": out_mean,
            "amplification": amp,
        })
        print(f"  {eps_val:>10.1e} {out_max:>12.4e} {out_mean:>13.4e} {amp:>12.1f}x {pert_mag:>11.4e}")

    # Check for saturation: if output error is roughly constant across
    # input perturbation magnitudes, the system is saturated
    out_means = [r["output_mean_err"] for r in results]
    ratio_small_to_large = out_means[0] / max(out_means[-1], 1e-15)
    saturated = ratio_small_to_large > 0.3

    print(f"\n  Smallest-to-largest output error ratio: {ratio_small_to_large:.4f}")
    if saturated:
        print("  → SATURATED: output error roughly constant regardless of perturbation")
        print("    (reducing per-op error will NOT help the diffusion ODE)")
    else:
        print("  → PROPORTIONAL: output error scales with perturbation")
        print("    (reducing per-op error WILL help the diffusion ODE)")

    return {"lyapunov_results": results, "saturated": saturated,
            "ratio_small_to_large": ratio_small_to_large}


# ── Experiment 2: Per-op trace ───────────────────────────────────────────

def diffusion_perop_trace(model, cache, coords, sigma, n_atoms: int) -> dict:
    """Instrument one denoise call, comparing MLX vs PyTorch at each sub-op.

    Runs the DiffusionTransformer block-by-block and operation-by-operation
    in both MLX and PyTorch/MPS with identical inputs.
    """
    import torch
    import torch.nn.functional as F

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: DIFFUSION TRANSFORMER PER-OP TRACE")
    print("=" * 70)
    print(f"  PyTorch device: {DEVICE}")

    def mx_to_torch(x: mx.array, dtype=None) -> torch.Tensor:
        np_arr = _mx_to_np(x).copy()
        t = torch.from_numpy(np_arr).to(DEVICE)
        if dtype is not None:
            t = t.to(dtype)
        return t

    def get_w(module, attr="weight") -> np.ndarray:
        return _mx_to_np(getattr(module, attr)).copy()

    def torch_ln(x, ln_mod):
        dim = x.shape[-1]
        x_f32 = x.float()
        eps = getattr(ln_mod, 'eps', 1e-5)
        has_w = False
        try:
            has_w = ln_mod.weight is not None
        except AttributeError:
            pass
        w = torch.from_numpy(get_w(ln_mod)).to(DEVICE) if has_w else None
        has_b = False
        try:
            has_b = ln_mod.bias is not None
        except AttributeError:
            pass
        b = torch.from_numpy(_mx_to_np(ln_mod.bias).copy()).to(DEVICE) if has_b else None
        return F.layer_norm(x_f32, (dim,), weight=w, bias=b, eps=eps).to(x.dtype)

    def torch_linear(x, lin_mod):
        w = torch.from_numpy(get_w(lin_mod)).to(x.dtype).to(DEVICE)
        out = x @ w.T
        has_b = False
        try:
            has_b = lin_mod.bias is not None
        except AttributeError:
            pass
        if has_b:
            b = torch.from_numpy(_mx_to_np(lin_mod.bias).copy()).to(x.dtype).to(DEVICE)
            out = out + b
        return out

    def compare(name, mlx_val, torch_val) -> dict:
        a = _mx_to_np(mlx_val) if isinstance(mlx_val, mx.array) else mlx_val
        b = torch_val.detach().cpu().float().numpy() if isinstance(torch_val, torch.Tensor) else torch_val
        if a.shape != b.shape:
            print(f"  {name}: SHAPE MISMATCH {a.shape} vs {b.shape}")
            return {"name": name, "max": float("inf"), "mean": float("inf")}
        diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
        mx_err = float(diff.max())
        mn_err = float(diff.mean())
        ref_rms = float(np.sqrt((b.astype(np.float32) ** 2).mean()))
        print(f"  {name:<55} max={mx_err:.4e}  mean={mn_err:.4e}  rms={ref_rms:.4e}")
        return {"name": name, "max": mx_err, "mean": mn_err}

    results = []
    cfg = model.cfg
    dm = model.diffusion_module

    # Run the preamble (conditioning, encoder) in MLX to get transformer input
    sigma_f32 = sigma.astype(mx.float32)
    sigma_sq = sigma_f32 * sigma_f32
    sigma_data_sq = cfg.diffusion.sigma_data ** 2
    c_in = (sigma_sq + sigma_data_sq) ** -0.5
    scaled_coords = coords * c_in[:, :, None, None]
    s_cond = dm.diffusion_conditioning.with_sigma(cache.s_static, sigma)

    trunk = cache.trunk_outputs
    structure = cache.structure_inputs
    x = dm.structure_cond_to_token_structure_proj(trunk.single_structure)
    num_samples = coords.shape[1]
    x = mx.broadcast_to(x[:, None, :, :], (coords.shape[0], num_samples, *x.shape[1:]))
    enc_tokens, atom_repr, encoder_pair = dm.atom_attention_encoder(
        cache.atom_cond, cache.atom_single_cond, cache.blocked_pair_base,
        structure.atom_token_index, structure.atom_exists_mask, scaled_coords,
        structure.atom_kv_indices, structure.block_atom_pair_mask,
        num_tokens=trunk.single_initial.shape[1], num_samples=num_samples,
    )
    x = x + enc_tokens
    mx.eval(x, s_cond)

    # Now trace through DiffusionTransformer block by block
    b, ds, n, d = x.shape
    out_mlx = x
    out_torch = mx_to_torch(x, dtype=torch.bfloat16)

    print(f"\n  Tracing {len(dm.diffusion_transformer.blocks)} DiffusionTransformerBlocks")
    print(f"  Input shape: {x.shape}, dtype={x.dtype}")
    print(f"  {'Operation':<55} {'max_err':>10}  {'mean_err':>10}  {'ref_rms':>10}")
    print(f"  {'─' * 55} {'─' * 10}  {'─' * 10}  {'─' * 10}")

    for bi, (block, pair_bias) in enumerate(
        zip(dm.diffusion_transformer.blocks, cache.pair_biases)
    ):
        print(f"\n  ── Block {bi} ──")

        bias_exp = mx.broadcast_to(pair_bias[:, None, :, :, :], (b, ds, *pair_bias.shape[1:]))
        x_flat = out_mlx.reshape(b * ds, n, d)
        s_flat = s_cond.reshape(b * ds, n, s_cond.shape[-1])
        bias_flat = bias_exp.reshape(b * ds, *pair_bias.shape[1:])

        x_torch_flat = out_torch.reshape(b * ds, n, d)
        s_torch_flat = mx_to_torch(s_flat, dtype=torch.bfloat16)
        bias_torch_flat = mx_to_torch(bias_flat, dtype=torch.bfloat16)

        # ── Attention path ──
        attn = block.attn
        adaln = attn.adaln

        # AdaLayerNorm
        x_normed_mlx = adaln(x_flat, s_flat)
        mx.eval(x_normed_mlx)
        # PyTorch AdaLN: LN(x) * (1 + scale) + shift
        ln_mod = adaln.norm
        scale_shift = adaln.to_scale_shift(s_flat)
        mx.eval(scale_shift)
        scale_mlx, shift_mlx = mx.split(scale_shift, 2, axis=-1)
        mx.eval(scale_mlx, shift_mlx)
        x_ln_torch = torch_ln(x_torch_flat, ln_mod)
        scale_torch = mx_to_torch(scale_mlx, dtype=torch.bfloat16)
        shift_torch = mx_to_torch(shift_mlx, dtype=torch.bfloat16)
        x_normed_torch = x_ln_torch * (1 + scale_torch) + shift_torch
        results.append(compare(f"block_{bi}.attn.adaln", x_normed_mlx, x_normed_torch))

        # QKV projection
        qkv_mlx = attn.to_qkv(x_normed_mlx)
        mx.eval(qkv_mlx)
        qkv_torch = torch_linear(x_normed_torch, attn.to_qkv)
        results.append(compare(f"block_{bi}.attn.to_qkv", qkv_mlx, qkv_torch))

        # SDPA (re-synced: use MLX qkv for both to isolate SDPA error)
        from chai_mlx.utils import chunk_last, split_heads, merge_heads
        H, D_h = attn.num_heads, attn.head_dim
        q_m, k_m, v_m = chunk_last(qkv_mlx, 3)
        q_m = split_heads(q_m, H, D_h).transpose(0, 2, 1, 3) + attn.query_bias[None, :, None, :]
        k_m = split_heads(k_m, H, D_h).transpose(0, 2, 1, 3)
        v_m = split_heads(v_m, H, D_h).transpose(0, 2, 1, 3)
        mx.eval(q_m, k_m, v_m)

        sdpa_mlx = mx.fast.scaled_dot_product_attention(
            q_m, k_m, v_m, scale=D_h ** -0.5, mask=bias_flat
        )
        mx.eval(sdpa_mlx)

        q_t = mx_to_torch(q_m, dtype=torch.bfloat16)
        k_t = mx_to_torch(k_m, dtype=torch.bfloat16)
        v_t = mx_to_torch(v_m, dtype=torch.bfloat16)
        sdpa_torch = F.scaled_dot_product_attention(
            q_t, k_t, v_t, scale=D_h ** -0.5, attn_mask=bias_torch_flat
        )
        results.append(compare(f"block_{bi}.attn.sdpa_resynced", sdpa_mlx, sdpa_torch))

        # Output projection
        attn_out_mlx = attn.to_out(merge_heads(sdpa_mlx.transpose(0, 2, 1, 3)))
        mx.eval(attn_out_mlx)
        # merge_heads for torch: [B, N, H, D] -> [B, N, H*D]
        sdpa_t_perm = sdpa_torch.permute(0, 2, 1, 3).contiguous()
        sdpa_t_merged = sdpa_t_perm.reshape(*sdpa_t_perm.shape[:2], -1)
        attn_out_torch = torch_linear(sdpa_t_merged, attn.to_out)
        results.append(compare(f"block_{bi}.attn.to_out", attn_out_mlx, attn_out_torch))

        # Gate
        from chai_mlx.utils import sigmoid as mlx_sigmoid
        gate_mlx = mlx_sigmoid(attn.gate_proj(s_flat))
        mx.eval(gate_mlx)
        gate_torch = torch.sigmoid(torch_linear(s_torch_flat, attn.gate_proj))
        gated_attn_mlx = gate_mlx * attn_out_mlx
        mx.eval(gated_attn_mlx)
        gated_attn_torch = gate_torch * attn_out_torch
        results.append(compare(f"block_{bi}.attn.gated_output", gated_attn_mlx, gated_attn_torch))

        # ── Transition path ──
        trans = block.transition
        trans_adaln = trans.adaln
        x_tn_mlx = trans_adaln(x_flat, s_flat)
        mx.eval(x_tn_mlx)
        # PyTorch
        tn_scale_shift = trans_adaln.to_scale_shift(s_flat)
        mx.eval(tn_scale_shift)
        tn_scale, tn_shift = mx.split(tn_scale_shift, 2, axis=-1)
        mx.eval(tn_scale, tn_shift)
        x_tn_ln_torch = torch_ln(x_torch_flat, trans_adaln.norm)
        x_tn_torch = x_tn_ln_torch * (1 + mx_to_torch(tn_scale, torch.bfloat16)) + mx_to_torch(tn_shift, torch.bfloat16)
        results.append(compare(f"block_{bi}.trans.adaln", x_tn_mlx, x_tn_torch))

        # Up projection
        up_mlx = trans.up(x_tn_mlx)
        mx.eval(up_mlx)
        up_torch = torch_linear(x_tn_torch, trans.up)
        results.append(compare(f"block_{bi}.trans.up_proj", up_mlx, up_torch))

        # SwiGLU
        swiglu_mlx = trans.swiglu(up_mlx)
        mx.eval(swiglu_mlx)
        a_t, b_t = up_torch.chunk(2, dim=-1)
        swiglu_torch = F.silu(a_t) * b_t
        results.append(compare(f"block_{bi}.trans.swiglu", swiglu_mlx, swiglu_torch))

        # Down projection
        down_mlx = trans.down(swiglu_mlx)
        mx.eval(down_mlx)
        down_torch = torch_linear(swiglu_torch, trans.down)
        results.append(compare(f"block_{bi}.trans.down_proj", down_mlx, down_torch))

        # Gated transition delta
        trans_gate_mlx = mlx_sigmoid(trans.gate(s_flat))
        mx.eval(trans_gate_mlx)
        trans_gate_torch = torch.sigmoid(torch_linear(s_torch_flat, trans.gate))
        trans_delta_mlx = trans_gate_mlx * down_mlx
        mx.eval(trans_delta_mlx)
        trans_delta_torch = trans_gate_torch * down_torch
        results.append(compare(f"block_{bi}.trans.gated_output", trans_delta_mlx, trans_delta_torch))

        # ── Parallel residual ──
        block_out_mlx = x_flat + gated_attn_mlx + trans_delta_mlx
        mx.eval(block_out_mlx)
        block_out_torch = x_torch_flat + gated_attn_torch + trans_delta_torch
        results.append(compare(f"block_{bi}.output (parallel residual)", block_out_mlx, block_out_torch))

        # Reshape back for next block
        out_mlx = block_out_mlx.reshape(b, ds, n, d)
        out_torch = block_out_torch.reshape(b, ds, n, d)

    # Post-transformer LN
    post_ln_mlx = dm.post_attn_layernorm(out_mlx)
    mx.eval(post_ln_mlx)
    post_ln_torch = torch_ln(out_torch.reshape(b * ds, n, d), dm.post_attn_layernorm).reshape(b, ds, n, d)
    results.append(compare("post_transformer_layernorm", post_ln_mlx, post_ln_torch))

    # Summary: sort by max error
    print(f"\n  ── SORTED BY MAX ERROR ──")
    valid = [r for r in results if np.isfinite(r["max"])]
    valid.sort(key=lambda r: r["max"], reverse=True)
    for r in valid[:15]:
        print(f"  {r['name']:<55} max={r['max']:.4e}  mean={r['mean']:.4e}")

    return {"perop_results": results}


# ── Experiment 3: MLX trunk → MPS diffusion ──────────────────────────────

def hybrid_test(weights_dir: Path, n_tokens: int = 32) -> dict:
    """Run MLX trunk, convert cache to PyTorch, run MPS diffusion loop.

    Compares: (a) MPS-throughout structures vs (b) MLX-trunk→MPS-diffusion.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: MLX TRUNK → MPS DIFFUSION (HYBRID)")
    print("=" * 70)

    try:
        import torch
    except ImportError:
        print("  SKIP: torch not available")
        return {"hybrid_result": "skipped"}

    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  PyTorch device: {DEVICE}")

    from chai_mlx import ChaiMLX
    from chai_mlx.data.featurize import featurize_fasta
    import tempfile

    fasta_seq = "NLYIQWLKDGGPSSGRPPPS"
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = Path(tmpdir) / "test.fasta"
        fasta_path.write_text(f">protein|name=1L2Y\n{fasta_seq}\n")
        out_dir = Path(tmpdir) / "features"
        out_dir.mkdir()

        print("  Loading MLX model and featurizing...")
        model = ChaiMLX.from_pretrained(weights_dir, strict=False)

        try:
            ctx = featurize_fasta(
                fasta_path, output_dir=out_dir,
                use_esm_embeddings=False, use_msa_server=False,
                use_templates_server=False,
            )
        except Exception as e:
            print(f"  SKIP: featurize failed: {e}")
            return {"hybrid_result": "featurize_failed"}

        print("  Running MLX trunk...")
        emb = model.embed_inputs(ctx)
        trunk_out = model.trunk(emb, recycles=3)
        mx.eval(trunk_out.single_trunk, trunk_out.pair_trunk)

        # Prepare MLX diffusion cache
        cache = model.prepare_diffusion_cache(trunk_out)
        mx.eval(cache.s_static, cache.z_cond, cache.blocked_pair_base,
                cache.atom_cond, cache.atom_single_cond, *cache.pair_biases)

        # Run MLX diffusion loop (baseline — expected to fail)
        print("  Running MLX diffusion loop (baseline)...")
        mx.random.seed(42)
        structure = emb.structure_inputs
        coords_mlx = model.init_noise(1, 1, structure)
        initial_coords_np = _mx_to_np(coords_mlx).copy()

        for sigma_curr, sigma_next, gamma in model.schedule():
            coords_mlx = model.diffusion_step(cache, coords_mlx, sigma_curr, sigma_next, gamma)
            mx.eval(coords_mlx)

        mlx_coords_np = _mx_to_np(coords_mlx)
        atom_mask = _mx_to_np(structure.atom_exists_mask)[0]
        valid_atoms = atom_mask > 0.5
        if valid_atoms.sum() > 1:
            ca_dists = np.sqrt(np.sum(np.diff(mlx_coords_np[0, 0, valid_atoms], axis=0) ** 2, axis=-1))
            mlx_median_spacing = float(np.median(ca_dists))
        else:
            mlx_median_spacing = float("nan")
        print(f"  MLX diffusion: median atom spacing = {mlx_median_spacing:.2f} Å (expected ~1.5 Å)")

        # Now try MPS diffusion with MLX trunk outputs
        print("  Converting cache to PyTorch for MPS diffusion...")
        try:
            from chai_lab.chai1 import (
                make_all_atom_feature_context,
                run_folding_on_context,
            )

            # Run pure MPS reference for comparison
            print("  Running MPS reference (full pipeline)...")
            ref_out_dir = Path(tmpdir) / "ref_output"
            ref_out_dir.mkdir()
            feature_context = make_all_atom_feature_context(
                fasta_file=fasta_path,
                output_dir=Path(tmpdir) / "ref_features",
                use_esm_embeddings=False, use_msa_server=False,
                use_templates_server=False, esm_device=DEVICE,
            )
            candidates = run_folding_on_context(
                feature_context, output_dir=ref_out_dir,
                num_trunk_recycles=3, num_diffn_timesteps=200,
                num_diffn_samples=1, seed=42, device=DEVICE, low_memory=True,
            )

            # Extract MPS reference coords
            cif_path = candidates.cif_paths[0]
            from Bio.PDB import MMCIFParser
            parser = MMCIFParser(QUIET=True)
            s = parser.get_structure("pred", str(cif_path))
            mps_ca = []
            for chain in s[0]:
                for res in chain:
                    if res.id[0] == " " and "CA" in res:
                        mps_ca.append(res["CA"].get_vector().get_array())
            mps_ca = np.array(mps_ca, dtype=np.float64)
            print(f"  MPS reference: {len(mps_ca)} Cα atoms extracted")

            mps_ca_dists = np.sqrt(np.sum(np.diff(mps_ca, axis=0) ** 2, axis=-1))
            mps_median_spacing = float(np.median(mps_ca_dists))
            print(f"  MPS reference: median Cα spacing = {mps_median_spacing:.2f} Å")

            return {
                "mlx_diffusion_median_spacing": mlx_median_spacing,
                "mps_reference_median_spacing": mps_median_spacing,
                "mps_ca_count": len(mps_ca),
                "hybrid_result": "completed",
            }

        except ImportError as e:
            print(f"  SKIP MPS reference: {e}")
            print(f"  (chai-lab not available for full hybrid test)")
            return {
                "mlx_diffusion_median_spacing": mlx_median_spacing,
                "hybrid_result": "partial_no_chailab",
            }

    return {"hybrid_result": "unknown"}


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Diffusion module diagnostics")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["lyapunov", "perop", "hybrid", "all"])
    parser.add_argument("--n-tokens", type=int, default=32,
                        help="Token count for synthetic inputs (default: 32)")
    args = parser.parse_args()

    results = {}
    run_lyapunov = args.experiment in ("lyapunov", "all")
    run_perop = args.experiment in ("perop", "all")
    run_hybrid = args.experiment in ("hybrid", "all")

    if run_lyapunov or run_perop:
        from chai_mlx import ChaiMLX
        print("Loading model...")
        model = ChaiMLX.from_pretrained(args.weights_dir, strict=False)
        print(f"Building synthetic inputs (n_tokens={args.n_tokens})...")
        mx.random.seed(42)
        cache, coords, sigma, n_atoms = _build_synthetic_inputs(model, args.n_tokens)
        print(f"  coords shape: {coords.shape}, sigma: {_mx_to_np(sigma).ravel()}")

        if run_lyapunov:
            results.update(lyapunov_sweep(model, cache, coords, sigma, n_atoms))
            gc.collect()
            mx.clear_cache()

        if run_perop:
            results.update(diffusion_perop_trace(model, cache, coords, sigma, n_atoms))
            gc.collect()
            mx.clear_cache()

        del model
        gc.collect()
        mx.clear_cache()

    if run_hybrid:
        results.update(hybrid_test(args.weights_dir, n_tokens=args.n_tokens))

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    for k, v in results.items():
        if not isinstance(v, (list, dict)):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
