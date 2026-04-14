"""Deep checkpoint trace for the diffusion denoiser.

This script replays one denoise call from reference trunk outputs in both
TorchScript and MLX, then compares aligned intermediate tensors in sequence.

It is intentionally focused on the diffusion encoder / decoder path where
layout and indexing bugs tend to hide:

- cache preparation (`blocked_pair_base`, `atom_cond`, `atom_single_cond`)
- encoder pair update + local atom transformer internals
- diffusion transformer block outputs
- decoder token-to-atom flow, local atom transformer internals, and `to_pos_updates`

Usage::

    python3 scripts/deep_denoise_trace.py \
        --weights-dir weights \
        --input-npz /tmp/chai_mlx_input.npz \
        --reference-npz /tmp/chai_mlx_reference.npz

Important caveat:
This is a manual replay harness, not the TorchScript top-level wrapper itself.
It is useful for localizing a known mismatch, but if it disagrees with direct
wrapper-level parity then the wrapper-level comparison should be treated as the
source of truth until the harness is repaired.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists():
    sys.path.insert(0, str(LOCAL_CHAI_LAB))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from chai_lab.chai1 import _component_moved_to  # type: ignore[import-not-found]

from chai_mlx import ChaiMLX
from chai_mlx.data.types import StructureInputs
from chai_mlx.utils import (
    chunk_last,
    gather_blocked_atom_values,
    gather_blocked_pair_values,
    gather_tokens_to_atoms,
    make_additive_mask,
    merge_heads,
    resolve_dtype,
    segment_mean,
    split_heads,
)
from layer_parity import _npz_dict, load_feature_context
from stage_isolation_parity import reconstruct_trunk_outputs


def _mx_np(x: mx.array) -> np.ndarray:
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    mx.eval(x)
    return np.array(x, copy=False)


def _pt_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _record_mx(
    store: OrderedDict[str, np.ndarray],
    name: str,
    value: mx.array,
) -> None:
    store[name] = _mx_np(value)


def _record_pt(
    store: OrderedDict[str, np.ndarray],
    name: str,
    value: torch.Tensor,
) -> None:
    store[name] = _pt_np(value)


def _to_torch(x, device: torch.device, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(x, mx.array):
        arr = _mx_np(x)
    else:
        arr = np.asarray(x)
    t = torch.from_numpy(arr).to(device)
    return t.to(dtype) if dtype is not None else t


def _torch_layer_norm(
    x: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(x.float(), (x.shape[-1],), weight=weight, bias=bias, eps=eps).to(x.dtype)


def _torch_linear_module(x: torch.Tensor, mod) -> torch.Tensor:
    weight = mod.weight.float()
    out = x.float() @ weight.T
    bias = getattr(mod, "bias", None)
    if bias is not None:
        out = out + bias.float()
    return out.to(x.dtype)


def _torch_adaln(x: torch.Tensor, cond: torch.Tensor, lin_s_merged) -> torch.Tensor:
    x_norm = _torch_layer_norm(x, eps=0.1)
    scale_shift = _torch_linear_module(cond, lin_s_merged)
    scale, shift = torch.chunk(scale_shift, 2, dim=-1)
    return x_norm * (1.0 + scale) + shift


def _torch_conditioned_transition_delta(x: torch.Tensor, cond: torch.Tensor, block) -> torch.Tensor:
    y = _torch_adaln(x, cond, block.ada_ln.lin_s_merged)
    up = _torch_linear_module(y, block.linear_a_nobias_double)
    a, b = torch.chunk(up, 2, dim=-1)
    swiglu = F.silu(a.float()).to(x.dtype) * b
    down = _torch_linear_module(swiglu, block.linear_b_nobias)
    gate = torch.sigmoid(_torch_linear_module(cond, block.linear_s_biasinit_m2))
    return gate * down


def _torch_segment_mean(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    num_segments: int,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    one_hot = F.one_hot(segment_ids, num_classes=num_segments).to(values.dtype)
    if mask is not None:
        one_hot = one_hot * mask[..., None].to(values.dtype)
    sums = torch.einsum("ban,bad->bnd", one_hot, values)
    counts = torch.einsum(
        "ban,bad->bnd",
        one_hot,
        torch.ones((*values.shape[:-1], 1), dtype=values.dtype, device=values.device),
    )
    return sums / torch.clamp(counts, min=1.0)


def _torch_gather_tokens_to_atoms(token_values: torch.Tensor, atom_token_index: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(token_values.shape[0], device=token_values.device)[:, None]
    return token_values[batch, atom_token_index]


def _torch_gather_blocked_atom_values(atom_values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(atom_values.shape[0], device=atom_values.device)[:, None, None]
    return atom_values[batch, indices]


def _local_attn_modules(container, idx: int):
    local_attn = getattr(container.local_attentions, str(idx))
    transition = getattr(container.transitions, str(idx))
    return local_attn, transition


def _trace_torch_local_atom_transformer(
    store: OrderedDict[str, np.ndarray],
    prefix: str,
    transformer,
    x: torch.Tensor,
    cond: torch.Tensor,
    pair: torch.Tensor,
    kv_idx: torch.Tensor,
    block_mask: torch.Tensor,
    *,
    atom_mask: torch.Tensor | None,
) -> torch.Tensor:
    pair_norm_mod = getattr(transformer.blocked_pairs2blocked_bias, "0")
    pair_bias_mod = getattr(transformer.blocked_pairs2blocked_bias, "1")
    num_heads = int(pair_bias_mod.weight.shape[1])
    head_dim = int(getattr(transformer.local_attentions, "0").q_bias.shape[-1])
    num_blocks = int(pair.shape[1])
    query_block = int(pair.shape[2])
    kv_block = int(pair.shape[3])
    pair_norm = pair_norm_mod(pair.float()).to(x.dtype)
    weight = pair_bias_mod.weight.float()

    _record_pt(store, f"{prefix}.pair_norm", pair_norm)

    pair_bias_blocks = []
    for block_idx in range(weight.shape[0]):
        bias = torch.einsum("blqkc,hc->blqkh", pair_norm.float(), weight[block_idx]).to(x.dtype)
        pair_bias_blocks.append(bias)
    pair_bias_all = torch.cat(pair_bias_blocks, dim=-1)
    _record_pt(store, f"{prefix}.pair_bias_all", pair_bias_all)

    out = x
    for block_idx in range(weight.shape[0]):
        local_attn, transition = _local_attn_modules(transformer, block_idx)

        if atom_mask is not None:
            out = out * atom_mask[..., None].to(out.dtype)
        _record_pt(store, f"{prefix}.block_{block_idx}.x_masked", out)

        bias_slice = pair_bias_all[..., block_idx * num_heads : (block_idx + 1) * num_heads]
        _record_pt(store, f"{prefix}.block_{block_idx}.bias_slice", bias_slice)

        x_norm = _torch_adaln(out, cond, local_attn.single_layer_norm.lin_s_merged)
        _record_pt(store, f"{prefix}.block_{block_idx}.attn.adaln", x_norm)

        q_raw, k_raw, v_raw = torch.unbind(local_attn.to_qkv(x_norm.float()).to(out.dtype), dim=0)

        batch = out.shape[0]
        seq = out.shape[1]
        q_all = q_raw.reshape(batch, num_heads, seq, head_dim).permute(0, 2, 1, 3).contiguous()
        k_all = k_raw.reshape(batch, num_heads, seq, head_dim).permute(0, 2, 1, 3).contiguous()
        v_all = v_raw.reshape(batch, num_heads, seq, head_dim).permute(0, 2, 1, 3).contiguous()
        _record_pt(store, f"{prefix}.block_{block_idx}.attn.q_all", q_all)
        _record_pt(store, f"{prefix}.block_{block_idx}.attn.k_all", k_all)
        _record_pt(store, f"{prefix}.block_{block_idx}.attn.v_all", v_all)

        q = q_all.reshape(batch, num_blocks, query_block, num_heads, head_dim).permute(0, 1, 3, 2, 4).contiguous()
        q_bias = local_attn.q_bias.float().to(out.dtype)
        q = q + q_bias[None, None, :, None, :]
        k = _torch_gather_blocked_atom_values(k_all, kv_idx).permute(0, 1, 3, 2, 4).contiguous()
        v = _torch_gather_blocked_atom_values(v_all, kv_idx).permute(0, 1, 3, 2, 4).contiguous()
        _record_pt(store, f"{prefix}.block_{block_idx}.attn.q_blocked", q)
        _record_pt(store, f"{prefix}.block_{block_idx}.attn.k_gathered", k)
        _record_pt(store, f"{prefix}.block_{block_idx}.attn.v_gathered", v)

        additive_bias = bias_slice.permute(0, 1, 4, 2, 3).contiguous()
        mask_add = torch.where(
            block_mask.bool(),
            torch.zeros_like(block_mask, dtype=additive_bias.dtype),
            torch.full_like(block_mask, -10000.0, dtype=additive_bias.dtype),
        )
        additive_bias = additive_bias + mask_add[:, :, None, :, :]
        _record_pt(store, f"{prefix}.block_{block_idx}.attn.bias_masked", additive_bias)

        q_flat = q.reshape(batch * num_blocks, num_heads, query_block, head_dim)
        k_flat = k.reshape(batch * num_blocks, num_heads, kv_block, head_dim)
        v_flat = v.reshape(batch * num_blocks, num_heads, kv_block, head_dim)
        bias_flat = additive_bias.reshape(batch * num_blocks, num_heads, query_block, kv_block)

        sdpa = F.scaled_dot_product_attention(
            q_flat.float(),
            k_flat.float(),
            v_flat.float(),
            attn_mask=bias_flat.float(),
            scale=head_dim ** -0.5,
        ).to(out.dtype)
        sdpa = sdpa.reshape(batch, num_blocks, num_heads, query_block, head_dim).contiguous()
        _record_pt(store, f"{prefix}.block_{block_idx}.attn.sdpa", sdpa)

        merged = sdpa.permute(0, 1, 3, 2, 4).reshape(batch, num_blocks * query_block, num_heads * head_dim)
        _record_pt(store, f"{prefix}.block_{block_idx}.attn.merged", merged)

        gate = torch.sigmoid(_torch_linear_module(cond, local_attn.out_proj))
        _record_pt(store, f"{prefix}.block_{block_idx}.attn.gate", gate)
        attn_delta = merged * gate
        _record_pt(store, f"{prefix}.block_{block_idx}.attn.delta", attn_delta)

        trans_delta = _torch_conditioned_transition_delta(out, cond, transition)
        _record_pt(store, f"{prefix}.block_{block_idx}.trans.delta", trans_delta)

        out = out + attn_delta + trans_delta
        _record_pt(store, f"{prefix}.block_{block_idx}.output", out)
    return out


def _trace_torch_denoise(
    jit,
    store: OrderedDict[str, np.ndarray],
    trunk,
    structure,
    coords: torch.Tensor,
    sigma: torch.Tensor,
) -> None:
    dm = jit
    enc = dm.atom_attention_encoder
    dec = dm.atom_attention_decoder

    atom_token_index = structure["atom_token_index"]
    atom_exists_mask = structure["atom_exists_mask"]
    token_pair_mask = structure["token_pair_mask"]
    atom_q_indices = structure["atom_q_indices"]
    atom_kv_indices = structure["atom_kv_indices"]
    block_atom_pair_mask = structure["block_atom_pair_mask"]

    pair_cat = torch.cat([trunk["pair_trunk"], trunk["pair_structure"]], dim=-1)
    z = dm.diffusion_conditioning.token_pair_proj(pair_cat.float())
    z = z + dm.diffusion_conditioning.pair_trans1(z)
    z = z + dm.diffusion_conditioning.pair_trans2(z)
    z_cond = dm.diffusion_conditioning.pair_ln(z)
    _record_pt(store, "cache.z_cond", z_cond)

    single_cat = torch.cat([trunk["single_structure"], trunk["single_trunk"]], dim=-1)
    s_proj = dm.diffusion_conditioning.token_in_proj(single_cat.float())
    _record_pt(store, "cache.s_static", s_proj)

    pair_biases = []
    for i in range(16):
        block = getattr(dm.diffusion_transformer.blocks, str(i))
        z_pair_norm = _torch_layer_norm(
            z_cond,
            weight=block.pair_layer_norm.weight.float(),
            bias=block.pair_layer_norm.bias.float(),
            eps=1e-5,
        )
        pair_bias = _torch_linear_module(z_pair_norm, block.pair_linear).permute(0, 3, 1, 2).contiguous()
        pair_bias = pair_bias + torch.where(
            token_pair_mask[:, None].bool(),
            torch.zeros_like(pair_bias),
            torch.full_like(pair_bias, -10000.0),
        )
        pair_biases.append(pair_bias)
        _record_pt(store, f"cache.pair_biases.{i}", pair_bias)

    token_atom_pair = enc.token_pair_to_atom_pair(z_cond.float())
    batch = torch.arange(atom_token_index.shape[0], device=atom_token_index.device)
    q_token_idx = atom_token_index[batch[:, None, None], atom_q_indices]
    kv_token_idx = atom_token_index[batch[:, None, None], atom_kv_indices]
    blocked_pair_base = token_atom_pair[
        batch[:, None, None, None],
        q_token_idx[:, :, :, None],
        kv_token_idx[:, :, None, :],
    ]
    blocked_pair_base = blocked_pair_base + trunk["atom_pair_structure_input"]
    _record_pt(store, "cache.blocked_pair_base", blocked_pair_base)

    atom_cond = _torch_linear_module(trunk["atom_single_structure_input"], enc.to_atom_cond)
    _record_pt(store, "cache.atom_cond", atom_cond)

    token_proj = enc.token_to_atom_single(trunk["single_trunk"].float())
    token_to_atom = _torch_gather_tokens_to_atoms(token_proj, atom_token_index)
    atom_single_cond = _torch_layer_norm(atom_cond + token_to_atom, eps=1e-5)
    _record_pt(store, "cache.atom_single_cond", atom_single_cond)

    sigma_sq = sigma.float() * sigma.float()
    sigma_data_sq = 16.0 ** 2
    c_in = (sigma_sq + sigma_data_sq).pow(-0.5)
    c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
    c_out = sigma.float() * 16.0 / torch.sqrt(sigma_sq + sigma_data_sq)
    _record_pt(store, "denoise.c_in", c_in)
    _record_pt(store, "denoise.c_skip", c_skip)
    _record_pt(store, "denoise.c_out", c_out)

    scaled_coords = coords * c_in[:, :, None, None]
    _record_pt(store, "denoise.scaled_coords", scaled_coords)

    c_noise = torch.log(torch.clamp(sigma.float(), min=torch.finfo(torch.float32).tiny)) * 0.25
    sigma_embed = torch.cos(
        (c_noise[..., None] * dm.diffusion_conditioning.fourier_embedding.weights.float()
         + dm.diffusion_conditioning.fourier_embedding.bias.float())
        * (2.0 * math.pi)
    )
    fourier_proj_norm = getattr(dm.diffusion_conditioning.fourier_proj, "0")
    sigma_embed = _torch_layer_norm(
        sigma_embed,
        weight=fourier_proj_norm.weight.float(),
        bias=fourier_proj_norm.bias.float(),
        eps=1e-5,
    )
    sigma_embed = _torch_linear_module(sigma_embed, getattr(dm.diffusion_conditioning.fourier_proj, "1"))
    s_cond = s_proj[:, None, :, :] + sigma_embed[:, :, None, :]
    s_cond = s_cond + dm.diffusion_conditioning.single_trans1(s_cond)
    s_cond = s_cond + dm.diffusion_conditioning.single_trans2(s_cond)
    s_cond = dm.diffusion_conditioning.single_ln(s_cond)
    _record_pt(store, "denoise.s_cond", s_cond)

    structure_proj = _torch_linear_module(s_cond, dm.structure_cond_to_token_structure_proj)
    _record_pt(store, "denoise.structure_proj", structure_proj)

    b = atom_cond.shape[0]
    ds = coords.shape[1]
    n_atoms = atom_cond.shape[1]
    num_blocks = int(n_atoms // 32)
    num_tokens = int(trunk["single_initial"].shape[1])

    atom_init = atom_cond[:, None, :, :].expand(-1, ds, -1, -1).reshape(b * ds, n_atoms, -1).contiguous()
    _record_pt(store, "encoder.atom_init", atom_init)
    pos_embed = _torch_linear_module(scaled_coords.reshape(b * ds, n_atoms, 3), enc.prev_pos_embed)
    _record_pt(store, "encoder.pos_embed", pos_embed)
    atom_single = atom_init + pos_embed
    _record_pt(store, "encoder.atom_single", atom_single)

    cond = atom_single_cond[:, None, :, :].expand(-1, ds, -1, -1).reshape(b * ds, n_atoms, -1).contiguous()
    _record_pt(store, "encoder.cond", cond)

    kv_idx_flat = atom_kv_indices[:, None, :, :].expand(-1, ds, -1, -1).reshape(b * ds, *atom_kv_indices.shape[1:]).contiguous()
    block_mask_flat = block_atom_pair_mask[:, None, :, :, :].expand(-1, ds, -1, -1, -1).reshape(
        b * ds, *block_atom_pair_mask.shape[1:]
    ).contiguous()
    blocked_pair = blocked_pair_base[:, None, :, :, :, :].expand(-1, ds, -1, -1, -1, -1).reshape(
        b * ds, *blocked_pair_base.shape[1:]
    ).contiguous()
    cond_q = cond.reshape(b * ds, num_blocks, 32, cond.shape[-1]).contiguous()
    cond_kv = _torch_gather_blocked_atom_values(cond, kv_idx_flat)
    _record_pt(store, "encoder.pair_update.cond_q", cond_q)
    _record_pt(store, "encoder.pair_update.cond_kv", cond_kv)
    _record_pt(store, "encoder.pair_update.blocked_pair_base", blocked_pair)

    proj_h = enc.pair_update_block.atom_single_to_atom_pair_proj_h(cond_q.float()).to(cond.dtype)
    proj_w = enc.pair_update_block.atom_single_to_atom_pair_proj_w(cond_kv.float()).to(cond.dtype)
    _record_pt(store, "encoder.pair_update.proj_h", proj_h)
    _record_pt(store, "encoder.pair_update.proj_w", proj_w)
    pair_input = blocked_pair + proj_h[:, :, :, None, :] + proj_w[:, :, None, :, :]
    _record_pt(store, "encoder.pair_update.input", pair_input)
    pair = pair_input + enc.pair_update_block.atom_pair_mlp(pair_input.float()).to(pair_input.dtype)
    _record_pt(store, "encoder.pair_update.output", pair)

    atom_mask_flat = atom_exists_mask[:, None, :].expand(-1, ds, -1).reshape(b * ds, n_atoms).contiguous()
    atom_repr = _trace_torch_local_atom_transformer(
        store,
        "encoder.atom_transformer",
        enc.atom_transformer.local_diffn_transformer,
        atom_single,
        cond,
        pair,
        kv_idx_flat,
        block_mask_flat,
        atom_mask=atom_mask_flat,
    )
    _record_pt(store, "denoise.encoder_atom_repr", atom_repr.reshape(b, ds, n_atoms, -1))

    token_to_token = _torch_linear_module(atom_repr, getattr(enc.to_token_single, "0"))
    _record_pt(store, "encoder.token_proj_pre_relu", token_to_token)
    token_to_token = torch.relu(token_to_token)
    _record_pt(store, "encoder.token_proj_post_relu", token_to_token)

    atom_token_index_flat = atom_token_index[:, None, :].expand(-1, ds, -1).reshape(b * ds, n_atoms).contiguous()
    enc_tokens = _torch_segment_mean(
        token_to_token,
        atom_token_index_flat,
        num_tokens,
        mask=atom_mask_flat,
    ).reshape(b, ds, num_tokens, -1)
    _record_pt(store, "denoise.encoder_tokens", enc_tokens)

    token_repr = structure_proj + enc_tokens
    _record_pt(store, "denoise.token_repr_pre_transformer", token_repr)

    x = token_repr
    s_cond_flat = s_cond.reshape(b * ds, num_tokens, -1).contiguous()
    for i in range(16):
        block = getattr(dm.diffusion_transformer.blocks, str(i))
        bias = pair_biases[i][:, None, :, :, :].expand(-1, ds, -1, -1, -1).reshape(b * ds, *pair_biases[i].shape[1:])
        x_flat = x.reshape(b * ds, num_tokens, -1)
        x_norm = _torch_adaln(x_flat, s_cond_flat, block.norm_in.lin_s_merged)
        qkv = _torch_linear_module(x_norm, block.to_qkv)
        qkv = qkv.reshape(b * ds, num_tokens, 16, 3, 48)
        q = qkv[:, :, :, 0].permute(0, 2, 1, 3).contiguous() + block.q_bias[None, :, None, :]
        k = qkv[:, :, :, 1].permute(0, 2, 1, 3).contiguous()
        v = qkv[:, :, :, 2].permute(0, 2, 1, 3).contiguous()
        attn = F.scaled_dot_product_attention(
            q.float(),
            k.float(),
            v.float(),
            attn_mask=bias.float(),
            scale=48 ** -0.5,
        ).to(x.dtype)
        attn = attn.permute(0, 2, 1, 3).reshape(b * ds, num_tokens, -1).contiguous()
        attn = _torch_linear_module(attn, block.to_out)
        attn = block.gate_proj(s_cond_flat.float()).to(x.dtype) * attn
        trans = _torch_conditioned_transition_delta(x_flat, s_cond_flat, block.transition)
        x = (x_flat + attn + trans).reshape(b, ds, num_tokens, -1).contiguous()
        _record_pt(store, f"denoise.transformer.block_{i}.token_repr", x)

    x = dm.post_attn_layernorm(x.float()).to(token_repr.dtype)
    _record_pt(store, "denoise.token_repr_post_transformer", x)

    decoder_cond = dm.post_atom_cond_layernorm(
        atom_single_cond[:, None, :, :].expand(-1, ds, -1, -1).float()
    ).to(token_repr.dtype)
    _record_pt(store, "denoise.decoder_cond", decoder_cond)

    token_to_atom = _torch_linear_module(x.reshape(b * ds, num_tokens, -1), dec.token_to_atom)
    _record_pt(store, "decoder.token_to_atom", token_to_atom)
    token_to_atom = _torch_gather_tokens_to_atoms(token_to_atom, atom_token_index_flat)
    _record_pt(store, "decoder.token_to_atom_gathered", token_to_atom)

    encoder_atom = atom_repr
    atom_single_dec = encoder_atom + token_to_atom
    _record_pt(store, "decoder.atom_single", atom_single_dec)

    atom_repr_dec = _trace_torch_local_atom_transformer(
        store,
        "decoder.atom_transformer",
        dec.atom_transformer.local_diffn_transformer,
        atom_single_dec,
        decoder_cond.reshape(b * ds, n_atoms, -1),
        pair,
        kv_idx_flat,
        block_mask_flat,
        atom_mask=atom_mask_flat,
    )

    output_norm = getattr(dec.to_pos_updates, "0")(atom_repr_dec.float()).to(token_repr.dtype)
    _record_pt(store, "decoder.output_norm", output_norm)
    pos_updates = _torch_linear_module(output_norm, getattr(dec.to_pos_updates, "1")).reshape(b, ds, n_atoms, 3)
    _record_pt(store, "denoise.pos_updates", pos_updates)

    out = c_skip[:, :, None, None] * coords + c_out[:, :, None, None] * pos_updates
    _record_pt(store, "denoise.output", out)


def _trace_mlx_local_atom_transformer(
    store: OrderedDict[str, np.ndarray],
    prefix: str,
    transformer,
    x: mx.array,
    cond: mx.array,
    pair: mx.array,
    kv_idx: mx.array,
    block_mask: mx.array,
    *,
    atom_mask: mx.array | None,
) -> mx.array:
    pair_norm = transformer.pair_norm(pair)
    _record_mx(store, f"{prefix}.pair_norm", pair_norm)
    pair_bias_all = transformer.blocked_pairs2blocked_bias(pair_norm)
    _record_mx(store, f"{prefix}.pair_bias_all", pair_bias_all)

    out = x
    num_blocks = int(pair.shape[1])
    query_block = int(pair.shape[2])
    kv_block = int(pair.shape[3])

    for block_idx, (local_attn, transition) in enumerate(zip(transformer.attn_blocks, transformer.transitions)):
        if atom_mask is not None:
            out = out * atom_mask[..., None].astype(out.dtype)
        _record_mx(store, f"{prefix}.block_{block_idx}.x_masked", out)

        bias_slice = pair_bias_all[..., block_idx * transformer.num_heads : (block_idx + 1) * transformer.num_heads]
        _record_mx(store, f"{prefix}.block_{block_idx}.bias_slice", bias_slice)

        x_norm = local_attn.adaln(out, cond)
        _record_mx(store, f"{prefix}.block_{block_idx}.attn.adaln", x_norm)

        q_raw, k_raw, v_raw = [
            split_heads(t, local_attn.num_heads, local_attn.head_dim)
            for t in mx.split(local_attn.to_qkv(x_norm), 3, axis=-1)
        ]
        _record_mx(store, f"{prefix}.block_{block_idx}.attn.q_all", q_raw)
        _record_mx(store, f"{prefix}.block_{block_idx}.attn.k_all", k_raw)
        _record_mx(store, f"{prefix}.block_{block_idx}.attn.v_all", v_raw)

        q = q_raw.reshape(out.shape[0], num_blocks, query_block, local_attn.num_heads, local_attn.head_dim).transpose(0, 1, 3, 2, 4)
        k = gather_blocked_atom_values(k_raw, kv_idx).transpose(0, 1, 3, 2, 4)
        v = gather_blocked_atom_values(v_raw, kv_idx).transpose(0, 1, 3, 2, 4)
        q = q + local_attn.q_bias[None, None, :, None, :]
        _record_mx(store, f"{prefix}.block_{block_idx}.attn.q_blocked", q)
        _record_mx(store, f"{prefix}.block_{block_idx}.attn.k_gathered", k)
        _record_mx(store, f"{prefix}.block_{block_idx}.attn.v_gathered", v)

        additive_bias = bias_slice.transpose(0, 1, 4, 2, 3)
        additive_bias = additive_bias + make_additive_mask(block_mask, dtype=additive_bias.dtype)[:, :, None, :, :]
        _record_mx(store, f"{prefix}.block_{block_idx}.attn.bias_masked", additive_bias)

        q_flat = q.reshape(out.shape[0] * num_blocks, local_attn.num_heads, query_block, local_attn.head_dim)
        k_flat = k.reshape(out.shape[0] * num_blocks, local_attn.num_heads, kv_block, local_attn.head_dim)
        v_flat = v.reshape(out.shape[0] * num_blocks, local_attn.num_heads, kv_block, local_attn.head_dim)
        bias_flat = additive_bias.reshape(out.shape[0] * num_blocks, local_attn.num_heads, query_block, kv_block)

        sdpa = mx.fast.scaled_dot_product_attention(
            q_flat,
            k_flat,
            v_flat,
            scale=local_attn.head_dim ** -0.5,
            mask=bias_flat,
        )
        sdpa = sdpa.reshape(out.shape[0], num_blocks, local_attn.num_heads, query_block, local_attn.head_dim)
        _record_mx(store, f"{prefix}.block_{block_idx}.attn.sdpa", sdpa)

        merged = merge_heads(sdpa.transpose(0, 1, 3, 2, 4)).reshape(out.shape[0], out.shape[1], out.shape[2])
        _record_mx(store, f"{prefix}.block_{block_idx}.attn.merged", merged)

        gate = mx.sigmoid(local_attn.output_proj(cond))
        _record_mx(store, f"{prefix}.block_{block_idx}.attn.gate", gate)
        attn_delta = merged * gate
        _record_mx(store, f"{prefix}.block_{block_idx}.attn.delta", attn_delta)

        trans_delta = transition.delta(out, cond)
        _record_mx(store, f"{prefix}.block_{block_idx}.trans.delta", trans_delta)

        out = out + attn_delta + trans_delta
        _record_mx(store, f"{prefix}.block_{block_idx}.output", out)
    return out


def _trace_mlx_denoise(
    model: ChaiMLX,
    store: OrderedDict[str, np.ndarray],
    trunk,
    coords: mx.array,
    sigma: mx.array,
) -> None:
    dm = model.diffusion_module
    structure = trunk.structure_inputs
    cache = dm.prepare_cache(trunk)

    _record_mx(store, "cache.s_static", cache.s_static)
    _record_mx(store, "cache.z_cond", cache.z_cond)
    for i, bias in enumerate(cache.pair_biases):
        _record_mx(store, f"cache.pair_biases.{i}", bias)
    _record_mx(store, "cache.blocked_pair_base", cache.blocked_pair_base)
    _record_mx(store, "cache.atom_cond", cache.atom_cond)
    _record_mx(store, "cache.atom_single_cond", cache.atom_single_cond)

    sigma = sigma.astype(mx.float32)
    sigma_sq = sigma * sigma
    sigma_data_sq = dm.cfg.diffusion.sigma_data ** 2
    c_in = (sigma_sq + sigma_data_sq) ** -0.5
    c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
    c_out = sigma * dm.cfg.diffusion.sigma_data / mx.sqrt(sigma_sq + sigma_data_sq)
    _record_mx(store, "denoise.c_in", c_in)
    _record_mx(store, "denoise.c_skip", c_skip)
    _record_mx(store, "denoise.c_out", c_out)

    num_samples = coords.shape[1]
    scaled_coords = coords * c_in[:, :, None, None]
    _record_mx(store, "denoise.scaled_coords", scaled_coords)

    s_cond = dm.diffusion_conditioning.with_sigma(cache.s_static, sigma)
    _record_mx(store, "denoise.s_cond", s_cond)

    structure_proj = dm.structure_cond_to_token_structure_proj(s_cond)
    _record_mx(store, "denoise.structure_proj", structure_proj)

    b = cache.atom_cond.shape[0]
    ds = num_samples
    n_atoms = cache.atom_cond.shape[1]
    num_blocks = n_atoms // 32

    atom_init = mx.broadcast_to(cache.atom_cond[:, None, :, :], (b, ds, n_atoms, cache.atom_cond.shape[-1])).reshape(
        b * ds, n_atoms, cache.atom_cond.shape[-1]
    )
    _record_mx(store, "encoder.atom_init", atom_init)
    pos_embed = dm.atom_attention_encoder.prev_pos_embed(
        scaled_coords.reshape(b * ds, n_atoms, 3).astype(mx.float32)
    )
    _record_mx(store, "encoder.pos_embed", pos_embed)
    atom_single = atom_init + pos_embed.astype(atom_init.dtype)
    _record_mx(store, "encoder.atom_single", atom_single)

    cond = mx.broadcast_to(cache.atom_single_cond[:, None, :, :], (b, ds, n_atoms, cache.atom_single_cond.shape[-1])).reshape(
        b * ds, n_atoms, cache.atom_single_cond.shape[-1]
    )
    _record_mx(store, "encoder.cond", cond)

    kv_idx_flat = mx.broadcast_to(structure.atom_kv_indices[:, None, :, :], (b, ds, *structure.atom_kv_indices.shape[1:])).reshape(
        b * ds, *structure.atom_kv_indices.shape[1:]
    )
    block_mask_flat = mx.broadcast_to(
        structure.block_atom_pair_mask[:, None, :, :, :],
        (b, ds, *structure.block_atom_pair_mask.shape[1:]),
    ).reshape(b * ds, *structure.block_atom_pair_mask.shape[1:])
    blocked_pair = mx.broadcast_to(
        cache.blocked_pair_base[:, None, :, :, :, :],
        (b, ds, *cache.blocked_pair_base.shape[1:]),
    ).reshape(b * ds, *cache.blocked_pair_base.shape[1:])
    cond_q = cond.reshape(b * ds, num_blocks, 32, cond.shape[-1])
    cond_kv = gather_blocked_atom_values(cond, kv_idx_flat)
    _record_mx(store, "encoder.pair_update.cond_q", cond_q)
    _record_mx(store, "encoder.pair_update.cond_kv", cond_kv)
    _record_mx(store, "encoder.pair_update.blocked_pair_base", blocked_pair)

    proj_h = dm.atom_attention_encoder.pair_update_block.proj_h(nn.relu(cond_q))
    proj_w = dm.atom_attention_encoder.pair_update_block.proj_w(nn.relu(cond_kv))
    _record_mx(store, "encoder.pair_update.proj_h", proj_h)
    _record_mx(store, "encoder.pair_update.proj_w", proj_w)
    pair_input = blocked_pair + proj_h[:, :, :, None, :] + proj_w[:, :, None, :, :]
    _record_mx(store, "encoder.pair_update.input", pair_input)
    pair = dm.atom_attention_encoder.pair_update_block(cond_q, cond_kv, blocked_pair)
    _record_mx(store, "encoder.pair_update.output", pair)

    atom_mask_flat = mx.broadcast_to(structure.atom_exists_mask[:, None, :], (b, ds, structure.atom_exists_mask.shape[-1])).reshape(
        b * ds, -1
    )
    atom_repr = _trace_mlx_local_atom_transformer(
        store,
        "encoder.atom_transformer",
        dm.atom_attention_encoder.atom_transformer,
        atom_single,
        cond,
        pair,
        kv_idx_flat,
        block_mask_flat,
        atom_mask=atom_mask_flat,
    )
    _record_mx(store, "denoise.encoder_atom_repr", atom_repr.reshape(b, ds, n_atoms, -1))

    token_pre_relu = atom_repr @ dm.atom_attention_encoder.to_token_single.weight.T
    _record_mx(store, "encoder.token_proj_pre_relu", token_pre_relu)
    token_to_token = mx.maximum(token_pre_relu, 0)
    _record_mx(store, "encoder.token_proj_post_relu", token_to_token)

    atom_token_index_flat = mx.broadcast_to(structure.atom_token_index[:, None, :], (b, ds, structure.atom_token_index.shape[-1])).reshape(
        b * ds, -1
    )
    enc_tokens = segment_mean(
        token_to_token,
        atom_token_index_flat,
        trunk.single_initial.shape[1],
        mask=atom_mask_flat,
    ).reshape(b, ds, trunk.single_initial.shape[1], -1)
    _record_mx(store, "denoise.encoder_tokens", enc_tokens)

    token_repr = structure_proj + enc_tokens
    _record_mx(store, "denoise.token_repr_pre_transformer", token_repr)

    out = token_repr
    for i, (block, pair_bias) in enumerate(zip(dm.diffusion_transformer.blocks, cache.pair_biases)):
        bias = mx.broadcast_to(pair_bias[:, None, :, :, :], (b, ds, *pair_bias.shape[1:]))
        out = block(
            out.reshape(b * ds, out.shape[2], out.shape[3]),
            s_cond.reshape(b * ds, s_cond.shape[2], s_cond.shape[3]),
            bias.reshape(b * ds, *pair_bias.shape[1:]),
        ).reshape(b, ds, out.shape[2], out.shape[3])
        _record_mx(store, f"denoise.transformer.block_{i}.token_repr", out)
    x = dm.post_attn_layernorm(out)
    _record_mx(store, "denoise.token_repr_post_transformer", x)

    decoder_cond = dm.post_atom_cond_layernorm(
        mx.broadcast_to(cache.atom_single_cond[:, None, :, :], (b, ds, *cache.atom_single_cond.shape[1:]))
    )
    _record_mx(store, "denoise.decoder_cond", decoder_cond)

    token_to_atom = dm.atom_attention_decoder.token_to_atom(x.reshape(b * ds, x.shape[2], x.shape[3]))
    _record_mx(store, "decoder.token_to_atom", token_to_atom)
    token_to_atom_gathered = gather_tokens_to_atoms(token_to_atom, atom_token_index_flat)
    _record_mx(store, "decoder.token_to_atom_gathered", token_to_atom_gathered)
    atom_single_dec = atom_repr + token_to_atom_gathered
    _record_mx(store, "decoder.atom_single", atom_single_dec)

    atom_repr_dec = _trace_mlx_local_atom_transformer(
        store,
        "decoder.atom_transformer",
        dm.atom_attention_decoder.atom_transformer,
        atom_single_dec,
        decoder_cond.reshape(b * ds, decoder_cond.shape[2], decoder_cond.shape[3]),
        pair,
        kv_idx_flat,
        block_mask_flat,
        atom_mask=atom_mask_flat,
    )

    output_norm = dm.atom_attention_decoder.output_norm(atom_repr_dec)
    _record_mx(store, "decoder.output_norm", output_norm)
    pos_updates = dm.atom_attention_decoder.to_pos_updates(output_norm).reshape(b, ds, n_atoms, 3)
    _record_mx(store, "denoise.pos_updates", pos_updates)

    output = c_skip[:, :, None, None] * coords + c_out[:, :, None, None] * pos_updates
    _record_mx(store, "denoise.output", output)


def _compare_traces(
    torch_trace: OrderedDict[str, np.ndarray],
    mlx_trace: OrderedDict[str, np.ndarray],
    structure: StructureInputs,
    *,
    num_samples: int,
    jump_threshold: float,
) -> int:
    atom_mask = np.array(structure.atom_exists_mask, copy=False).astype(bool)
    token_mask = np.array(structure.token_exists_mask, copy=False).astype(bool)
    token_pair_mask = np.array(structure.token_pair_mask, copy=False).astype(bool)
    block_pair_mask = np.array(structure.block_atom_pair_mask, copy=False).astype(bool)

    atom_mask_flat = np.broadcast_to(
        atom_mask[:, None, :],
        (atom_mask.shape[0], num_samples, atom_mask.shape[1]),
    ).reshape(atom_mask.shape[0] * num_samples, atom_mask.shape[1])
    token_mask_flat = np.broadcast_to(
        token_mask[:, None, :],
        (token_mask.shape[0], num_samples, token_mask.shape[1]),
    ).reshape(token_mask.shape[0] * num_samples, token_mask.shape[1])
    q_mask_flat = atom_mask_flat.reshape(atom_mask_flat.shape[0], atom_mask_flat.shape[1] // 32, 32)

    kv_idx = np.array(structure.atom_kv_indices, copy=False)
    kv_idx_flat = np.broadcast_to(
        kv_idx[:, None, :, :],
        (kv_idx.shape[0], num_samples, *kv_idx.shape[1:]),
    ).reshape(kv_idx.shape[0] * num_samples, *kv_idx.shape[1:])
    kv_mask_flat = atom_mask_flat[np.arange(atom_mask_flat.shape[0])[:, None, None], kv_idx_flat]

    block_pair_mask_flat = np.broadcast_to(
        block_pair_mask[:, None, :, :, :],
        (block_pair_mask.shape[0], num_samples, *block_pair_mask.shape[1:]),
    ).reshape(block_pair_mask.shape[0] * num_samples, *block_pair_mask.shape[1:])

    def trace_mask(name: str, shape: tuple[int, ...]) -> np.ndarray | None:
        if name == "cache.z_cond":
            return np.broadcast_to(token_pair_mask[..., None], shape)
        if name == "cache.s_static":
            return np.broadcast_to(token_mask[..., None], shape)
        if name.startswith("cache.pair_biases."):
            return np.broadcast_to(token_pair_mask[:, None, :, :], shape)
        if name in {
            "denoise.s_cond",
            "denoise.structure_proj",
            "denoise.encoder_tokens",
            "denoise.token_repr_pre_transformer",
            "denoise.token_repr_post_transformer",
        } or name.startswith("denoise.transformer.block_"):
            return np.broadcast_to(token_mask[:, None, :, None], shape)
        if name in {
            "cache.atom_cond",
            "cache.atom_single_cond",
            "denoise.scaled_coords",
            "denoise.encoder_atom_repr",
            "denoise.decoder_cond",
            "decoder.atom_single",
            "decoder.output_norm",
            "denoise.pos_updates",
            "denoise.output",
            "encoder.atom_init",
            "encoder.pos_embed",
            "encoder.atom_single",
            "encoder.cond",
            "encoder.token_proj_pre_relu",
            "encoder.token_proj_post_relu",
            "decoder.token_to_atom_gathered",
        }:
            base = atom_mask[:, None, :, None] if len(shape) == 4 else atom_mask_flat[:, :, None]
            return np.broadcast_to(base, shape)
        if name == "decoder.token_to_atom":
            return np.broadcast_to(token_mask_flat[:, :, None], shape)
        if name in {
            "cache.blocked_pair_base",
            "encoder.pair_update.blocked_pair_base",
            "encoder.pair_update.input",
            "encoder.pair_update.output",
            "encoder.atom_transformer.pair_norm",
            "encoder.atom_transformer.pair_bias_all",
            "decoder.atom_transformer.pair_norm",
            "decoder.atom_transformer.pair_bias_all",
        } or name.endswith(".bias_slice"):
            base = (
                block_pair_mask[:, :, :, :, None]
                if len(shape) == 5 and shape[0] == block_pair_mask.shape[0]
                else block_pair_mask_flat[:, :, :, :, None]
            )
            return np.broadcast_to(base, shape)
        if name == "encoder.pair_update.cond_q":
            return np.broadcast_to(q_mask_flat[:, :, :, None], shape)
        if name == "encoder.pair_update.cond_kv":
            return np.broadcast_to(kv_mask_flat[:, :, :, None], shape)
        if name.endswith(".attn.q_blocked") or name.endswith(".attn.sdpa"):
            return np.broadcast_to(q_mask_flat[:, :, None, :, None], shape)
        if name.endswith(".attn.k_gathered") or name.endswith(".attn.v_gathered"):
            return np.broadcast_to(kv_mask_flat[:, :, None, :, None], shape)
        if name.endswith(".attn.bias_masked"):
            return np.broadcast_to(block_pair_mask_flat[:, :, None, :, :], shape)
        if name.endswith(".attn.merged") or name.endswith(".attn.gate") or name.endswith(".attn.delta") or name.endswith(".trans.delta") or name.endswith(".output"):
            return np.broadcast_to(atom_mask_flat[:, :, None], shape)
        return None

    failures = 0
    first_jump = None
    for name, ref in torch_trace.items():
        got = mlx_trace.get(name)
        if got is None:
            print(f"[MISSING] {name}")
            failures += 1
            continue
        if ref.shape != got.shape:
            print(f"[SHAPE] {name}: torch={ref.shape} mlx={got.shape}")
            failures += 1
            continue
        diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
        full_max = float(diff.max()) if diff.size else 0.0
        full_mean = float(diff.mean()) if diff.size else 0.0
        full_p99 = float(np.quantile(diff, 0.99)) if diff.size else 0.0
        mask = trace_mask(name, diff.shape)
        cmp_label = "full"
        cmp_max = full_max
        cmp_mean = full_mean
        cmp_p99 = full_p99
        invalid_summary = ""
        if mask is not None and np.any(mask):
            valid = diff[mask]
            cmp_label = "valid"
            cmp_max = float(valid.max()) if valid.size else 0.0
            cmp_mean = float(valid.mean()) if valid.size else 0.0
            cmp_p99 = float(np.quantile(valid, 0.99)) if valid.size else 0.0
            invalid = diff[~mask]
            if invalid.size:
                invalid_summary = (
                    f"  invalid_mean={float(invalid.mean()):9.4e}"
                    f"  invalid_max={float(invalid.max()):9.4e}"
                )
        print(
            f"{name:<52} {cmp_label}_max={cmp_max:9.4e}  {cmp_label}_mean={cmp_mean:9.4e}  "
            f"{cmp_label}_p99={cmp_p99:9.4e}  full_max={full_max:9.4e}"
            f"{invalid_summary}  shape={ref.shape}  torch={ref.dtype} mlx={got.dtype}"
        )
        if first_jump is None and cmp_p99 >= jump_threshold:
            first_jump = (name, cmp_label, cmp_max, cmp_mean, cmp_p99)
    extra = sorted(set(mlx_trace) - set(torch_trace))
    if extra:
        print(f"[INFO] extra MLX checkpoints: {len(extra)}")
        for name in extra[:20]:
            print(f"  {name}")
    if first_jump is not None:
        name, cmp_label, cmp_max, cmp_mean, cmp_p99 = first_jump
        print(
            f"\nFirst checkpoint over threshold {jump_threshold:.3g}: "
            f"{name} ({cmp_label}_max={cmp_max:.4e}, {cmp_label}_mean={cmp_mean:.4e}, {cmp_label}_p99={cmp_p99:.4e})"
        )
    else:
        print(f"\nNo checkpoint exceeded threshold {jump_threshold:.3g}.")
    return failures


def _torch_structure(ctx: StructureInputs, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "atom_exists_mask": _to_torch(ctx.atom_exists_mask, device, dtype=torch.bool),
        "token_pair_mask": _to_torch(ctx.token_pair_mask, device, dtype=torch.bool),
        "atom_token_index": _to_torch(ctx.atom_token_index, device, dtype=torch.long),
        "atom_q_indices": _to_torch(ctx.atom_q_indices, device, dtype=torch.long),
        "atom_kv_indices": _to_torch(ctx.atom_kv_indices, device, dtype=torch.long),
        "block_atom_pair_mask": _to_torch(ctx.block_atom_pair_mask, device, dtype=torch.bool),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deep denoise checkpoint trace")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--coords-mode", choices=("zero", "input"), default="zero")
    parser.add_argument(
        "--torch-device",
        default="cpu",
        help="Torch device for the TorchScript replay (default: cpu)",
    )
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=0.1,
        help=(
            "First checkpoint whose valid/full p99 abs diff exceeds this is "
            "reported as the first jump"
        ),
    )
    parser.add_argument("--write-torch-dump", type=Path, default=None)
    parser.add_argument("--write-mlx-dump", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False)
    ctx, extras = load_feature_context(args.input_npz)
    ref = _npz_dict(args.reference_npz)
    dtype = resolve_dtype(model.cfg)
    trunk = reconstruct_trunk_outputs(ref, ctx.structure_inputs, dtype=dtype)

    coords = extras["coords"]
    if args.coords_mode == "zero":
        coords = mx.zeros(coords.shape, dtype=mx.float32)
    sigma = mx.full(extras["sigma"].shape, float(args.sigma), dtype=mx.float32)

    torch_device = torch.device(args.torch_device)
    torch_trunk = {
        "single_initial": _to_torch(trunk.single_initial, torch_device, dtype=torch.float32),
        "single_trunk": _to_torch(trunk.single_trunk, torch_device, dtype=torch.float32),
        "single_structure": _to_torch(trunk.single_structure, torch_device, dtype=torch.float32),
        "pair_initial": _to_torch(trunk.pair_initial, torch_device, dtype=torch.float32),
        "pair_trunk": _to_torch(trunk.pair_trunk, torch_device, dtype=torch.float32),
        "pair_structure": _to_torch(trunk.pair_structure, torch_device, dtype=torch.float32),
        "atom_single_structure_input": _to_torch(trunk.atom_single_structure_input, torch_device, dtype=torch.float32),
        "atom_pair_structure_input": _to_torch(trunk.atom_pair_structure_input, torch_device, dtype=torch.float32),
    }
    torch_structure = _torch_structure(ctx.structure_inputs, torch_device)
    torch_coords = _to_torch(coords, torch_device, dtype=torch.float32)
    torch_sigma = _to_torch(sigma, torch_device, dtype=torch.float32)

    mlx_trace: OrderedDict[str, np.ndarray] = OrderedDict()
    _trace_mlx_denoise(model, mlx_trace, trunk, coords, sigma)

    torch_trace: OrderedDict[str, np.ndarray] = OrderedDict()
    with torch.no_grad():
        with _component_moved_to("diffusion_module.pt", device=torch_device) as diffusion_module:
            _trace_torch_denoise(
                diffusion_module.jit_module,
                torch_trace,
                torch_trunk,
                torch_structure,
                torch_coords,
                torch_sigma,
            )

    if args.write_torch_dump is not None:
        args.write_torch_dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.write_torch_dump, **torch_trace)
    if args.write_mlx_dump is not None:
        args.write_mlx_dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.write_mlx_dump, **mlx_trace)

    failures = _compare_traces(
        torch_trace,
        mlx_trace,
        ctx.structure_inputs,
        num_samples=int(coords.shape[1]),
        jump_threshold=args.jump_threshold,
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
