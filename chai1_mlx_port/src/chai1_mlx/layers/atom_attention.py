from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..kernels.blocked_local_attention import blocked_local_attention as blocked_local_attention_kernel
from ..utils import (
    gather_blocked_atom_values,
    gather_tokens_to_atoms,
    make_additive_mask,
    merge_heads,
    segment_mean,
    split_heads,
)
from .common import AdaLayerNorm, ConditionedTransition, ResidualMLP


class PairUpdateBlock(nn.Module):
    def __init__(self, atom_dim: int, pair_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(atom_dim, eps=eps)
        self.norm_kv = nn.LayerNorm(atom_dim, eps=eps)
        self.proj_h = nn.Linear(atom_dim, pair_dim, bias=False)
        self.proj_w = nn.Linear(atom_dim, pair_dim, bias=False)
        self.mlp = ResidualMLP(pair_dim)

    def __call__(self, q_atoms: mx.array, kv_atoms: mx.array, pair: mx.array) -> mx.array:
        h = self.proj_h(self.norm_q(q_atoms))
        w = self.proj_w(self.norm_kv(kv_atoms))
        pair = pair + h[:, :, :, None, :] + w[:, :, None, :, :]
        return self.mlp(pair)


class LocalAttentionPairBiasBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, num_heads: int, head_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.adaln = AdaLayerNorm(dim, cond_dim, eps=eps)
        self.to_qkv = nn.Linear(dim, 3 * num_heads * head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * head_dim, dim, bias=True)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_bias = mx.zeros((num_heads, head_dim), dtype=mx.float32)

    def __call__(
        self,
        x: mx.array,
        cond: mx.array,
        kv_idx: mx.array,
        additive_bias: mx.array,
        block_mask: mx.array,
        *,
        use_custom_kernel: bool = False,
    ) -> mx.array:
        b, a, d = x.shape
        num_blocks = a // 32
        x_norm = self.adaln(x, cond, use_kernel=use_custom_kernel)
        q_all, k_all, v_all = [
            split_heads(t, self.num_heads, self.head_dim) for t in mx.split(self.to_qkv(x_norm), 3, axis=-1)
        ]

        q = q_all.reshape(b, num_blocks, 32, self.num_heads, self.head_dim).transpose(0, 1, 3, 2, 4)
        k = gather_blocked_atom_values(k_all, kv_idx).transpose(0, 1, 3, 2, 4)
        v = gather_blocked_atom_values(v_all, kv_idx).transpose(0, 1, 3, 2, 4)

        additive_bias = additive_bias.transpose(0, 1, 4, 2, 3)
        additive_bias = additive_bias + make_additive_mask(block_mask)[:, :, None, :, :]

        q = q + self.q_bias[None, None, :, None, :]
        q_flat = q.reshape(b * num_blocks, self.num_heads, 32, self.head_dim)
        k_flat = k.reshape(b * num_blocks, self.num_heads, 128, self.head_dim)
        v_flat = v.reshape(b * num_blocks, self.num_heads, 128, self.head_dim)
        bias_flat = additive_bias.reshape(b * num_blocks, self.num_heads, 32, 128)

        if use_custom_kernel:
            out = blocked_local_attention_kernel(
                q_flat.astype(mx.float32),
                k_flat.astype(mx.float32),
                v_flat.astype(mx.float32),
                bias_flat.astype(mx.float32),
                self.head_dim ** -0.5,
            ).astype(x.dtype)
        else:
            out = mx.fast.scaled_dot_product_attention(
                q_flat,
                k_flat,
                v_flat,
                scale=self.head_dim ** -0.5,
                mask=bias_flat,
            )
        out = out.reshape(b, num_blocks, self.num_heads, 32, self.head_dim).transpose(0, 1, 3, 2, 4)
        out = self.output_proj(merge_heads(out)).reshape(b, a, d)
        return x + out


class LocalAtomTransformer(nn.Module):
    def __init__(self, dim: int, pair_dim: int, cond_dim: int, num_blocks: int, num_heads: int, head_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.pair_norm = nn.LayerNorm(pair_dim, eps=eps)
        self.blocked_pairs2blocked_bias = nn.Linear(pair_dim, num_blocks * num_heads, bias=True)
        self.attn_blocks = [
            LocalAttentionPairBiasBlock(dim, cond_dim, num_heads, head_dim, eps=eps)
            for _ in range(num_blocks)
        ]
        self.transitions = [
            ConditionedTransition(dim, cond_dim, expansion=2, eps=eps) for _ in range(num_blocks)
        ]
        self.num_heads = num_heads
        self.num_blocks = num_blocks

    def __call__(
        self,
        x: mx.array,
        cond: mx.array,
        pair: mx.array,
        kv_idx: mx.array,
        block_mask: mx.array,
        *,
        use_custom_kernel: bool = False,
    ) -> mx.array:
        pair_bias = self.blocked_pairs2blocked_bias(self.pair_norm(pair))
        for i, (attn, ff) in enumerate(zip(self.attn_blocks, self.transitions)):
            bias_slice = pair_bias[..., i * self.num_heads : (i + 1) * self.num_heads]
            x = attn(
                x,
                cond,
                kv_idx,
                bias_slice,
                block_mask,
                use_custom_kernel=use_custom_kernel,
            )
            x = ff(x, cond, use_kernel=use_custom_kernel)
        return x


class TokenInputAtomEncoder(nn.Module):
    def __init__(self, atom_dim: int, pair_dim: int, token_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.to_atom_cond = nn.Linear(atom_dim, atom_dim, bias=False)
        self.pair_update_block = PairUpdateBlock(atom_dim, pair_dim, eps=eps)
        self.atom_transformer = LocalAtomTransformer(
            atom_dim,
            pair_dim,
            atom_dim,
            num_blocks=3,
            num_heads=4,
            head_dim=32,
            eps=eps,
        )
        self.to_token_single = nn.Linear(atom_dim, token_dim, bias=False)

    def __call__(
        self,
        atom_single_input: mx.array,
        atom_pair_input: mx.array,
        atom_token_index: mx.array,
        atom_mask: mx.array,
        kv_idx: mx.array,
        block_mask: mx.array,
        *,
        num_tokens: int,
        use_custom_kernel: bool = False,
    ) -> mx.array:
        b, a, d = atom_single_input.shape
        num_blocks = a // 32
        cond = self.to_atom_cond(atom_single_input)
        q_atoms = atom_single_input.reshape(b, num_blocks, 32, d)
        kv_atoms = gather_blocked_atom_values(atom_single_input, kv_idx)
        pair = self.pair_update_block(q_atoms, kv_atoms, atom_pair_input)
        atom_repr = self.atom_transformer(
            atom_single_input,
            cond,
            pair,
            kv_idx,
            block_mask,
            use_custom_kernel=use_custom_kernel,
        )
        token_repr = mx.maximum(self.to_token_single(atom_repr), 0)
        return segment_mean(token_repr, atom_token_index, num_tokens, mask=atom_mask)


class DiffusionAtomAttentionEncoder(nn.Module):
    def __init__(self, atom_dim: int, pair_dim: int, token_dim: int, diffusion_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.to_atom_cond = nn.Linear(atom_dim, atom_dim, bias=False)
        self.token_to_atom_single_norm = nn.LayerNorm(token_dim, eps=eps)
        self.token_to_atom_single = nn.Linear(token_dim, atom_dim, bias=False)
        self.prev_pos_embed = nn.Linear(3, atom_dim, bias=False)
        self.pair_update_block = PairUpdateBlock(atom_dim, pair_dim, eps=eps)
        self.atom_transformer = LocalAtomTransformer(
            atom_dim,
            pair_dim,
            atom_dim,
            num_blocks=3,
            num_heads=4,
            head_dim=32,
            eps=eps,
        )
        self.to_token_single = nn.Linear(atom_dim, diffusion_dim, bias=False)

    def __call__(
        self,
        s_cond: mx.array,
        atom_cond_projected: mx.array,
        blocked_pair_base: mx.array,
        atom_token_index: mx.array,
        atom_mask: mx.array,
        coords: mx.array,
        kv_idx: mx.array,
        block_mask: mx.array,
        *,
        num_tokens: int,
        use_custom_kernel: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]:
        b, ds, n, _ = s_cond.shape
        atom_cond = atom_cond_projected
        token_to_atom = gather_tokens_to_atoms(
            self.token_to_atom_single(self.token_to_atom_single_norm(s_cond.reshape(b * ds, n, -1))),
            mx.broadcast_to(atom_token_index[:, None, :], (b, ds, atom_token_index.shape[-1])).reshape(b * ds, -1),
        )
        atom_cond = mx.broadcast_to(atom_cond[:, None, :, :], (b, ds, *atom_cond.shape[1:])).reshape(b * ds, atom_cond.shape[1], atom_cond.shape[2])
        atom_single = atom_cond + token_to_atom + self.prev_pos_embed(coords.reshape(b * ds, coords.shape[-2], 3))

        num_blocks = atom_single.shape[1] // 32
        q_atoms = atom_single.reshape(b * ds, num_blocks, 32, atom_single.shape[-1])
        kv_idx_flat = mx.broadcast_to(kv_idx[:, None, :, :], (b, ds, *kv_idx.shape[1:])).reshape(b * ds, *kv_idx.shape[1:])
        block_mask_flat = mx.broadcast_to(block_mask[:, None, :, :, :], (b, ds, *block_mask.shape[1:])).reshape(b * ds, *block_mask.shape[1:])
        blocked_pair = mx.broadcast_to(blocked_pair_base[:, None, :, :, :, :], (b, ds, *blocked_pair_base.shape[1:])).reshape(b * ds, *blocked_pair_base.shape[1:])
        kv_atoms = gather_blocked_atom_values(atom_single, kv_idx_flat)
        pair = self.pair_update_block(q_atoms, kv_atoms, blocked_pair)
        atom_repr = self.atom_transformer(
            atom_single,
            atom_cond,
            pair,
            kv_idx_flat,
            block_mask_flat,
            use_custom_kernel=use_custom_kernel,
        )
        atom_token_index_flat = mx.broadcast_to(atom_token_index[:, None, :], (b, ds, atom_token_index.shape[-1])).reshape(b * ds, -1)
        atom_mask_flat = mx.broadcast_to(atom_mask[:, None, :], (b, ds, atom_mask.shape[-1])).reshape(b * ds, -1)
        token_repr = segment_mean(
            mx.maximum(self.to_token_single(atom_repr), 0),
            atom_token_index_flat,
            num_tokens,
            mask=atom_mask_flat,
        )
        return token_repr.reshape(b, ds, num_tokens, -1), atom_repr.reshape(b, ds, atom_repr.shape[1], -1), atom_cond.reshape(b, ds, atom_cond.shape[1], -1)


class DiffusionAtomAttentionDecoder(nn.Module):
    def __init__(self, diffusion_dim: int, atom_dim: int, pair_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.token_to_atom = nn.Linear(diffusion_dim, atom_dim, bias=False)
        self.atom_transformer = LocalAtomTransformer(
            atom_dim,
            pair_dim,
            atom_dim,
            num_blocks=3,
            num_heads=4,
            head_dim=32,
            eps=eps,
        )
        self.output_norm = nn.LayerNorm(atom_dim, eps=eps)
        self.to_pos_updates = nn.Linear(atom_dim, 3, bias=False)

    def __call__(
        self,
        token_repr: mx.array,
        atom_cond: mx.array,
        blocked_pair_base: mx.array,
        atom_token_index: mx.array,
        kv_idx: mx.array,
        block_mask: mx.array,
        *,
        use_custom_kernel: bool = False,
    ) -> mx.array:
        b, ds, n, _ = token_repr.shape
        atom_token_index_flat = mx.broadcast_to(atom_token_index[:, None, :], (b, ds, atom_token_index.shape[-1])).reshape(b * ds, -1)
        atom_cond_flat = atom_cond.reshape(b * ds, atom_cond.shape[-2], atom_cond.shape[-1])
        atom_single = gather_tokens_to_atoms(
            self.token_to_atom(token_repr.reshape(b * ds, n, -1)),
            atom_token_index_flat,
        )
        kv_idx_flat = mx.broadcast_to(kv_idx[:, None, :, :], (b, ds, *kv_idx.shape[1:])).reshape(b * ds, *kv_idx.shape[1:])
        block_mask_flat = mx.broadcast_to(block_mask[:, None, :, :, :], (b, ds, *block_mask.shape[1:])).reshape(b * ds, *block_mask.shape[1:])
        blocked_pair = mx.broadcast_to(blocked_pair_base[:, None, :, :, :, :], (b, ds, *blocked_pair_base.shape[1:])).reshape(b * ds, *blocked_pair_base.shape[1:])
        atom_repr = self.atom_transformer(
            atom_single,
            atom_cond_flat,
            blocked_pair,
            kv_idx_flat,
            block_mask_flat,
            use_custom_kernel=use_custom_kernel,
        )
        return self.to_pos_updates(self.output_norm(atom_repr)).reshape(b, ds, atom_repr.shape[1], 3)
