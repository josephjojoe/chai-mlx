from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator

import mlx.core as mx
import mlx.nn as nn

from chai_mlx.config import ChaiConfig
from chai_mlx.nn.layers.attention import DiffusionSelfAttention
from chai_mlx.nn.layers.atom_attention import DiffusionAtomAttentionDecoder, DiffusionAtomAttentionEncoder
from chai_mlx.nn.layers.common import ConditionedTransition, Transition
from chai_mlx.data.types import DiffusionCache, StructureInputs, TrunkOutputs
from chai_mlx.utils import (
    center_random_augmentation,
    edm_gammas,
    edm_sigmas,
    gather_blocked_pair_values,
)


class FourierEmbedding(nn.Module):
    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.weights = mx.zeros((dim,), dtype=mx.float32)
        self.bias = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, sigma: mx.array) -> mx.array:
        c_noise = mx.log(sigma) * 0.25
        return mx.cos((c_noise[..., None] * self.weights + self.bias) * (2.0 * math.pi))


class DiffusionConditioning(nn.Module):
    def __init__(self, cfg: ChaiConfig) -> None:
        super().__init__()
        self.token_pair_norm = nn.LayerNorm(2 * cfg.hidden.token_pair, eps=cfg.layer_norm_eps)
        self.token_pair_proj = nn.Linear(2 * cfg.hidden.token_pair, cfg.hidden.token_pair, bias=False)
        self.pair_trans1 = Transition(cfg.hidden.token_pair, expansion=2, eps=cfg.layer_norm_eps)
        self.pair_trans2 = Transition(cfg.hidden.token_pair, expansion=2, eps=cfg.layer_norm_eps)
        self.pair_ln = nn.LayerNorm(cfg.hidden.token_pair, eps=cfg.layer_norm_eps)

        self.token_in_norm = nn.LayerNorm(2 * cfg.hidden.token_single, eps=cfg.layer_norm_eps)
        self.token_in_proj = nn.Linear(2 * cfg.hidden.token_single, cfg.hidden.token_single, bias=False)
        self.single_trans1 = Transition(cfg.hidden.token_single, expansion=2, eps=cfg.layer_norm_eps)
        self.single_trans2 = Transition(cfg.hidden.token_single, expansion=2, eps=cfg.layer_norm_eps)
        self.fourier_embedding = FourierEmbedding(dim=256)
        self.fourier_proj_norm = nn.LayerNorm(256, eps=cfg.layer_norm_eps)
        self.fourier_proj = nn.Linear(256, cfg.hidden.token_single, bias=False)
        self.single_ln = nn.LayerNorm(cfg.hidden.token_single, eps=cfg.layer_norm_eps)

    def prepare_static(self, trunk: TrunkOutputs) -> tuple[mx.array, mx.array]:
        # The reference diffusion module concatenates the *structure-path*
        # initial representations with trunk outputs — NOT the trunk-path
        # initial representations.  See chai1.py static_diffusion_inputs:
        #   token_pair_initial_repr  = token_pair_structure_input_feats
        #   token_single_initial_repr = token_single_structure_input
        pair_cat = mx.concatenate([trunk.pair_trunk, trunk.pair_structure], axis=-1)
        z = self.token_pair_proj(self.token_pair_norm(pair_cat))
        z = z + self.pair_trans1(z)
        z = z + self.pair_trans2(z)
        z_cond = self.pair_ln(z)

        single_cat = mx.concatenate([trunk.single_structure, trunk.single_trunk], axis=-1)
        s = self.token_in_proj(self.token_in_norm(single_cat))
        s = s + self.single_trans1(s)
        return s, z_cond

    def with_sigma(self, s_static: mx.array, sigma: mx.array) -> mx.array:
        sigma_embed = self.fourier_proj(self.fourier_proj_norm(self.fourier_embedding(sigma)))
        s = s_static[:, None, :, :] + sigma_embed[:, :, None, :]
        s = s + self.single_trans2(s)
        return self.single_ln(s)


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, cfg: ChaiConfig) -> None:
        super().__init__()
        self.attn = DiffusionSelfAttention(
            cfg.hidden.diffusion,
            cfg.hidden.token_single,
            cfg.hidden.token_pair,
            cfg.diffusion.num_heads,
            cfg.diffusion.head_dim,
            eps=cfg.layer_norm_eps,
        )
        self.transition = ConditionedTransition(
            cfg.hidden.diffusion,
            cfg.hidden.token_single,
            expansion=2,
            eps=cfg.layer_norm_eps,
        )

    def __call__(self, x: mx.array, s_cond: mx.array, pair_bias: mx.array, *, use_kernel: bool = False) -> mx.array:
        x = self.attn(x, s_cond, pair_bias=pair_bias, use_kernel=use_kernel)
        x = self.transition(x, s_cond, use_kernel=use_kernel)
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, cfg: ChaiConfig) -> None:
        super().__init__()
        self.blocks = [DiffusionTransformerBlock(cfg) for _ in range(cfg.diffusion.num_blocks)]

    def precompute_pair_biases(self, z_cond: mx.array, pair_mask: mx.array | None = None) -> tuple[mx.array, ...]:
        return tuple(block.attn.pair_bias(z_cond, pair_mask=pair_mask) for block in self.blocks)

    def __call__(self, x: mx.array, s_cond: mx.array, pair_biases: tuple[mx.array, ...], *, use_kernel: bool = False) -> mx.array:
        b, ds, n, d = x.shape
        out = x
        for block, pair_bias in zip(self.blocks, pair_biases):
            bias = mx.broadcast_to(pair_bias[:, None, :, :, :], (b, ds, *pair_bias.shape[1:]))
            out = block(
                out.reshape(b * ds, n, d),
                s_cond.reshape(b * ds, n, s_cond.shape[-1]),
                bias.reshape(b * ds, *pair_bias.shape[1:]),
                use_kernel=use_kernel,
            ).reshape(b, ds, n, d)
        return out


class DiffusionModule(nn.Module):
    def __init__(self, cfg: ChaiConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.diffusion_conditioning = DiffusionConditioning(cfg)
        self.token_pair_to_atom_pair_norm = nn.LayerNorm(cfg.hidden.token_pair, eps=cfg.layer_norm_eps)
        self.token_pair_to_atom_pair = nn.Linear(cfg.hidden.token_pair, cfg.hidden.atom_pair, bias=False)
        self.atom_attention_encoder = DiffusionAtomAttentionEncoder(
            cfg.hidden.atom_single,
            cfg.hidden.atom_pair,
            cfg.hidden.token_single,
            cfg.hidden.diffusion,
            eps=cfg.layer_norm_eps,
        )
        self.diffusion_transformer = DiffusionTransformer(cfg)
        self.atom_attention_decoder = DiffusionAtomAttentionDecoder(
            cfg.hidden.diffusion,
            cfg.hidden.atom_single,
            cfg.hidden.atom_pair,
            eps=cfg.layer_norm_eps,
        )
        self.structure_cond_to_token_structure_proj = nn.Linear(
            cfg.hidden.token_single, cfg.hidden.diffusion, bias=False
        )
        self.post_attn_layernorm = nn.LayerNorm(cfg.hidden.diffusion, eps=cfg.layer_norm_eps)
        self.post_atom_cond_layernorm = nn.LayerNorm(cfg.hidden.atom_single, eps=cfg.layer_norm_eps)

    def prepare_cache(self, trunk: TrunkOutputs) -> DiffusionCache:
        structure = trunk.structure_inputs
        s_static, z_cond = self.diffusion_conditioning.prepare_static(trunk)
        pair_biases = self.diffusion_transformer.precompute_pair_biases(
            z_cond, pair_mask=structure.token_pair_mask
        )
        token_atom_pair = self.token_pair_to_atom_pair(self.token_pair_to_atom_pair_norm(z_cond))
        blocked_pair_base = gather_blocked_pair_values(
            token_atom_pair,
            structure.atom_q_indices,
            structure.atom_kv_indices,
        )
        blocked_pair_base = blocked_pair_base + trunk.atom_pair_structure_input

        atom_cond = self.atom_attention_encoder.to_atom_cond(
            trunk.atom_single_structure_input
        )
        atom_single_cond = self.atom_attention_encoder.prepare_cond(
            atom_cond,
            trunk.single_trunk,
            structure.atom_token_index,
        )

        return DiffusionCache(
            s_static=s_static,
            z_cond=z_cond,
            pair_biases=pair_biases,
            blocked_pair_base=blocked_pair_base,
            atom_cond=atom_cond,
            atom_single_cond=atom_single_cond,
            trunk_outputs=trunk,
            structure_inputs=structure,
        )

    def denoise(
        self,
        cache: DiffusionCache,
        coords: mx.array,
        sigma: mx.array,
        *,
        use_kernel: bool = False,
    ) -> mx.array:
        trunk = cache.trunk_outputs
        structure = cache.structure_inputs
        sigma = sigma.astype(mx.float32)
        sigma_sq = sigma * sigma
        sigma_data_sq = self.cfg.diffusion.sigma_data ** 2
        c_in = (sigma_sq + sigma_data_sq) ** -0.5
        c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
        c_out = sigma * self.cfg.diffusion.sigma_data / mx.sqrt(sigma_sq + sigma_data_sq)

        num_samples = coords.shape[1]
        scaled_coords = coords * c_in[:, :, None, None]
        s_cond = self.diffusion_conditioning.with_sigma(cache.s_static, sigma)

        x = self.structure_cond_to_token_structure_proj(trunk.single_structure)
        x = mx.broadcast_to(x[:, None, :, :], (coords.shape[0], num_samples, *x.shape[1:]))
        enc_tokens, atom_repr = self.atom_attention_encoder(
            cache.atom_cond,
            cache.atom_single_cond,
            cache.blocked_pair_base,
            structure.atom_token_index,
            structure.atom_exists_mask,
            scaled_coords,
            structure.atom_kv_indices,
            structure.block_atom_pair_mask,
            num_tokens=trunk.single_initial.shape[1],
            num_samples=num_samples,
            use_kernel=use_kernel,
        )
        x = x + enc_tokens
        x = self.diffusion_transformer(x, s_cond, cache.pair_biases, use_kernel=use_kernel)
        x = self.post_attn_layernorm(x)
        decoder_cond = self.post_atom_cond_layernorm(
            mx.broadcast_to(
                cache.atom_single_cond[:, None, :, :],
                (coords.shape[0], num_samples, *cache.atom_single_cond.shape[1:]),
            )
        )
        pos_updates = self.atom_attention_decoder(
            x,
            atom_repr,
            decoder_cond,
            cache.blocked_pair_base,
            structure.atom_token_index,
            structure.atom_kv_indices,
            structure.block_atom_pair_mask,
            use_kernel=use_kernel,
        )
        return c_skip[:, :, None, None] * coords + c_out[:, :, None, None] * pos_updates

    def schedule(self, num_steps: int | None = None) -> Iterator[tuple[mx.array, mx.array, mx.array]]:
        num_steps = self.cfg.diffusion.num_steps if num_steps is None else num_steps
        sigmas = edm_sigmas(
            num_steps,
            self.cfg.diffusion.sigma_data,
            self.cfg.diffusion.s_min,
            self.cfg.diffusion.s_max,
            self.cfg.diffusion.p,
        )
        gammas = edm_gammas(
            sigmas,
            self.cfg.diffusion.s_churn,
            self.cfg.diffusion.s_tmin,
            self.cfg.diffusion.s_tmax,
        )
        for sigma_curr, sigma_next, gamma in zip(sigmas[:-1], sigmas[1:], gammas[:-1]):
            yield sigma_curr, sigma_next, gamma

    def init_noise(self, batch_size: int, num_samples: int, structure: StructureInputs) -> mx.array:
        num_atoms = structure.atom_exists_mask.shape[-1]
        sigmas = edm_sigmas(
            self.cfg.diffusion.num_steps,
            self.cfg.diffusion.sigma_data,
            self.cfg.diffusion.s_min,
            self.cfg.diffusion.s_max,
            self.cfg.diffusion.p,
        )
        return float(sigmas[0]) * mx.random.normal((batch_size, num_samples, num_atoms, 3)).astype(mx.float32)

    def diffusion_step(
        self,
        cache: DiffusionCache,
        coords: mx.array,
        sigma_curr: mx.array | float,
        sigma_next: mx.array | float,
        gamma: mx.array | float,
        *,
        use_kernel: bool = False,
    ) -> mx.array:
        structure = cache.structure_inputs
        sigma_curr = mx.full((coords.shape[0], coords.shape[1]), float(sigma_curr), dtype=mx.float32)
        sigma_next = mx.full((coords.shape[0], coords.shape[1]), float(sigma_next), dtype=mx.float32)
        gamma = mx.full((coords.shape[0], coords.shape[1]), float(gamma), dtype=mx.float32)

        # SE(3) augmentation before noise injection.
        coords_aug = center_random_augmentation(
            coords.reshape(coords.shape[0] * coords.shape[1], coords.shape[2], 3),
            mx.broadcast_to(structure.atom_exists_mask[:, None, :], (coords.shape[0], coords.shape[1], structure.atom_exists_mask.shape[-1])).reshape(
                coords.shape[0] * coords.shape[1], structure.atom_exists_mask.shape[-1]
            ),
            centroid_eps=self.cfg.centroid_eps,
        ).reshape(coords.shape)

        sigma_hat = sigma_curr + gamma * sigma_curr
        noise = self.cfg.diffusion.s_noise * mx.random.normal(coords.shape).astype(mx.float32)
        sigma_delta = mx.maximum(sigma_hat * sigma_hat - sigma_curr * sigma_curr, self.cfg.diffusion_sqrt_eps)
        coords_hat = coords_aug + noise * mx.sqrt(sigma_delta)[:, :, None, None]

        denoised = self.denoise(cache, coords_hat, sigma_hat, use_kernel=use_kernel)
        d_i = (coords_hat - denoised) / sigma_hat[:, :, None, None]
        coords_euler = coords_hat + (sigma_next - sigma_hat)[:, :, None, None] * d_i

        sigma_next_nonzero = bool(mx.any(sigma_next != 0).item())
        if self.cfg.diffusion.second_order and sigma_next_nonzero:
            denoised_next = self.denoise(
                cache, coords_euler, sigma_next, use_kernel=use_kernel
            )
            d_i_prime = (coords_euler - denoised_next) / sigma_next[:, :, None, None]
            coords_euler = coords_euler + (sigma_next - sigma_hat)[:, :, None, None] * (d_i_prime + d_i) / 2.0
        return coords_euler
