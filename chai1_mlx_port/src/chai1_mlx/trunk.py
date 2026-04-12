from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import Chai1Config
from .layers.attention import MSAPairWeightedAveraging
from .layers.common import Transition
from .layers.pairformer import PairformerBlock, PairformerStack
from .types import EmbeddingOutputs, TrunkOutputs
from .utils import masked_mean


class RecycleProjection(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.proj = nn.Linear(dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(self.norm(x))


class TemplateEmbedder(nn.Module):
    def __init__(self, cfg: Chai1Config) -> None:
        super().__init__()
        tdim = cfg.hidden.template_pair
        self.proj_in_norm = nn.LayerNorm(cfg.templates.max_templates * tdim, eps=cfg.layer_norm_eps)
        self.proj_in = nn.Linear(cfg.templates.max_templates * tdim, tdim, bias=False)
        self.blocks = [
            PairformerBlock(
                pair_dim=tdim,
                single_dim=None,
                triangle_heads=cfg.templates.triangle_heads,
                triangle_head_dim=cfg.templates.triangle_head_dim,
                eps=cfg.layer_norm_eps,
            )
            for _ in range(cfg.templates.num_blocks)
        ]
        self.template_layernorm = nn.LayerNorm(tdim, eps=cfg.layer_norm_eps)
        self.proj_out = nn.Linear(tdim, cfg.hidden.token_pair, bias=False)

    def __call__(self, pair: mx.array, templates: mx.array, *, pair_mask: mx.array | None = None) -> mx.array:
        b, t, n, _, c = templates.shape
        x = templates.transpose(0, 2, 3, 1, 4).reshape(b, n, n, t * c)
        x = self.proj_in(self.proj_in_norm(x))
        for block in self.blocks:
            x, _ = block(x, None, pair_mask=pair_mask)
        return pair + self.proj_out(self.template_layernorm(x))


class OuterProductMean(nn.Module):
    """Outer-product-mean from MSA to pair representation.

    Reference: ARCHITECTURE.md §6.3.2.
    weight_ab: [2, 8, 8, msa_dim] -- two projections mapping MSA-dim to 8x8.
    Einsum 1 (projection): "abc, defc -> abdef" where a=batch*depth, b=tokens,
        c=msa_dim and d=2 (proj index), e=8 (row), f=8 (col).
    Einsum 2 (outer product): "abcde, afcdg -> cegabf" -- contracts over batch
        and depth dimensions, producing the [n, n, 8, 8] outer product that is
        reshaped to [n, n, 512] before LN + linear.
    """

    def __init__(self, msa_dim: int, pair_dim: int, *, eps: float = 1e-5) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(msa_dim, eps=eps)
        self.weight_ab = mx.zeros((2, 8, 8, msa_dim), dtype=mx.float32)
        self.ln_out = nn.LayerNorm(512, eps=eps)
        self.linear_out = nn.Linear(512, pair_dim, bias=True)

    def __call__(self, msa: mx.array, msa_mask: mx.array | None = None) -> mx.array:
        x = self.norm(msa)
        # x: [b, m, n, msa_dim]; weight_ab: [2, 8, 8, msa_dim]
        # proj: [b, m, n, 2, 8, 8] via einsum "bmnc, defc -> bmnde f" (contract c)
        proj = mx.einsum("bmnc,defc->bmndef", x, self.weight_ab)

        if msa_mask is not None:
            m = msa_mask.astype(x.dtype)[..., None, None, None]
            proj = proj * m

        # Outer product over depth dimension:
        # a_proj = proj[..., 0, :, :]: [b, m, n, 8, 8]  (proj index 0)
        # b_proj = proj[..., 1, :, :]: [b, m, n, 8, 8]  (proj index 1)
        a_proj = proj[:, :, :, 0, :, :]
        b_proj = proj[:, :, :, 1, :, :]
        # outer product: contract batch*depth, produce [n_i, n_j, 8_a, 8_b, 8_c, 8_d]
        # but we want [b, n_i, n_j, 8, 8] from contracting m.
        # Reference einsum 2: "abcde, afcdg -> cegabf" is specific to the
        # TorchScript implementation shape conventions.  The core operation is:
        #   op[b, i, j, ra, cb] = sum_m  a_proj[b, m, i, ra, :] * b_proj[b, m, j, :, cb]
        # which is an outer product of rows/cols across MSA depth.
        op = mx.einsum("bmiae,bmjbe->bijab", a_proj, b_proj)

        if msa_mask is not None:
            pair_count = mx.einsum(
                "bmi,bmj->bij",
                msa_mask.astype(x.dtype),
                msa_mask.astype(x.dtype),
            )
            denom = mx.maximum(pair_count[..., None, None], 1.0)
            op = op / denom
        else:
            op = op / float(msa.shape[1])

        op = op.reshape(op.shape[0], op.shape[1], op.shape[2], 512)
        return self.linear_out(self.ln_out(op))


class MSAModule(nn.Module):
    def __init__(self, cfg: Chai1Config) -> None:
        super().__init__()
        self.linear_s2m = nn.Linear(cfg.hidden.token_single, cfg.hidden.msa, bias=False)
        self.outer_product_mean = [
            OuterProductMean(cfg.hidden.msa, cfg.hidden.token_pair, eps=cfg.layer_norm_eps)
            for _ in range(cfg.msa.num_outer_product_mean)
        ]
        self.msa_pair_weighted_averaging = [
            MSAPairWeightedAveraging(
                cfg.hidden.msa,
                cfg.hidden.token_pair,
                num_heads=cfg.msa.pair_weight_heads,
                value_dim=cfg.msa.pair_weight_value_dim,
                eps=cfg.layer_norm_eps,
            )
            for _ in range(cfg.msa.num_pair_weighted_avg)
        ]
        self.msa_transition = [
            Transition(cfg.hidden.msa, expansion=4, bias_out=True, eps=cfg.layer_norm_eps)
            for _ in range(cfg.msa.num_msa_transition)
        ]
        self.pair_transition = [
            Transition(cfg.hidden.token_pair, expansion=4, bias_out=True, eps=cfg.layer_norm_eps)
            for _ in range(cfg.msa.num_pair_transition)
        ]
        self.triangular_multiplication = [
            PairformerBlock(
                pair_dim=cfg.hidden.token_pair,
                single_dim=None,
                triangle_heads=cfg.pairformer.triangle_heads,
                triangle_head_dim=cfg.pairformer.triangle_head_dim,
                eps=cfg.layer_norm_eps,
            ).triangle_multiplication
            for _ in range(cfg.msa.num_tri_mult)
        ]
        self.triangular_attention = [
            PairformerBlock(
                pair_dim=cfg.hidden.token_pair,
                single_dim=None,
                triangle_heads=cfg.pairformer.triangle_heads,
                triangle_head_dim=cfg.pairformer.triangle_head_dim,
                eps=cfg.layer_norm_eps,
            ).triangle_attention
            for _ in range(cfg.msa.num_tri_attn)
        ]

    def __call__(
        self,
        single: mx.array,
        pair: mx.array,
        msa_input: mx.array,
        *,
        token_pair_mask: mx.array | None = None,
        msa_mask: mx.array | None = None,
    ) -> mx.array:
        msa = msa_input
        if msa.shape[1] > 0:
            first = msa[:, :1] + self.linear_s2m(single)[:, None, :, :]
            msa = mx.concatenate([first, msa[:, 1:]], axis=1)

        for i in range(len(self.outer_product_mean)):
            pair = pair + self.outer_product_mean[i](msa, msa_mask=msa_mask)
            pair = pair + self.pair_transition[i](pair)
            if i < len(self.msa_pair_weighted_averaging):
                msa = msa + self.msa_pair_weighted_averaging[i](msa, pair, token_pair_mask=token_pair_mask)
                msa = msa + self.msa_transition[i](msa)
            pair = self.triangular_multiplication[i](pair, pair_mask=token_pair_mask)
            pair = self.triangular_attention[i](pair, pair_mask=token_pair_mask)
        return pair


class Trunk(nn.Module):
    def __init__(self, cfg: Chai1Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_single_recycle_proj = RecycleProjection(cfg.hidden.token_single, eps=cfg.layer_norm_eps)
        self.token_pair_recycle_proj = RecycleProjection(cfg.hidden.token_pair, eps=cfg.layer_norm_eps)
        self.template_embedder = TemplateEmbedder(cfg)
        self.msa_module = MSAModule(cfg)
        self.pairformer_stack = PairformerStack(
            [
                PairformerBlock(
                    pair_dim=cfg.hidden.token_pair,
                    single_dim=cfg.hidden.token_single,
                    single_heads=cfg.pairformer.single_heads,
                    single_head_dim=cfg.pairformer.single_head_dim,
                    triangle_heads=cfg.pairformer.triangle_heads,
                    triangle_head_dim=cfg.pairformer.triangle_head_dim,
                    eps=cfg.layer_norm_eps,
                )
                for _ in range(cfg.pairformer.num_blocks)
            ]
        )

    def __call__(
        self,
        emb: EmbeddingOutputs,
        *,
        recycles: int = 3,
        msa_mask: mx.array | None = None,
    ) -> TrunkOutputs:
        single_init = emb.single_initial
        pair_init = emb.pair_initial
        single = single_init
        pair = pair_init
        token_pair_mask = emb.structure_inputs.token_pair_mask

        prev_single = None
        prev_pair = None
        for _ in range(recycles):
            if prev_single is not None:
                single = single_init + self.token_single_recycle_proj(prev_single)
                pair = pair_init + self.token_pair_recycle_proj(prev_pair)
            pair = self.template_embedder(pair, emb.template_input, pair_mask=token_pair_mask)
            pair = self.msa_module(
                single,
                pair,
                emb.msa_input,
                token_pair_mask=token_pair_mask,
                msa_mask=msa_mask,
            )
            single, pair = self.pairformer_stack(single, pair, pair_mask=token_pair_mask)
            prev_single, prev_pair = single, pair

        return TrunkOutputs(
            single_initial=single_init,
            single_trunk=single,
            single_structure=emb.single_structure,
            pair_initial=pair_init,
            pair_trunk=pair,
            pair_structure=emb.pair_structure,
            atom_single_structure_input=emb.atom_single_structure_input,
            atom_pair_structure_input=emb.atom_pair_structure_input,
            msa_input=emb.msa_input,
            template_input=emb.template_input,
            structure_inputs=emb.structure_inputs,
        )
