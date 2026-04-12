from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import Chai1Config
from .layers.atom_attention import TokenInputAtomEncoder
from .types import EmbeddingOutputs, FeatureContext
from .utils import chunk_last


class FeatureEmbedding(nn.Module):
    def __init__(self, cfg: Chai1Config) -> None:
        super().__init__()
        fd = cfg.feature_dims
        hd = cfg.hidden
        self.token_proj = nn.Linear(fd.token, hd.token_single, bias=True)
        self.token_pair_proj = nn.Linear(fd.token_pair, 2 * hd.token_pair, bias=True)
        self.atom_proj = nn.Linear(fd.atom, 2 * hd.atom_single, bias=True)
        self.atom_pair_proj = nn.Linear(fd.atom_pair, 2 * hd.atom_pair, bias=True)
        self.msa_proj = nn.Linear(fd.msa, hd.msa, bias=True)
        self.template_proj = nn.Linear(fd.templates, hd.template_pair, bias=True)

    def __call__(self, ctx: FeatureContext) -> dict[str, mx.array]:
        token_single = self.token_proj(ctx.token_features)
        token_pair = self.token_pair_proj(ctx.token_pair_features)
        atom_single = self.atom_proj(ctx.atom_features)
        atom_pair = self.atom_pair_proj(ctx.atom_pair_features)
        msa = self.msa_proj(ctx.msa_features)
        templates = self.template_proj(ctx.template_features)

        token_pair_trunk, token_pair_structure = chunk_last(token_pair, 2)
        atom_single_trunk, atom_single_structure = chunk_last(atom_single, 2)
        atom_pair_trunk, atom_pair_structure = chunk_last(atom_pair, 2)
        return {
            "token_single": token_single,
            "token_pair_trunk": token_pair_trunk,
            "token_pair_structure": token_pair_structure,
            "atom_single_trunk": atom_single_trunk,
            "atom_single_structure": atom_single_structure,
            "atom_pair_trunk": atom_pair_trunk,
            "atom_pair_structure": atom_pair_structure,
            "msa": msa,
            "templates": templates,
        }


class BondProjection(nn.Module):
    def __init__(self, cfg: Chai1Config) -> None:
        super().__init__()
        self.proj = nn.Linear(cfg.feature_dims.bond, 2 * cfg.hidden.token_pair, bias=False)

    def __call__(self, bond_adjacency: mx.array) -> tuple[mx.array, mx.array]:
        trunk, structure = chunk_last(self.proj(bond_adjacency), 2)
        return trunk, structure


class TokenInputEmbedding(nn.Module):
    def __init__(self, cfg: Chai1Config) -> None:
        super().__init__()
        self.atom_encoder = TokenInputAtomEncoder(
            cfg.hidden.atom_single,
            cfg.hidden.atom_pair,
            cfg.hidden.token_single,
            eps=cfg.layer_norm_eps,
        )
        self.token_single_proj_in_trunk = nn.Linear(
            2 * cfg.hidden.token_single, cfg.hidden.token_single, bias=False
        )
        self.token_single_proj_in_structure = nn.Linear(
            2 * cfg.hidden.token_single, cfg.hidden.token_single, bias=False
        )
        self.single_to_pair_proj = nn.Linear(cfg.hidden.token_single, 2 * cfg.hidden.token_pair, bias=False)
        self.token_pair_proj_in_trunk = nn.Linear(cfg.hidden.token_pair, cfg.hidden.token_pair, bias=False)

    def __call__(
        self,
        token_single_input: mx.array,
        token_pair_input: mx.array,
        atom_single_input: mx.array,
        atom_pair_input: mx.array,
        *,
        atom_token_index: mx.array,
        atom_mask: mx.array,
        kv_idx: mx.array,
        block_mask: mx.array,
        use_custom_kernel: bool = False,
    ) -> tuple[mx.array, mx.array, mx.array]:
        num_tokens = token_single_input.shape[1]
        atom_agg = self.atom_encoder(
            atom_single_input,
            atom_pair_input,
            atom_token_index,
            atom_mask,
            kv_idx,
            block_mask,
            num_tokens=num_tokens,
            use_custom_kernel=use_custom_kernel,
        )
        token_concat = mx.concatenate([token_single_input, atom_agg], axis=-1)
        single_initial = self.token_single_proj_in_trunk(token_concat)
        single_structure = self.token_single_proj_in_structure(token_concat)

        row, col = chunk_last(self.single_to_pair_proj(single_initial), 2)
        pair = token_pair_input + row[:, :, None, :] + col[:, None, :, :]
        pair_initial = self.token_pair_proj_in_trunk(pair)
        return single_initial, single_structure, pair_initial


class InputEmbedder(nn.Module):
    def __init__(self, cfg: Chai1Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.feature_embedding = FeatureEmbedding(cfg)
        self.bond_projection = BondProjection(cfg)
        self.token_input = TokenInputEmbedding(cfg)

    def __call__(self, ctx: FeatureContext, *, use_custom_kernel: bool = False) -> EmbeddingOutputs:
        feats = self.feature_embedding(ctx)

        bond_adjacency = ctx.bond_adjacency
        if bond_adjacency is None:
            bond_adjacency = ctx.structure_inputs.bond_adjacency
        if bond_adjacency is not None:
            bond_trunk, bond_structure = self.bond_projection(bond_adjacency)
            feats["token_pair_trunk"] = feats["token_pair_trunk"] + bond_trunk
            feats["token_pair_structure"] = feats["token_pair_structure"] + bond_structure

        structure = ctx.structure_inputs
        single_initial, single_structure, pair_initial = self.token_input(
            feats["token_single"],
            feats["token_pair_trunk"],
            feats["atom_single_trunk"],
            feats["atom_pair_trunk"],
            atom_token_index=structure.atom_token_index,
            atom_mask=structure.atom_exists_mask,
            kv_idx=structure.atom_kv_indices,
            block_mask=structure.block_atom_pair_mask,
            use_custom_kernel=use_custom_kernel,
        )

        return EmbeddingOutputs(
            token_single_input=feats["token_single"],
            token_pair_input=feats["token_pair_trunk"],
            token_pair_structure_input=feats["token_pair_structure"],
            atom_single_input=feats["atom_single_trunk"],
            atom_single_structure_input=feats["atom_single_structure"],
            atom_pair_input=feats["atom_pair_trunk"],
            atom_pair_structure_input=feats["atom_pair_structure"],
            msa_input=feats["msa"],
            template_input=feats["templates"],
            single_initial=single_initial,
            single_structure=single_structure,
            pair_initial=pair_initial,
            pair_structure=feats["token_pair_structure"],
            structure_inputs=structure,
        )
