from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import Chai1Config
from .layers.atom_attention import TokenInputAtomEncoder
from .types import EmbeddingOutputs, FeatureContext
from .utils import chunk_last

_TEMPLATE_RESTYPE_START = 41
_TEMPLATE_RESTYPE_END = 73
_TOKEN_PAIR_RBF_START = 149


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

        self.template_restype_embedding = nn.Embedding(
            cfg.template_restype_vocab, cfg.template_restype_embed_dim
        )
        self.distance_restraint_radii = mx.zeros((cfg.num_rbf_radii,))
        self.pocket_restraint_radii = mx.zeros((cfg.num_rbf_radii,))
        self._dist_rbf_scale = cfg.distance_rbf_scale
        self._pocket_rbf_scale = cfg.pocket_rbf_scale

    def _encode_rbf(
        self, raw: mx.array, radii: mx.array, scale: float
    ) -> mx.array:
        """Gaussian RBF encoding matching the TorchScript feature_embedding.pt.

        Args:
            raw: (B, N, N, 1) float — distance or -1 sentinel.
            radii: (num_radii,) learned centre positions.
            scale: scalar denominator.

        Returns:
            (B, N, N, num_radii + 1) — RBF channels + mask channel.
        """
        data = mx.expand_dims(raw, axis=-1)
        r = radii.reshape((1, 1, 1, 1, -1))
        diff = (r - data) / scale
        exp_val = diff * diff
        clamped = mx.minimum(exp_val, mx.array(16.0))
        encoding = mx.exp(-clamped)
        encoding = mx.where(clamped >= 16.0, mx.array(0.0), encoding)
        should_mask = (data == -1.0).astype(mx.float32)
        encoding = encoding * (1.0 - should_mask)
        result = mx.concatenate([encoding, should_mask], axis=-1)
        return result.reshape(*result.shape[:-2], result.shape[-1])

    def _encode_template_restype(self, indices: mx.array) -> mx.array:
        """Learned embedding outer-sum for TemplateResType.

        Args:
            indices: (B, T, N) int — residue type indices 0..vocab-1.

        Returns:
            (B, T, N, N, embed_dim) — pairwise outer-sum of embeddings.
        """
        emb = self.template_restype_embedding(indices)
        row = mx.expand_dims(emb, axis=-2)
        col = mx.expand_dims(emb, axis=-3)
        return row + col

    def __call__(self, ctx: FeatureContext) -> dict[str, mx.array]:
        token_pair_feats = ctx.token_pair_features
        if ctx.distance_restraint_data is not None:
            rbf_dist = self._encode_rbf(
                ctx.distance_restraint_data,
                self.distance_restraint_radii,
                self._dist_rbf_scale,
            )
            rbf_pocket = self._encode_rbf(
                ctx.pocket_restraint_data,
                self.pocket_restraint_radii,
                self._pocket_rbf_scale,
            )
            pre = token_pair_feats[..., :_TOKEN_PAIR_RBF_START]
            token_pair_feats = mx.concatenate(
                [pre, rbf_dist, rbf_pocket], axis=-1
            )

        template_feats = ctx.template_features
        if ctx.template_restype_indices is not None:
            encoded_rt = self._encode_template_restype(
                ctx.template_restype_indices
            )
            pre = template_feats[..., :_TEMPLATE_RESTYPE_START]
            post = template_feats[..., _TEMPLATE_RESTYPE_END:]
            template_feats = mx.concatenate(
                [pre, encoded_rt, post], axis=-1
            )

        token_single = self.token_proj(ctx.token_features)
        token_pair = self.token_pair_proj(token_pair_feats)
        atom_single = self.atom_proj(ctx.atom_features)
        atom_pair = self.atom_pair_proj(ctx.atom_pair_features)
        msa = self.msa_proj(ctx.msa_features)
        templates = self.template_proj(template_feats)

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
        use_kernel: bool = False,
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
            use_kernel=use_kernel,
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

    def __call__(self, ctx: FeatureContext, *, use_kernel: bool = False) -> EmbeddingOutputs:
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
            use_kernel=use_kernel,
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
