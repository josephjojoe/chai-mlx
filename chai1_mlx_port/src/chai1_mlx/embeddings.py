from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import Chai1Config
from .layers.atom_attention import TokenInputAtomEncoder
from .types import EmbeddingOutputs, FeatureContext, RawFeatures
from .utils import chunk_last


# ── per-feature encoding specs ────────────────────────────────────────────
# Verified against every torch.one_hot / reshape / cat in
# feature_embedding_forward256.py.  Order is alphabetical within each type
# group, matching the TorchScript concatenation order.

# (name, encoding, encoded_width)
#   encoding = "oh" (one-hot), "id" (identity), "rbf", "emb_outersum"
#   encoded_width = number of columns in the concatenated encoded tensor

_TOKEN_FEATURES = [
    ("ChainIsCropped",     "id",  1),
    ("ESMEmbeddings",      "id",  2560),
    ("IsDistillation",     "oh",  2),
    ("MSADeletionMean",    "id",  1),
    ("MSAProfile",         "id",  33),
    ("MissingChainContact","id",  1),
    ("ResidueType",        "oh",  33),
    ("TokenBFactor",       "oh",  3),
    ("TokenPLDDT",         "oh",  4),
]  # total = 2638

_TOKEN_PAIR_FEATURES = [
    ("DockingConstraintGenerator",    "oh",  6),
    ("RelativeChain",                 "oh",  6),
    ("RelativeEntity",                "oh",  3),
    ("RelativeSequenceSeparation",    "oh",  67),
    ("RelativeTokenSeparation",       "oh",  67),
    ("TokenDistanceRestraint",        "rbf", 7),
    ("TokenPairPocketRestraint",      "rbf", 7),
]  # total = 163

_ATOM_FEATURES = [
    ("AtomNameOneHot",  "oh",  65),   # 4 components × 65 → 260 after flatten
    ("AtomRefCharge",   "id",  1),
    ("AtomRefElement",  "oh",  130),
    ("AtomRefMask",     "id",  1),
    ("AtomRefPos",      "id",  3),
]  # total = 260 + 1 + 130 + 1 + 3 = 395

_ATOM_PAIR_FEATURES = [
    ("BlockedAtomPairDistogram",              "oh", 12),
    ("InverseSquaredBlockedAtomPairDistances", "id", 2),
]  # total = 14

_MSA_FEATURES = [
    ("IsPairedMSA",       "id",  1),
    ("MSADataSource",     "oh",  6),
    ("MSADeletionValue",  "id",  1),
    ("MSAHasDeletion",    "id",  1),
    ("MSAOneHot",         "oh",  33),
]  # total = 42

_TEMPLATE_FEATURES = [
    ("TemplateDistogram", "oh",          39),
    ("TemplateMask",      "id",          2),
    ("TemplateResType",   "emb_outersum", 32),
    ("TemplateUnitVector","id",          3),
]  # total = 76


class FeatureEmbedding(nn.Module):
    """Encodes raw features and projects to hidden dims in one fused pass.

    For ONE_HOT features, ``one_hot(idx, W) @ weight_slice`` is equivalent
    to an embedding lookup into the corresponding slice of the projection
    weight — no dense one-hot tensor is ever materialised.  This matches the
    TorchScript's memory profile where the wide concatenated tensor is only
    transient, and is strictly better because the embedding-gather path
    avoids allocating the wide tensor at all.
    """

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

    # ── encoding helpers ──────────────────────────────────────────────

    @staticmethod
    def _encode_one_hot(idx: mx.array, width: int) -> mx.array:
        """ONE_HOT encoding, handling multi-component features (AtomNameOneHot).

        For single-component inputs ``(…, 1)``, squeezes the trailing dim
        so one-hot produces ``(…, width)``.  For multi-component inputs
        like AtomNameOneHot ``(B, A, 4)``, the resulting ``(B, A, 4, 65)``
        is flattened to ``(B, A, 260)`` to match TorchScript.
        """
        idx = idx.astype(mx.int32)
        multi_component = idx.shape[-1] > 1
        if idx.shape[-1] == 1:
            idx = idx.squeeze(-1)
        result = mx.eye(width, dtype=mx.float32)[mx.clip(idx, 0, width - 1)]
        if multi_component:
            result = result.reshape(*idx.shape[:-1], -1)
        return result

    @staticmethod
    def _encode_identity(feat: mx.array) -> mx.array:
        out = feat.astype(mx.float32)
        if out.ndim >= 2 and out.shape[-1] > 1:
            return out
        while out.ndim < 3:
            out = mx.expand_dims(out, axis=-1)
        return out

    def _encode_rbf(
        self, raw: mx.array, radii: mx.array, scale: float
    ) -> mx.array:
        data = mx.expand_dims(raw, axis=-1)
        r = radii.reshape((1,) * (data.ndim - 1) + (-1,))
        diff = (r - data) / scale
        exp_val = diff * diff
        clamped = mx.minimum(exp_val, 16.0)
        encoding = mx.exp(-clamped)
        encoding = mx.where(clamped >= 16.0, 0.0, encoding)
        should_mask = (data == -1.0).astype(mx.float32)
        encoding = encoding * (1.0 - should_mask)
        return mx.concatenate([encoding, should_mask], axis=-1).reshape(
            *raw.shape[:-1], -1
        )

    def _encode_template_restype(self, idx: mx.array) -> mx.array:
        idx = idx.astype(mx.int32)
        if idx.shape[-1] == 1:
            idx = idx.squeeze(-1)
        emb = self.template_restype_embedding(idx)
        return mx.expand_dims(emb, axis=-2) + mx.expand_dims(emb, axis=-3)

    def _encode_group(
        self,
        spec: list[tuple[str, str, int]],
        raw: dict[str, mx.array],
    ) -> list[mx.array]:
        """Encode every feature in *spec* from raw data and return a list."""
        parts: list[mx.array] = []
        for name, enc, width in spec:
            feat = raw[name]
            if enc == "oh":
                parts.append(self._encode_one_hot(feat, width))
            elif enc == "id":
                parts.append(self._encode_identity(feat))
            elif enc == "rbf":
                if name == "TokenDistanceRestraint":
                    parts.append(self._encode_rbf(
                        feat, self.distance_restraint_radii, self._dist_rbf_scale,
                    ))
                else:
                    parts.append(self._encode_rbf(
                        feat, self.pocket_restraint_radii, self._pocket_rbf_scale,
                    ))
            elif enc == "emb_outersum":
                parts.append(self._encode_template_restype(feat))
            else:
                raise ValueError(f"Unknown encoding {enc!r}")
        return parts

    # ── main forward ─────────────────────────────────────────────────

    def __call__(self, ctx: FeatureContext) -> dict[str, mx.array]:
        if ctx.raw_features is not None:
            return self._forward_raw(ctx.raw_features)
        return self._forward_precomputed(ctx)

    def _forward_precomputed(self, ctx: FeatureContext) -> dict[str, mx.array]:
        """Fast path: features already encoded and concatenated."""
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

    def _forward_raw(self, raw: dict[str, mx.array]) -> dict[str, mx.array]:
        """Memory-efficient path: encode per-feature, project, discard encoded.

        For each feature group, we encode + concatenate + project in one shot.
        This still materialises the concatenated encoded tensor, but only one
        group at a time (so peak memory is max(single group) not sum(all)).
        """
        def _project(spec, proj):
            parts = self._encode_group(spec, raw)
            cat = mx.concatenate(parts, axis=-1)
            return proj(cat)

        token_single = _project(_TOKEN_FEATURES, self.token_proj)
        token_pair = _project(_TOKEN_PAIR_FEATURES, self.token_pair_proj)
        atom_single = _project(_ATOM_FEATURES, self.atom_proj)
        atom_pair = _project(_ATOM_PAIR_FEATURES, self.atom_pair_proj)
        msa = _project(_MSA_FEATURES, self.msa_proj)
        templates = _project(_TEMPLATE_FEATURES, self.template_proj)

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
