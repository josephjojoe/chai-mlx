from __future__ import annotations

import numpy as np

import mlx.core as mx

from chai_mlx.data.types import FeatureContext
from chai_mlx.model.embeddings import (
    _ATOM_FEATURES,
    _ATOM_PAIR_FEATURES,
    _MSA_FEATURES,
    _TEMPLATE_FEATURES,
    _TOKEN_FEATURES,
    _TOKEN_PAIR_FEATURES,
    FeatureEmbedding,
    InputEmbedder,
)

from tests.helpers import make_precomputed_feature_context, make_structure_inputs


def _feature_shape(name: str) -> tuple[int, ...]:
    batch_size = 1
    n_tokens = 4
    n_atoms = 8
    msa_depth = 4
    n_templates = 2
    raw_widths = {
        "ChainIsCropped": 1,
        "ESMEmbeddings": 2560,
        "IsDistillation": 1,
        "MSADeletionMean": 1,
        "MSAProfile": 33,
        "MissingChainContact": 1,
        "ResidueType": 1,
        "TokenBFactor": 1,
        "TokenPLDDT": 1,
        "DockingConstraintGenerator": 1,
        "RelativeChain": 1,
        "RelativeEntity": 1,
        "RelativeSequenceSeparation": 1,
        "RelativeTokenSeparation": 1,
        "TokenDistanceRestraint": 1,
        "TokenPairPocketRestraint": 1,
        "AtomNameOneHot": 4,
        "AtomRefCharge": 1,
        "AtomRefElement": 1,
        "AtomRefMask": 1,
        "AtomRefPos": 3,
        "BlockedAtomPairDistogram": 1,
        "InverseSquaredBlockedAtomPairDistances": 2,
        "IsPairedMSA": 1,
        "MSADataSource": 1,
        "MSADeletionValue": 1,
        "MSAHasDeletion": 1,
        "MSAOneHot": 1,
        "TemplateDistogram": 1,
        "TemplateMask": 2,
        "TemplateResType": 1,
        "TemplateUnitVector": 3,
    }
    if name in {item[0] for item in _TOKEN_FEATURES}:
        return (batch_size, n_tokens, raw_widths[name])
    if name in {item[0] for item in _TOKEN_PAIR_FEATURES}:
        return (batch_size, n_tokens, n_tokens, raw_widths[name])
    if name in {item[0] for item in _ATOM_FEATURES}:
        return (batch_size, n_atoms, raw_widths[name])
    if name in {item[0] for item in _ATOM_PAIR_FEATURES}:
        return (batch_size, 1, 32, 128, raw_widths[name])
    if name in {item[0] for item in _MSA_FEATURES}:
        return (batch_size, msa_depth, n_tokens, raw_widths[name])
    if name in {item[0] for item in _TEMPLATE_FEATURES}:
        if name == "TemplateResType":
            return (batch_size, n_templates, n_tokens, 1)
        return (batch_size, n_templates, n_tokens, n_tokens, raw_widths[name])
    raise KeyError(name)


def _make_raw_features() -> dict[str, mx.array]:
    raw: dict[str, mx.array] = {}
    specs = (
        _TOKEN_FEATURES
        + _TOKEN_PAIR_FEATURES
        + _ATOM_FEATURES
        + _ATOM_PAIR_FEATURES
        + _MSA_FEATURES
        + _TEMPLATE_FEATURES
    )
    for idx, (name, enc, width) in enumerate(specs, start=1):
        shape = _feature_shape(name)
        size = 1
        for dim in shape:
            size *= dim
        values = mx.arange(size, dtype=mx.float32).reshape(shape)
        if enc == "oh":
            raw[name] = (values % width).astype(mx.int32)
        elif enc == "emb_outersum":
            raw[name] = (values % 33).astype(mx.int32)
        elif enc == "rbf":
            raw[name] = (values / (idx + 1)).astype(mx.float32)
        else:
            raw[name] = (values / 10.0).astype(mx.float32)
    return raw


def _encode_precomputed_context(
    feature_embedding: FeatureEmbedding,
    structure_inputs,
    raw_features: dict[str, mx.array],
) -> FeatureContext:
    return FeatureContext(
        token_features=mx.concatenate(
            feature_embedding._encode_group(_TOKEN_FEATURES, raw_features), axis=-1
        ),
        token_pair_features=mx.concatenate(
            feature_embedding._encode_group(_TOKEN_PAIR_FEATURES, raw_features), axis=-1
        ),
        atom_features=mx.concatenate(
            feature_embedding._encode_group(_ATOM_FEATURES, raw_features), axis=-1
        ),
        atom_pair_features=mx.concatenate(
            feature_embedding._encode_group(_ATOM_PAIR_FEATURES, raw_features), axis=-1
        ),
        msa_features=mx.concatenate(
            feature_embedding._encode_group(_MSA_FEATURES, raw_features), axis=-1
        ),
        template_features=mx.concatenate(
            feature_embedding._encode_group(_TEMPLATE_FEATURES, raw_features), axis=-1
        ),
        structure_inputs=structure_inputs,
    )


def test_feature_embedding_raw_and_precomputed_paths_match(cfg) -> None:
    structure_inputs = make_structure_inputs(msa_depth=4)
    raw_features = _make_raw_features()
    feature_embedding = FeatureEmbedding(cfg)
    raw_ctx = FeatureContext(
        token_features=mx.zeros((1, 0), dtype=mx.float32),
        token_pair_features=mx.zeros((1, 0), dtype=mx.float32),
        atom_features=mx.zeros((1, 0), dtype=mx.float32),
        atom_pair_features=mx.zeros((1, 0), dtype=mx.float32),
        msa_features=mx.zeros((1, 0), dtype=mx.float32),
        template_features=mx.zeros((1, 0), dtype=mx.float32),
        structure_inputs=structure_inputs,
        raw_features=raw_features,
    )
    precomputed_ctx = _encode_precomputed_context(
        feature_embedding, structure_inputs, raw_features
    )

    raw_outputs = feature_embedding(raw_ctx)
    precomputed_outputs = feature_embedding(precomputed_ctx)

    assert raw_outputs.keys() == precomputed_outputs.keys()
    for key in raw_outputs:
        np.testing.assert_allclose(
            np.array(raw_outputs[key]),
            np.array(precomputed_outputs[key]),
            rtol=1e-5,
            atol=1e-5,
        )


def test_input_embedder_trims_empty_msa_rows(cfg) -> None:
    ctx = make_precomputed_feature_context(cfg)
    raw_features = _make_raw_features()
    structure = ctx.structure_inputs
    trimmed_mask = mx.array(
        [[[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]],
        dtype=mx.float32,
    )
    structure.msa_mask = trimmed_mask
    ctx.raw_features = raw_features
    ctx.msa_features = mx.arange(1 * 4 * 4 * cfg.feature_dims.msa, dtype=mx.float32).reshape(
        1, 4, 4, cfg.feature_dims.msa
    )

    trimmed = InputEmbedder(cfg)._trim_empty_msa_rows(ctx)

    assert trimmed.structure_inputs.msa_mask.shape[1] == 2
    assert trimmed.msa_features.shape[1] == 2
    assert trimmed.raw_features is not None
    for name, feature in trimmed.raw_features.items():
        if name in {item[0] for item in _MSA_FEATURES}:
            assert feature.shape[1] == 2
