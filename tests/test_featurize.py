from __future__ import annotations

import pytest

from chai_mlx.data.featurize import featurize
from chai_mlx.data.types import FeatureContext, InputBundle, StructureInputs

from tests.helpers import make_precomputed_feature_context, make_structure_inputs


def test_featurize_returns_existing_feature_context(cfg) -> None:
    ctx = make_precomputed_feature_context(cfg)

    actual = featurize(ctx)

    assert actual is ctx


def test_featurize_accepts_input_bundle_with_features(cfg) -> None:
    ctx = make_precomputed_feature_context(cfg)
    bundle = InputBundle(features=ctx)

    actual = featurize(bundle)

    assert actual is ctx


def test_featurize_builds_feature_context_from_raw_dict(cfg) -> None:
    ctx = make_precomputed_feature_context(cfg)
    payload = {
        "token_features": ctx.token_features,
        "token_pair_features": ctx.token_pair_features,
        "atom_features": ctx.atom_features,
        "atom_pair_features": ctx.atom_pair_features,
        "msa_features": ctx.msa_features,
        "template_features": ctx.template_features,
        "bond_adjacency": ctx.bond_adjacency,
        "structure_inputs": {
            "atom_exists_mask": ctx.structure_inputs.atom_exists_mask,
            "token_exists_mask": ctx.structure_inputs.token_exists_mask,
            "token_pair_mask": ctx.structure_inputs.token_pair_mask,
            "atom_token_index": ctx.structure_inputs.atom_token_index,
            "atom_within_token_index": ctx.structure_inputs.atom_within_token_index,
            "token_reference_atom_index": ctx.structure_inputs.token_reference_atom_index,
            "token_centre_atom_index": ctx.structure_inputs.token_centre_atom_index,
            "token_asym_id": ctx.structure_inputs.token_asym_id,
            "token_entity_id": ctx.structure_inputs.token_entity_id,
            "token_chain_id": ctx.structure_inputs.token_chain_id,
            "token_is_polymer": ctx.structure_inputs.token_is_polymer,
        },
    }

    actual = featurize(payload)

    assert isinstance(actual, FeatureContext)
    assert isinstance(actual.structure_inputs, StructureInputs)
    assert actual.structure_inputs.token_centre_atom_index is not None
    assert actual.structure_inputs.token_centre_atom_index.shape == ctx.structure_inputs.token_centre_atom_index.shape


def test_featurize_accepts_input_bundle_raw_payload(cfg) -> None:
    ctx = make_precomputed_feature_context(cfg)
    bundle = InputBundle(
        raw={
            "token_features": ctx.token_features,
            "token_pair_features": ctx.token_pair_features,
            "atom_features": ctx.atom_features,
            "atom_pair_features": ctx.atom_pair_features,
            "msa_features": ctx.msa_features,
            "template_features": ctx.template_features,
            "structure_inputs": make_structure_inputs(),
        }
    )

    actual = featurize(bundle)

    assert isinstance(actual, FeatureContext)


def test_featurize_rejects_missing_required_keys() -> None:
    with pytest.raises(ValueError, match="Missing keys"):
        featurize({"token_features": 1})


def test_featurize_rejects_invalid_structure_inputs(cfg) -> None:
    ctx = make_precomputed_feature_context(cfg)
    payload = {
        "token_features": ctx.token_features,
        "token_pair_features": ctx.token_pair_features,
        "atom_features": ctx.atom_features,
        "atom_pair_features": ctx.atom_pair_features,
        "msa_features": ctx.msa_features,
        "template_features": ctx.template_features,
        "structure_inputs": object(),
    }

    with pytest.raises(TypeError, match="structure_inputs"):
        featurize(payload)
