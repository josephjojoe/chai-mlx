from __future__ import annotations

import importlib
import importlib.util
from functools import lru_cache

import mlx.core as mx

from chai_mlx.config import ChaiConfig
from chai_mlx.data.types import FeatureContext, StructureInputs


@lru_cache(maxsize=None)
def module_importable(module_name: str) -> bool:
    """Return True when *module_name* can be imported successfully."""
    if importlib.util.find_spec(module_name) is None:
        return False
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True


def has_chai_lab_runtime(*, require_esm_mlx: bool = False) -> bool:
    """Whether the optional chai-lab stack is actually usable."""
    if importlib.util.find_spec('torch') is None:
        return False
    if not module_importable('chai_lab.chai1'):
        return False
    if require_esm_mlx and not module_importable('esm_mlx'):
        return False
    return True


def make_structure_inputs(
    *,
    batch_size: int = 1,
    n_tokens: int = 4,
    n_atoms: int = 8,
    msa_depth: int = 3,
    n_templates: int = 2,
    with_ranker_fields: bool = False,
) -> StructureInputs:
    """Build a ``StructureInputs`` with masks/indices sized to the given shape.

    When ``with_ranker_fields`` is true, also populates the fields consumed
    by the ranking pipeline (``token_residue_index``, ``token_entity_type``,
    ``token_backbone_frame_mask``, ``token_backbone_frame_index``). Those
    default to ``None`` to keep the majority of tests that don't exercise
    ranking on their existing happy path.
    """
    token_exists = mx.ones((batch_size, n_tokens), dtype=mx.float32)
    atom_exists = mx.ones((batch_size, n_atoms), dtype=mx.float32)
    atom_token_index = (mx.arange(n_atoms)[None, :] % n_tokens).astype(mx.int32)
    atom_within_token_index = mx.zeros((batch_size, n_atoms), dtype=mx.int32)
    token_reference_atom_index = (mx.arange(n_tokens)[None, :] % n_atoms).astype(mx.int32)
    token_centre_atom_index = ((mx.arange(n_tokens)[None, :] + 1) % n_atoms).astype(mx.int32)
    token_pair_mask = token_exists[:, :, None] * token_exists[:, None, :]
    msa_mask = mx.ones((batch_size, msa_depth, n_tokens), dtype=mx.float32)
    template_input_masks = mx.ones(
        (batch_size, n_templates, n_tokens, n_tokens), dtype=mx.float32
    )

    ranker_extras: dict = {}
    if with_ranker_fields:
        ranker_extras = {
            "token_residue_index": mx.arange(n_tokens, dtype=mx.int32)[None, :],
            "token_entity_type": mx.zeros((batch_size, n_tokens), dtype=mx.int32),
            "token_backbone_frame_mask": mx.ones((batch_size, n_tokens), dtype=mx.bool_),
            "token_backbone_frame_index": mx.zeros((batch_size, n_tokens, 3), dtype=mx.int32),
        }

    return StructureInputs(
        atom_exists_mask=atom_exists,
        token_exists_mask=token_exists,
        token_pair_mask=token_pair_mask,
        atom_token_index=atom_token_index,
        atom_within_token_index=atom_within_token_index,
        token_reference_atom_index=token_reference_atom_index,
        token_centre_atom_index=token_centre_atom_index,
        token_asym_id=mx.zeros((batch_size, n_tokens), dtype=mx.int32),
        token_entity_id=mx.zeros((batch_size, n_tokens), dtype=mx.int32),
        token_chain_id=mx.zeros((batch_size, n_tokens), dtype=mx.int32),
        token_is_polymer=mx.ones((batch_size, n_tokens), dtype=mx.float32),
        msa_mask=msa_mask,
        template_input_masks=template_input_masks,
        **ranker_extras,
    )


def make_precomputed_feature_context(cfg: ChaiConfig) -> FeatureContext:
    structure = make_structure_inputs()
    batch_size, n_tokens = structure.token_exists_mask.shape
    n_atoms = structure.atom_exists_mask.shape[1]
    msa_depth = structure.msa_mask.shape[1]
    n_templates = structure.template_input_masks.shape[1]
    atom_blocks = 1

    def seq(shape: tuple[int, ...]) -> mx.array:
        size = 1
        for dim in shape:
            size *= dim
        return mx.arange(size, dtype=mx.float32).reshape(shape) / 100.0

    return FeatureContext(
        token_features=seq((batch_size, n_tokens, cfg.feature_dims.token)),
        token_pair_features=seq(
            (batch_size, n_tokens, n_tokens, cfg.feature_dims.token_pair)
        ),
        atom_features=seq((batch_size, n_atoms, cfg.feature_dims.atom)),
        atom_pair_features=seq(
            (
                batch_size,
                atom_blocks,
                cfg.atom_blocks.query_block,
                cfg.atom_blocks.kv_block,
                cfg.feature_dims.atom_pair,
            )
        ),
        msa_features=seq((batch_size, msa_depth, n_tokens, cfg.feature_dims.msa)),
        template_features=seq(
            (
                batch_size,
                n_templates,
                n_tokens,
                n_tokens,
                cfg.feature_dims.templates,
            )
        ),
        structure_inputs=structure,
        bond_adjacency=mx.zeros((batch_size, n_tokens, n_tokens, 1), dtype=mx.float32),
    )
