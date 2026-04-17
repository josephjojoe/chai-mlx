"""Compare MLX intermediate tensors against a reference runtime dump.

Usage::

    python scripts/layer_parity.py \
        --weights-dir /path/to/safetensors_dir \
        --input-npz /path/to/input_context.npz \
        --reference-npz /path/to/reference_tensors.npz

The input NPZ should contain precomputed feature tensors using keys like:

- ``token_features``, ``token_pair_features``, ``atom_features``,
  ``atom_pair_features``, ``msa_features``, ``template_features``
- optional ``bond_adjacency``
- ``structure_inputs.<field_name>`` for ``StructureInputs``
- optional ``coords`` and ``sigma`` for diffusion / confidence checks

When ``--reference-npz`` is omitted, this script can still write the MLX
capture to ``--write-mlx-dump`` for later comparison.

For diffusion captures, this script follows the same top-level denoise dataflow
as the runtime module, including the sigma-conditioned token-structure
projection path.
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from chai_mlx import ChaiMLX
from chai_mlx.data.types import (
    ConfidenceOutputs,
    DiffusionCache,
    EmbeddingOutputs,
    FeatureContext,
    RankingOutputs,
    StructureInputs,
    TrunkOutputs,
)
from chai_mlx.utils import cdist, expand_plddt_to_atoms, one_hot_binned, representative_atom_coords


_FEATURE_KEYS = (
    "token_features",
    "token_pair_features",
    "atom_features",
    "atom_pair_features",
    "msa_features",
    "template_features",
)
_REQUIRED_STRUCTURE_KEYS = {
    "atom_exists_mask",
    "token_exists_mask",
    "token_pair_mask",
    "atom_token_index",
    "atom_within_token_index",
    "token_reference_atom_index",
    "token_asym_id",
    "token_entity_id",
    "token_chain_id",
    "token_is_polymer",
}


def _npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def _mx_array(x: np.ndarray) -> mx.array:
    return mx.array(x)


def _record(store: dict[str, np.ndarray], key: str, value: mx.array) -> None:
    mx.eval(value)
    v = value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
    store[key] = np.array(v, copy=False)


def _record_dataclass(store: dict[str, np.ndarray], prefix: str, value) -> None:
    for field in fields(value):
        field_value = getattr(value, field.name)
        if isinstance(field_value, mx.array):
            _record(store, f"{prefix}.{field.name}", field_value)


def _zero_placeholder(batch_size: int) -> mx.array:
    return mx.zeros((batch_size, 0), dtype=mx.float32)


def load_feature_context(path: Path) -> tuple[FeatureContext, dict[str, mx.array]]:
    data = _npz_dict(path)
    structure_data = {
        key.split(".", 1)[1]: _mx_array(value)
        for key, value in data.items()
        if key.startswith("structure_inputs.")
    }
    missing_structure = sorted(_REQUIRED_STRUCTURE_KEYS - set(structure_data))
    if missing_structure:
        raise ValueError(
            "Missing required structure_inputs keys in input NPZ: "
            + ", ".join(missing_structure)
        )
    if "token_centre_atom_index" not in structure_data:
        warnings.warn(
            "input NPZ is missing structure_inputs.token_centre_atom_index; "
            "regenerate the bundle with the current chai_lab_reference_dump.py "
            "for correct Cα metrics.",
            RuntimeWarning,
        )

    raw_features = {
        key.split(".", 1)[1]: _mx_array(value)
        for key, value in data.items()
        if key.startswith("raw_features.")
    }
    if raw_features:
        batch_size = int(structure_data["token_exists_mask"].shape[0])
        feature_values = {
            key: _mx_array(data[key]) if key in data else _zero_placeholder(batch_size)
            for key in _FEATURE_KEYS
        }
    else:
        missing_features = sorted(set(_FEATURE_KEYS) - set(data))
        if missing_features:
            raise ValueError(
                "Missing required feature keys in input NPZ: "
                + ", ".join(missing_features)
            )
        feature_values = {key: _mx_array(data[key]) for key in _FEATURE_KEYS}

    ctx = FeatureContext(
        token_features=feature_values["token_features"],
        token_pair_features=feature_values["token_pair_features"],
        atom_features=feature_values["atom_features"],
        atom_pair_features=feature_values["atom_pair_features"],
        msa_features=feature_values["msa_features"],
        template_features=feature_values["template_features"],
        structure_inputs=StructureInputs(**structure_data),
        bond_adjacency=_mx_array(data["bond_adjacency"]) if "bond_adjacency" in data else None,
        raw_features=raw_features or None,
    )
    consumed = set(_FEATURE_KEYS) | {"bond_adjacency"}
    consumed |= {f"structure_inputs.{key}" for key in structure_data}
    consumed |= {f"raw_features.{key}" for key in raw_features}
    extras = {
        key: _mx_array(value)
        for key, value in data.items()
        if key not in consumed
    }
    return ctx, extras


def _capture_template_embedder(
    module,
    pair: mx.array,
    templates: mx.array,
    *,
    template_input_masks: mx.array | None,
    token_pair_mask: mx.array | None,
    tensors: dict[str, np.ndarray],
    prefix: str,
) -> mx.array:
    b, t, _, _, _ = templates.shape
    z_base = module.proj_in(module.proj_in_norm(pair))
    _record(tensors, f"{prefix}.z_base", z_base)

    if template_input_masks is not None and token_pair_mask is not None:
        combined_mask = template_input_masks * token_pair_mask[:, None, :, :]
    elif template_input_masks is not None:
        combined_mask = template_input_masks
    else:
        combined_mask = None

    if combined_mask is not None:
        _record(tensors, f"{prefix}.combined_mask", combined_mask)
    if template_input_masks is not None:
        has_any = mx.any(template_input_masks, axis=(-2, -1))
        n_valid = mx.maximum(has_any.astype(mx.float32).sum(axis=1), 1.0)
    else:
        n_valid = mx.full((b,), float(t))
    _record(tensors, f"{prefix}.n_valid", n_valid)

    per_template_outputs: list[mx.array] = []
    for ti in range(t):
        z = z_base + templates[:, ti]
        _record(tensors, f"{prefix}.template_{ti}.input", z)
        tmask = combined_mask[:, ti] if combined_mask is not None else None
        for bi, block in enumerate(module.blocks):
            z, _ = block(z, None, pair_mask=tmask)
            _record(tensors, f"{prefix}.template_{ti}.block_{bi}.pair", z)
        per_template_outputs.append(z)

    stacked = mx.stack(per_template_outputs, axis=1)
    normed = module.template_layernorm(stacked)
    if combined_mask is not None:
        normed = normed * combined_mask[..., None]
    averaged = normed.sum(axis=1) / n_valid[:, None, None, None]
    out = pair + module.proj_out(nn.relu(averaged))
    _record(tensors, f"{prefix}.stacked", stacked)
    _record(tensors, f"{prefix}.averaged", averaged)
    _record(tensors, f"{prefix}.pair", out)
    return out


def _capture_msa_module(
    module,
    single: mx.array,
    pair: mx.array,
    msa_input: mx.array,
    *,
    token_pair_mask: mx.array | None,
    msa_mask: mx.array | None,
    tensors: dict[str, np.ndarray],
    prefix: str,
) -> mx.array:
    msa = msa_input
    if msa.shape[1] > 0:
        msa = msa + module.linear_s2m(single)[:, None, :, :]
    _record(tensors, f"{prefix}.msa_input", msa)

    for i in range(len(module.outer_product_mean)):
        pair = pair + module.outer_product_mean[i](msa, msa_mask=msa_mask)
        _record(tensors, f"{prefix}.iter_{i}.pair_after_opm", pair)
        if i < len(module.msa_transition):
            msa = msa + module.msa_transition[i](msa)
            _record(tensors, f"{prefix}.iter_{i}.msa_after_transition", msa)
            msa = msa + module.msa_pair_weighted_averaging[i](
                msa,
                pair,
                token_pair_mask=token_pair_mask,
                msa_mask=msa_mask,
            )
            _record(tensors, f"{prefix}.iter_{i}.msa_after_pair_weight", msa)
        pair_transition_out = module.pair_transition[i](pair)
        _record(tensors, f"{prefix}.iter_{i}.pair_transition", pair_transition_out)
        pair = module.triangular_multiplication[i](pair, pair_mask=token_pair_mask) + pair_transition_out
        _record(tensors, f"{prefix}.iter_{i}.pair_after_triangle_mult", pair)
        pair = module.triangular_attention[i](pair, pair_mask=token_pair_mask)
        _record(tensors, f"{prefix}.iter_{i}.pair_after_triangle_attn", pair)
    return pair


def _capture_pairformer_stack(
    stack,
    single: mx.array,
    pair: mx.array,
    *,
    pair_mask: mx.array | None,
    single_mask: mx.array | None,
    tensors: dict[str, np.ndarray],
    prefix: str,
    block_start: int | None = None,
    block_end: int | None = None,
    stop_after_block: int | None = None,
) -> tuple[mx.array, mx.array]:
    s = single
    z = pair
    for i, block in enumerate(stack.blocks):
        z, s = block(z, s, pair_mask=pair_mask, single_mask=single_mask)
        assert s is not None
        if (block_start is None or i >= block_start) and (block_end is None or i <= block_end):
            _record(tensors, f"{prefix}.block_{i}.pair", z)
            _record(tensors, f"{prefix}.block_{i}.single", s)
        if stop_after_block is not None and i >= stop_after_block:
            break
    return s, z


def capture_embeddings(
    model: ChaiMLX,
    ctx: FeatureContext,
    *,
    tensors: dict[str, np.ndarray],
) -> EmbeddingOutputs:
    ctx = model.input_embedder._trim_empty_msa_rows(ctx)
    feats = model.input_embedder.feature_embedding(ctx)
    for key, value in feats.items():
        _record(tensors, f"embedding.features.{key}", value)

    bond_adjacency = (
        ctx.bond_adjacency
        if ctx.bond_adjacency is not None
        else ctx.structure_inputs.bond_adjacency
    )
    if bond_adjacency is not None:
        bond_trunk, bond_structure = model.input_embedder.bond_projection(bond_adjacency)
        feats["token_pair_trunk"] = feats["token_pair_trunk"] + bond_trunk
        feats["token_pair_structure"] = feats["token_pair_structure"] + bond_structure
        _record(tensors, "embedding.bond_trunk", bond_trunk)
        _record(tensors, "embedding.bond_structure", bond_structure)

    structure = ctx.structure_inputs
    single_initial, single_structure, pair_initial = model.input_embedder.token_input(
        feats["token_single"],
        feats["token_pair_trunk"],
        feats["atom_single_trunk"],
        feats["atom_pair_trunk"],
        atom_token_index=structure.atom_token_index,
        atom_mask=structure.atom_exists_mask,
        kv_idx=structure.atom_kv_indices,
        block_mask=structure.block_atom_pair_mask,
    )
    emb = EmbeddingOutputs(
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
    _record_dataclass(tensors, "embedding.outputs", emb)
    return emb


def capture_trunk(
    model: ChaiMLX,
    emb: EmbeddingOutputs,
    *,
    recycles: int,
    tensors: dict[str, np.ndarray],
    capture_detail: str = "full",
    capture_pre_pairformer: bool = False,
    pairformer_block_start: int | None = None,
    pairformer_block_end: int | None = None,
    stop_after_pairformer_block: int | None = None,
    record_outputs: bool = True,
) -> TrunkOutputs:
    single_init = emb.single_initial
    pair_init = emb.pair_initial
    si = emb.structure_inputs
    token_pair_mask = si.token_pair_mask
    msa_mask = si.msa_mask
    template_input_masks = si.template_input_masks
    token_single_mask = si.token_exists_mask

    prev_single = single_init
    prev_pair = pair_init
    for recycle_idx in range(recycles):
        single = single_init + model.trunk_module.token_single_recycle_proj(prev_single)
        pair = pair_init + model.trunk_module.token_pair_recycle_proj(prev_pair)
        if capture_detail == "full":
            _record(tensors, f"trunk.recycle_{recycle_idx}.single_after_recycle", single)
            _record(tensors, f"trunk.recycle_{recycle_idx}.pair_after_recycle", pair)
            pair = _capture_template_embedder(
                model.trunk_module.template_embedder,
                pair,
                emb.template_input,
                template_input_masks=template_input_masks,
                token_pair_mask=token_pair_mask,
                tensors=tensors,
                prefix=f"trunk.recycle_{recycle_idx}.template",
            )
            pair = _capture_msa_module(
                model.trunk_module.msa_module,
                single,
                pair,
                emb.msa_input,
                token_pair_mask=token_pair_mask,
                msa_mask=msa_mask,
                tensors=tensors,
                prefix=f"trunk.recycle_{recycle_idx}.msa",
            )
        elif capture_detail == "pairformer":
            if capture_pre_pairformer:
                _record(tensors, f"trunk.recycle_{recycle_idx}.single_after_recycle", single)
                _record(tensors, f"trunk.recycle_{recycle_idx}.pair_after_recycle", pair)
            pair = model.trunk_module.template_embedder(
                pair,
                emb.template_input,
                template_input_masks=template_input_masks,
                token_pair_mask=token_pair_mask,
            )
            if capture_pre_pairformer:
                _record(tensors, f"trunk.recycle_{recycle_idx}.pair_after_template", pair)
            pair = model.trunk_module.msa_module(
                single,
                pair,
                emb.msa_input,
                token_pair_mask=token_pair_mask,
                msa_mask=msa_mask,
            )
            if capture_pre_pairformer:
                _record(tensors, f"trunk.recycle_{recycle_idx}.pair_after_msa", pair)
        else:
            raise ValueError(f"unknown capture_detail: {capture_detail}")
        single, pair = _capture_pairformer_stack(
            model.trunk_module.pairformer_stack,
            single,
            pair,
            pair_mask=token_pair_mask,
            single_mask=token_single_mask,
            tensors=tensors,
            prefix=f"trunk.recycle_{recycle_idx}.pairformer",
            block_start=pairformer_block_start,
            block_end=pairformer_block_end,
            stop_after_block=stop_after_pairformer_block,
        )
        prev_single, prev_pair = single, pair

    out = TrunkOutputs(
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
    if record_outputs:
        _record_dataclass(tensors, "trunk.outputs", out)
    return out


def capture_cache(
    model: ChaiMLX,
    trunk_out: TrunkOutputs,
    *,
    tensors: dict[str, np.ndarray],
) -> DiffusionCache:
    cache = model.diffusion_module.prepare_cache(trunk_out)
    _record(tensors, "cache.s_static", cache.s_static)
    _record(tensors, "cache.z_cond", cache.z_cond)
    for i, bias in enumerate(cache.pair_biases):
        _record(tensors, f"cache.pair_biases.{i}", bias)
    _record(tensors, "cache.blocked_pair_base", cache.blocked_pair_base)
    _record(tensors, "cache.atom_cond", cache.atom_cond)
    _record(tensors, "cache.atom_single_cond", cache.atom_single_cond)
    return cache


def _capture_diffusion_transformer(
    transformer,
    x: mx.array,
    s_cond: mx.array,
    pair_biases: tuple[mx.array, ...],
    *,
    tensors: dict[str, np.ndarray],
    prefix: str,
) -> mx.array:
    b, ds, n, d = x.shape
    out = x
    for i, (block, pair_bias) in enumerate(zip(transformer.blocks, pair_biases)):
        bias = mx.broadcast_to(pair_bias[:, None, :, :, :], (b, ds, *pair_bias.shape[1:]))
        out = block(
            out.reshape(b * ds, n, d),
            s_cond.reshape(b * ds, n, s_cond.shape[-1]),
            bias.reshape(b * ds, *pair_bias.shape[1:]),
        ).reshape(b, ds, n, d)
        _record(tensors, f"{prefix}.block_{i}.token_repr", out)
    return out


def capture_denoise(
    model: ChaiMLX,
    cache: DiffusionCache,
    coords: mx.array,
    sigma: mx.array,
    *,
    tensors: dict[str, np.ndarray],
) -> mx.array:
    trunk = cache.trunk_outputs
    structure = cache.structure_inputs
    sigma = sigma.astype(mx.float32)
    sigma_sq = sigma * sigma
    sigma_data_sq = model.cfg.diffusion.sigma_data ** 2
    c_in = (sigma_sq + sigma_data_sq) ** -0.5
    c_skip = sigma_data_sq / (sigma_sq + sigma_data_sq)
    c_out = sigma * model.cfg.diffusion.sigma_data / mx.sqrt(sigma_sq + sigma_data_sq)
    num_samples = coords.shape[1]
    scaled_coords = coords * c_in[:, :, None, None]
    s_cond = model.diffusion_module.diffusion_conditioning.with_sigma(cache.s_static, sigma)
    x = model.diffusion_module.structure_cond_to_token_structure_proj(s_cond)
    enc_tokens, atom_repr, encoder_pair = model.diffusion_module.atom_attention_encoder(
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
    )
    x = x + enc_tokens
    _record(tensors, "denoise.scaled_coords", scaled_coords)
    _record(tensors, "denoise.s_cond", s_cond)
    _record(tensors, "denoise.encoder_tokens", enc_tokens)
    _record(tensors, "denoise.encoder_atom_repr", atom_repr)
    _record(tensors, "denoise.token_repr_pre_transformer", x)
    x = _capture_diffusion_transformer(
        model.diffusion_module.diffusion_transformer,
        x,
        s_cond,
        cache.pair_biases,
        tensors=tensors,
        prefix="denoise.transformer",
    )
    x = model.diffusion_module.post_attn_layernorm(x)
    decoder_cond = model.diffusion_module.post_atom_cond_layernorm(
        mx.broadcast_to(
            cache.atom_single_cond[:, None, :, :],
            (coords.shape[0], num_samples, *cache.atom_single_cond.shape[1:]),
        )
    )
    pos_updates = model.diffusion_module.atom_attention_decoder(
        x,
        atom_repr,
        decoder_cond,
        encoder_pair,
        structure.atom_token_index,
        structure.atom_exists_mask,
        structure.atom_kv_indices,
        structure.block_atom_pair_mask,
    )
    denoised = c_skip[:, :, None, None] * coords + c_out[:, :, None, None] * pos_updates
    _record(tensors, "denoise.token_repr_post_transformer", x)
    _record(tensors, "denoise.decoder_cond", decoder_cond)
    _record(tensors, "denoise.pos_updates", pos_updates)
    _record(tensors, "denoise.output", denoised)
    return denoised


def capture_confidence(
    model: ChaiMLX,
    trunk_out: TrunkOutputs,
    coords: mx.array,
    *,
    tensors: dict[str, np.ndarray],
) -> ConfidenceOutputs:
    if coords.ndim == 4:
        if coords.shape[1] != 1:
            raise ValueError("layer parity capture expects coords with a single sample")
        coords = coords[:, 0]

    structure = trunk_out.structure_inputs
    pair = trunk_out.pair_trunk
    row, col = mx.split(model.confidence_head.single_to_pair_proj(trunk_out.single_initial), 2, axis=-1)
    pair = pair + row[:, :, None, :] + col[:, None, :, :]
    ref_coords = representative_atom_coords(coords, structure.token_reference_atom_index)
    dists = cdist(ref_coords)
    dist_bins = one_hot_binned(dists, model.cfg.confidence.distance_bin_edges)
    pair = pair + model.confidence_head.atom_distance_bins_projection(dist_bins)
    token_mask = structure.token_exists_mask.astype(mx.float32)
    s = trunk_out.single_trunk
    z = pair
    _record(tensors, "confidence.pair_input", z)
    for i, block in enumerate(model.confidence_head.blocks):
        z, s = block(z, s, pair_mask=structure.token_pair_mask, single_mask=token_mask)
        assert s is not None
        _record(tensors, f"confidence.block_{i}.pair", z)
        _record(tensors, f"confidence.block_{i}.single", s)
    s = s * token_mask[..., None]
    s_normed = model.confidence_head.single_output_norm(s)
    z_normed = model.confidence_head.pair_output_norm(z)
    plddt_token = model.confidence_head.plddt_projection(s_normed)
    plddt_logits = expand_plddt_to_atoms(
        plddt_token,
        structure.atom_token_index,
        structure.atom_within_token_index,
        model.cfg.confidence.plddt_bins,
    )
    pae_logits = model.confidence_head.pae_projection(z_normed)
    z_sym = z_normed + z_normed.transpose(0, 2, 1, 3)
    pde_logits = model.confidence_head.pde_projection(z_sym)
    out = ConfidenceOutputs(
        pae_logits=pae_logits,
        pde_logits=pde_logits,
        plddt_logits=plddt_logits,
        token_single=s,
        token_pair=z,
        structure_inputs=structure,
    )
    _record_dataclass(tensors, "confidence.outputs", out)
    return out


def capture_ranking(
    model: ChaiMLX,
    confidence: ConfidenceOutputs,
    coords: mx.array,
    *,
    tensors: dict[str, np.ndarray],
) -> RankingOutputs:
    if coords.ndim == 4:
        if coords.shape[1] != 1:
            raise ValueError("layer parity capture expects coords with a single sample")
        coords = coords[:, 0]
    out = model.ranker(confidence, coords, confidence.structure_inputs)
    _record_dataclass(tensors, "ranking.outputs", out)
    return out


def capture_model_tensors(
    model: ChaiMLX,
    ctx: FeatureContext,
    extras: dict[str, mx.array],
    *,
    recycles: int,
) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    emb = capture_embeddings(model, ctx, tensors=tensors)
    trunk_out = capture_trunk(model, emb, recycles=recycles, tensors=tensors)
    cache = capture_cache(model, trunk_out, tensors=tensors)

    coords = extras.get("coords")
    if coords is not None:
        sigma = extras.get("sigma")
        if sigma is not None:
            capture_denoise(
                model,
                cache,
                coords,
                sigma,
                tensors=tensors,
            )
        confidence = capture_confidence(model, trunk_out, coords, tensors=tensors)
        capture_ranking(model, confidence, coords, tensors=tensors)
    return tensors


def compare_tensors(
    reference: dict[str, np.ndarray],
    actual: dict[str, np.ndarray],
    *,
    tol: float,
    key_pattern: str | None,
    fail_on_extra_actual_keys: bool,
) -> int:
    pattern = re.compile(key_pattern) if key_pattern is not None else None
    ref_keys = sorted(key for key in reference if pattern is None or pattern.search(key))
    actual_keys = sorted(key for key in actual if pattern is None or pattern.search(key))
    missing = sorted(set(ref_keys) - set(actual_keys))
    extra = sorted(set(actual_keys) - set(ref_keys))
    failures = 0

    if missing:
        failures += len(missing)
        print(f"[FAIL] missing MLX tensors ({len(missing)}): {', '.join(missing[:10])}")
    if extra and fail_on_extra_actual_keys:
        failures += len(extra)
        print(f"[FAIL] extra MLX tensors ({len(extra)}): {', '.join(extra[:10])}")
    elif extra:
        print(f"[*] ignoring extra MLX tensors ({len(extra)})")

    for key in sorted(set(ref_keys) & set(actual_keys)):
        ref = reference[key]
        got = actual[key]
        if ref.shape != got.shape:
            failures += 1
            print(f"[FAIL] {key}: shape mismatch reference={ref.shape} mlx={got.shape}")
            continue
        diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
        max_diff = float(diff.max()) if diff.size else 0.0
        mean_diff = float(diff.mean()) if diff.size else 0.0
        if max_diff > tol:
            failures += 1
            print(f"[FAIL] {key}: max={max_diff:.3e} mean={mean_diff:.3e}")
        else:
            print(f"[PASS] {key}: max={max_diff:.3e} mean={mean_diff:.3e}")
    return failures


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare MLX intermediate tensors against a reference dump",
    )
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, default=None)
    parser.add_argument("--write-mlx-dump", type=Path, default=None)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--key-pattern", type=str, default=None)
    parser.add_argument(
        "--fail-on-extra-actual-keys",
        action="store_true",
        help="Treat reference/MLX keyset mismatches as failures in both directions",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    # strict=False: confidence_head.atom_distance_v_bins is a hardcoded
    # config tensor (distance bin edges), not a learned weight.  It lives in
    # the model but is absent from the safetensors index.
    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False)
    ctx, extras = load_feature_context(args.input_npz)
    actual = capture_model_tensors(
        model,
        ctx,
        extras,
        recycles=args.recycles,
    )

    if args.write_mlx_dump is not None:
        args.write_mlx_dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.write_mlx_dump, **actual)
        print(f"[*] Wrote MLX tensor dump to {args.write_mlx_dump}")

    if args.reference_npz is None:
        print("[*] No reference dump provided; capture complete")
        return

    reference = _npz_dict(args.reference_npz)
    failures = compare_tensors(
        reference,
        actual,
        tol=args.tol,
        key_pattern=args.key_pattern,
        fail_on_extra_actual_keys=args.fail_on_extra_actual_keys,
    )
    if failures:
        raise SystemExit(1)
    print("[PASS] all compared tensors matched within tolerance")


if __name__ == "__main__":  # pragma: no cover
    main()
