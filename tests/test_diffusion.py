"""Regression tests for diffusion top-level dataflow.

These tests are intentionally narrow: they do not try to prove numerical parity
with TorchScript. They protect specific denoise wiring decisions that have
already caused structural bugs, such as feeding the token-structure projection
from the wrong representation.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

import numpy as np

import mlx.core as mx
from mlx.utils import tree_flatten

from chai_mlx.config import ChaiConfig
from chai_mlx.data.types import DiffusionCache, TrunkOutputs
from chai_mlx.model.core import ChaiMLX, _cast_weights
from chai_mlx.model.diffusion import DiffusionModule
from tests.helpers import make_structure_inputs


class _ConditioningStub:
    def __init__(self, s_cond: mx.array) -> None:
        self.s_cond = s_cond

    def with_sigma(self, s_static: mx.array, sigma: mx.array) -> mx.array:
        return self.s_cond


class _ProjectionRecorder:
    def __init__(self, out_dim: int) -> None:
        self.out_dim = out_dim
        self.last_input: mx.array | None = None

    def __call__(self, x: mx.array) -> mx.array:
        self.last_input = x
        return mx.zeros((*x.shape[:-1], self.out_dim), dtype=mx.float32)


class _EncoderStub:
    def __init__(self, token_dim: int, atom_dim: int) -> None:
        self.token_dim = token_dim
        self.atom_dim = atom_dim

    def __call__(
        self,
        atom_cond: mx.array,
        atom_single_cond: mx.array,
        blocked_pair_base: mx.array,
        atom_token_index: mx.array,
        atom_exists_mask: mx.array,
        scaled_coords: mx.array,
        atom_kv_indices: mx.array | None,
        block_atom_pair_mask: mx.array | None,
        *,
        num_tokens: int,
        num_samples: int,
    ) -> tuple[mx.array, mx.array, mx.array]:
        batch, n_atoms = atom_cond.shape[:2]
        enc_tokens = mx.zeros((batch, num_samples, num_tokens, self.token_dim), dtype=mx.float32)
        atom_repr = mx.zeros((batch, num_samples, n_atoms, self.atom_dim), dtype=mx.float32)
        encoder_pair = mx.zeros((1,), dtype=mx.float32)
        return enc_tokens, atom_repr, encoder_pair


class _Identity:
    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        return x


class _DecoderStub:
    def __call__(
        self,
        x: mx.array,
        atom_repr: mx.array,
        decoder_cond: mx.array,
        encoder_pair: mx.array,
        atom_token_index: mx.array,
        atom_exists_mask: mx.array,
        atom_kv_indices: mx.array | None,
        block_atom_pair_mask: mx.array | None,
    ) -> mx.array:
        return mx.zeros((*atom_repr.shape[:-1], 3), dtype=mx.float32)


class _FakeDiffusionModule:
    def __init__(self, cfg: ChaiConfig, s_cond: mx.array, recorder: _ProjectionRecorder) -> None:
        self.cfg = cfg
        self.diffusion_conditioning = _ConditioningStub(s_cond)
        self.structure_cond_to_token_structure_proj = recorder
        self.atom_attention_encoder = _EncoderStub(token_dim=8, atom_dim=4)
        self.diffusion_transformer = _Identity()
        self.post_attn_layernorm = _Identity()
        self.post_atom_cond_layernorm = _Identity()
        self.atom_attention_decoder = _DecoderStub()


def _make_diffusion_trunk(cfg: ChaiConfig, *, dtype: mx.Dtype) -> TrunkOutputs:
    batch_size = 1
    n_tokens = 4
    n_atoms = 32
    atom_blocks = n_atoms // 32

    structure = make_structure_inputs(
        batch_size=batch_size,
        n_tokens=n_tokens,
        n_atoms=n_atoms,
        msa_depth=2,
        n_templates=1,
    )
    q_idx = mx.arange(n_atoms, dtype=mx.int32).reshape(batch_size, atom_blocks, 32)
    kv_idx = mx.clip(
        q_idx[:, :, :1] + mx.arange(128, dtype=mx.int32)[None, None, :] - 48,
        0,
        n_atoms - 1,
    )
    block_mask = mx.ones((batch_size, atom_blocks, 32, 128), dtype=mx.float32)
    structure = dc_replace(
        structure,
        atom_q_indices=q_idx,
        atom_kv_indices=kv_idx,
        block_atom_pair_mask=block_mask,
    )

    return TrunkOutputs(
        single_initial=mx.zeros((batch_size, n_tokens, cfg.hidden.token_single), dtype=dtype),
        single_trunk=mx.zeros((batch_size, n_tokens, cfg.hidden.token_single), dtype=dtype),
        single_structure=mx.zeros((batch_size, n_tokens, cfg.hidden.token_single), dtype=dtype),
        pair_initial=mx.zeros(
            (batch_size, n_tokens, n_tokens, cfg.hidden.token_pair), dtype=dtype
        ),
        pair_trunk=mx.zeros(
            (batch_size, n_tokens, n_tokens, cfg.hidden.token_pair), dtype=dtype
        ),
        pair_structure=mx.zeros(
            (batch_size, n_tokens, n_tokens, cfg.hidden.token_pair), dtype=dtype
        ),
        atom_single_structure_input=mx.zeros(
            (batch_size, n_atoms, cfg.hidden.atom_single), dtype=dtype
        ),
        atom_pair_structure_input=mx.zeros(
            (
                batch_size,
                atom_blocks,
                cfg.atom_blocks.query_block,
                cfg.atom_blocks.kv_block,
                cfg.hidden.atom_pair,
            ),
            dtype=dtype,
        ),
        msa_input=mx.zeros((batch_size, 2, n_tokens, cfg.hidden.msa), dtype=dtype),
        template_input=mx.zeros(
            (batch_size, 1, n_tokens, n_tokens, cfg.hidden.template_pair), dtype=dtype
        ),
        structure_inputs=structure,
    )


def test_denoise_projects_sigma_conditioned_single_repr() -> None:
    """Ensure ``denoise`` projects from ``s_cond`` rather than trunk structure."""
    cfg = ChaiConfig()
    structure = make_structure_inputs(n_tokens=4, n_atoms=8)
    trunk = TrunkOutputs(
        single_initial=mx.zeros((1, 4, 8), dtype=mx.float32),
        single_trunk=mx.zeros((1, 4, 8), dtype=mx.float32),
        single_structure=mx.full((1, 4, 6), 3.0, dtype=mx.float32),
        pair_initial=mx.zeros((1, 4, 4, 8), dtype=mx.float32),
        pair_trunk=mx.zeros((1, 4, 4, 8), dtype=mx.float32),
        pair_structure=mx.zeros((1, 4, 4, 8), dtype=mx.float32),
        atom_single_structure_input=mx.zeros((1, 8, 4), dtype=mx.float32),
        atom_pair_structure_input=mx.zeros((1, 1, 32, 128, 16), dtype=mx.float32),
        msa_input=mx.zeros((1, 1, 4, 8), dtype=mx.float32),
        template_input=mx.zeros((1, 1, 4, 4, 8), dtype=mx.float32),
        structure_inputs=structure,
    )
    s_cond = mx.full((1, 2, 4, 6), 7.0, dtype=mx.float32)
    recorder = _ProjectionRecorder(out_dim=8)
    cache = DiffusionCache(
        s_static=mx.zeros((1, 4, 6), dtype=mx.float32),
        z_cond=mx.zeros((1, 4, 4, 8), dtype=mx.float32),
        pair_biases=(mx.zeros((1, 1, 4, 4), dtype=mx.float32),),
        blocked_pair_base=mx.zeros((1, 1, 32, 128, 16), dtype=mx.float32),
        atom_cond=mx.zeros((1, 8, 4), dtype=mx.float32),
        atom_single_cond=mx.zeros((1, 8, 4), dtype=mx.float32),
        trunk_outputs=trunk,
        structure_inputs=structure,
    )
    coords = mx.zeros((1, 2, 8, 3), dtype=mx.float32)
    sigma = mx.full((1, 2), 0.5, dtype=mx.float32)

    out = DiffusionModule.denoise(
        _FakeDiffusionModule(cfg, s_cond, recorder),
        cache,
        coords,
        sigma,
    )
    mx.eval(out, recorder.last_input)

    assert recorder.last_input is not None
    np.testing.assert_allclose(
        np.array(recorder.last_input, copy=False),
        np.array(s_cond, copy=False),
    )


def test_cast_weights_keeps_diffusion_module_fp32() -> None:
    model = ChaiMLX(ChaiConfig(compute_dtype="bfloat16"))
    _cast_weights(model, mx.bfloat16)

    params = {
        key: value
        for key, value in tree_flatten(model.parameters())
        if isinstance(value, mx.array)
    }
    diffusion_dtypes = {
        key: value.dtype
        for key, value in params.items()
        if key.startswith("diffusion_module.")
    }
    trunk_dtypes = {
        key: value.dtype
        for key, value in params.items()
        if key.startswith("trunk_module.")
    }

    assert diffusion_dtypes
    assert trunk_dtypes
    assert all(dtype == mx.float32 for dtype in diffusion_dtypes.values())
    assert any(dtype == mx.bfloat16 for dtype in trunk_dtypes.values())


def test_prepare_cache_upcasts_trunk_conditioning_to_fp32() -> None:
    cfg = ChaiConfig(compute_dtype="bfloat16")
    module = DiffusionModule(cfg)
    trunk = _make_diffusion_trunk(cfg, dtype=mx.bfloat16)

    cache = module.prepare_cache(trunk)
    tensors = (
        cache.s_static,
        cache.z_cond,
        cache.blocked_pair_base,
        cache.atom_cond,
        cache.atom_single_cond,
        *cache.pair_biases,
    )
    mx.eval(*tensors)

    assert trunk.single_trunk.dtype == mx.bfloat16
    assert all(t.dtype == mx.float32 for t in tensors)


def test_diffusion_step_returns_fp32_under_bfloat16_compute_dtype() -> None:
    cfg = ChaiConfig(compute_dtype="bfloat16")
    module = DiffusionModule(cfg)
    trunk = _make_diffusion_trunk(cfg, dtype=mx.bfloat16)
    cache = module.prepare_cache(trunk)
    coords = module.init_noise(1, 1, trunk.structure_inputs)
    sigma_curr, sigma_next, gamma = next(module.schedule(num_steps=2))

    out = module.diffusion_step(cache, coords, sigma_curr, sigma_next, gamma)
    mx.eval(out)

    assert out.dtype == mx.float32
