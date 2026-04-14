"""Trace the real post-trunk downstream pipeline in chai-lab and MLX.

This harness feeds the same chai-lab reference trunk outputs into:

- chai-lab's downstream path (cache prep + diffusion loop) in fp32
- the MLX downstream path (cache prep + init_noise + diffusion_step) in fp32

It uses a shared sampler randomness stream so that any divergence comes from
cache prep / schedule / downstream math rather than RNG implementation
differences.

It also runs a quick control that compares the MLX runtime cache prep against
the old manual replay path from ``deep_denoise_trace.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists():
    sys.path.insert(0, str(LOCAL_CHAI_LAB))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from chai_lab.chai1 import DiffusionConfig, _component_moved_to  # type: ignore[import-not-found]
from chai_lab.model.diffusion_schedules import InferenceNoiseSchedule  # type: ignore[import-not-found]
from chai_lab.model.utils import calc_centroid, random_rotations  # type: ignore[import-not-found]
from chai_lab.utils.tensor_utils import set_seed  # type: ignore[import-not-found]

from chai_mlx import ChaiMLX
from chai_mlx.data.types import StructureInputs
from chai_mlx.utils import masked_mean, resolve_dtype
from deep_denoise_trace import (
    _to_torch,
    _torch_gather_tokens_to_atoms,
    _torch_layer_norm,
    _torch_linear_module,
    _trace_mlx_denoise,
)
from layer_parity import _npz_dict, load_feature_context
from stage_isolation_parity import reconstruct_trunk_outputs


def _mx_np(x: mx.array) -> np.ndarray:
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    mx.eval(x)
    return np.array(x, copy=False)


def _pt_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _record_mx(store: OrderedDict[str, np.ndarray], name: str, value: mx.array) -> None:
    store[name] = _mx_np(value)


def _record_pt(store: OrderedDict[str, np.ndarray], name: str, value: torch.Tensor) -> None:
    store[name] = _pt_np(value)


def _broadcast_step_scalar(value: float, batch_size: int, num_samples: int) -> mx.array:
    return mx.full((batch_size, num_samples), value, dtype=mx.float32)


def _compare_stats(diff: np.ndarray) -> tuple[float, float, float]:
    if diff.size == 0:
        return 0.0, 0.0, 0.0
    return float(diff.max()), float(diff.mean()), float(np.quantile(diff, 0.99))


def _ca_median(coords: np.ndarray, structure: StructureInputs) -> float:
    atom_mask = np.array(structure.atom_exists_mask.astype(mx.float32))[0] > 0.5
    token_centre_atom_idx = getattr(structure, "token_centre_atom_index", None)
    if token_centre_atom_idx is None:
        token_centre_atom_idx = structure.token_reference_atom_index
    token_centre_atom_idx = np.array(token_centre_atom_idx)[0]
    token_mask = np.array(structure.token_exists_mask.astype(mx.float32))[0] > 0.5
    sample = coords[0, 0]
    ca = []
    for token_idx in np.where(token_mask)[0]:
        centre_atom = int(token_centre_atom_idx[token_idx])
        if 0 <= centre_atom < len(sample) and atom_mask[centre_atom]:
            ca.append(sample[centre_atom])
    ca = np.array(ca, dtype=np.float64)
    if len(ca) < 2:
        return float("nan")
    d = np.sqrt(np.sum(np.diff(ca, axis=0) ** 2, axis=-1))
    return float(np.median(d))


@dataclass(frozen=True)
class SharedRandomness:
    init_noise: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    noise: np.ndarray


def _sample_shared_randomness(
    *,
    batch_size: int,
    num_samples: int,
    num_atoms: int,
    num_steps: int,
    seed: int,
) -> SharedRandomness:
    flat_batch = batch_size * num_samples
    set_seed([seed])
    init_noise = torch.randn((batch_size, num_samples, num_atoms, 3), dtype=torch.float32)
    rotations = []
    translations = []
    noise = []
    for _ in range(num_steps - 1):
        rotations.append(_pt_np(random_rotations(flat_batch, dtype=torch.float32, device=torch.device("cpu"))))
        translations.append(_pt_np(torch.randn((flat_batch, 1, 3), dtype=torch.float32)))
        noise.append(_pt_np(torch.randn((batch_size, num_samples, num_atoms, 3), dtype=torch.float32)))
    return SharedRandomness(
        init_noise=_pt_np(init_noise),
        rotations=np.stack(rotations, axis=0),
        translations=np.stack(translations, axis=0),
        noise=np.stack(noise, axis=0),
    )


def _trace_mask(name: str, structure: StructureInputs, shape: tuple[int, ...], num_samples: int) -> np.ndarray | None:
    atom_mask = np.array(structure.atom_exists_mask, copy=False).astype(bool)
    token_mask = np.array(structure.token_exists_mask, copy=False).astype(bool)
    token_pair_mask = np.array(structure.token_pair_mask, copy=False).astype(bool)
    block_pair_mask = np.array(structure.block_atom_pair_mask, copy=False).astype(bool)

    if name.startswith("boundary.trunk.single") or name in {
        "cache.s_static",
        "sample.step_001.s_cond",
    }:
        base = token_mask[..., None] if len(shape) == 3 else token_mask[:, None, :, None]
        return np.broadcast_to(base, shape)
    if name.startswith("boundary.trunk.pair") or name == "cache.z_cond":
        return np.broadcast_to(token_pair_mask[..., None], shape)
    if name.startswith("cache.pair_biases."):
        return np.broadcast_to(token_pair_mask[:, None, :, :], shape)
    if name in {
        "boundary.trunk.atom_single_structure_input",
        "cache.atom_cond",
        "cache.atom_single_cond",
    }:
        return np.broadcast_to(atom_mask[..., None], shape)
    if name in {
        "boundary.trunk.atom_pair_structure_input",
        "cache.blocked_pair_base",
    }:
        return np.broadcast_to(block_pair_mask[..., None], shape)
    if name.startswith("schedule."):
        return None
    if "coords" in name or "atom_pos" in name or "atom_coords" in name or "structure_tensor" in name:
        if len(shape) == 4:
            return np.broadcast_to(atom_mask[:, None, :, None], shape)
        if len(shape) == 3:
            return np.broadcast_to(atom_mask[:, :, None], shape)
    return None


def _compare_checkpoint_traces(
    torch_trace: OrderedDict[str, np.ndarray],
    mlx_trace: OrderedDict[str, np.ndarray],
    structure: StructureInputs,
    *,
    num_samples: int,
    jump_threshold: float,
) -> tuple[int, tuple[str, str, float, float, float] | None]:
    failures = 0
    first_jump = None
    for name, ref in torch_trace.items():
        got = mlx_trace.get(name)
        if got is None:
            print(f"[MISSING] {name}")
            failures += 1
            continue
        if ref.shape != got.shape:
            print(f"[SHAPE] {name}: torch={ref.shape} mlx={got.shape}")
            failures += 1
            continue
        diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
        full_max, full_mean, full_p99 = _compare_stats(diff)
        mask = _trace_mask(name, structure, diff.shape, num_samples)
        cmp_label = "full"
        cmp_max = full_max
        cmp_mean = full_mean
        cmp_p99 = full_p99
        invalid_summary = ""
        if mask is not None and np.any(mask):
            valid = diff[mask]
            cmp_label = "valid"
            cmp_max, cmp_mean, cmp_p99 = _compare_stats(valid)
            invalid = diff[~mask]
            if invalid.size:
                invalid_summary = (
                    f"  invalid_mean={float(invalid.mean()):9.4e}"
                    f"  invalid_max={float(invalid.max()):9.4e}"
                )
        print(
            f"{name:<44} {cmp_label}_max={cmp_max:9.4e}  "
            f"{cmp_label}_mean={cmp_mean:9.4e}  {cmp_label}_p99={cmp_p99:9.4e}  "
            f"full_max={full_max:9.4e}{invalid_summary}  shape={ref.shape}"
        )
        if first_jump is None and not name.startswith("schedule.") and cmp_p99 >= jump_threshold:
            first_jump = (name, cmp_label, cmp_max, cmp_mean, cmp_p99)
    extra = sorted(set(mlx_trace) - set(torch_trace))
    if extra:
        print(f"[INFO] extra MLX checkpoints: {len(extra)}")
        for name in extra[:20]:
            print(f"  {name}")
    if first_jump is None:
        print(f"\nNo checkpoint exceeded threshold {jump_threshold:.3g}.")
    else:
        name, label, cmp_max, cmp_mean, cmp_p99 = first_jump
        print(
            f"\nFirst checkpoint over threshold {jump_threshold:.3g}: "
            f"{name} ({label}_max={cmp_max:.4e}, {label}_mean={cmp_mean:.4e}, {label}_p99={cmp_p99:.4e})"
        )
    return failures, first_jump


def _record_boundary(store: OrderedDict[str, np.ndarray], trunk, *, framework: str) -> None:
    del framework
    store["boundary.trunk.single_initial"] = _mx_np(trunk.single_initial)
    store["boundary.trunk.single_trunk"] = _mx_np(trunk.single_trunk)
    store["boundary.trunk.single_structure"] = _mx_np(trunk.single_structure)
    store["boundary.trunk.pair_trunk"] = _mx_np(trunk.pair_trunk)
    store["boundary.trunk.pair_structure"] = _mx_np(trunk.pair_structure)
    store["boundary.trunk.atom_single_structure_input"] = _mx_np(trunk.atom_single_structure_input)
    store["boundary.trunk.atom_pair_structure_input"] = _mx_np(trunk.atom_pair_structure_input)


def _record_cache_fields(store: OrderedDict[str, np.ndarray], cache) -> None:
    _record_mx(store, "cache.s_static", cache.s_static)
    _record_mx(store, "cache.z_cond", cache.z_cond)
    for i, bias in enumerate(cache.pair_biases):
        _record_mx(store, f"cache.pair_biases.{i}", bias)
    _record_mx(store, "cache.blocked_pair_base", cache.blocked_pair_base)
    _record_mx(store, "cache.atom_cond", cache.atom_cond)
    _record_mx(store, "cache.atom_single_cond", cache.atom_single_cond)


def _compare_manual_vs_runtime_cache(
    model: ChaiMLX,
    ref_trunk,
    coords: mx.array,
    sigma: mx.array,
    structure: StructureInputs,
    *,
    jump_threshold: float,
) -> None:
    print("\nQuick cache-prep control: real MLX cache vs old manual replay cache")
    runtime = OrderedDict()
    manual = OrderedDict()

    cache = model.prepare_diffusion_cache(ref_trunk)
    _record_cache_fields(runtime, cache)

    _trace_mlx_denoise(model, manual, ref_trunk, coords, sigma)
    manual = OrderedDict((k, v) for k, v in manual.items() if k.startswith("cache."))

    _compare_checkpoint_traces(runtime, manual, structure, num_samples=int(coords.shape[1]), jump_threshold=jump_threshold)


def _torch_static_inputs(ref_trunk, structure: StructureInputs, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "token_single_initial_repr": _to_torch(ref_trunk.single_structure, device, dtype=torch.float32),
        "token_pair_initial_repr": _to_torch(ref_trunk.pair_structure, device, dtype=torch.float32),
        "token_single_trunk_repr": _to_torch(ref_trunk.single_trunk, device, dtype=torch.float32),
        "token_pair_trunk_repr": _to_torch(ref_trunk.pair_trunk, device, dtype=torch.float32),
        "atom_single_input_feats": _to_torch(ref_trunk.atom_single_structure_input, device, dtype=torch.float32),
        "atom_block_pair_input_feats": _to_torch(ref_trunk.atom_pair_structure_input, device, dtype=torch.float32),
        "atom_single_mask": _to_torch(structure.atom_exists_mask, device, dtype=torch.bool),
        "atom_block_pair_mask": _to_torch(structure.block_atom_pair_mask, device, dtype=torch.bool),
        "token_single_mask": _to_torch(structure.token_exists_mask, device, dtype=torch.bool),
        "block_indices_h": _to_torch(structure.atom_q_indices, device, dtype=torch.long).squeeze(0),
        "block_indices_w": _to_torch(structure.atom_kv_indices, device, dtype=torch.long).squeeze(0),
        "atom_token_indices": _to_torch(structure.atom_token_index, device, dtype=torch.long),
    }


def _torch_cache_from_ref_trunk(
    store: OrderedDict[str, np.ndarray],
    diffusion_module,
    ref_trunk,
    structure: StructureInputs,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    atom_token_index = _to_torch(structure.atom_token_index, device, dtype=torch.long)
    atom_q_indices = _to_torch(structure.atom_q_indices, device, dtype=torch.long)
    atom_kv_indices = _to_torch(structure.atom_kv_indices, device, dtype=torch.long)
    token_pair_mask = _to_torch(structure.token_pair_mask, device, dtype=torch.bool)

    trunk = {
        "single_initial": _to_torch(ref_trunk.single_initial, device, dtype=torch.float32),
        "single_trunk": _to_torch(ref_trunk.single_trunk, device, dtype=torch.float32),
        "single_structure": _to_torch(ref_trunk.single_structure, device, dtype=torch.float32),
        "pair_trunk": _to_torch(ref_trunk.pair_trunk, device, dtype=torch.float32),
        "pair_structure": _to_torch(ref_trunk.pair_structure, device, dtype=torch.float32),
        "atom_single_structure_input": _to_torch(ref_trunk.atom_single_structure_input, device, dtype=torch.float32),
        "atom_pair_structure_input": _to_torch(ref_trunk.atom_pair_structure_input, device, dtype=torch.float32),
    }
    dm = diffusion_module
    enc = dm.atom_attention_encoder

    pair_cat = torch.cat([trunk["pair_trunk"], trunk["pair_structure"]], dim=-1)
    z = dm.diffusion_conditioning.token_pair_proj(pair_cat.float())
    z = z + dm.diffusion_conditioning.pair_trans1(z)
    z = z + dm.diffusion_conditioning.pair_trans2(z)
    z_cond = dm.diffusion_conditioning.pair_ln(z)
    _record_pt(store, "cache.z_cond", z_cond)

    single_cat = torch.cat([trunk["single_structure"], trunk["single_trunk"]], dim=-1)
    s_static = dm.diffusion_conditioning.token_in_proj(single_cat.float())
    _record_pt(store, "cache.s_static", s_static)

    pair_biases = []
    for i in range(16):
        block = getattr(dm.diffusion_transformer.blocks, str(i))
        z_pair_norm = _torch_layer_norm(
            z_cond,
            weight=block.pair_layer_norm.weight.float(),
            bias=block.pair_layer_norm.bias.float(),
            eps=1e-5,
        )
        pair_bias = _torch_linear_module(z_pair_norm, block.pair_linear).permute(0, 3, 1, 2).contiguous()
        pair_bias = pair_bias + torch.where(
            token_pair_mask[:, None],
            torch.zeros_like(pair_bias),
            torch.full_like(pair_bias, -10000.0),
        )
        pair_biases.append(pair_bias)
        _record_pt(store, f"cache.pair_biases.{i}", pair_bias)

    token_atom_pair = enc.token_pair_to_atom_pair(z_cond.float())
    batch = torch.arange(atom_token_index.shape[0], device=device)
    q_token_idx = atom_token_index[batch[:, None, None], atom_q_indices]
    kv_token_idx = atom_token_index[batch[:, None, None], atom_kv_indices]
    blocked_pair_base = token_atom_pair[
        batch[:, None, None, None],
        q_token_idx[:, :, :, None],
        kv_token_idx[:, :, None, :],
    ]
    blocked_pair_base = blocked_pair_base + trunk["atom_pair_structure_input"]
    _record_pt(store, "cache.blocked_pair_base", blocked_pair_base)

    atom_cond = _torch_linear_module(trunk["atom_single_structure_input"], enc.to_atom_cond)
    _record_pt(store, "cache.atom_cond", atom_cond)

    token_proj = enc.token_to_atom_single(trunk["single_trunk"].float())
    token_to_atom = _torch_gather_tokens_to_atoms(token_proj, atom_token_index)
    atom_single_cond = _torch_layer_norm(atom_cond + token_to_atom, eps=1e-5)
    _record_pt(store, "cache.atom_single_cond", atom_single_cond)

    return {
        "s_static": s_static,
        "z_cond": z_cond,
        "pair_biases": pair_biases,
        "blocked_pair_base": blocked_pair_base,
        "atom_cond": atom_cond,
        "atom_single_cond": atom_single_cond,
    }


def _torch_s_cond(diffusion_module, s_static: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    c_noise = torch.log(torch.clamp(sigma.float(), min=torch.finfo(torch.float32).tiny)) * 0.25
    sigma_embed = torch.cos(
        (
            c_noise[..., None] * diffusion_module.diffusion_conditioning.fourier_embedding.weights.float()
            + diffusion_module.diffusion_conditioning.fourier_embedding.bias.float()
        )
        * (2.0 * math.pi)
    )
    fourier_proj_norm = getattr(diffusion_module.diffusion_conditioning.fourier_proj, "0")
    sigma_embed = _torch_layer_norm(
        sigma_embed,
        weight=fourier_proj_norm.weight.float(),
        bias=fourier_proj_norm.bias.float(),
        eps=1e-5,
    )
    sigma_embed = _torch_linear_module(sigma_embed, getattr(diffusion_module.diffusion_conditioning.fourier_proj, "1"))
    s_cond = s_static[:, None, :, :] + sigma_embed[:, :, None, :]
    s_cond = s_cond + diffusion_module.diffusion_conditioning.single_trans1(s_cond)
    s_cond = s_cond + diffusion_module.diffusion_conditioning.single_trans2(s_cond)
    return diffusion_module.diffusion_conditioning.single_ln(s_cond)


def _torch_schedule(num_steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sigmas = InferenceNoiseSchedule(
        s_max=DiffusionConfig.S_tmax,
        s_min=4e-4,
        p=7.0,
        sigma_data=DiffusionConfig.sigma_data,
    ).get_schedule(device=device, num_timesteps=num_steps)
    gammas = torch.where(
        (sigmas >= DiffusionConfig.S_tmin) & (sigmas <= DiffusionConfig.S_tmax),
        torch.tensor(min(DiffusionConfig.S_churn / num_steps, math.sqrt(2.0) - 1.0), dtype=torch.float32, device=device),
        torch.tensor(0.0, dtype=torch.float32, device=device),
    )
    return sigmas.float(), gammas.float()


def _apply_torch_augmentation(
    coords_flat: torch.Tensor,
    atom_mask_flat: torch.Tensor,
    *,
    rotations: np.ndarray,
    translations: np.ndarray,
) -> torch.Tensor:
    centroid = calc_centroid(coords_flat, atom_mask_flat)
    centered = coords_flat - centroid[:, None, :]
    rot = torch.from_numpy(rotations).to(coords_flat.device, dtype=torch.float32)
    trans = torch.from_numpy(translations).to(coords_flat.device, dtype=torch.float32)
    return torch.einsum("bij,baj->bai", rot, centered) + trans


def _run_torch_downstream(
    store: OrderedDict[str, np.ndarray],
    ref_trunk,
    structure: StructureInputs,
    shared: SharedRandomness,
    *,
    num_steps: int,
    num_samples: int,
    device: torch.device,
) -> np.ndarray:
    _record_boundary(store, ref_trunk, framework="torch")
    with torch.no_grad():
        with _component_moved_to("diffusion_module.pt", device=device) as diffusion_module:
            cache = _torch_cache_from_ref_trunk(store, diffusion_module.jit_module, ref_trunk, structure, device)
            sigmas, gammas = _torch_schedule(num_steps, device)
            _record_pt(store, "schedule.sigmas", sigmas.cpu())
            _record_pt(store, "schedule.gammas", gammas.cpu())

            static_inputs = _torch_static_inputs(ref_trunk, structure, device)
            batch_size = int(structure.atom_exists_mask.shape[0])
            num_atoms = int(structure.atom_exists_mask.shape[-1])
            flat_batch = batch_size * num_samples
            atom_mask_flat = _to_torch(structure.atom_exists_mask, device, dtype=torch.bool)
            atom_mask_flat = atom_mask_flat[:, None, :].expand(batch_size, num_samples, num_atoms).reshape(flat_batch, num_atoms)

            atom_pos = sigmas[0] * torch.from_numpy(shared.init_noise).to(device=device, dtype=torch.float32)
            _record_pt(store, "sample.initial_atom_pos", atom_pos)

            for step_idx, (sigma_curr, sigma_next, gamma_curr) in enumerate(zip(sigmas[:-1], sigmas[1:], gammas[:-1]), start=1):
                atom_pos_flat = atom_pos.reshape(flat_batch, num_atoms, 3).contiguous()
                coords_aug = _apply_torch_augmentation(
                    atom_pos_flat,
                    atom_mask_flat,
                    rotations=shared.rotations[step_idx - 1],
                    translations=shared.translations[step_idx - 1],
                )
                noise = DiffusionConfig.S_noise * torch.from_numpy(shared.noise[step_idx - 1]).to(device=device, dtype=torch.float32)
                sigma_hat = sigma_curr + gamma_curr * sigma_curr
                noise_scale = (sigma_hat.square() - sigma_curr.square()).clamp_min(1e-6).sqrt()
                atom_pos_hat = coords_aug.reshape(batch_size, num_samples, num_atoms, 3) + noise * noise_scale
                if step_idx == 1:
                    _record_pt(store, "sample.step_001.coords_before_denoise", atom_pos_hat)
                    sigma_hat_t = torch.full((batch_size, num_samples), float(sigma_hat.item()), dtype=torch.float32, device=device)
                    s_cond = _torch_s_cond(diffusion_module.jit_module, cache["s_static"], sigma_hat_t)
                    _record_pt(store, "sample.step_001.s_cond", s_cond)

                sigma_hat_t = torch.full((batch_size, num_samples), float(sigma_hat.item()), dtype=torch.float32, device=device)
                denoised = diffusion_module.forward(
                    atom_noised_coords=atom_pos_hat.float(),
                    noise_sigma=sigma_hat_t.float(),
                    crop_size=int(static_inputs["token_single_mask"].shape[-1]),
                    **static_inputs,
                )
                if denoised.ndim == 3:
                    denoised = denoised[:, None]
                d_i = (atom_pos_hat - denoised) / sigma_hat_t[:, :, None, None]
                atom_pos = atom_pos_hat + (sigma_next - sigma_hat) * d_i

                if float(sigma_next.item()) != 0.0 and DiffusionConfig.second_order:
                    sigma_next_t = torch.full((batch_size, num_samples), float(sigma_next.item()), dtype=torch.float32, device=device)
                    denoised_next = diffusion_module.forward(
                        atom_noised_coords=atom_pos.float(),
                        noise_sigma=sigma_next_t.float(),
                        crop_size=int(static_inputs["token_single_mask"].shape[-1]),
                        **static_inputs,
                    )
                    if denoised_next.ndim == 3:
                        denoised_next = denoised_next[:, None]
                    d_i_prime = (atom_pos - denoised_next) / sigma_next_t[:, :, None, None]
                    atom_pos = atom_pos + (sigma_next - sigma_hat) * ((d_i_prime + d_i) / 2.0)

                if step_idx in {1, 50, 100, 150, 199}:
                    _record_pt(store, f"trajectory.step_{step_idx:03d}.coords", atom_pos)

            _record_pt(store, "final.atom_coords", atom_pos)
            _record_pt(store, "final.structure_tensor", atom_pos)
            return _pt_np(atom_pos)


@contextlib.contextmanager
def _mlx_shared_randomness(shared: SharedRandomness):
    import chai_mlx.model.diffusion as diffusion_mod

    orig_center_random_augmentation = diffusion_mod.center_random_augmentation
    orig_random_normal = diffusion_mod.mx.random.normal
    step_idx = {"value": 0}
    noise_queue = [shared.init_noise] + [shared.noise[i] for i in range(shared.noise.shape[0])]

    def patched_center_random_augmentation(
        coords: mx.array,
        atom_mask: mx.array,
        *,
        centroid_eps: float = 1e-4,
        translation_scale: float = 1.0,
    ) -> mx.array:
        idx = step_idx["value"]
        centroid = masked_mean(coords, atom_mask, axis=1, keepdims=True, eps=centroid_eps)
        centered = coords - centroid
        rot = mx.array(shared.rotations[idx], dtype=mx.float32)
        trans = translation_scale * mx.array(shared.translations[idx], dtype=mx.float32)
        step_idx["value"] += 1
        return mx.einsum("bij,baj->bai", rot, centered) + trans

    def patched_random_normal(shape):
        if not noise_queue:
            raise RuntimeError(f"Random queue exhausted for shape {shape}")
        arr = noise_queue.pop(0)
        if tuple(shape) != tuple(arr.shape):
            raise RuntimeError(f"Random shape mismatch: expected {arr.shape}, got {shape}")
        return mx.array(arr, dtype=mx.float32)

    diffusion_mod.center_random_augmentation = patched_center_random_augmentation
    diffusion_mod.mx.random.normal = patched_random_normal
    try:
        yield
    finally:
        diffusion_mod.center_random_augmentation = orig_center_random_augmentation
        diffusion_mod.mx.random.normal = orig_random_normal


def _run_mlx_downstream(
    store: OrderedDict[str, np.ndarray],
    model: ChaiMLX,
    ref_trunk,
    structure: StructureInputs,
    shared: SharedRandomness,
    *,
    num_steps: int,
    num_samples: int,
) -> np.ndarray:
    _record_boundary(store, ref_trunk, framework="mlx")

    cache = model.prepare_diffusion_cache(ref_trunk)
    _record_cache_fields(store, cache)

    schedule = list(model.schedule(num_steps=num_steps))
    sigmas = mx.array([float(sigma_curr) for sigma_curr, _, _ in schedule] + [float(schedule[-1][1])], dtype=mx.float32)
    gammas = mx.array([float(gamma) for _, _, gamma in schedule] + [0.0], dtype=mx.float32)
    _record_mx(store, "schedule.sigmas", sigmas)
    _record_mx(store, "schedule.gammas", gammas)

    dm = model.diffusion_module
    orig_denoise = dm.denoise
    current_step = {"value": 0, "captured": False}

    def traced_denoise(cache_arg, coords, sigma, *, use_kernel: bool = False):
        if current_step["value"] == 1 and not current_step["captured"]:
            _record_mx(store, "sample.step_001.coords_before_denoise", coords)
            s_cond = dm.diffusion_conditioning.with_sigma(cache_arg.s_static, sigma.astype(mx.float32))
            _record_mx(store, "sample.step_001.s_cond", s_cond)
            current_step["captured"] = True
        return orig_denoise(cache_arg, coords, sigma, use_kernel=use_kernel)

    dm.denoise = traced_denoise  # type: ignore[method-assign]
    try:
        with _mlx_shared_randomness(shared):
            batch_size = int(structure.atom_exists_mask.shape[0])
            coords = model.init_noise(batch_size, num_samples, structure).astype(mx.float32)
            _record_mx(store, "sample.initial_atom_pos", coords)
            for step_idx, (sigma_curr, sigma_next, gamma) in enumerate(schedule, start=1):
                current_step["value"] = step_idx
                current_step["captured"] = False
                coords = model.diffusion_step(cache, coords, sigma_curr, sigma_next, gamma)
                mx.eval(coords)
                if step_idx in {1, 50, 100, 150, 199}:
                    _record_mx(store, f"trajectory.step_{step_idx:03d}.coords", coords)
            _record_mx(store, "final.atom_coords", coords)
            _record_mx(store, "final.structure_tensor", coords)
            return _mx_np(coords)
    finally:
        dm.denoise = orig_denoise  # type: ignore[method-assign]


def _write_npz(path: Path | None, trace: OrderedDict[str, np.ndarray], label: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **trace)
    print(f"Wrote {label}: {path}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trace the real downstream pipeline from reference trunk outputs")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-device", default="cpu")
    parser.add_argument("--jump-threshold", type=float, default=1e-5)
    parser.add_argument("--write-torch-dump", type=Path, default=None)
    parser.add_argument("--write-mlx-dump", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False, compute_dtype="float32")
    ctx, extras = load_feature_context(args.input_npz)
    ref = _npz_dict(args.reference_npz)
    ref_trunk = reconstruct_trunk_outputs(ref, ctx.structure_inputs, dtype=mx.float32)

    coords = extras.get("coords")
    sigma = extras.get("sigma")
    if coords is None or sigma is None:
        raise ValueError("input NPZ must contain coords and sigma for the quick cache control")

    _compare_manual_vs_runtime_cache(
        model,
        ref_trunk,
        coords.astype(mx.float32),
        sigma.astype(mx.float32),
        ctx.structure_inputs,
        jump_threshold=args.jump_threshold,
    )

    batch_size = int(ctx.structure_inputs.atom_exists_mask.shape[0])
    num_atoms = int(ctx.structure_inputs.atom_exists_mask.shape[-1])
    shared = _sample_shared_randomness(
        batch_size=batch_size,
        num_samples=args.num_samples,
        num_atoms=num_atoms,
        num_steps=args.num_steps,
        seed=args.seed,
    )

    torch_trace: OrderedDict[str, np.ndarray] = OrderedDict()
    mlx_trace: OrderedDict[str, np.ndarray] = OrderedDict()

    torch_coords = _run_torch_downstream(
        torch_trace,
        ref_trunk,
        ctx.structure_inputs,
        shared,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        device=torch.device(args.torch_device),
    )
    mlx_coords = _run_mlx_downstream(
        mlx_trace,
        model,
        ref_trunk,
        ctx.structure_inputs,
        shared,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
    )

    _write_npz(args.write_torch_dump, torch_trace, "torch downstream trace")
    _write_npz(args.write_mlx_dump, mlx_trace, "MLX downstream trace")

    print("\nCheckpoint comparison: chai-lab downstream vs MLX downstream")
    _compare_checkpoint_traces(
        torch_trace,
        mlx_trace,
        ctx.structure_inputs,
        num_samples=args.num_samples,
        jump_threshold=args.jump_threshold,
    )

    print("\nFinal Cα medians")
    print(f"  Torch/chai-lab downstream: {_ca_median(torch_coords, ctx.structure_inputs):.6f} Å")
    print(f"  MLX downstream:            {_ca_median(mlx_coords, ctx.structure_inputs):.6f} Å")


if __name__ == "__main__":
    main()
