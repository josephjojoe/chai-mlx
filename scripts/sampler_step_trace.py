"""Trace one diffusion sampler step in MLX and chai-lab/TorchScript.

This script is the sampler-side analogue of ``deep_denoise_trace.py``.
It compares one full diffusion integration step, including:

- coordinate augmentation
- per-step noise generation
- sigma / gamma schedule values
- the first denoise call
- Euler update
- the second denoise call (Heun correction)
- the final step output

Two randomness modes are supported:

- ``native``: MLX and Torch each use their own RNG after seeding
- ``torch_shared``: Torch samples rotations / translations / noise once and
  the exact same tensors are injected into MLX

The second mode isolates sampler math from backend RNG differences.

Usage::

    python3 scripts/sampler_step_trace.py \
        --weights-dir weights \
        --input-npz /tmp/chai_mlx_phase0/input.npz \
        --reference-npz /tmp/chai_mlx_phase0/reference.npz \
        --step-index 100 \
        --randomness native

    python3 scripts/sampler_step_trace.py \
        --weights-dir weights \
        --input-npz /tmp/chai_mlx_phase0/input.npz \
        --reference-npz /tmp/chai_mlx_phase0/reference.npz \
        --step-index 100 \
        --randomness torch_shared
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from chai_lab.chai1 import DiffusionConfig, _component_moved_to  # type: ignore[import-not-found]
from chai_lab.model.diffusion_schedules import InferenceNoiseSchedule  # type: ignore[import-not-found]
from chai_lab.model.utils import calc_centroid, random_rotations  # type: ignore[import-not-found]
from chai_lab.utils.tensor_utils import set_seed  # type: ignore[import-not-found]

from chai_mlx import ChaiMLX
from chai_mlx.data.types import StructureInputs
from chai_mlx.utils import edm_gammas, edm_sigmas, masked_mean, random_rotation, resolve_dtype
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


def _to_torch(x, device: torch.device, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(x, mx.array):
        arr = _mx_np(x)
    else:
        arr = np.asarray(x)
    out = torch.from_numpy(arr).to(device)
    return out.to(dtype) if dtype is not None else out


def _broadcast_step_scalar(value: float, batch_size: int, num_samples: int) -> mx.array:
    return mx.full((batch_size, num_samples), value, dtype=mx.float32)


def _seed_torch(seed: int) -> None:
    # Chai-lab does not use the raw integer directly; it derives a torch seed
    # via SeedSequence and seeds numpy / stdlib alongside torch.
    set_seed([seed])


def _compare_stats(diff: np.ndarray) -> tuple[float, float, float]:
    if diff.size == 0:
        return 0.0, 0.0, 0.0
    return float(diff.max()), float(diff.mean()), float(np.quantile(diff, 0.99))


_ATOM_MASKED_KEYS = {
    "step.coords_in",
    "step.coords_aug",
    "step.noise",
    "step.coords_hat",
    "step.denoised_1",
    "step.d_i",
    "step.coords_euler",
    "step.denoised_2",
    "step.d_prime",
    "step.coords_out",
}


def _compare_sampler_traces(
    torch_trace: OrderedDict[str, np.ndarray],
    mlx_trace: OrderedDict[str, np.ndarray],
    structure: StructureInputs,
    *,
    num_samples: int,
    jump_threshold: float,
) -> None:
    atom_mask = np.array(structure.atom_exists_mask, copy=False).astype(bool)
    atom_mask_4d = np.broadcast_to(atom_mask[:, None, :, None], (atom_mask.shape[0], num_samples, atom_mask.shape[1], 3))

    first_jump: tuple[str, str, float, float, float] | None = None

    for name, ref in torch_trace.items():
        got = mlx_trace.get(name)
        if got is None:
            print(f"[MISSING] {name}")
            continue
        if ref.shape != got.shape:
            print(f"[SHAPE] {name}: torch={ref.shape} mlx={got.shape}")
            continue
        diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
        full_max, full_mean, full_p99 = _compare_stats(diff)

        if name in _ATOM_MASKED_KEYS:
            valid = diff[atom_mask_4d]
            cmp_max, cmp_mean, cmp_p99 = _compare_stats(valid)
            label = "valid"
        else:
            cmp_max, cmp_mean, cmp_p99 = full_max, full_mean, full_p99
            label = "full"

        print(
            f"{name:<28} {label}_max={cmp_max:9.4e}  "
            f"{label}_mean={cmp_mean:9.4e}  {label}_p99={cmp_p99:9.4e}  "
            f"full_max={full_max:9.4e}  shape={ref.shape}"
        )

        if first_jump is None and cmp_p99 >= jump_threshold:
            first_jump = (name, label, cmp_max, cmp_mean, cmp_p99)

    if first_jump is None:
        print(f"\nNo sampler checkpoint exceeded threshold {jump_threshold:.3g}.")
    else:
        name, label, cmp_max, cmp_mean, cmp_p99 = first_jump
        print(
            f"\nFirst sampler checkpoint over threshold {jump_threshold:.3g}: "
            f"{name} ({label}_max={cmp_max:.4e}, {label}_mean={cmp_mean:.4e}, {label}_p99={cmp_p99:.4e})"
        )


def _torch_schedule(num_steps: int) -> tuple[np.ndarray, np.ndarray]:
    sigmas = InferenceNoiseSchedule(
        s_max=DiffusionConfig.S_tmax,
        s_min=4e-4,
        p=7.0,
        sigma_data=DiffusionConfig.sigma_data,
    ).get_schedule(device=torch.device("cpu"), num_timesteps=num_steps)
    gammas = torch.where(
        (sigmas >= DiffusionConfig.S_tmin) & (sigmas <= DiffusionConfig.S_tmax),
        torch.tensor(min(DiffusionConfig.S_churn / num_steps, math.sqrt(2.0) - 1.0), dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float32),
    )
    return sigmas.numpy().astype(np.float32), gammas.numpy().astype(np.float32)


def _mlx_schedule(model: ChaiMLX, num_steps: int) -> tuple[np.ndarray, np.ndarray]:
    sigmas = _mx_np(
        edm_sigmas(
            num_steps,
            model.cfg.diffusion.sigma_data,
            model.cfg.diffusion.s_min,
            model.cfg.diffusion.s_max,
            model.cfg.diffusion.p,
        )
    ).astype(np.float32)
    gammas = _mx_np(
        edm_gammas(
            mx.array(sigmas),
            model.cfg.diffusion.s_churn,
            model.cfg.diffusion.s_tmin,
            model.cfg.diffusion.s_tmax,
        )
    ).astype(np.float32)
    return sigmas, gammas


def _coords_source(
    mode: str,
    extras: dict[str, mx.array],
    model: ChaiMLX,
    structure: StructureInputs,
    *,
    seed: int,
) -> mx.array:
    if mode == "input":
        return extras["coords"].astype(mx.float32)
    if mode == "zero":
        return mx.zeros(extras["coords"].shape, dtype=mx.float32)
    if mode == "init_noise":
        mx.random.seed(seed)
        return model.init_noise(1, 1, structure).astype(mx.float32)
    raise ValueError(f"Unknown coords mode {mode!r}")


@dataclass(frozen=True)
class SharedRandomness:
    rotations: np.ndarray
    translations: np.ndarray
    noise: np.ndarray


def _sample_shared_randomness(
    shape_flat: tuple[int, int, int],
    *,
    seed: int,
    device: torch.device,
) -> SharedRandomness:
    flat_batch, _, _ = shape_flat
    _seed_torch(seed)
    rotations = random_rotations(flat_batch, dtype=torch.float32, device=device)
    translations = torch.randn((flat_batch, 1, 3), dtype=torch.float32, device=device)
    noise = DiffusionConfig.S_noise * torch.randn(shape_flat, dtype=torch.float32, device=device)
    return SharedRandomness(
        rotations=_pt_np(rotations),
        translations=_pt_np(translations),
        noise=_pt_np(noise),
    )


def _torch_step_trace(
    store: OrderedDict[str, np.ndarray],
    diffusion_module,
    static_inputs: dict[str, torch.Tensor],
    structure: StructureInputs,
    coords_in: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
    gamma: float,
    seed: int,
    randomness: str,
    shared: SharedRandomness | None,
) -> None:
    batch_size, num_samples, num_atoms, _ = coords_in.shape
    coords_flat = coords_in.reshape(batch_size * num_samples, num_atoms, 3).contiguous()
    atom_mask = _to_torch(structure.atom_exists_mask, coords_in.device, dtype=torch.bool)
    atom_mask_flat = atom_mask[:, None, :].expand(batch_size, num_samples, num_atoms).reshape(batch_size * num_samples, num_atoms)

    _record_pt(store, "step.coords_in", coords_in)
    _record_pt(store, "step.sigma_curr", torch.tensor([sigma_curr], dtype=torch.float32, device=coords_in.device))
    _record_pt(store, "step.sigma_next", torch.tensor([sigma_next], dtype=torch.float32, device=coords_in.device))
    _record_pt(store, "step.gamma", torch.tensor([gamma], dtype=torch.float32, device=coords_in.device))

    if randomness == "torch_shared":
        assert shared is not None
        rotations = torch.from_numpy(shared.rotations).to(coords_in.device)
        translations = torch.from_numpy(shared.translations).to(coords_in.device)
        noise = torch.from_numpy(shared.noise).to(coords_in.device)
    else:
        _seed_torch(seed)
        rotations = random_rotations(coords_flat.shape[0], dtype=torch.float32, device=coords_in.device)
        translations = torch.randn((coords_flat.shape[0], 1, 3), dtype=torch.float32, device=coords_in.device)
        noise = DiffusionConfig.S_noise * torch.randn(coords_flat.shape, dtype=torch.float32, device=coords_in.device)

    centroid = calc_centroid(coords_flat, atom_mask_flat)
    centered = coords_flat - centroid[:, None, :]
    coords_aug_flat = torch.einsum("bij,baj->bai", rotations, centered) + translations

    sigma_hat = sigma_curr + gamma * sigma_curr
    noise_scale = math.sqrt(max(sigma_hat * sigma_hat - sigma_curr * sigma_curr, 1e-6))
    coords_hat_flat = coords_aug_flat + noise * noise_scale

    sigma_hat_t = torch.full((batch_size, num_samples), sigma_hat, dtype=torch.float32, device=coords_in.device)
    coords_hat = coords_hat_flat.reshape(batch_size, num_samples, num_atoms, 3).contiguous()
    denoised_1 = diffusion_module.forward(
        atom_noised_coords=coords_hat.float(),
        noise_sigma=sigma_hat_t.float(),
        crop_size=int(static_inputs["token_single_mask"].shape[-1]),
        **static_inputs,
    )
    denoised_1 = denoised_1[:, None] if denoised_1.ndim == 3 else denoised_1
    denoised_1_flat = denoised_1.reshape(batch_size * num_samples, num_atoms, 3)
    d_i_flat = (coords_hat_flat - denoised_1_flat) / sigma_hat
    coords_euler_flat = coords_hat_flat + (sigma_next - sigma_hat) * d_i_flat
    coords_euler = coords_euler_flat.reshape(batch_size, num_samples, num_atoms, 3).contiguous()

    _record_pt(store, "step.rotations", rotations.reshape(batch_size, num_samples, 3, 3))
    _record_pt(store, "step.translations", translations.reshape(batch_size, num_samples, 1, 3))
    _record_pt(store, "step.coords_aug", coords_aug_flat.reshape(batch_size, num_samples, num_atoms, 3))
    _record_pt(store, "step.noise", noise.reshape(batch_size, num_samples, num_atoms, 3))
    _record_pt(store, "step.sigma_hat", torch.tensor([sigma_hat], dtype=torch.float32, device=coords_in.device))
    _record_pt(store, "step.noise_scale", torch.tensor([noise_scale], dtype=torch.float32, device=coords_in.device))
    _record_pt(store, "step.coords_hat", coords_hat)
    _record_pt(store, "step.denoised_1", denoised_1)
    _record_pt(store, "step.d_i", d_i_flat.reshape(batch_size, num_samples, num_atoms, 3))
    _record_pt(store, "step.coords_euler", coords_euler)

    if sigma_next != 0.0 and DiffusionConfig.second_order:
        sigma_next_t = torch.full((batch_size, num_samples), sigma_next, dtype=torch.float32, device=coords_in.device)
        denoised_2 = diffusion_module.forward(
            atom_noised_coords=coords_euler.float(),
            noise_sigma=sigma_next_t.float(),
            crop_size=int(static_inputs["token_single_mask"].shape[-1]),
            **static_inputs,
        )
        denoised_2 = denoised_2[:, None] if denoised_2.ndim == 3 else denoised_2
        denoised_2_flat = denoised_2.reshape(batch_size * num_samples, num_atoms, 3)
        d_prime_flat = (coords_euler_flat - denoised_2_flat) / sigma_next
        coords_out_flat = coords_euler_flat + (sigma_next - sigma_hat) * ((d_prime_flat + d_i_flat) / 2.0)
        _record_pt(store, "step.denoised_2", denoised_2)
        _record_pt(store, "step.d_prime", d_prime_flat.reshape(batch_size, num_samples, num_atoms, 3))
    else:
        coords_out_flat = coords_euler_flat

    _record_pt(store, "step.coords_out", coords_out_flat.reshape(batch_size, num_samples, num_atoms, 3))


def _mlx_step_trace(
    store: OrderedDict[str, np.ndarray],
    model: ChaiMLX,
    cache,
    structure: StructureInputs,
    coords_in: mx.array,
    *,
    sigma_curr: float,
    sigma_next: float,
    gamma: float,
    seed: int,
    randomness: str,
    shared: SharedRandomness | None,
) -> None:
    batch_size, num_samples, num_atoms, _ = coords_in.shape
    coords_flat = coords_in.reshape(batch_size * num_samples, num_atoms, 3)
    atom_mask = structure.atom_exists_mask
    atom_mask_flat = mx.broadcast_to(atom_mask[:, None, :], (batch_size, num_samples, num_atoms)).reshape(batch_size * num_samples, num_atoms)

    _record_mx(store, "step.coords_in", coords_in)
    _record_mx(store, "step.sigma_curr", mx.array([sigma_curr], dtype=mx.float32))
    _record_mx(store, "step.sigma_next", mx.array([sigma_next], dtype=mx.float32))
    _record_mx(store, "step.gamma", mx.array([gamma], dtype=mx.float32))

    if randomness == "torch_shared":
        assert shared is not None
        rotations = mx.array(shared.rotations, dtype=mx.float32)
        translations = mx.array(shared.translations, dtype=mx.float32)
        noise = mx.array(shared.noise, dtype=mx.float32)
    else:
        mx.random.seed(seed)
        rotations = random_rotation(coords_flat.shape[0]).astype(mx.float32)
        translations = mx.random.normal((coords_flat.shape[0], 1, 3)).astype(mx.float32)
        noise = (model.cfg.diffusion.s_noise * mx.random.normal(coords_flat.shape)).astype(mx.float32)

    centroid = masked_mean(coords_flat, atom_mask_flat, axis=1, keepdims=True, eps=model.cfg.centroid_eps)
    centered = coords_flat - centroid
    coords_aug_flat = mx.einsum("bij,baj->bai", rotations, centered) + translations

    sigma_hat = sigma_curr + gamma * sigma_curr
    sigma_delta = max(sigma_hat * sigma_hat - sigma_curr * sigma_curr, float(model.cfg.diffusion_sqrt_eps))
    noise_scale = math.sqrt(sigma_delta)
    coords_hat_flat = coords_aug_flat + noise * noise_scale

    sigma_hat_mx = _broadcast_step_scalar(sigma_hat, batch_size, num_samples)
    coords_hat = coords_hat_flat.reshape(batch_size, num_samples, num_atoms, 3)
    denoised_1 = model.denoise(cache, coords_hat, sigma_hat_mx)
    denoised_1_flat = denoised_1.reshape(batch_size * num_samples, num_atoms, 3)
    d_i_flat = (coords_hat_flat - denoised_1_flat) / sigma_hat
    coords_euler_flat = coords_hat_flat + (sigma_next - sigma_hat) * d_i_flat
    coords_euler = coords_euler_flat.reshape(batch_size, num_samples, num_atoms, 3)

    _record_mx(store, "step.rotations", rotations.reshape(batch_size, num_samples, 3, 3))
    _record_mx(store, "step.translations", translations.reshape(batch_size, num_samples, 1, 3))
    _record_mx(store, "step.coords_aug", coords_aug_flat.reshape(batch_size, num_samples, num_atoms, 3))
    _record_mx(store, "step.noise", noise.reshape(batch_size, num_samples, num_atoms, 3))
    _record_mx(store, "step.sigma_hat", mx.array([sigma_hat], dtype=mx.float32))
    _record_mx(store, "step.noise_scale", mx.array([noise_scale], dtype=mx.float32))
    _record_mx(store, "step.coords_hat", coords_hat)
    _record_mx(store, "step.denoised_1", denoised_1)
    _record_mx(store, "step.d_i", d_i_flat.reshape(batch_size, num_samples, num_atoms, 3))
    _record_mx(store, "step.coords_euler", coords_euler)

    if sigma_next != 0.0 and model.cfg.diffusion.second_order:
        sigma_next_mx = _broadcast_step_scalar(sigma_next, batch_size, num_samples)
        denoised_2 = model.denoise(cache, coords_euler, sigma_next_mx)
        denoised_2_flat = denoised_2.reshape(batch_size * num_samples, num_atoms, 3)
        d_prime_flat = (coords_euler_flat - denoised_2_flat) / sigma_next
        coords_out_flat = coords_euler_flat + (sigma_next - sigma_hat) * ((d_prime_flat + d_i_flat) / 2.0)
        _record_mx(store, "step.denoised_2", denoised_2)
        _record_mx(store, "step.d_prime", d_prime_flat.reshape(batch_size, num_samples, num_atoms, 3))
    else:
        coords_out_flat = coords_euler_flat

    _record_mx(store, "step.coords_out", coords_out_flat.reshape(batch_size, num_samples, num_atoms, 3))


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Trace one diffusion sampler step in MLX and chai-lab/TorchScript")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--step-index", type=int, default=100)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--coords-mode", choices=("input", "zero", "init_noise"), default="input")
    parser.add_argument("--randomness", choices=("native", "torch_shared"), default="native")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-device", default="mps")
    parser.add_argument("--jump-threshold", type=float, default=1e-4)
    parser.add_argument("--write-torch-dump", type=Path, default=None)
    parser.add_argument("--write-mlx-dump", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False, compute_dtype="float32")
    ctx, extras = load_feature_context(args.input_npz)
    ref = _npz_dict(args.reference_npz)

    mlx_sigmas, mlx_gammas = _mlx_schedule(model, args.num_steps)
    torch_sigmas, torch_gammas = _torch_schedule(args.num_steps)

    print("Schedule comparison")
    print(f"  sigma max diff: {float(np.max(np.abs(mlx_sigmas - torch_sigmas))):.4e}")
    print(f"  gamma max diff: {float(np.max(np.abs(mlx_gammas - torch_gammas))):.4e}")

    if not (0 <= args.step_index < args.num_steps - 1):
        raise ValueError(f"--step-index must satisfy 0 <= idx < {args.num_steps - 1}")

    sigma_curr = float(torch_sigmas[args.step_index])
    sigma_next = float(torch_sigmas[args.step_index + 1])
    gamma = float(torch_gammas[args.step_index])
    print(
        f"Tracing step {args.step_index}: sigma_curr={sigma_curr:.6f} "
        f"sigma_next={sigma_next:.6f} gamma={gamma:.6f} "
        f"coords_mode={args.coords_mode} randomness={args.randomness}"
    )

    coords_in = _coords_source(
        args.coords_mode,
        extras,
        model,
        ctx.structure_inputs,
        seed=args.seed,
    )
    batch_size, num_samples, num_atoms, _ = coords_in.shape

    dtype = resolve_dtype(model.cfg)
    trunk = reconstruct_trunk_outputs(ref, ctx.structure_inputs, dtype=dtype)
    cache = model.prepare_diffusion_cache(trunk)
    mx.eval(cache.s_static, cache.z_cond, cache.blocked_pair_base, cache.atom_cond, cache.atom_single_cond, *cache.pair_biases)

    torch_device = torch.device(args.torch_device)
    static_inputs = {
        "token_single_initial_repr": torch.from_numpy(ref["trunk.outputs.single_structure"]).float().to(torch_device),
        "token_pair_initial_repr": torch.from_numpy(ref["trunk.outputs.pair_structure"]).float().to(torch_device),
        "token_single_trunk_repr": torch.from_numpy(ref["trunk.outputs.single_trunk"]).float().to(torch_device),
        "token_pair_trunk_repr": torch.from_numpy(ref["trunk.outputs.pair_trunk"]).float().to(torch_device),
        "atom_single_input_feats": torch.from_numpy(ref["trunk.outputs.atom_single_structure_input"]).float().to(torch_device),
        "atom_block_pair_input_feats": torch.from_numpy(ref["trunk.outputs.atom_pair_structure_input"]).float().to(torch_device),
        "atom_single_mask": _to_torch(ctx.structure_inputs.atom_exists_mask, torch_device, dtype=torch.bool),
        "atom_block_pair_mask": _to_torch(ctx.structure_inputs.block_atom_pair_mask, torch_device, dtype=torch.bool),
        "token_single_mask": _to_torch(ctx.structure_inputs.token_exists_mask, torch_device, dtype=torch.bool),
        "block_indices_h": _to_torch(ctx.structure_inputs.atom_q_indices, torch_device, dtype=torch.long).squeeze(0),
        "block_indices_w": _to_torch(ctx.structure_inputs.atom_kv_indices, torch_device, dtype=torch.long).squeeze(0),
        "atom_token_indices": _to_torch(ctx.structure_inputs.atom_token_index, torch_device, dtype=torch.long),
    }

    shared = None
    if args.randomness == "torch_shared":
        shared = _sample_shared_randomness(
            (batch_size * num_samples, num_atoms, 3),
            seed=args.seed,
            device=torch_device,
        )

    torch_trace: OrderedDict[str, np.ndarray] = OrderedDict()
    mlx_trace: OrderedDict[str, np.ndarray] = OrderedDict()

    with torch.no_grad():
        with _component_moved_to("diffusion_module.pt", device=torch_device) as diffusion_module:
            _torch_step_trace(
                torch_trace,
                diffusion_module,
                static_inputs,
                ctx.structure_inputs,
                _to_torch(coords_in, torch_device, dtype=torch.float32),
                sigma_curr=sigma_curr,
                sigma_next=sigma_next,
                gamma=gamma,
                seed=args.seed,
                randomness=args.randomness,
                shared=shared,
            )

    _mlx_step_trace(
        mlx_trace,
        model,
        cache,
        ctx.structure_inputs,
        coords_in.astype(mx.float32),
        sigma_curr=sigma_curr,
        sigma_next=sigma_next,
        gamma=gamma,
        seed=args.seed,
        randomness=args.randomness,
        shared=shared,
    )

    if args.write_torch_dump is not None:
        args.write_torch_dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.write_torch_dump, **torch_trace)
        print(f"Wrote torch sampler trace: {args.write_torch_dump}")
    if args.write_mlx_dump is not None:
        args.write_mlx_dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.write_mlx_dump, **mlx_trace)
        print(f"Wrote MLX sampler trace: {args.write_mlx_dump}")

    _compare_sampler_traces(
        torch_trace,
        mlx_trace,
        ctx.structure_inputs,
        num_samples=num_samples,
        jump_threshold=args.jump_threshold,
    )


if __name__ == "__main__":
    main()
