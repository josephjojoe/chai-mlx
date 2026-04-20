"""Capture chai-lab CUDA intermediate tensors for numerical-parity analysis.

This harness runs chai-lab's pipeline in Modal on CUDA and records the
boundary tensors at every stage as a single NPZ per seed:

* **inputs** — the raw collator features and token bond features the MLX
  port ingests.  Enough information is captured to reconstruct a
  ``FeatureContext`` locally without re-running the featurizer.
* **embedding** — feature-embedding outputs, bond-projection outputs, and
  the token embedder's three initial representations.
* **trunk** — the trunk's outputs after each recycle.  Includes the single
  and pair representations that go into the diffusion module.
* **diffusion** — the initial noisy atom positions, the schedule
  ``(sigma, gamma)`` tuple, and a deterministic per-step snapshot (only
  a configurable subset; by default the first, last, and a middle step)
  of ``atom_pos``, ``atom_pos_hat``, and ``denoised_pos``.  This is what
  enables per-step error-accumulation analysis from the MLX side without
  having to keep all 200×N atoms × float32 tensors.
* **confidence** — the three confidence heads' logits per sample.
* **ranking** — the scalar scores (ptm, iptm, aggregate, has_clashes) per
  sample.

The resulting bundle is a drop-in input for :mod:`scripts.cuda_parity`.

Usage
-----

::

    modal run -m cuda_harness.run_intermediates \\
        --targets 1L2Y \\
        --seeds 42 \\
        --num-recycles 3 \\
        --num-steps 200 \\
        --snapshot-steps 1,100,200 \\
        --output-dir /tmp/chai_mlx_cuda/intermediates

Notes
-----

Precision policy of the reference implementation (verified from the
exported TorchScript graphs in ``findings/graphs/`` and cross-checked
with ``cuda_harness._probe_jit_precision``):

* All model parameters ship as **fp32** on disk (``*.pt`` bundles).
* The trunk, token embedder and confidence head graphs bake in an
  autocast-equivalent cast chain: every ``aten::linear`` /
  ``aten::einsum`` is preceded by ``weight.to(bfloat16)`` and
  ``input.to(bfloat16)`` (``aten::to`` with scalar-type constant 15),
  while every ``aten::layer_norm`` / ``aten::softmax`` is preceded by
  ``aten::to`` with scalar-type 6 (fp32). Concretely, the trunk graph
  alone contains ~18k bf16 casts vs ~5k fp32 casts, i.e. matmuls run
  in bf16 and the layer norms / softmaxes compute their reductions
  in fp32. Functionally this is ``torch.autocast("cuda",
  dtype=torch.bfloat16)`` with an explicit fp32 fallback around
  normalisation-like ops, compiled into the scripted module.
* The diffusion module graph has **zero bf16 casts** and runs in pure
  fp32 end-to-end (``static_diffusion_inputs`` are explicitly
  ``.float()``'d just above).
* The feature embedding and bond projection run in fp32; they are
  tiny feature-generator stubs with no reduced-precision constants.

In short: "chai-lab precision" is bf16 autocast for the trunk + token
embedder + confidence head (fp32 weights, bf16 activations/matmuls,
fp32 layer-norm / softmax reductions), and fp32 for the diffusion
module. The ``--precision`` knob below only toggles TF32 / cuDNN
atomics; it does *not* move anything between bf16 and fp32.

Implications for the MLX parity experiments:

* At ``--compute-dtype bfloat16`` the MLX trunk and confidence head
  match the reference's dtype policy directly.
* At ``--compute-dtype float32`` the MLX side runs matmuls in fp32
  while CUDA's scripted graph still casts to bf16 inside the module
  before every linear. Non-zero error at the trunk boundary is
  expected even though both sides nominally ingest fp32 tensors.

We reproduce the main chai-lab inference flow explicitly rather than
monkey-patching ``run_folding_on_context``.  This is intentional: the
code is verbose but it mirrors ``chai-lab/chai_lab/chai1.py`` line-for-line
and gives us clean hook points to dump intermediates without reaching into
TorchScript graphs.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import modal

from cuda_harness.modal_common import (
    MINUTES,
    MODELS_DIR,
    OUTPUTS_DIR,
    app,
    chai_model_volume,
    chai_outputs_volume,
    DEFAULT_TARGETS,
    download_inference_dependencies,
    image,
)


N_DIFFUSION_SAMPLES = 5

CONSTRAINTS_DIR = Path(__file__).resolve().parent / "constraints"


def _load_constraint_bytes(resource_name: str | None) -> bytes | None:
    if resource_name is None:
        return None
    path = CONSTRAINTS_DIR / resource_name
    if not path.is_file():
        raise FileNotFoundError(
            f"Constraint resource {resource_name!r} not found at {path}"
        )
    return path.read_bytes()


def _apply_precision_policy(policy: str) -> dict[str, object]:
    """Apply a CUDA precision policy before chai-lab runs anything.

    Mirrors :func:`cuda_harness.run_determinism._apply_precision_policy`
    so a determinism experiment and an intermediates-capture experiment
    can be run under matching conditions.
    """
    import os

    import torch

    settings: dict[str, object] = {}
    if policy == "default":
        settings["note"] = "chai-lab defaults; TF32 typically on for Ampere/Hopper"
    elif policy == "tf32_off":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        settings["tf32_matmul"] = False
        settings["tf32_cudnn"] = False
    elif policy == "deterministic":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        settings["CUBLAS_WORKSPACE_CONFIG"] = os.environ["CUBLAS_WORKSPACE_CONFIG"]
        settings["deterministic_algorithms"] = True
        settings["cudnn_deterministic"] = True
        settings["tf32_matmul"] = False
        settings["tf32_cudnn"] = False
    else:
        raise ValueError(f"unknown precision policy: {policy!r}")
    return settings


@app.function(
    timeout=25 * MINUTES,
    gpu="H100",
    volumes={MODELS_DIR: chai_model_volume, OUTPUTS_DIR: chai_outputs_volume},
    image=image,
)
def cuda_intermediates(
    target: str,
    fasta: str,
    seed: int,
    num_recycles: int,
    num_steps: int,
    snapshot_steps: list[int],
    run_id: str,
    precision: str = "default",
    constraint_csv_bytes: bytes | None = None,
    use_esm_embeddings: bool = False,
) -> bytes:
    """Run chai-lab end-to-end on CUDA and bundle the intermediates as NPZ bytes."""
    import math
    import time

    import numpy as np
    import torch
    from einops import rearrange, repeat

    precision_settings = _apply_precision_policy(precision)

    from chai_lab.chai1 import (
        Collate,
        DiffusionConfig,
        TokenBondRestraint,
        _component_moved_to,
        feature_factory,
        make_all_atom_feature_context,
    )
    from chai_lab.model.diffusion_schedules import InferenceNoiseSchedule
    from chai_lab.model.utils import center_random_augmentation
    from chai_lab.ranking.frames import get_frames_and_mask
    from chai_lab.ranking.rank import rank
    from chai_lab.utils.tensor_utils import move_data_to_device, set_seed, und_self

    from chai_lab.data.collate.utils import AVAILABLE_MODEL_SIZES

    cfg = DiffusionConfig

    # chai-lab's ``run_folding_on_context`` is decorated with ``@torch.no_grad``
    # so PyTorch doesn't retain the autograd graph for the 48-block trunk and
    # 20-block diffusion transformer. Without this, the graph alone blows past
    # 80 GB on model_size=256 crops when the MSA is padded to 16384 rows.
    torch.set_grad_enabled(False)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    work_dir = OUTPUTS_DIR / run_id / target / f"seed_{seed}"
    work_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = work_dir / "input.fasta"
    fasta_path.write_text(fasta.strip() + "\n")

    constraint_path: Path | None = None
    if constraint_csv_bytes is not None:
        constraint_path = work_dir / "constraints.csv"
        constraint_path.write_bytes(constraint_csv_bytes)

    feature_context = make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=work_dir / "features",
        entity_name_as_subchain=True,
        use_esm_embeddings=use_esm_embeddings,
        use_msa_server=False,
        use_templates_server=False,
        constraint_path=constraint_path,
        esm_device=device,
    )
    set_seed([seed])

    collator = Collate(
        feature_factory=feature_factory,
        num_key_atoms=128,
        num_query_atoms=32,
    )
    batch = collator([feature_context])
    features = {k: v for k, v in batch["features"].items()}
    inputs = batch["inputs"]

    block_indices_h = inputs["block_atom_pair_q_idces"]
    block_indices_w = inputs["block_atom_pair_kv_idces"]
    atom_single_mask = inputs["atom_exists_mask"]
    atom_token_indices = inputs["atom_token_index"].long()
    token_single_mask = inputs["token_exists_mask"]
    token_pair_mask = und_self(token_single_mask, "b i, b j -> b i j")
    token_reference_atom_index = inputs["token_ref_atom_index"]
    atom_within_token_index = inputs["atom_within_token_index"]
    msa_mask = inputs["msa_mask"]
    template_input_masks = und_self(
        inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2"
    )
    block_atom_pair_mask = inputs["block_atom_pair_mask"]
    _, _, model_size = msa_mask.shape
    assert model_size in AVAILABLE_MODEL_SIZES

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().float().numpy() if t.is_floating_point() else t.detach().cpu().numpy()

    dump: dict[str, np.ndarray] = {}
    meta: dict = {
        "target": target,
        "seed": seed,
        "fasta": fasta,
        "constraints_attached": constraint_csv_bytes is not None,
        "use_esm_embeddings": use_esm_embeddings,
        "num_recycles": num_recycles,
        "num_steps": num_steps,
        "snapshot_steps": list(snapshot_steps),
        "num_diffusion_samples": N_DIFFUSION_SAMPLES,
        "n_tokens": int(token_single_mask.sum().item()),
        "model_size": int(model_size),
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "precision": precision,
        "precision_settings": precision_settings,
    }

    for name, tensor in features.items():
        dump[f"inputs.features.{name}"] = _np(tensor)
    for name, tensor in inputs.items():
        if isinstance(tensor, torch.Tensor):
            dump[f"inputs.batch.{name}"] = _np(tensor)

    bond_ft = TokenBondRestraint().generate(batch=batch).data
    dump["inputs.bond_ft"] = _np(bond_ft)

    # chai-lab's ``run_folding_on_context`` uses ``low_memory=True`` by default,
    # which means every module forward returns its outputs on CPU and the next
    # module moves its own inputs back to GPU transiently. Staying on GPU for
    # every intermediate OOMs even on an 80 GB H100 at model_size=256 because
    # the trunk's masked_fill / einsum paths over the full 16k-row MSA are
    # huge. We mirror chai-lab's pattern here (``return_on_cpu=True``) to keep
    # intermediates off-GPU between stages.
    t0 = time.perf_counter()
    with _component_moved_to("feature_embedding.pt", device) as feature_embedding:
        embedded_features = feature_embedding.forward(
            crop_size=model_size,
            move_to_device=device,
            return_on_cpu=True,
            **features,
        )
    t_feat = time.perf_counter() - t0

    token_single_input_feats = embedded_features["TOKEN"]
    token_pair_input_feats, token_pair_structure_input_feats = embedded_features[
        "TOKEN_PAIR"
    ].chunk(2, dim=-1)
    atom_single_input_feats, atom_single_structure_input_feats = embedded_features[
        "ATOM"
    ].chunk(2, dim=-1)
    block_atom_pair_input_feats, block_atom_pair_structure_input_feats = (
        embedded_features["ATOM_PAIR"].chunk(2, dim=-1)
    )
    template_input_feats = embedded_features["TEMPLATES"]
    msa_input_feats = embedded_features["MSA"]

    dump["embedding.token_single"] = _np(token_single_input_feats)
    dump["embedding.token_pair_trunk"] = _np(token_pair_input_feats)
    dump["embedding.token_pair_structure"] = _np(token_pair_structure_input_feats)
    dump["embedding.atom_single_trunk"] = _np(atom_single_input_feats)
    dump["embedding.atom_single_structure"] = _np(atom_single_structure_input_feats)
    dump["embedding.atom_pair_trunk"] = _np(block_atom_pair_input_feats)
    dump["embedding.atom_pair_structure"] = _np(block_atom_pair_structure_input_feats)
    dump["embedding.templates"] = _np(template_input_feats)
    dump["embedding.msa"] = _np(msa_input_feats)

    t0 = time.perf_counter()
    with _component_moved_to("bond_loss_input_proj.pt", device) as bond_loss_input_proj:
        trunk_bond_feat, structure_bond_feat = bond_loss_input_proj.forward(
            return_on_cpu=True,
            move_to_device=device,
            crop_size=model_size,
            input=bond_ft,
        ).chunk(2, dim=-1)
    t_bond = time.perf_counter() - t0
    dump["embedding.bond_trunk"] = _np(trunk_bond_feat)
    dump["embedding.bond_structure"] = _np(structure_bond_feat)
    token_pair_input_feats = token_pair_input_feats + trunk_bond_feat.to(
        token_pair_input_feats.dtype
    )
    token_pair_structure_input_feats = token_pair_structure_input_feats + structure_bond_feat.to(
        token_pair_structure_input_feats.dtype
    )
    # Also dump the POST-bond versions so the local parity script can feed
    # them straight through without re-computing the addition.
    dump["embedding.token_pair_trunk_with_bond"] = _np(token_pair_input_feats)
    dump["embedding.token_pair_structure_with_bond"] = _np(token_pair_structure_input_feats)

    t0 = time.perf_counter()
    with _component_moved_to("token_embedder.pt", device) as token_input_embedder:
        (
            token_single_initial_repr,
            token_single_structure_input,
            token_pair_initial_repr,
        ) = token_input_embedder.forward(
            return_on_cpu=True,
            move_to_device=device,
            token_single_input_feats=token_single_input_feats,
            token_pair_input_feats=token_pair_input_feats,
            atom_single_input_feats=atom_single_input_feats,
            block_atom_pair_feat=block_atom_pair_input_feats,
            block_atom_pair_mask=block_atom_pair_mask,
            block_indices_h=block_indices_h,
            block_indices_w=block_indices_w,
            atom_single_mask=atom_single_mask,
            atom_token_indices=atom_token_indices,
            crop_size=model_size,
        )
    t_tok = time.perf_counter() - t0
    dump["embedding.token_single_initial"] = _np(token_single_initial_repr)
    dump["embedding.token_single_structure"] = _np(token_single_structure_input)
    dump["embedding.token_pair_initial"] = _np(token_pair_initial_repr)

    torch.cuda.synchronize()
    t_recycles: list[float] = []
    token_single_trunk_repr = token_single_initial_repr
    token_pair_trunk_repr = token_pair_initial_repr
    for rec_idx in range(num_recycles):
        t0 = time.perf_counter()
        with _component_moved_to("trunk.pt", device) as trunk:
            (token_single_trunk_repr, token_pair_trunk_repr) = trunk.forward(
                move_to_device=device,
                token_single_trunk_initial_repr=token_single_initial_repr,
                token_pair_trunk_initial_repr=token_pair_initial_repr,
                token_single_trunk_repr=token_single_trunk_repr,
                token_pair_trunk_repr=token_pair_trunk_repr,
                msa_input_feats=msa_input_feats,
                msa_mask=msa_mask,
                template_input_feats=template_input_feats,
                template_input_masks=template_input_masks,
                token_single_mask=token_single_mask,
                token_pair_mask=token_pair_mask,
                crop_size=model_size,
            )
        torch.cuda.synchronize()
        t_recycles.append(time.perf_counter() - t0)
        dump[f"trunk.recycle_{rec_idx}.single"] = _np(token_single_trunk_repr)
        dump[f"trunk.recycle_{rec_idx}.pair"] = _np(token_pair_trunk_repr)

    dump["trunk.final.single"] = _np(token_single_trunk_repr)
    dump["trunk.final.pair"] = _np(token_pair_trunk_repr)

    torch.cuda.empty_cache()

    atom_single_mask_dev = atom_single_mask.to(device)
    static_diffusion_inputs = dict(
        token_single_initial_repr=token_single_structure_input.float(),
        token_pair_initial_repr=token_pair_structure_input_feats.float(),
        token_single_trunk_repr=token_single_trunk_repr.float(),
        token_pair_trunk_repr=token_pair_trunk_repr.float(),
        atom_single_input_feats=atom_single_structure_input_feats.float(),
        atom_block_pair_input_feats=block_atom_pair_structure_input_feats.float(),
        atom_single_mask=atom_single_mask_dev,
        atom_block_pair_mask=block_atom_pair_mask,
        token_single_mask=token_single_mask,
        block_indices_h=block_indices_h,
        block_indices_w=block_indices_w,
        atom_token_indices=atom_token_indices,
    )
    static_diffusion_inputs = move_data_to_device(static_diffusion_inputs, device=device)

    inference_noise_schedule = InferenceNoiseSchedule(
        s_max=cfg.S_tmax, s_min=4e-4, p=7.0, sigma_data=cfg.sigma_data
    )
    sigmas = inference_noise_schedule.get_schedule(device=device, num_timesteps=num_steps)
    gammas = torch.where(
        (sigmas >= cfg.S_tmin) & (sigmas <= cfg.S_tmax),
        min(cfg.S_churn / num_steps, math.sqrt(2) - 1),
        0.0,
    )
    sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))
    dump["diffusion.sigmas"] = _np(sigmas)
    dump["diffusion.gammas"] = _np(gammas)

    _, num_atoms = atom_single_mask.shape
    atom_pos = sigmas[0] * torch.randn(
        1 * N_DIFFUSION_SAMPLES, num_atoms, 3, device=device
    )
    dump["diffusion.atom_pos_init"] = _np(atom_pos)

    def _denoise(diff_mod, atom_pos_in: torch.Tensor, sigma: torch.Tensor, ds: int) -> torch.Tensor:
        atom_noised_coords = rearrange(
            atom_pos_in, "(b ds) ... -> b ds ...", ds=ds
        ).contiguous()
        noise_sigma = repeat(sigma, " -> b ds", b=1, ds=ds)
        return diff_mod.forward(
            atom_noised_coords=atom_noised_coords.float(),
            noise_sigma=noise_sigma.float(),
            crop_size=model_size,
            **static_diffusion_inputs,
        )

    snapshot_set = {int(s) for s in snapshot_steps if 1 <= int(s) <= num_steps}
    step_times: list[float] = []

    torch.cuda.synchronize()
    with _component_moved_to("diffusion_module.pt", device=device) as diffusion_module:
        for step_idx, (sigma_curr, sigma_next, gamma_curr) in enumerate(sigmas_and_gammas, start=1):
            t0 = time.perf_counter()
            atom_pos = center_random_augmentation(
                atom_pos,
                atom_single_mask=repeat(
                    atom_single_mask_dev, "b a -> (b ds) a", ds=N_DIFFUSION_SAMPLES
                ),
            )
            noise = cfg.S_noise * torch.randn(atom_pos.shape, device=atom_pos.device)
            sigma_hat = sigma_curr + gamma_curr * sigma_curr
            atom_pos_noise = (sigma_hat**2 - sigma_curr**2).clamp_min(1e-6).sqrt()
            atom_pos_hat = atom_pos + noise * atom_pos_noise

            denoised_pos = _denoise(
                diff_mod=diffusion_module,
                atom_pos_in=atom_pos_hat,
                sigma=sigma_hat,
                ds=N_DIFFUSION_SAMPLES,
            )
            d_i = (atom_pos_hat - denoised_pos) / sigma_hat
            atom_pos = atom_pos_hat + (sigma_next - sigma_hat) * d_i

            if sigma_next != 0 and cfg.second_order:
                denoised_pos2 = _denoise(
                    diff_mod=diffusion_module,
                    atom_pos_in=atom_pos,
                    sigma=sigma_next,
                    ds=N_DIFFUSION_SAMPLES,
                )
                d_i_prime = (atom_pos - denoised_pos2) / sigma_next
                atom_pos = atom_pos + (sigma_next - sigma_hat) * ((d_i_prime + d_i) / 2)

            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

            if step_idx in snapshot_set:
                dump[f"diffusion.step_{step_idx:04d}.atom_pos_hat"] = _np(atom_pos_hat)
                dump[f"diffusion.step_{step_idx:04d}.denoised"] = _np(denoised_pos)
                dump[f"diffusion.step_{step_idx:04d}.atom_pos_after"] = _np(atom_pos)
                dump[f"diffusion.step_{step_idx:04d}.sigma_curr"] = np.array(float(sigma_curr.item()))
                dump[f"diffusion.step_{step_idx:04d}.sigma_next"] = np.array(float(sigma_next.item()))
                dump[f"diffusion.step_{step_idx:04d}.gamma"] = np.array(float(gamma_curr.item()))

    dump["diffusion.atom_pos_final"] = _np(atom_pos)
    dump["diffusion.step_times_seconds"] = np.array(step_times, dtype=np.float64)

    del static_diffusion_inputs
    torch.cuda.empty_cache()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with _component_moved_to("confidence_head.pt", device=device) as confidence_head:
        confidence_outputs = [
            confidence_head.forward(
                move_to_device=device,
                token_single_input_repr=token_single_initial_repr,
                token_single_trunk_repr=token_single_trunk_repr,
                token_pair_trunk_repr=token_pair_trunk_repr,
                token_single_mask=token_single_mask,
                atom_single_mask=atom_single_mask_dev,
                atom_coords=atom_pos[ds : ds + 1],
                token_reference_atom_index=token_reference_atom_index,
                atom_token_index=atom_token_indices,
                atom_within_token_index=atom_within_token_index,
                crop_size=model_size,
            )
            for ds in range(N_DIFFUSION_SAMPLES)
        ]
    torch.cuda.synchronize()
    t_conf = time.perf_counter() - t0

    pae_logits, pde_logits, plddt_logits = [
        torch.cat(single_sample, dim=0)
        for single_sample in zip(*confidence_outputs, strict=True)
    ]
    dump["confidence.pae_logits"] = _np(pae_logits)
    dump["confidence.pde_logits"] = _np(pde_logits)
    dump["confidence.plddt_logits"] = _np(plddt_logits)

    inputs_cpu = move_data_to_device(inputs, torch.device("cpu"))
    atom_pos_cpu = atom_pos.cpu()
    plddt_logits_cpu = plddt_logits.cpu()
    pae_logits_cpu = pae_logits.cpu()

    pae_bin_centers = torch.linspace(0.0, 32.0, 2 * 64 + 1)[1::2]
    lddt_bin_centers = torch.linspace(0.0, 1.0, 2 * plddt_logits.shape[-1] + 1)[1::2]

    ranking_summaries = []
    for idx in range(N_DIFFUSION_SAMPLES):
        _, valid_frames_mask = get_frames_and_mask(
            atom_pos_cpu[idx : idx + 1],
            inputs_cpu["token_asym_id"],
            inputs_cpu["token_residue_index"],
            inputs_cpu["token_backbone_frame_mask"],
            inputs_cpu["token_centre_atom_index"],
            inputs_cpu["token_exists_mask"],
            inputs_cpu["atom_exists_mask"],
            inputs_cpu["token_backbone_frame_index"],
            inputs_cpu["atom_token_index"],
        )
        rd = rank(
            atom_pos_cpu[idx : idx + 1],
            atom_mask=inputs_cpu["atom_exists_mask"],
            atom_token_index=inputs_cpu["atom_token_index"],
            token_exists_mask=inputs_cpu["token_exists_mask"],
            token_asym_id=inputs_cpu["token_asym_id"],
            token_entity_type=inputs_cpu["token_entity_type"],
            token_valid_frames_mask=valid_frames_mask,
            lddt_logits=plddt_logits_cpu[idx : idx + 1],
            lddt_bin_centers=lddt_bin_centers,
            pae_logits=pae_logits_cpu[idx : idx + 1],
            pae_bin_centers=pae_bin_centers,
        )
        ranking_summaries.append(
            {
                "aggregate_score": float(rd.aggregate_score.item()),
                "complex_ptm": float(rd.ptm_scores.complex_ptm.item()),
                "interface_ptm": float(rd.ptm_scores.interface_ptm.item()),
                "has_inter_chain_clashes": bool(rd.clash_scores.has_inter_chain_clashes.item()),
            }
        )
    meta["ranking"] = ranking_summaries
    meta["timings"] = {
        "feature_embedding_seconds": t_feat,
        "bond_projection_seconds": t_bond,
        "token_embedder_seconds": t_tok,
        "trunk_recycle_seconds": t_recycles,
        "confidence_seconds": t_conf,
        "diffusion_step_seconds_sum": float(sum(step_times)),
        "diffusion_step_seconds_count": len(step_times),
    }

    dump["_manifest_json"] = np.frombuffer(json.dumps(meta, indent=2).encode(), dtype=np.uint8)

    buf = io.BytesIO()
    np.savez_compressed(buf, **dump)
    chai_outputs_volume.commit()
    return buf.getvalue()


def _parse_steps(spec: str, total: int) -> list[int]:
    items = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "first":
            items.append(1)
        elif tok == "last":
            items.append(total)
        elif tok == "mid":
            items.append(total // 2)
        else:
            items.append(int(tok))
    return sorted(set(max(1, min(total, s)) for s in items))


@app.local_entrypoint()
def run_intermediates(
    targets: str = "1L2Y",
    seeds: str = "42",
    num_recycles: int = 3,
    num_steps: int = 200,
    snapshot_steps: str = "first,mid,last",
    output_dir: str = "/tmp/chai_mlx_cuda/intermediates",
    run_id: str | None = None,
    ensure_weights: bool = True,
    precision: str = "default",
    constraint_resource: str | None = None,
    use_esm_embeddings: bool = False,
) -> None:
    targets_list = [t.strip() for t in targets.split(",") if t.strip()]
    seeds_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    rid = run_id or f"intm-{precision}-{num_recycles}r-{num_steps}s"
    steps = _parse_steps(snapshot_steps, num_steps)
    print(f"[modal] snapshot steps = {steps}  precision={precision}")

    if ensure_weights:
        print("[modal] ensuring weights are on the volume")
        download_inference_dependencies.remote(force=False)

    for name in targets_list:
        if name not in DEFAULT_TARGETS:
            raise KeyError(
                f"Unknown target {name!r}. Known: {sorted(DEFAULT_TARGETS)}"
            )
        target = DEFAULT_TARGETS[name]
        resource = constraint_resource or target.constraint_resource
        constraint_bytes = _load_constraint_bytes(resource)
        for seed in seeds_list:
            label = f"{name} seed={seed} steps={steps}"
            if resource:
                label += f" constraints={resource}"
            print(f"[modal] -> {label}")
            payload = cuda_intermediates.remote(
                target=name,
                fasta=target.to_fasta(),
                seed=seed,
                num_recycles=num_recycles,
                num_steps=num_steps,
                snapshot_steps=steps,
                run_id=rid,
                precision=precision,
                constraint_csv_bytes=constraint_bytes,
                use_esm_embeddings=use_esm_embeddings,
            )
            suffix = "" if precision == "default" else f"_{precision}"
            dst = output_dir_path / name / f"seed_{seed}{suffix}.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(payload)
            print(f"[modal]    wrote {len(payload) / (1 << 20):.1f} MB -> {dst}")
    print("[modal] done")
