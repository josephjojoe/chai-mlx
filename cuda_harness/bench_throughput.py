"""Per-module CUDA throughput benchmark on Modal.

For a set of targets this harness measures wall-clock time for every
module in the chai-1 inference pipeline, with proper warmup and explicit
``cuda.synchronize`` calls so we get a fair comparison to the local MLX
numbers from :mod:`scripts.mlx_throughput`.

Reported fields (per target, averaged across ``--num-repeats`` runs with
``warmup`` runs discarded):

* ``feature_embedding_ms``
* ``bond_projection_ms``
* ``token_embedder_ms``
* ``trunk_recycle_ms`` (per-recycle)
* ``diffusion_step_ms`` (mean per-step; second-order denoise counted as
  two denoises)
* ``diffusion_total_ms`` (full ``num_steps``-step roll-out)
* ``confidence_ms`` (all five samples)
* ``end_to_end_ms`` (sum of the above)

Saves a single ``throughput.json`` per target and a combined CSV for
easy ingestion by plotting / reporting scripts.

Usage
-----

::

    modal run -m cuda_harness.bench_throughput \\
        --targets 1L2Y,1CRN,1UBQ \\
        --num-steps 200 \\
        --num-recycles 3 \\
        --num-repeats 3 \\
        --warmup 1 \\
        --output-dir /tmp/chai_mlx_cuda/throughput
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

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
    fasta_for,
    image,
)


N_DIFFUSION_SAMPLES = 5


@app.function(
    timeout=30 * MINUTES,
    gpu="H100",
    volumes={MODELS_DIR: chai_model_volume, OUTPUTS_DIR: chai_outputs_volume},
    image=image,
)
def cuda_benchmark(
    target: str,
    sequence: str,
    num_recycles: int,
    num_steps: int,
    num_repeats: int,
    warmup: int,
    run_id: str,
) -> dict:
    """Benchmark one target on CUDA and return per-module timings."""
    import math
    import time

    import numpy as np
    import torch
    from einops import rearrange, repeat

    from chai_lab.chai1 import (
        Collate,
        DiffusionConfig as _Cfg,
        TokenBondRestraint,
        _component_moved_to,
        feature_factory,
        make_all_atom_feature_context,
    )
    from chai_lab.data.collate.utils import AVAILABLE_MODEL_SIZES
    from chai_lab.model.diffusion_schedules import InferenceNoiseSchedule
    from chai_lab.model.utils import center_random_augmentation
    from chai_lab.utils.tensor_utils import move_data_to_device, set_seed, und_self

    # Match chai-lab's ``@torch.no_grad`` on ``run_folding_on_context``
    # so we don't accumulate a 70+ GB autograd graph.
    torch.set_grad_enabled(False)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    fasta_path = Path("/tmp/input.fasta")
    fasta_path.write_text(fasta_for(target, sequence).strip())

    work_dir = OUTPUTS_DIR / run_id / target
    work_dir.mkdir(parents=True, exist_ok=True)
    feature_context = make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=work_dir / "features",
        use_esm_embeddings=False,
        use_msa_server=False,
        use_templates_server=False,
        esm_device=device,
    )

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

    bond_ft = TokenBondRestraint().generate(batch=batch).data

    inference_noise_schedule = InferenceNoiseSchedule(
        s_max=_Cfg.S_tmax, s_min=4e-4, p=7.0, sigma_data=_Cfg.sigma_data
    )

    module_times: dict[str, list[float]] = {
        "feature_embedding_ms": [],
        "bond_projection_ms": [],
        "token_embedder_ms": [],
        "trunk_recycle_ms": [],
        "diffusion_step_ms": [],
        "diffusion_total_ms": [],
        "confidence_ms": [],
        "end_to_end_ms": [],
    }

    total_trials = warmup + num_repeats
    for trial in range(total_trials):
        is_warmup = trial < warmup
        label = "warmup" if is_warmup else "repeat"
        print(f"[bench] {target} trial {trial + 1}/{total_trials} ({label})")
        set_seed([42 + trial])
        torch.cuda.synchronize()
        trial_t0 = time.perf_counter()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with _component_moved_to("feature_embedding.pt", device) as feature_embedding:
            embedded_features = feature_embedding.forward(
                crop_size=model_size,
                move_to_device=device,
                return_on_cpu=False,
                **features,
            )
        torch.cuda.synchronize()
        feat_ms = (time.perf_counter() - t0) * 1000

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

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with _component_moved_to("bond_loss_input_proj.pt", device) as bond_loss_input_proj:
            trunk_bond_feat, structure_bond_feat = bond_loss_input_proj.forward(
                return_on_cpu=False,
                move_to_device=device,
                crop_size=model_size,
                input=bond_ft,
            ).chunk(2, dim=-1)
        torch.cuda.synchronize()
        bond_ms = (time.perf_counter() - t0) * 1000

        token_pair_input_feats = token_pair_input_feats + trunk_bond_feat.to(
            token_pair_input_feats.dtype
        )
        token_pair_structure_input_feats = token_pair_structure_input_feats + structure_bond_feat.to(
            token_pair_structure_input_feats.dtype
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with _component_moved_to("token_embedder.pt", device) as token_input_embedder:
            (
                token_single_initial_repr,
                token_single_structure_input,
                token_pair_initial_repr,
            ) = token_input_embedder.forward(
                return_on_cpu=False,
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
        torch.cuda.synchronize()
        tok_ms = (time.perf_counter() - t0) * 1000

        token_single_trunk_repr = token_single_initial_repr
        token_pair_trunk_repr = token_pair_initial_repr
        recycle_ms_list: list[float] = []
        for _ in range(num_recycles):
            torch.cuda.synchronize()
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
            recycle_ms_list.append((time.perf_counter() - t0) * 1000)

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

        sigmas = inference_noise_schedule.get_schedule(device=device, num_timesteps=num_steps)
        gammas = torch.where(
            (sigmas >= _Cfg.S_tmin) & (sigmas <= _Cfg.S_tmax),
            min(_Cfg.S_churn / num_steps, math.sqrt(2) - 1),
            0.0,
        )
        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        _, num_atoms = atom_single_mask.shape
        atom_pos = sigmas[0] * torch.randn(
            1 * N_DIFFUSION_SAMPLES, num_atoms, 3, device=device
        )

        def _denoise(diff_mod, atom_pos_in, sigma, ds):
            noised = rearrange(atom_pos_in, "(b ds) ... -> b ds ...", ds=ds).contiguous()
            noise_sigma = repeat(sigma, " -> b ds", b=1, ds=ds)
            return diff_mod.forward(
                atom_noised_coords=noised.float(),
                noise_sigma=noise_sigma.float(),
                crop_size=model_size,
                **static_diffusion_inputs,
            )

        step_times: list[float] = []
        torch.cuda.synchronize()
        diff_t0 = time.perf_counter()
        with _component_moved_to("diffusion_module.pt", device=device) as diffusion_module:
            for sigma_curr, sigma_next, gamma_curr in sigmas_and_gammas:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                atom_pos = center_random_augmentation(
                    atom_pos,
                    atom_single_mask=repeat(
                        atom_single_mask_dev, "b a -> (b ds) a", ds=N_DIFFUSION_SAMPLES
                    ),
                )
                noise = _Cfg.S_noise * torch.randn(atom_pos.shape, device=atom_pos.device)
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
                if sigma_next != 0 and _Cfg.second_order:
                    denoised_pos2 = _denoise(
                        diff_mod=diffusion_module,
                        atom_pos_in=atom_pos,
                        sigma=sigma_next,
                        ds=N_DIFFUSION_SAMPLES,
                    )
                    d_i_prime = (atom_pos - denoised_pos2) / sigma_next
                    atom_pos = atom_pos + (sigma_next - sigma_hat) * ((d_i_prime + d_i) / 2)
                torch.cuda.synchronize()
                step_times.append((time.perf_counter() - t0) * 1000)
        torch.cuda.synchronize()
        diff_total_ms = (time.perf_counter() - diff_t0) * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with _component_moved_to("confidence_head.pt", device=device) as confidence_head:
            for ds in range(N_DIFFUSION_SAMPLES):
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
        torch.cuda.synchronize()
        conf_ms = (time.perf_counter() - t0) * 1000

        torch.cuda.synchronize()
        e2e_ms = (time.perf_counter() - trial_t0) * 1000

        del static_diffusion_inputs
        torch.cuda.empty_cache()

        if is_warmup:
            continue
        module_times["feature_embedding_ms"].append(feat_ms)
        module_times["bond_projection_ms"].append(bond_ms)
        module_times["token_embedder_ms"].append(tok_ms)
        module_times["trunk_recycle_ms"].append(statistics.mean(recycle_ms_list))
        module_times["diffusion_step_ms"].append(statistics.mean(step_times))
        module_times["diffusion_total_ms"].append(diff_total_ms)
        module_times["confidence_ms"].append(conf_ms)
        module_times["end_to_end_ms"].append(e2e_ms)

    def _summarize(xs: list[float]) -> dict:
        if not xs:
            return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
        return {
            "mean": float(statistics.fmean(xs)),
            "std": float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0,
            "min": float(min(xs)),
            "max": float(max(xs)),
            "n": len(xs),
        }

    return {
        "target": target,
        "sequence": sequence,
        "n_tokens": len(sequence),
        "model_size": int(model_size),
        "num_recycles": num_recycles,
        "num_steps": num_steps,
        "num_repeats": num_repeats,
        "warmup": warmup,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "summary": {k: _summarize(v) for k, v in module_times.items()},
        "raw": module_times,
    }


def _fmt_summary(s: dict, key: str) -> str:
    block = s["summary"][key]
    if block["n"] == 0:
        return "—"
    return f"{block['mean']:.1f} ± {block['std']:.1f}"


@app.local_entrypoint()
def bench_throughput(
    targets: str = "1L2Y,1VII,1CRN,1UBQ",
    num_recycles: int = 3,
    num_steps: int = 200,
    num_repeats: int = 3,
    warmup: int = 1,
    output_dir: str = "/tmp/chai_mlx_cuda/throughput",
    run_id: str | None = None,
    ensure_weights: bool = True,
) -> None:
    import csv

    targets_list = [t.strip() for t in targets.split(",") if t.strip()]
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    rid = run_id or f"bench-{num_recycles}r-{num_steps}s"

    if ensure_weights:
        print("[modal] ensuring weights are on the volume")
        download_inference_dependencies.remote(force=False)

    all_rows: list[dict] = []
    for target in targets_list:
        if target not in DEFAULT_TARGETS:
            raise KeyError(
                f"Unknown target {target!r}. Known: {sorted(DEFAULT_TARGETS)}"
            )
        sequence = DEFAULT_TARGETS[target]
        print(f"[modal] bench -> {target} ({len(sequence)} residues)")
        result = cuda_benchmark.remote(
            target=target,
            sequence=sequence,
            num_recycles=num_recycles,
            num_steps=num_steps,
            num_repeats=num_repeats,
            warmup=warmup,
            run_id=rid,
        )
        dst = output_dir_path / f"{target}_throughput.json"
        dst.write_text(json.dumps(result, indent=2))
        all_rows.append(result)
        print(f"[modal]   saved {dst}")
        print(
            f"[modal]   trunk/recycle={_fmt_summary(result, 'trunk_recycle_ms')} ms  "
            f"diffusion/step={_fmt_summary(result, 'diffusion_step_ms')} ms  "
            f"end-to-end={_fmt_summary(result, 'end_to_end_ms')} ms"
        )

    csv_path = output_dir_path / "throughput.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "target",
                "n_tokens",
                "model_size",
                "gpu_name",
                "num_recycles",
                "num_steps",
                "num_repeats",
                "feature_embedding_ms_mean",
                "bond_projection_ms_mean",
                "token_embedder_ms_mean",
                "trunk_recycle_ms_mean",
                "diffusion_step_ms_mean",
                "diffusion_total_ms_mean",
                "confidence_ms_mean",
                "end_to_end_ms_mean",
            ]
        )
        for row in all_rows:
            s = row["summary"]
            writer.writerow(
                [
                    row["target"],
                    row["n_tokens"],
                    row["model_size"],
                    row["gpu_name"],
                    row["num_recycles"],
                    row["num_steps"],
                    row["num_repeats"],
                    s["feature_embedding_ms"]["mean"],
                    s["bond_projection_ms"]["mean"],
                    s["token_embedder_ms"]["mean"],
                    s["trunk_recycle_ms"]["mean"],
                    s["diffusion_step_ms"]["mean"],
                    s["diffusion_total_ms"]["mean"],
                    s["confidence_ms"]["mean"],
                    s["end_to_end_ms"]["mean"],
                ]
            )
    print(f"[modal] aggregate CSV -> {csv_path}")
