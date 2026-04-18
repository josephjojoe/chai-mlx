"""Measure CUDA's own run-to-run determinism for Chai-1 on Modal.

When interpreting an MLX-vs-CUDA Cα RMSD number, the natural question is:
*how different are two CUDA runs of the same seed?*  If CUDA is bit-exact
on replay, the full gap is "real" MLX-vs-CUDA drift; if CUDA disagrees
with itself by a measurable amount, some fraction of the gap is just
CUDA's own non-determinism (atomic reductions, cuDNN autotune, TF32 on
matmul epilogues).

This harness runs chai-lab end-to-end **twice in the same container**
on the same target + seed and dumps a compact NPZ with per-run:

* ``diffusion.atom_pos_final`` (the 5 sampled structures, (S, A, 3))
* ``confidence.pae`` / ``confidence.pde`` / ``confidence.plddt`` (the
  pre-ranking token-level tensors from ``StructureCandidates``)
* per-sample ``ranking`` summary (pTM, iPTM, aggregate, clashes)
* the per-sample CIF text, stored as raw bytes so the local companion
  can Kabsch-align and compute RMSD directly

We can optionally toggle two CUDA-side precision knobs:

* ``--precision default``         — out-of-the-box chai-lab behaviour
* ``--precision tf32_off``        — ``torch.backends.{cuda.matmul,cudnn}.allow_tf32 = False``
* ``--precision deterministic``   — sets ``CUBLAS_WORKSPACE_CONFIG`` and
                                    ``torch.use_deterministic_algorithms(True, warn_only=True)``

Together these let us answer:

1. Is CUDA deterministic under the default policy?  (``--precision default``)
2. If not, is TF32 the culprit?                     (``--precision tf32_off``)
3. If still not, is cuDNN atomic reduction the culprit? (``--precision deterministic``)

The local companion :mod:`scripts.cuda_determinism_report` loads the NPZ,
computes per-run Cα RMSDs, tensor-level deltas, and prints a summary.

Usage
-----

::

    # Quickest: two back-to-back default-precision runs of 1L2Y seed 42.
    modal run -m cuda_harness.run_determinism --targets 1L2Y --seeds 42

    # Check whether TF32 is contributing to any observed non-determinism.
    modal run -m cuda_harness.run_determinism \\
        --targets 1L2Y --seeds 42 --precision tf32_off

    # Full deterministic mode (slower, some ops may fall back).
    modal run -m cuda_harness.run_determinism \\
        --targets 1L2Y --seeds 42 --precision deterministic
"""

from __future__ import annotations

import io
import json
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


def _apply_precision_policy(policy: str) -> dict[str, object]:
    """Apply a CUDA precision policy before chai-lab runs anything.

    Returns the dict of settings we actually ended up with, for logging.
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
    timeout=45 * MINUTES,
    gpu="H100",
    volumes={MODELS_DIR: chai_model_volume, OUTPUTS_DIR: chai_outputs_volume},
    image=image,
)
def cuda_determinism(
    target: str,
    sequence: str,
    seed: int,
    num_recycles: int,
    num_steps: int,
    run_id: str,
    precision: str,
    n_repeats: int = 2,
) -> bytes:
    """Run chai-lab ``n_repeats`` times in the same container; return an NPZ.

    We run everything in one container to keep the GPU, driver, and
    allocator state constant across replays — this is the tightest
    possible definition of "same CUDA runtime". The only thing that
    changes between run A and run B is the PyTorch RNG state, which
    we re-seed identically, and anything cuDNN or cuBLAS chooses to
    do internally (workspace allocation, algorithm autotune, atomic
    reductions) that is not covered by the seed.
    """
    import time

    import numpy as np
    import torch

    from chai_lab.chai1 import run_inference

    settings = _apply_precision_policy(precision)

    torch.set_grad_enabled(False)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    fasta_path = Path("/tmp/input.fasta")
    fasta_path.write_text(fasta_for(target, sequence).strip())

    meta: dict = {
        "target": target,
        "seed": seed,
        "sequence": sequence,
        "num_recycles": num_recycles,
        "num_steps": num_steps,
        "n_diffusion_samples": N_DIFFUSION_SAMPLES,
        "n_repeats": n_repeats,
        "precision": precision,
        "precision_settings": settings,
        "gpu_name": gpu_name,
        "torch_version": torch.__version__,
        "n_tokens": len(sequence),
        "runs": [],
    }

    dump: dict[str, np.ndarray] = {}

    for run_idx in range(n_repeats):
        # chai-lab's ``run_inference`` reseeds inside ``run_folding_on_context``
        # via ``set_seed([seed])``.  We *also* reseed the global RNGs here to
        # be defensive — if CUDA is deterministic under this policy, run A
        # and run B should match bit-exactly regardless.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        work_dir = OUTPUTS_DIR / run_id / target / f"seed_{seed}" / f"run_{run_idx}"
        # ``run_inference`` asserts the output dir is empty; recreate each time.
        if work_dir.exists():
            import shutil

            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        candidates = run_inference(
            fasta_file=fasta_path,
            output_dir=work_dir,
            num_trunk_recycles=num_recycles,
            num_diffn_timesteps=num_steps,
            num_diffn_samples=N_DIFFUSION_SAMPLES,
            seed=seed,
            device=str(device),
            use_esm_embeddings=False,
            use_msa_server=False,
            use_templates_server=False,
            low_memory=True,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Per-sample CIF text. Store as raw bytes so the local side can diff
        # Cα coordinates directly using gemmi without re-running any model.
        for s, cif_path in enumerate(candidates.cif_paths):
            cif_text = Path(cif_path).read_text()
            dump[f"run_{run_idx}.cif_{s}"] = np.frombuffer(cif_text.encode(), dtype=np.uint8)

        # The pae/pde/plddt tensors on StructureCandidates are the scalar
        # scores used for ranking (token-level summaries), not the full logit
        # tensors. They still provide a sensitive determinism signal: if two
        # runs agree bit-exactly on 5 × 20 × 20 pae, CUDA is deterministic at
        # the confidence-head level.
        dump[f"run_{run_idx}.pae"] = candidates.pae.cpu().float().numpy()
        dump[f"run_{run_idx}.pde"] = candidates.pde.cpu().float().numpy()
        dump[f"run_{run_idx}.plddt"] = candidates.plddt.cpu().float().numpy()

        rank_records = []
        for rd in candidates.ranking_data:
            rank_records.append(
                {
                    "aggregate_score": float(rd.aggregate_score.item()),
                    "complex_ptm": float(rd.ptm_scores.complex_ptm.item()),
                    "interface_ptm": float(rd.ptm_scores.interface_ptm.item()),
                    "has_inter_chain_clashes": bool(
                        rd.clash_scores.has_inter_chain_clashes.item()
                    ),
                }
            )

        meta["runs"].append({
            "run_idx": run_idx,
            "seconds": elapsed,
            "ranking": rank_records,
            "cif_paths": [str(p) for p in candidates.cif_paths],
        })

    dump["_manifest_json"] = np.frombuffer(
        json.dumps(meta, indent=2).encode(), dtype=np.uint8
    )
    buf = io.BytesIO()
    np.savez_compressed(buf, **dump)
    chai_outputs_volume.commit()
    return buf.getvalue()


@app.local_entrypoint()
def run_determinism(
    targets: str = "1L2Y",
    seeds: str = "42",
    num_recycles: int = 3,
    num_steps: int = 200,
    precision: str = "default",
    n_repeats: int = 2,
    output_dir: str = "/tmp/chai_mlx_cuda/determinism",
    run_id: str | None = None,
    ensure_weights: bool = True,
) -> None:
    targets_list = [t.strip() for t in targets.split(",") if t.strip()]
    seeds_list = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    rid = run_id or f"det-{precision}-{num_recycles}r-{num_steps}s"

    if ensure_weights:
        print("[modal] ensuring weights are on the volume")
        download_inference_dependencies.remote(force=False)

    for target in targets_list:
        if target not in DEFAULT_TARGETS:
            raise KeyError(
                f"Unknown target {target!r}. Known: {sorted(DEFAULT_TARGETS)}"
            )
        sequence = DEFAULT_TARGETS[target]
        for seed in seeds_list:
            print(
                f"[modal] -> {target} seed={seed} precision={precision} "
                f"n_repeats={n_repeats}"
            )
            payload = cuda_determinism.remote(
                target=target,
                sequence=sequence,
                seed=seed,
                num_recycles=num_recycles,
                num_steps=num_steps,
                run_id=rid,
                precision=precision,
                n_repeats=n_repeats,
            )
            dst = output_dir_path / target / f"seed_{seed}_{precision}.npz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(payload)
            print(
                f"[modal]    wrote {len(payload) / (1 << 20):.1f} MB -> {dst}"
            )
    print("[modal] done")
