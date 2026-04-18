"""Run a reference end-to-end chai-lab inference on CUDA via Modal.

This is the CUDA-side equivalent of :mod:`scripts.cif_seed_sweep` and is the
primary input for :mod:`scripts.cuda_structure_sweep`.  For one or more
targets and seeds, it:

* Runs ``chai_lab.chai1.run_inference`` on a CUDA GPU on Modal.
* Captures per-sample ``pred.model_idx_*.cif`` bytes + ``scores.model_idx_*.npz``
  bytes in memory.
* Optionally copies the full output directory into the shared
  ``chai-mlx-cuda-outputs`` volume so later runs can diff against it.
* Returns the artifacts to the local caller and writes them to
  ``--output-dir`` so they can be compared against local MLX runs.

Usage
-----

Run the default sweep (1L2Y, three seeds)::

    modal run -m cuda_harness.run_reference

Run a custom sweep::

    modal run -m cuda_harness.run_reference \\
        --targets 1L2Y,1VII \\
        --seeds 0,42,123 \\
        --num-steps 200 \\
        --num-recycles 3 \\
        --output-dir /tmp/chai_mlx_cuda/reference

Notes
-----

The function uses the same deterministic seed path as chai-lab does locally
(``set_seed``), so the CIFs produced here are bitwise reproducible for a
given target/seed/step-count on the same GPU model (we pin H100).
"""

from __future__ import annotations

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
    fasta_for,
    image,
)


N_DIFFUSION_SAMPLES = 5  # hard-coded in chai-lab


@dataclass
class CudaRun:
    target: str
    seed: int
    sequence: str
    cifs: list[bytes]
    scores: list[bytes]
    n_tokens: int
    wall_seconds: float
    gpu_name: str


@app.function(
    timeout=20 * MINUTES,
    gpu="H100",
    volumes={MODELS_DIR: chai_model_volume, OUTPUTS_DIR: chai_outputs_volume},
    image=image,
)
def cuda_inference(
    target: str,
    sequence: str,
    seed: int,
    num_recycles: int,
    num_steps: int,
    run_id: str,
) -> dict:
    """Run one CUDA chai-lab inference and return bytes for every sample."""
    import time

    import torch
    from chai_lab import chai1

    fasta_path = Path("/tmp/input.fasta")
    fasta_path.write_text(fasta_for(target, sequence).strip())
    output_dir = OUTPUTS_DIR / run_id / target / f"seed_{seed}"
    if output_dir.exists():
        # Empty it; run_inference requires an empty directory.
        for child in sorted(output_dir.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                try:
                    child.rmdir()
                except OSError:
                    pass
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    chai1.run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        use_esm_embeddings=False,
        use_msa_server=False,
        use_templates_server=False,
        num_trunk_recycles=num_recycles,
        num_diffn_timesteps=num_steps,
        num_diffn_samples=N_DIFFUSION_SAMPLES,
        num_trunk_samples=1,
        seed=seed,
        device="cuda:0",
        # chai-lab's default (True) keeps intermediates off GPU between
        # modules so even a 256-token crop fits on an 80 GB H100. Setting
        # this to False OOMs reliably.
        low_memory=True,
    )
    wall_seconds = time.perf_counter() - t0

    cifs: list[bytes] = []
    scores: list[bytes] = []
    for ii in range(N_DIFFUSION_SAMPLES):
        cifs.append((output_dir / f"pred.model_idx_{ii}.cif").read_bytes())
        scores.append((output_dir / f"scores.model_idx_{ii}.npz").read_bytes())

    n_tokens = len(sequence)
    chai_outputs_volume.commit()

    return {
        "target": target,
        "seed": seed,
        "sequence": sequence,
        "n_tokens": n_tokens,
        "wall_seconds": wall_seconds,
        "gpu_name": gpu_name,
        "cifs": cifs,
        "scores": scores,
    }


def _save_run(result: dict, output_dir: Path) -> Path:
    target = result["target"]
    seed = result["seed"]
    dst = output_dir / target / f"seed_{seed}"
    dst.mkdir(parents=True, exist_ok=True)
    for i, (cif, score) in enumerate(zip(result["cifs"], result["scores"])):
        (dst / f"pred.model_idx_{i}.cif").write_bytes(cif)
        (dst / f"scores.model_idx_{i}.npz").write_bytes(score)
    manifest = {
        "target": target,
        "seed": seed,
        "sequence": result["sequence"],
        "n_tokens": result["n_tokens"],
        "wall_seconds": result["wall_seconds"],
        "gpu_name": result["gpu_name"],
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return dst


@app.local_entrypoint()
def run_reference(
    targets: str = "1L2Y",
    seeds: str = "0,42,123",
    num_recycles: int = 3,
    num_steps: int = 200,
    output_dir: str = "/tmp/chai_mlx_cuda/reference",
    run_id: str | None = None,
    ensure_weights: bool = True,
) -> None:
    targets_list: list[str] = [t.strip() for t in targets.split(",") if t.strip()]
    seeds_list: list[int] = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    rid = run_id or f"ref-{num_recycles}r-{num_steps}s"

    if ensure_weights:
        print("[modal] ensuring weights are on the volume")
        download_inference_dependencies.remote(force=False)

    print(f"[modal] targets={targets_list} seeds={seeds_list}")
    for target in targets_list:
        if target not in DEFAULT_TARGETS:
            raise KeyError(
                f"Unknown target {target!r}. Known: {sorted(DEFAULT_TARGETS)}"
            )
        sequence = DEFAULT_TARGETS[target]
        for seed in seeds_list:
            print(f"[modal] -> {target} seed={seed}")
            result = cuda_inference.remote(
                target=target,
                sequence=sequence,
                seed=seed,
                num_recycles=num_recycles,
                num_steps=num_steps,
                run_id=rid,
            )
            dst = _save_run(result, output_dir_path)
            print(
                f"[modal]    wrote {len(result['cifs'])} CIFs + "
                f"{len(result['scores'])} score files -> {dst} "
                f"({result['wall_seconds']:.1f}s on {result['gpu_name']})"
            )

    print(f"[modal] done. reference outputs at {output_dir_path}")
