"""Run a reference end-to-end chai-lab inference on CUDA via Modal.

This is the primary input for :mod:`scripts.cuda_structure_sweep`. For one
or more targets and seeds, it:

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

Attach a constraint CSV (must be a resource name under
``cuda_harness/constraints/``)::

    modal run -m cuda_harness.run_reference \\
        --targets 1CRN_CONSTR --seeds 42 \\
        --constraint-resource 1CRN_all_three.csv

``constraint_resource`` may also be pre-baked on the target (the
``1CRN_CONSTR`` target ships with its own default).  The CLI flag, when
provided, overrides the per-target default for every listed target.

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

import modal

from cuda_harness.modal_common import (
    MINUTES,
    MODELS_DIR,
    OUTPUTS_DIR,
    DEFAULT_TARGETS,
    Target,
    app,
    chai_model_volume,
    chai_outputs_volume,
    download_inference_dependencies,
    image,
)


N_DIFFUSION_SAMPLES = 5  # hard-coded in chai-lab

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


@dataclass
class CudaRun:
    target: str
    seed: int
    fasta: str
    cifs: list[bytes]
    scores: list[bytes]
    n_protein_residues: int
    n_nucleic_residues: int
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
    fasta: str,
    seed: int,
    num_recycles: int,
    num_steps: int,
    run_id: str,
    constraint_csv_bytes: bytes | None = None,
    use_esm_embeddings: bool = False,
    use_msa_server: bool = False,
    use_templates_server: bool = False,
) -> dict:
    """Run one CUDA chai-lab inference and return bytes for every sample."""
    import time

    import torch
    from chai_lab import chai1

    work_dir = Path("/tmp") / target / f"seed_{seed}"
    work_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = work_dir / "input.fasta"
    fasta_path.write_text(fasta.strip() + "\n")

    constraint_path: Path | None = None
    if constraint_csv_bytes is not None:
        constraint_path = work_dir / "constraints.csv"
        constraint_path.write_bytes(constraint_csv_bytes)

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
        use_esm_embeddings=use_esm_embeddings,
        use_msa_server=use_msa_server,
        use_templates_server=use_templates_server,
        constraint_path=constraint_path,
        # Use FASTA entity names as chain IDs so constraint CSVs can
        # reference chains by the labels the user wrote in the FASTA
        # header (matches chai_mlx.data.featurize.featurize_fasta).
        fasta_names_as_cif_chains=True,
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

    chai_outputs_volume.commit()

    return {
        "target": target,
        "seed": seed,
        "fasta": fasta,
        "wall_seconds": wall_seconds,
        "gpu_name": gpu_name,
        "cifs": cifs,
        "scores": scores,
    }


def _save_run(result: dict, target_meta: Target, output_dir: Path) -> Path:
    target = result["target"]
    seed = result["seed"]
    dst = output_dir / target / f"seed_{seed}"
    dst.mkdir(parents=True, exist_ok=True)
    for i, (cif, score) in enumerate(zip(result["cifs"], result["scores"])):
        (dst / f"pred.model_idx_{i}.cif").write_bytes(cif)
        (dst / f"scores.model_idx_{i}.npz").write_bytes(score)
    (dst / "input.fasta").write_text(result["fasta"])
    manifest = {
        "target": target,
        "seed": seed,
        "records": [
            {"kind": r.kind, "name": r.name, "sequence": r.sequence}
            for r in target_meta.records
        ],
        "kinds": sorted(target_meta.kinds),
        "description": target_meta.description,
        "n_protein_residues": target_meta.n_protein_residues,
        "n_nucleic_residues": target_meta.n_nucleic_residues,
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
    constraint_resource: str | None = None,
    use_esm_embeddings: bool = False,
    use_msa_server: bool = False,
    use_templates_server: bool = False,
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
    for name in targets_list:
        if name not in DEFAULT_TARGETS:
            raise KeyError(
                f"Unknown target {name!r}. Known: {sorted(DEFAULT_TARGETS)}"
            )
        target = DEFAULT_TARGETS[name]
        resource = constraint_resource or target.constraint_resource
        constraint_bytes = _load_constraint_bytes(resource)
        for seed in seeds_list:
            label = f"{name} seed={seed}"
            if resource:
                label += f" constraints={resource}"
            print(f"[modal] -> {label}")
            result = cuda_inference.remote(
                target=name,
                fasta=target.to_fasta(),
                seed=seed,
                num_recycles=num_recycles,
                num_steps=num_steps,
                run_id=rid,
                constraint_csv_bytes=constraint_bytes,
                use_esm_embeddings=use_esm_embeddings,
                use_msa_server=use_msa_server,
                use_templates_server=use_templates_server,
            )
            dst = _save_run(result, target, output_dir_path)
            print(
                f"[modal]    wrote {len(result['cifs'])} CIFs + "
                f"{len(result['scores'])} score files -> {dst} "
                f"({result['wall_seconds']:.1f}s on {result['gpu_name']})"
            )

    print(f"[modal] done. reference outputs at {output_dir_path}")
