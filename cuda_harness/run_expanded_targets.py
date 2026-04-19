"""Run CUDA reference inference across the expanded validation slate.

This harness is the CUDA half of the "ship-validation" sweep. It invokes
:func:`cuda_harness.run_reference.cuda_inference` with
``use_esm_embeddings=True`` (chai-lab's traced 3B fp16 checkpoint) on
every target matched by a ``--target-kinds`` filter, attaching the
target's default ``constraint_resource`` when one is present.

The five new validation axes we care about are encoded as kind tags on
:data:`cuda_harness.modal_common.DEFAULT_TARGETS`:

* ``multimer``   → 1BRS (barnase-barstar heterodimer, also tags 1BNA)
* ``ligand``     → 1FKB (FKBP-12 + FK506)
* ``long``       → 7TIM (248-residue TIM barrel)
* ``dna``        → 1BNA (Dickerson dodecamer)
* ``esm``        → 1UBQ_ESM (paired with a local MLX-ESM rerun)
* ``constraints``→ 1CRN_CONSTR (contact + pocket + covalent)

MSA and template servers stay off throughout (offline-only per design);
ESM-on-CUDA uses chai-lab's own traced checkpoint.

Usage
-----

Run the full expanded slate on a single seed::

    modal run -m cuda_harness.run_expanded_targets --seeds 42

Narrow to one axis::

    modal run -m cuda_harness.run_expanded_targets \\
        --target-kinds multimer,ligand --seeds 42

Override constraints globally (applies to every target for this run)::

    modal run -m cuda_harness.run_expanded_targets \\
        --target-kinds constraints --seeds 42 \\
        --constraint-resource 1CRN_all_three.csv
"""

from __future__ import annotations

from pathlib import Path

from cuda_harness.modal_common import (
    DEFAULT_TARGETS,
    app,
    download_inference_dependencies,
    filter_targets,
)
from cuda_harness.run_reference import (
    _load_constraint_bytes,
    _save_run,
    cuda_inference,
)


@app.local_entrypoint()
def run_expanded_targets(
    target_kinds: str = "multimer,ligand,long,dna,esm,constraints",
    seeds: str = "42",
    num_recycles: int = 3,
    num_steps: int = 200,
    output_dir: str = "/tmp/chai_mlx_cuda/expanded",
    run_id: str | None = None,
    ensure_weights: bool = True,
    constraint_resource: str | None = None,
    use_esm_embeddings: bool = True,
) -> None:
    targets = filter_targets(target_kinds)
    if not targets:
        raise ValueError(
            f"No targets match --target-kinds={target_kinds!r}. "
            f"Known kinds: "
            + ",".join(
                sorted({k for t in DEFAULT_TARGETS.values() for k in t.kinds})
            )
        )

    seeds_list: list[int] = [int(s.strip()) for s in seeds.split(",") if s.strip()]
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    rid = run_id or f"expanded-{num_recycles}r-{num_steps}s"

    if ensure_weights:
        print("[modal] ensuring weights are on the volume")
        download_inference_dependencies.remote(force=False)

    print(
        f"[modal] expanded sweep: {len(targets)} target(s), {len(seeds_list)} seed(s), "
        f"use_esm_embeddings={use_esm_embeddings}"
    )
    for name, target in targets.items():
        resource = constraint_resource or target.constraint_resource
        constraint_bytes = _load_constraint_bytes(resource)
        for seed in seeds_list:
            label = f"{name} [{','.join(sorted(target.kinds))}] seed={seed}"
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
            )
            dst = _save_run(result, target, output_dir_path)
            print(
                f"[modal]    wrote {len(result['cifs'])} CIFs + "
                f"{len(result['scores'])} score files -> {dst} "
                f"({result['wall_seconds']:.1f}s on {result['gpu_name']})"
            )

    print(f"[modal] done. expanded outputs at {output_dir_path}")
