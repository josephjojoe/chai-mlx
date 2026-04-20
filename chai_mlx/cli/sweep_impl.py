"""Drive the local MLX side of the expanded ship-validation sweep.

Canonical implementation of the ``chai-mlx-sweep`` console script
declared in ``pyproject.toml`` and of the legacy
``scripts/run_mlx_sweep.py`` forwarder. Logic lives here so the
binary works after ``pip install chai-mlx`` from PyPI, without needing
the ``scripts/`` directory on disk.

Runs ``ChaiMLX.run_inference`` on each target in turn, paired with ESM
embeddings from a pre-computed cache (see
:mod:`chai_mlx.cli.precompute_esm_impl`). One subprocess per target so
the MLX Metal allocator is fully reclaimed between targets -- essential
on 16 GB Macs where ESM-3B + chai-mlx would otherwise compete for
memory.

Each subprocess writes:

* ``<mlx-dir>/<target>/seed_<seed>/pred.model_idx_*.cif`` -- one per
  sample, in chai-lab CIF format (with per-atom pLDDT B-factors, so
  PyMOL / ChimeraX can colour by confidence).
* ``<mlx-dir>/<target>/seed_<seed>/scores.json`` -- ``aggregate_score``,
  ``ptm``, ``iptm`` per sample.
* ``<mlx-dir>/<target>/seed_<seed>/manifest.json`` -- dtype, recycles,
  steps, wall-clock seconds.

The caller drives the subprocesses via ``subprocess.run`` so a crash
in one target does not take down the whole sweep.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# Canonical repo root (resolved via the ``chai_mlx`` package install
# location).  ``Path(__file__).resolve().parents[2]`` walks
# chai_mlx/cli/sweep_impl.py -> chai_mlx/cli -> chai_mlx -> repo root.
# Under a non-editable install this lands inside site-packages, which
# is fine because the subprocess worker below only needs it to pin
# sys.path for chai-lab/esm-mlx submodules when those are present.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _single_target_script(repo_root: Path) -> str:
    """Return the worker script as a string (no separate file needed)."""
    header = (
        f"from __future__ import annotations\n\n"
        f"import argparse\nimport json\nimport sys\nimport time\n"
        f"from pathlib import Path\n\n"
        f"REPO_ROOT = Path({str(repo_root)!r})\n"
    )
    body = r'''
for sub in ("chai-lab", "esm-mlx"):
    p = REPO_ROOT / sub
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Install the RDKit-timeout workaround *before* the first chai-lab import
# so both the ``make_all_atom_feature_context`` call below and the
# downstream ``featurize_fasta`` go through the patched decorator on macOS.
from chai_mlx.data._rdkit_timeout_patch import apply_rdkit_timeout_patch
apply_rdkit_timeout_patch()

import mlx.core as mx

from chai_lab.chai1 import Collate, feature_factory, make_all_atom_feature_context
from chai_lab.data.io.cif_utils import get_chain_letter, save_to_cif

from chai_mlx import ChaiMLX
from chai_mlx.data.featurize import featurize_fasta
from cuda_harness.modal_common import DEFAULT_TARGETS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--num-steps", type=int, required=True)
    p.add_argument("--num-recycles", type=int, required=True)
    p.add_argument("--num-samples", type=int, required=True)
    p.add_argument("--dtype", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--feature-dir", type=Path, required=True)
    p.add_argument("--esm-backend", choices=["off", "mlx", "mlx_cache"], required=True)
    p.add_argument("--esm-cache-dir", type=Path, default=None)
    args = p.parse_args()

    target = DEFAULT_TARGETS[args.target]
    fasta_path = args.feature_dir / f"{args.target}.fasta"
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(target.to_fasta())

    # Reference chai-lab feature context (needed for save_to_cif).
    import torch
    ref_ctx = make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=args.feature_dir / "ref_features",
        entity_name_as_subchain=True,
        use_esm_embeddings=False,
        use_msa_server=False,
        use_templates_server=False,
        esm_device=torch.device("cpu"),
    )
    collator = Collate(feature_factory=feature_factory, num_key_atoms=128, num_query_atoms=32)
    output_batch = collator([ref_ctx])["inputs"]
    asym_entity_names = {i: get_chain_letter(i) for i, _ in enumerate(ref_ctx.chains, start=1)}

    ctx = featurize_fasta(
        fasta_path,
        output_dir=args.feature_dir / "mlx_features",
        esm_backend=args.esm_backend,
        esm_cache_dir=args.esm_cache_dir,
        use_msa_server=False,
        use_templates_server=False,
    )

    mx.random.seed(args.seed)
    t_load = time.perf_counter()
    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False, compute_dtype=args.dtype)
    t_load = time.perf_counter() - t_load

    t0 = time.perf_counter()
    result = model.run_inference(
        ctx,
        recycles=args.num_recycles,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
    )
    wall = time.perf_counter() - t0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    coords = np.array(result.coords.astype(mx.float32))  # (B, S, A, 3)
    agg = np.array(result.ranking.aggregate_score.astype(mx.float32))
    ptm = np.array(result.ranking.ptm.astype(mx.float32))
    iptm = np.array(result.ranking.iptm.astype(mx.float32))
    per_atom_plddt_np = (
        np.array(result.ranking.per_atom_plddt.astype(mx.float32))
        if result.ranking.per_atom_plddt is not None
        else None
    )

    n_samples = coords.shape[1]
    cif_paths = []
    for s in range(n_samples):
        cif_path = args.out_dir / f"pred.model_idx_{s}.cif"
        bfactors = None
        if per_atom_plddt_np is not None:
            bfactors = torch.from_numpy(
                per_atom_plddt_np[:, s].astype("float32") * 100.0
            )
        save_to_cif(
            coords=torch.from_numpy(coords[:, s]),
            bfactors=bfactors,
            output_batch=output_batch,
            write_path=cif_path,
            asym_entity_names=asym_entity_names,
        )
        cif_paths.append(str(cif_path))

    scores_path = args.out_dir / "scores.json"
    scores_path.write_text(json.dumps({
        "aggregate_score": agg.reshape(-1).tolist(),
        "ptm": ptm.reshape(-1).tolist(),
        "iptm": iptm.reshape(-1).tolist(),
    }, indent=2))

    manifest = {
        "target": args.target,
        "seed": args.seed,
        "dtype": args.dtype,
        "num_recycles": args.num_recycles,
        "num_steps": args.num_steps,
        "num_samples": args.num_samples,
        "esm_backend": args.esm_backend,
        "esm_cache_dir": str(args.esm_cache_dir) if args.esm_cache_dir else None,
        "wall_seconds": wall,
        "weights_load_seconds": t_load,
        "cif_paths": cif_paths,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[mlx-worker] {args.target} seed={args.seed} -> {args.out_dir} ({wall:.1f}s)")


if __name__ == "__main__":
    main()
'''
    return header + body


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument(
        "--targets",
        required=True,
        help="Comma-separated target names (keys of DEFAULT_TARGETS).",
    )
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--mlx-dir", type=Path, default=Path("/tmp/chai_mlx_cuda/mlx_expanded"))
    parser.add_argument("--feature-dir", type=Path, default=Path("/tmp/chai_mlx_cuda/mlx_expanded_features"))
    parser.add_argument("--esm-backend", choices=["off", "mlx", "mlx_cache"], default="off")
    parser.add_argument("--esm-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--esm-targets",
        default=None,
        help="Optional comma-separated subset of targets that should use the "
             "esm backend (remaining targets fall back to --esm-backend=off). "
             "Handy when one target is the designated ESM-evaluation target.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip targets whose manifest.json already exists under --mlx-dir.",
    )
    args = parser.parse_args(argv)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    esm_targets = None
    if args.esm_targets is not None:
        esm_targets = {t.strip() for t in args.esm_targets.split(",") if t.strip()}

    worker = args.feature_dir / "_mlx_single_target.py"
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    worker.write_text(_single_target_script(REPO_ROOT))

    overall_t0 = time.perf_counter()
    rows: list[dict] = []
    for name in targets:
        for seed in seeds:
            out_dir = args.mlx_dir / args.dtype / name / f"seed_{seed}"
            if args.skip_existing and (out_dir / "manifest.json").is_file():
                print(f"[mlx-sweep] skipping {name} seed={seed} (manifest exists)")
                continue

            effective_backend = args.esm_backend
            if esm_targets is not None and name not in esm_targets:
                effective_backend = "off"

            cmd = [
                sys.executable, str(worker),
                "--weights-dir", str(args.weights_dir),
                "--target", name,
                "--seed", str(seed),
                "--num-steps", str(args.num_steps),
                "--num-recycles", str(args.num_recycles),
                "--num-samples", str(args.num_samples),
                "--dtype", args.dtype,
                "--out-dir", str(out_dir),
                "--feature-dir", str(args.feature_dir / name),
                "--esm-backend", effective_backend,
            ]
            if effective_backend == "mlx_cache":
                if args.esm_cache_dir is None:
                    raise SystemExit("--esm-backend mlx_cache requires --esm-cache-dir")
                cmd.extend(["--esm-cache-dir", str(args.esm_cache_dir)])

            print(f"[mlx-sweep] -> {name} seed={seed} esm={effective_backend}")
            t0 = time.perf_counter()
            try:
                subprocess.run(cmd, check=True, cwd=REPO_ROOT)
                status = "ok"
            except subprocess.CalledProcessError as e:
                status = f"fail (rc={e.returncode})"
            elapsed = time.perf_counter() - t0

            row = {
                "target": name,
                "seed": seed,
                "esm_backend": effective_backend,
                "dtype": args.dtype,
                "wall_seconds": elapsed,
                "status": status,
                "out_dir": str(out_dir),
            }
            rows.append(row)
            print(f"[mlx-sweep]    {status} in {elapsed:.1f}s")

    total = time.perf_counter() - overall_t0
    summary = {
        "total_seconds": total,
        "runs": rows,
    }
    summary_path = args.mlx_dir / "sweep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[mlx-sweep] done in {total:.1f}s; summary -> {summary_path}")


if __name__ == "__main__":
    main()
