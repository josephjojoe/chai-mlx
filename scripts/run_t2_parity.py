"""Run MLX-side inference for the T2 parity targets (PTM1, LYSM, BIGL).

Mirrors the ``modal run -m cuda_harness.run_reference --targets
PTM1,LYSM,BIGL --seeds 42`` invocation on the CUDA side: same target
definitions (from ``cuda_harness.modal_common.DEFAULT_TARGETS``), same
diffusion knobs, same seed. Writes outputs under
``<output-dir>/<target>/seed_<seed>/`` so
``scripts/cuda_structure_sweep.py`` can pair them against the CUDA
reference CIFs.

Required environment: ``CHAI_MLX_ALLOW_MODIFIED_RESIDUES=1`` to fold
PTM1 (the MLX fail-fast validator rejects inline ``[SEP]`` tokens by
default; the check exists to stop users who don't know they need to
set the env var, not to deny the path).

Usage::

    export CHAI_MLX_ALLOW_MODIFIED_RESIDUES=1
    python3 scripts/run_t2_parity.py \\
        --weights-dir weights \\
        --output-dir /tmp/chai_mlx_cuda/t2_parity/mlx

Runs one target after the other (not in parallel) so the MLX Metal
allocator is fully reclaimed between targets on 16 GB Macs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("chai-lab", "esm-mlx"):
    p = REPO_ROOT / sub
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cuda_harness.modal_common import DEFAULT_TARGETS  # noqa: E402


T2_TARGETS = ("PTM1", "LYSM", "BIGL")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/chai_mlx_cuda/t2_parity/mlx"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--dtype", default="reference", choices=["reference", "float32"])
    parser.add_argument(
        "--targets",
        default=",".join(T2_TARGETS),
        help="Comma-separated subset of {PTM1,LYSM,BIGL}.",
    )
    args = parser.parse_args()

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir = args.output_dir / "_fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("CHAI_MLX_ALLOW_MODIFIED_RESIDUES", "1")

    summary: dict[str, dict] = {}
    for target in targets:
        if target not in DEFAULT_TARGETS:
            print(f"[skip] {target} not in DEFAULT_TARGETS", file=sys.stderr)
            continue
        tgt = DEFAULT_TARGETS[target]
        fasta_path = fasta_dir / f"{target}.fasta"
        fasta_path.write_text(tgt.to_fasta())

        out_dir = args.output_dir / target / f"seed_{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "inference.py"),
            "--weights-dir", str(args.weights_dir),
            "--fasta", str(fasta_path),
            "--output-dir", str(out_dir),
            "--seed", str(args.seed),
            "--recycles", str(args.recycles),
            "--num-steps", str(args.num_steps),
            "--num-samples", str(args.num_samples),
            "--dtype", args.dtype,
            "--esm-backend", "off",
            # CUDA side uses fasta_names_as_cif_chains=True in
            # cuda_harness/run_reference.py::cuda_inference; match
            # that here so cuda_structure_sweep.py pairs chains by
            # the same label on both sides.
            "--fasta-chain-names",
        ]
        print(f"[mlx] {target}: {' '.join(cmd)}")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, cwd=REPO_ROOT)
        wall = time.perf_counter() - t0
        summary[target] = {
            "returncode": proc.returncode,
            "wall_seconds": wall,
            "out_dir": str(out_dir),
        }
        print(f"[mlx] {target}: rc={proc.returncode} wall={wall:.1f}s")

    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    failed = [t for t, r in summary.items() if r["returncode"] != 0]
    if failed:
        print(f"[mlx] FAILED: {failed}", file=sys.stderr)
        sys.exit(1)
    print(f"[mlx] all done -> {args.output_dir}")


if __name__ == "__main__":
    main()
