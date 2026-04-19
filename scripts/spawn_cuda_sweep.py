"""Fan out every CUDA-side experiment to Modal in parallel.

Modal can run many ``@app.function(gpu="H100")`` invocations concurrently
(each gets its own container + GPU), so spawning all the independent
jobs at once and harvesting their results as they finish is far faster
than running them serially.  This script does exactly that.

Jobs spawned
------------

1. ``cuda_determinism(target=1L2Y, seed=42, precision=default)``
2. ``cuda_determinism(target=1L2Y, seed=42, precision=tf32_off)``
3. ``cuda_determinism(target=1L2Y, seed=42, precision=deterministic)``
4. ``cuda_intermediates(target=1L2Y, seed=42, precision=tf32_off)``
5. ``cuda_reference(target=1VII, seeds=0,42,123)``
6. ``cuda_reference(target=1UBQ, seeds=0,42,123)``
7. ``cuda_mlx(target=1VII, seeds=0,42,123)``      (local MLX runs)
8. ``cuda_mlx(target=1UBQ, seeds=0,42,123)``      (local MLX runs)

Jobs 1-6 are Modal H100; job 4 also pairs with a local ``cuda_parity``
run when the payload lands.  Jobs 7-8 run locally after the reference
CIFs come back; they re-use ``scripts/cuda_structure_sweep.py`` to
extract Cα RMSDs / GDT-TS / lDDT against both CUDA and the PDB.

Usage
-----

::

    python scripts/spawn_cuda_sweep.py --output-root /tmp/chai_mlx_cuda

The script writes a ``_sweep_handles.json`` manifest alongside the
outputs so you can re-harvest without re-spawning if the local side
crashes mid-run.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Handle:
    name: str
    call_id: str
    target: str
    seed: int | None
    precision: str | None
    kind: str  # "determinism" | "intermediates" | "reference"
    # For ``determinism`` / ``intermediates``: path of the NPZ to write.
    # For ``reference``: directory where per-sample CIFs + scores get written.
    output_path: str


def _spawn_all(output_root: Path) -> list[Handle]:
    """Fire all Modal jobs via ``.spawn()``; return their FunctionCall handles."""
    from cuda_harness.run_determinism import cuda_determinism
    from cuda_harness.run_intermediates import cuda_intermediates
    from cuda_harness.run_reference import cuda_inference
    from cuda_harness.modal_common import DEFAULT_TARGETS

    handles: list[Handle] = []

    baseline = DEFAULT_TARGETS["1L2Y"]

    # ---- determinism sweeps (3 policies) ----
    for precision in ("default", "tf32_off", "deterministic"):
        run_id = f"det-{precision}-3r-200s"
        call = cuda_determinism.spawn(
            target=baseline.name,
            fasta=baseline.to_fasta(),
            seed=42,
            num_recycles=3,
            num_steps=200,
            run_id=run_id,
            precision=precision,
            n_repeats=2,
        )
        out = output_root / "determinism" / baseline.name / f"seed_42_{precision}.npz"
        handles.append(
            Handle(
                name=f"determinism_{precision}",
                call_id=call.object_id,
                target=baseline.name,
                seed=42,
                precision=precision,
                kind="determinism",
                output_path=str(out),
            )
        )
        print(f"[spawn] determinism {precision!r} -> call {call.object_id}")

    # ---- intermediates @ tf32_off (pair with cuda_parity.py locally) ----
    call = cuda_intermediates.spawn(
        target=baseline.name,
        fasta=baseline.to_fasta(),
        seed=42,
        num_recycles=3,
        num_steps=200,
        snapshot_steps=[1, 25, 50, 100, 150, 199, 200],
        run_id="intm-tf32_off-3r-200s",
        precision="tf32_off",
    )
    out = output_root / "intermediates" / baseline.name / "seed_42_tf32_off.npz"
    handles.append(
        Handle(
            name="intermediates_tf32_off",
            call_id=call.object_id,
            target=baseline.name,
            seed=42,
            precision="tf32_off",
            kind="intermediates",
            output_path=str(out),
        )
    )
    print(f"[spawn] intermediates tf32_off -> call {call.object_id}")

    # ---- reference runs for larger targets (1VII, 1UBQ) x (0, 42, 123) ----
    for name in ("1VII", "1UBQ"):
        target = DEFAULT_TARGETS[name]
        for seed in (0, 42, 123):
            call = cuda_inference.spawn(
                target=name,
                fasta=target.to_fasta(),
                seed=seed,
                num_recycles=3,
                num_steps=200,
                run_id="ref-3r-200s",
            )
            dst_dir = output_root / "reference" / name / f"seed_{seed}"
            handles.append(
                Handle(
                    name=f"reference_{name}_s{seed}",
                    call_id=call.object_id,
                    target=name,
                    seed=seed,
                    precision="default",
                    kind="reference",
                    output_path=str(dst_dir),
                )
            )
            print(f"[spawn] reference {name} seed={seed} -> call {call.object_id}")

    return handles


def _write_reference_bundle(result: dict, dst_dir: Path) -> int:
    """Write the per-sample CIFs + scores returned by ``cuda_inference``."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for i, (cif, score) in enumerate(zip(result["cifs"], result["scores"])):
        (dst_dir / f"pred.model_idx_{i}.cif").write_bytes(cif)
        (dst_dir / f"scores.model_idx_{i}.npz").write_bytes(score)
        total += len(cif) + len(score)
    manifest = {
        "target": result["target"],
        "seed": result["seed"],
        "sequence": result["sequence"],
        "n_tokens": result["n_tokens"],
        "wall_seconds": result["wall_seconds"],
        "gpu_name": result["gpu_name"],
    }
    (dst_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return total


def _harvest(handles: list[Handle], output_root: Path, poll_interval: int = 15) -> None:
    """Block until every FunctionCall returns; write payloads to disk."""
    from modal import FunctionCall

    pending = [h for h in handles]
    t0 = time.perf_counter()

    while pending:
        still_pending: list[Handle] = []
        for h in pending:
            fc = FunctionCall.from_id(h.call_id)
            try:
                payload = fc.get(timeout=0)
            except TimeoutError:
                still_pending.append(h)
                continue
            except Exception as exc:
                print(f"[harvest] {h.name}: FAILED: {type(exc).__name__}: {exc}")
                continue

            if h.kind == "reference":
                dst_dir = Path(h.output_path)
                n_bytes = _write_reference_bundle(payload, dst_dir)
                print(
                    f"[harvest] {h.name}: wrote {n_bytes / (1 << 20):.1f} MB "
                    f"-> {dst_dir}  (elapsed {time.perf_counter() - t0:.1f}s)"
                )
            else:
                dst = Path(h.output_path)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(payload)
                print(
                    f"[harvest] {h.name}: wrote {len(payload) / (1 << 20):.1f} MB "
                    f"-> {dst}  (elapsed {time.perf_counter() - t0:.1f}s)"
                )

        if still_pending:
            print(
                f"[harvest] {len(still_pending)}/{len(handles)} still running; "
                f"sleeping {poll_interval}s..."
            )
            time.sleep(poll_interval)
        pending = still_pending

    print(f"[harvest] all jobs complete in {time.perf_counter() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/chai_mlx_cuda"))
    parser.add_argument(
        "--skip-spawn",
        action="store_true",
        help="Re-harvest from an existing _sweep_handles.json",
    )
    parser.add_argument("--poll-interval", type=int, default=15)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    handles_path = args.output_root / "_sweep_handles.json"

    # Modal's ``.spawn()`` / ``.remote()`` calls need to run inside an
    # ``app.run()`` context (or from a deployed app).  We import the app
    # *and every function module* so all the ``@app.function`` decorators
    # have attached before entering the context manager.
    import modal
    from cuda_harness.modal_common import app, download_inference_dependencies
    from cuda_harness import run_determinism as _rd  # noqa: F401
    from cuda_harness import run_intermediates as _ri  # noqa: F401
    from cuda_harness import run_reference as _rr  # noqa: F401

    with modal.enable_output():
        with app.run():
            if args.skip_spawn:
                if not handles_path.exists():
                    raise SystemExit(f"no handles file at {handles_path}")
                print(f"[load] {handles_path}")
                handles = [Handle(**h) for h in json.loads(handles_path.read_text())]
            else:
                print("[modal] ensuring weights cache is populated")
                download_inference_dependencies.remote(force=False)

                handles = _spawn_all(args.output_root)
                handles_path.write_text(
                    json.dumps([asdict(h) for h in handles], indent=2)
                )
                print(f"[save] {handles_path}")

            _harvest(handles, args.output_root, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
