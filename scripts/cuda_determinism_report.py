"""Report CUDA run-to-run determinism for Chai-1.

Consumes an NPZ produced by
``modal run -m cuda_harness.run_determinism`` (which runs chai-lab
``n_repeats`` times in the same container on the same seed) and prints:

* Per-sample Cα RMSD between runs (Kabsch-aligned).
* Per-sample pae / pde / plddt scalar-tensor deltas (which are the
  values actually used for ranking).
* Per-sample ranking scalar deltas (aggregate, pTM, iPTM, clashes).

The key number is **max Cα RMSD across all matched (run_A[s],
run_B[s]) pairs**. Interpretation:

* If max ≤ 0.01 Å → CUDA is deterministic under this precision policy.
  Any observed MLX-vs-CUDA gap is "real" — it cannot be blamed on CUDA
  disagreeing with itself.
* If max is larger (say 0.1–0.5 Å) → CUDA is non-deterministic under
  this policy, and that much of the MLX-vs-CUDA 0.75 Å is "CUDA vs
  itself" noise rather than "MLX drift from CUDA".

We also print the pae/plddt max abs delta so the confidence-side
determinism is visible separately from the diffusion-side.

Usage
-----

::

    python scripts/cuda_determinism_report.py \\
        --npz /tmp/chai_mlx_cuda/determinism/1L2Y/seed_42_default.npz

    # Dump a JSON summary for diffing against other precision policies:
    python scripts/cuda_determinism_report.py \\
        --npz /tmp/chai_mlx_cuda/determinism/1L2Y/seed_42_default.npz \\
        --summary-json /tmp/chai_mlx_cuda/determinism/1L2Y_default.json
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class PairStat:
    run_a: int
    run_b: int
    sample: int
    ca_rmsd: float
    pae_max_abs: float
    pde_max_abs: float
    plddt_max_abs: float
    agg_delta: float
    ptm_delta: float


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in f.files}


def _read_manifest(data: dict[str, np.ndarray]) -> dict:
    raw = data.get("_manifest_json")
    if raw is None:
        return {}
    return json.loads(bytes(raw).decode())


def _ca_from_cif_text(cif_text: str) -> np.ndarray:
    """Extract Cα coordinates from a CIF string.

    We use ``gemmi`` since that's chai-lab's own writer. Parsing from the
    text avoids having to persist temp files and keeps the report
    self-contained.
    """
    import gemmi

    # ``read_structure_string`` takes a single positional argument (the raw
    # CIF text); earlier gemmi releases exposed a different-named helper.
    reader = getattr(gemmi, "read_structure_from_string", None) or gemmi.read_structure_string
    st = reader(cif_text)
    coords = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.name == "CA":
                    p = atom.pos
                    coords.append([p.x, p.y, p.z])
    if not coords:
        raise RuntimeError("No Cα atoms found in CIF text")
    return np.asarray(coords, dtype=np.float64)


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    a_rot = a @ R.T
    return float(np.sqrt(((a_rot - b) ** 2).sum(axis=1).mean()))


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())


def _run_sample_cifs(data: dict[str, np.ndarray], run_idx: int) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    s = 0
    while True:
        key = f"run_{run_idx}.cif_{s}"
        if key not in data:
            break
        text = bytes(data[key]).decode()
        out.append(_ca_from_cif_text(text))
        s += 1
    return out


def _run_scalars(data: dict[str, np.ndarray], run_idx: int) -> dict[str, np.ndarray]:
    return {
        "pae": data[f"run_{run_idx}.pae"],
        "pde": data[f"run_{run_idx}.pde"],
        "plddt": data[f"run_{run_idx}.plddt"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    data = _load_npz(args.npz)
    manifest = _read_manifest(data)
    runs = manifest.get("runs", [])
    n_repeats = len(runs) or manifest.get("n_repeats", 0)
    if n_repeats < 2:
        raise SystemExit(
            f"NPZ has {n_repeats} runs; need ≥2 to compute determinism"
        )

    precision = manifest.get("precision", "unknown")
    target = manifest.get("target", "?")
    seed = manifest.get("seed", "?")
    print(f"[load] {args.npz}")
    print(
        f"  target={target} seed={seed} precision={precision} "
        f"n_repeats={n_repeats} gpu={manifest.get('gpu_name', '?')}"
    )

    run_cifs: list[list[np.ndarray]] = [
        _run_sample_cifs(data, r) for r in range(n_repeats)
    ]
    run_scalars: list[dict[str, np.ndarray]] = [
        _run_scalars(data, r) for r in range(n_repeats)
    ]
    ranking_per_run = [r.get("ranking", []) for r in runs]

    n_samples = min(len(c) for c in run_cifs)
    pairs: list[PairStat] = []
    print(
        f"\n{'pair':>6} {'sample':>6}  {'Cα RMSD':>9}  {'pae|Δ|':>9}  "
        f"{'pde|Δ|':>9}  {'plddt|Δ|':>9}  {'agg|Δ|':>9}  {'pTM|Δ|':>9}"
    )
    for a in range(n_repeats):
        for b in range(a + 1, n_repeats):
            for s in range(n_samples):
                rmsd = _kabsch_rmsd(run_cifs[a][s], run_cifs[b][s])
                pae_d = _max_abs(run_scalars[a]["pae"][s], run_scalars[b]["pae"][s])
                pde_d = _max_abs(run_scalars[a]["pde"][s], run_scalars[b]["pde"][s])
                plddt_d = _max_abs(
                    run_scalars[a]["plddt"][s], run_scalars[b]["plddt"][s]
                )
                agg_d = (
                    abs(
                        ranking_per_run[a][s]["aggregate_score"]
                        - ranking_per_run[b][s]["aggregate_score"]
                    )
                    if ranking_per_run and ranking_per_run[a] and ranking_per_run[b]
                    else float("nan")
                )
                ptm_d = (
                    abs(
                        ranking_per_run[a][s]["complex_ptm"]
                        - ranking_per_run[b][s]["complex_ptm"]
                    )
                    if ranking_per_run and ranking_per_run[a] and ranking_per_run[b]
                    else float("nan")
                )
                pairs.append(
                    PairStat(
                        run_a=a,
                        run_b=b,
                        sample=s,
                        ca_rmsd=rmsd,
                        pae_max_abs=pae_d,
                        pde_max_abs=pde_d,
                        plddt_max_abs=plddt_d,
                        agg_delta=agg_d,
                        ptm_delta=ptm_d,
                    )
                )
                print(
                    f"  {a}↔{b}  {s:>6}  {rmsd:>9.4f}  {pae_d:>9.3e}  "
                    f"{pde_d:>9.3e}  {plddt_d:>9.3e}  {agg_d:>9.3e}  {ptm_d:>9.3e}"
                )

    rmsds = np.array([p.ca_rmsd for p in pairs])
    paes = np.array([p.pae_max_abs for p in pairs])
    plddts = np.array([p.plddt_max_abs for p in pairs])

    print("\n" + "=" * 70)
    print(f"Summary  ({target} seed={seed} precision={precision})")
    print("=" * 70)
    print(
        f"  Cα RMSD:        mean={rmsds.mean():.4f} Å  "
        f"median={np.median(rmsds):.4f} Å  max={rmsds.max():.4f} Å"
    )
    print(
        f"  pae  max|Δ|:    mean={paes.mean():.3e}  max={paes.max():.3e}"
    )
    print(
        f"  plddt max|Δ|:   mean={plddts.mean():.3e}  max={plddts.max():.3e}"
    )

    bit_exact_structure = rmsds.max() < 1e-3
    bit_exact_confidence = max(paes.max(), plddts.max()) < 1e-6
    print()
    print(
        f"  CUDA is {'deterministic' if bit_exact_structure else 'NON-deterministic'} "
        f"at the structure level under policy {precision!r} (max Cα RMSD = "
        f"{rmsds.max():.4f} Å, threshold 1e-3 Å)."
    )
    print(
        f"  CUDA is {'deterministic' if bit_exact_confidence else 'NON-deterministic'} "
        f"at the confidence level under policy {precision!r}."
    )

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "npz": str(args.npz),
            "manifest": manifest,
            "n_samples": n_samples,
            "pairs": [asdict(p) for p in pairs],
            "summary": {
                "ca_rmsd_mean": float(rmsds.mean()),
                "ca_rmsd_median": float(np.median(rmsds)),
                "ca_rmsd_max": float(rmsds.max()),
                "pae_max_delta": float(paes.max()),
                "plddt_max_delta": float(plddts.max()),
                "bit_exact_structure": bool(bit_exact_structure),
                "bit_exact_confidence": bool(bit_exact_confidence),
            },
        }
        args.summary_json.write_text(json.dumps(payload, indent=2))
        print(f"\n[save] summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
