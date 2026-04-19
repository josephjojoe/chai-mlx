"""Compare MLX and CUDA predictions against experimental PDB structures.

Handles the three awkward cases that defeat ``cuda_structure_sweep.py
--compare-pdb`` naively:

* Synthetic target names that don't map to real PDB IDs (e.g.
  ``1UBQ_ESM`` → ``1UBQ``, ``1CRN_CONSTR`` → ``1CRN``).
* Multimers where the PDB deposit has multiple copies of each chain
  and our prediction is two chains.  We pick the shortest matching
  representative chain per entity by length.
* Nucleic-acid targets where the canonical metric is P-backbone not
  Cα.

For each (target, sample) we emit:

* **mlx_vs_pdb_A** — Kabsch-aligned RMSD of the MLX prediction against
  the experimental coordinates (Cα for proteins, P for DNA/RNA).
* **cuda_vs_pdb_A** — same for CUDA.
* **mlx_minus_cuda_A** — negative means MLX is closer to the PDB.

Results are printed per-sample and saved as CSV + JSON under
``/tmp/chai_mlx_cuda/findings/``.

Usage::

    python scripts/compare_vs_pdb.py \\
        --reference-dir /tmp/chai_mlx_cuda/expanded_noesm \\
        --mlx-dir       /tmp/chai_mlx_cuda/mlx_expanded/bfloat16
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for sub in ("chai-lab",):
    p = REPO_ROOT / sub
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Map synthetic target names to real PDB IDs and which entities map to
# which PDB chains.  ``chain_lengths`` is used only to pick the right
# representative when the PDB deposit has multiple copies of a chain.
_TARGET_TO_PDB = {
    "1L2Y": {"pdb_id": "1L2Y", "entity": "protein", "chain_lengths": [20]},
    "1VII": {"pdb_id": "1VII", "entity": "protein", "chain_lengths": [35]},
    "1CRN": {"pdb_id": "1CRN", "entity": "protein", "chain_lengths": [46]},
    "1UBQ": {"pdb_id": "1UBQ", "entity": "protein", "chain_lengths": [76]},
    "1UBQ_ESM": {"pdb_id": "1UBQ", "entity": "protein", "chain_lengths": [76]},
    "1CRN_CONSTR": {"pdb_id": "1CRN", "entity": "protein", "chain_lengths": [46]},
    # Barnase (110 aa) + barstar (89 aa); 1BRS has three copies of each.
    "1BRS": {"pdb_id": "1BRS", "entity": "protein", "chain_lengths": [110, 89]},
    # FKBP-12 (107 residues) + FK506 ligand.  For the protein side
    # compare against the 107-aa chain.
    "1FKB": {"pdb_id": "1FKB", "entity": "protein", "chain_lengths": [107]},
    # TIM (7TIM is the dimer; the monomer is 249 residues).
    "7TIM": {"pdb_id": "7TIM", "entity": "protein", "chain_lengths": [249]},
    "1BNA": {"pdb_id": "1BNA", "entity": "dna", "chain_lengths": [12, 12]},
}


def _fetch_pdb(pdb_id: str) -> Path:
    cache = REPO_ROOT / ".pdb_cache"
    cache.mkdir(exist_ok=True)
    path = cache / f"{pdb_id.upper()}.pdb"
    if not path.is_file():
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", path
        )
    return path


def _pdb_chains_atoms(pdb_id: str, which: str) -> dict[str, np.ndarray]:
    """Extract per-chain reference atoms from the PDB deposit.

    ``which="ca"`` returns Cα (proteins).  ``which="p"`` returns the
    phosphate backbone (DNA/RNA).
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, _fetch_pdb(pdb_id))

    target_atom = "CA" if which == "ca" else "P"
    out: dict[str, np.ndarray] = {}
    for model in structure:
        for chain in model:
            coords: list[tuple[float, float, float]] = []
            for residue in chain:
                # Skip heteroatoms / water.
                if residue.id[0] != " ":
                    continue
                if target_atom in residue:
                    coords.append(tuple(residue[target_atom].get_vector().get_array()))
            if coords:
                out[chain.id] = np.array(coords, dtype=np.float64)
        break  # one NMR / X-ray model is enough
    return out


def _predicted_chains_atoms(cif_path: Path, which: str) -> dict[str, np.ndarray]:
    import gemmi

    s = gemmi.read_structure(str(cif_path))
    target_atom = "CA" if which == "ca" else "P"
    out: dict[str, list[tuple[float, float, float]]] = {}
    for model in s:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.name.strip() == target_atom:
                        out.setdefault(chain.name, []).append(
                            (atom.pos.x, atom.pos.y, atom.pos.z)
                        )
        break
    return {k: np.array(v, dtype=np.float64) for k, v in out.items() if v}


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.shape != b.shape or a.shape[0] < 3:
        return None
    a_c = a - a.mean(0, keepdims=True)
    b_c = b - b.mean(0, keepdims=True)
    h = a_c.T @ b_c
    u, _s, vt = np.linalg.svd(h)
    det = np.linalg.det(vt.T @ u.T)
    r = vt.T @ np.diag([1.0, 1.0, np.sign(det)]) @ u.T
    pred = a_c @ r.T
    return float(np.sqrt(((pred - b_c) ** 2).sum(-1).mean()))


def _pick_representative_chains(
    pdb_chains: dict[str, np.ndarray],
    expected_lengths: list[int],
) -> list[np.ndarray]:
    """Pick one PDB chain per expected entity by length.

    For each expected length, sort PDB chains by how close they are to
    that length and by chain letter (deterministic tie-break), picking
    the first unused match.  Returns a list of length
    ``len(expected_lengths)``; entries whose length does not match any
    PDB chain are ``None``.
    """
    used: set[str] = set()
    picks: list[np.ndarray | None] = []
    chains_by_length = sorted(
        pdb_chains.items(), key=lambda kv: (-len(kv[1]), kv[0])
    )
    for want in expected_lengths:
        candidates = [
            (k, v) for k, v in chains_by_length
            if k not in used and len(v) == want
        ]
        if candidates:
            name, arr = candidates[0]
            used.add(name)
            picks.append(arr)
            continue
        # Fall back to the closest length if no exact match.
        closest = sorted(
            ((k, v) for k, v in chains_by_length if k not in used),
            key=lambda kv: abs(len(kv[1]) - want),
        )
        if closest:
            name, arr = closest[0]
            used.add(name)
            picks.append(arr)
        else:
            picks.append(None)
    return picks


def _pair_pred_to_pdb(
    pred_chains: dict[str, np.ndarray],
    pdb_picks: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (pred, pdb) array pairs, matched positionally by length.

    Prediction chains are sorted by length (descending) and PDB picks
    are assumed already length-sorted.  Each pair is truncated to the
    shorter of the two.
    """
    pred_sorted = sorted(pred_chains.values(), key=lambda a: -len(a))
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for p, q in zip(pred_sorted, pdb_picks, strict=False):
        if q is None:
            continue
        n = min(len(p), len(q))
        if n < 3:
            continue
        pairs.append((p[:n], q[:n]))
    return pairs


def _rmsd_vs_pdb(
    pred_chains: dict[str, np.ndarray],
    pdb_picks: list[np.ndarray],
) -> float | None:
    """Joint Kabsch-aligned RMSD over all matched chains."""
    pairs = _pair_pred_to_pdb(pred_chains, pdb_picks)
    if not pairs:
        return None
    pred_cat = np.concatenate([p for p, _ in pairs], axis=0)
    pdb_cat = np.concatenate([q for _, q in pairs], axis=0)
    return _kabsch_rmsd(pred_cat, pdb_cat)


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("/tmp/chai_mlx_cuda/expanded_noesm"),
    )
    parser.add_argument(
        "--mlx-dir",
        type=Path,
        default=Path("/tmp/chai_mlx_cuda/mlx_expanded/bfloat16"),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed subdirectory name to read (seed_<seed>).",
    )
    parser.add_argument(
        "--targets",
        default=",".join(_TARGET_TO_PDB),
        help="Comma-separated list of target names (keys of _TARGET_TO_PDB).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/tmp/chai_mlx_cuda/findings/vs_pdb.csv"),
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("/tmp/chai_mlx_cuda/findings/vs_pdb.json"),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    rows: list[dict] = []

    for target in targets:
        if target not in _TARGET_TO_PDB:
            print(f"[skip] {target}: not in target map")
            continue
        info = _TARGET_TO_PDB[target]
        pdb_id = info["pdb_id"]
        entity = info["entity"]
        expected = info["chain_lengths"]
        which = "ca" if entity == "protein" else "p"

        try:
            pdb_chains = _pdb_chains_atoms(pdb_id, which)
        except Exception as exc:
            print(f"[skip] {target}: could not fetch or parse {pdb_id}: {exc}")
            continue

        pdb_picks = _pick_representative_chains(pdb_chains, expected)
        picked_lengths = [len(p) if p is not None else None for p in pdb_picks]
        print(
            f"[{target}] PDB={pdb_id}  entity={entity}  "
            f"expected_lengths={expected}  picked_lengths={picked_lengths}"
        )

        cuda_dir = args.reference_dir / target / f"seed_{args.seed}"
        mlx_dir = args.mlx_dir / target / f"seed_{args.seed}"
        if not cuda_dir.is_dir():
            print(f"  [skip] no CUDA output at {cuda_dir}")
            continue

        for sample_idx in range(5):
            cuda_cif = cuda_dir / f"pred.model_idx_{sample_idx}.cif"
            mlx_cif = mlx_dir / f"pred.model_idx_{sample_idx}.cif"
            if not cuda_cif.is_file():
                continue
            cuda_chains = _predicted_chains_atoms(cuda_cif, which)
            cuda_rmsd = _rmsd_vs_pdb(cuda_chains, pdb_picks)

            mlx_chains = (
                _predicted_chains_atoms(mlx_cif, which)
                if mlx_cif.is_file() else None
            )
            mlx_rmsd = _rmsd_vs_pdb(mlx_chains, pdb_picks) if mlx_chains else None

            rows.append({
                "target": target,
                "pdb_id": pdb_id,
                "entity": entity,
                "sample_idx": sample_idx,
                "pred_lengths_cuda": sorted((len(v) for v in cuda_chains.values()), reverse=True),
                "pred_lengths_mlx": (
                    sorted((len(v) for v in mlx_chains.values()), reverse=True)
                    if mlx_chains else None
                ),
                "pdb_lengths_picked": [len(p) if p is not None else None for p in pdb_picks],
                "mlx_vs_pdb_A": mlx_rmsd,
                "cuda_vs_pdb_A": cuda_rmsd,
                "mlx_minus_cuda_A": (
                    mlx_rmsd - cuda_rmsd
                    if (mlx_rmsd is not None and cuda_rmsd is not None) else None
                ),
            })

    # ---------- report ----------
    print("\n" + "=" * 110)
    header_cols = ("target", "pdb_id", "N", "sample", "MLX_vs_PDB", "CUDA_vs_PDB", "ΔMLX_CUDA")
    print(f"  {header_cols[0]:<12} {header_cols[1]:<6} {header_cols[2]:>4} "
          f"{header_cols[3]:>6} {header_cols[4]:>12} {header_cols[5]:>12} {header_cols[6]:>12}")
    print("  " + "-" * 106)

    def _f(x: float | None) -> str:
        return f"{x:9.2f}Å" if x is not None else "     --   "

    for r in rows:
        n = r["pdb_lengths_picked"][0] if r["pdb_lengths_picked"] else None
        n_str = f"{n:>4}" if n is not None else "   -"
        delta = r["mlx_minus_cuda_A"]
        delta_str = f"{delta:+9.2f}Å" if delta is not None else "     --   "
        print(
            f"  {r['target']:<12} {r['pdb_id']:<6} {n_str} "
            f"{r['sample_idx']:>6} {_f(r['mlx_vs_pdb_A']):>12} "
            f"{_f(r['cuda_vs_pdb_A']):>12} {delta_str:>12}"
        )

    # ---------- aggregate per target ----------
    print("\nPer-target means (smaller = closer to PDB):")
    by_target: dict[str, list[dict]] = {}
    for r in rows:
        by_target.setdefault(r["target"], []).append(r)
    summary: dict[str, dict] = {}
    for target, entries in by_target.items():
        mlx_vals = [e["mlx_vs_pdb_A"] for e in entries if e["mlx_vs_pdb_A"] is not None]
        cuda_vals = [e["cuda_vs_pdb_A"] for e in entries if e["cuda_vs_pdb_A"] is not None]
        summary[target] = {
            "n_samples": len(entries),
            "mlx_vs_pdb_mean_A": float(np.mean(mlx_vals)) if mlx_vals else None,
            "mlx_vs_pdb_best_A": float(np.min(mlx_vals)) if mlx_vals else None,
            "cuda_vs_pdb_mean_A": float(np.mean(cuda_vals)) if cuda_vals else None,
            "cuda_vs_pdb_best_A": float(np.min(cuda_vals)) if cuda_vals else None,
        }
        mean_m = summary[target]["mlx_vs_pdb_mean_A"]
        mean_c = summary[target]["cuda_vs_pdb_mean_A"]
        best_m = summary[target]["mlx_vs_pdb_best_A"]
        best_c = summary[target]["cuda_vs_pdb_best_A"]
        verdict = "—"
        if mean_m is not None and mean_c is not None:
            if mean_m < mean_c:
                verdict = f"MLX closer by {mean_c - mean_m:.2f}Å (mean)"
            else:
                verdict = f"CUDA closer by {mean_m - mean_c:.2f}Å (mean)"
        print(
            f"  {target:<12} MLX mean={mean_m:>6.2f}Å best={best_m:>6.2f}Å  "
            f"CUDA mean={mean_c:>6.2f}Å best={best_c:>6.2f}Å  →  {verdict}"
            if mean_m is not None and mean_c is not None
            else f"  {target:<12} (insufficient data)"
        )

    # ---------- save ----------
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=(
                "target", "pdb_id", "entity", "sample_idx",
                "pred_lengths_cuda", "pred_lengths_mlx", "pdb_lengths_picked",
                "mlx_vs_pdb_A", "cuda_vs_pdb_A", "mlx_minus_cuda_A",
            ),
        )
        writer.writeheader()
        for r in rows:
            row_out = dict(r)
            row_out["pred_lengths_cuda"] = json.dumps(row_out["pred_lengths_cuda"])
            row_out["pred_lengths_mlx"] = json.dumps(row_out["pred_lengths_mlx"])
            row_out["pdb_lengths_picked"] = json.dumps(row_out["pdb_lengths_picked"])
            writer.writerow(row_out)
    args.json.write_text(json.dumps({"rows": rows, "summary": summary}, indent=2))
    print(f"\n[save] csv  -> {args.csv}")
    print(f"[save] json -> {args.json}")


if __name__ == "__main__":
    main()
