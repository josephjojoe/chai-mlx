"""Structural validation: compare MLX vs MPS reference vs PDB ground truth.

For each test protein, runs both the MLX port and the PyTorch/MPS reference
through the full pipeline (FASTA → predicted 3D coordinates), then compares
predicted Cα coordinates against the experimental PDB structure.

Usage::

    python3 scripts/structural_validation.py --weights-dir weights/

Requires: biopython, torch, chai-lab (in chai-lab/), mlx, chai_mlx

The script:
  1. Downloads experimental PDB structures from RCSB
  2. Runs the PyTorch/MPS reference pipeline (chai-lab)
  3. Runs the MLX pipeline (chai_mlx)
  4. Extracts Cα coordinates from all three
  5. Computes RMSD, GDT-TS, and per-residue Cα error for each pair
  6. Prints a comparison table
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists():
    sys.path.insert(0, str(LOCAL_CHAI_LAB))


# ── Test proteins ────────────────────────────────────────────────────────
# Small, well-resolved, single-chain proteins that fit in 256-512 tokens.


@dataclass
class TestProtein:
    name: str
    pdb_id: str
    chain: str
    sequence: str
    description: str


TEST_PROTEINS = [
    TestProtein(
        name="1L2Y (Trp-cage)",
        pdb_id="1L2Y",
        chain="A",
        sequence="NLYIQWLKDGGPSSGRPPPS",
        description="20-residue miniprotein, NMR, ultra-small",
    ),
    TestProtein(
        name="1VII (villin HP35)",
        pdb_id="1VII",
        chain="A",
        sequence="LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        description="35-residue villin headpiece, well-studied fold",
    ),
    TestProtein(
        name="1CRN (crambin)",
        pdb_id="1CRN",
        chain="A",
        sequence="TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN",
        description="46-residue plant protein, 0.54Å resolution",
    ),
]


# ── PDB ground truth ────────────────────────────────────────────────────


def fetch_pdb_ca_coords(pdb_id: str, chain_id: str) -> np.ndarray:
    """Download PDB file and extract Cα coordinates for the given chain.

    Returns array of shape [num_residues, 3].
    """
    from Bio.PDB import PDBParser

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    cache_dir = REPO_ROOT / ".pdb_cache"
    cache_dir.mkdir(exist_ok=True)
    pdb_path = cache_dir / f"{pdb_id.upper()}.pdb"

    if not pdb_path.exists():
        print(f"    Downloading {pdb_id} from RCSB...")
        urllib.request.urlretrieve(url, pdb_path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    model = structure[0]
    chain = model[chain_id]

    ca_coords = []
    for residue in chain:
        if residue.id[0] != " ":
            continue  # skip heteroatoms
        if "CA" in residue:
            ca_coords.append(residue["CA"].get_vector().get_array())

    return np.array(ca_coords, dtype=np.float64)


# ── Metrics ──────────────────────────────────────────────────────────────


def kabsch_rmsd(pred: np.ndarray, true: np.ndarray) -> tuple[float, np.ndarray]:
    """Kabsch-aligned RMSD and per-residue distances.

    Both inputs: [N, 3] float64. Returns (rmsd, per_residue_distances).
    """
    assert pred.shape == true.shape, f"Shape mismatch: {pred.shape} vs {true.shape}"
    n = pred.shape[0]

    # Center both
    pred_c = pred - pred.mean(axis=0)
    true_c = true - true.mean(axis=0)

    # Kabsch rotation
    H = pred_c.T @ true_c
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    pred_aligned = pred_c @ R.T
    dists = np.sqrt(((pred_aligned - true_c) ** 2).sum(axis=-1))
    rmsd = float(np.sqrt((dists ** 2).mean()))
    return rmsd, dists


def gdt_ts(per_residue_dists: np.ndarray) -> float:
    """GDT-TS: average fraction of residues within 1, 2, 4, 8 Å."""
    thresholds = [1.0, 2.0, 4.0, 8.0]
    n = len(per_residue_dists)
    return float(np.mean([np.sum(per_residue_dists < t) / n for t in thresholds]))


def lddt_score(pred: np.ndarray, true: np.ndarray, cutoff: float = 15.0) -> float:
    """Simplified lDDT (local distance difference test).

    For each residue pair within `cutoff` in the true structure, check if the
    predicted distance is within 0.5, 1, 2, 4 Å of the true distance.
    """
    n = pred.shape[0]
    true_dists = np.sqrt(((true[:, None] - true[None, :]) ** 2).sum(axis=-1))
    pred_dists = np.sqrt(((pred[:, None] - pred[None, :]) ** 2).sum(axis=-1))

    mask = (true_dists < cutoff) & (np.eye(n) == 0)
    if mask.sum() == 0:
        return 0.0

    diff = np.abs(pred_dists - true_dists)
    thresholds = [0.5, 1.0, 2.0, 4.0]
    scores = [float((diff[mask] < t).mean()) for t in thresholds]
    return float(np.mean(scores))


# ── MLX prediction ───────────────────────────────────────────────────────


def predict_mlx(
    fasta_path: Path,
    weights_dir: Path,
    work_dir: Path,
    *,
    num_samples: int = 1,
    num_steps: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Run full MLX pipeline, return Cα coords [num_residues, 3]."""
    import mlx.core as mx

    mx.random.seed(seed)

    from chai_mlx import ChaiMLX
    from chai_mlx.data.featurize import featurize_fasta

    print("    Loading MLX model...")
    model = ChaiMLX.from_pretrained(weights_dir, strict=False)

    print("    Featurizing (via chai-lab pipeline)...")
    mlx_out_dir = work_dir / "mlx_features"
    mlx_out_dir.mkdir(parents=True, exist_ok=True)
    ctx = featurize_fasta(
        fasta_path,
        output_dir=mlx_out_dir,
        use_esm_embeddings=False,
        use_msa_server=False,
        use_templates_server=False,
    )

    print(f"    Running fold (samples={num_samples}, steps={num_steps})...")
    result = model.fold(ctx, recycles=3, num_samples=num_samples, num_steps=num_steps)

    # Pick best sample by aggregate score
    scores = np.array(result.ranking.aggregate_score.astype(mx.float32)).ravel()
    best_idx = int(scores.argmax())
    print(f"    Best sample: {best_idx} (score={float(scores[best_idx]):.4f})")

    coords = np.array(result.coords.astype(mx.float32))[0, best_idx]  # [num_atoms, 3]
    si = result.embeddings.structure_inputs
    atom_mask = np.array(si.atom_exists_mask.astype(mx.float32))[0]
    token_centre_atom_idx = getattr(si, "token_centre_atom_index", None)
    if token_centre_atom_idx is None:
        token_centre_atom_idx = si.token_reference_atom_index
    token_centre_atom_idx = np.array(token_centre_atom_idx)[0]
    token_mask = np.array(si.token_exists_mask.astype(mx.float32))[0]

    del model, result
    gc.collect()
    mx.clear_cache()

    return _extract_ca_coords(coords, atom_mask, token_centre_atom_idx, token_mask)


# ── Reference (MPS) prediction ──────────────────────────────────────────


def predict_reference(
    fasta_path: Path,
    work_dir: Path,
    *,
    num_samples: int = 1,
    num_steps: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Run full PyTorch/MPS reference pipeline, return Cα coords."""
    import torch
    from chai_lab.chai1 import (
        make_all_atom_feature_context,
        run_folding_on_context,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"    Running reference on {device}...")

    output_dir = work_dir / "reference_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_context = make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=work_dir / "ref_features",
        use_esm_embeddings=False,
        use_msa_server=False,
        use_templates_server=False,
        esm_device=device,
    )

    candidates = run_folding_on_context(
        feature_context,
        output_dir=output_dir,
        num_trunk_recycles=3,
        num_diffn_timesteps=num_steps,
        num_diffn_samples=num_samples,
        seed=seed,
        device=device,
        low_memory=True,
    )

    # Read best CIF by ranking score
    best_idx = 0
    best_score = -1.0
    for i, rd in enumerate(candidates.ranking_data):
        score = float(rd.aggregate_score.item())
        if score > best_score:
            best_score = score
            best_idx = i
    print(f"    Best sample: {best_idx} (score={best_score:.4f})")

    cif_path = candidates.cif_paths[best_idx]
    return _extract_ca_from_cif(cif_path)


def _extract_ca_from_cif(cif_path: Path) -> np.ndarray:
    """Extract Cα coordinates from a CIF file produced by chai-lab."""
    from Bio.PDB import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("pred", str(cif_path))
    model = structure[0]

    ca_coords = []
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            if "CA" in residue:
                ca_coords.append(residue["CA"].get_vector().get_array())

    return np.array(ca_coords, dtype=np.float64)


def _extract_ca_coords(
    all_coords: np.ndarray,
    atom_mask: np.ndarray,
    token_centre_atom_idx: np.ndarray,
    token_mask: np.ndarray,
) -> np.ndarray:
    """Extract Cα-like coordinates from the internal atom representation.

    For polymer residues, the token centre atom is the Cα.
    """
    n_tokens = int(token_mask.sum())
    ca_coords = []
    for t in range(n_tokens):
        if token_mask[t] < 0.5:
            continue
        centre_atom = int(token_centre_atom_idx[t])
        if centre_atom >= 0 and centre_atom < len(all_coords) and atom_mask[centre_atom] > 0.5:
            ca_coords.append(all_coords[centre_atom])

    return np.array(ca_coords, dtype=np.float64)


# ── Harness ──────────────────────────────────────────────────────────────


@dataclass
class ComparisonResult:
    protein: str
    pair: str  # "MLX vs PDB", "MPS vs PDB", "MLX vs MPS"
    n_residues: int
    rmsd: float
    gdt: float
    lddt: float
    median_dist: float
    max_dist: float


def compare_structures(
    name: str,
    pair_label: str,
    pred_ca: np.ndarray,
    true_ca: np.ndarray,
) -> ComparisonResult | None:
    """Compare two Cα coordinate arrays, handling length mismatches."""
    n_pred = len(pred_ca)
    n_true = len(true_ca)

    if n_pred == 0 or n_true == 0:
        print(f"    {pair_label}: empty coordinates (pred={n_pred}, true={n_true})")
        return None

    # Truncate to shorter length (padding tokens may add extra)
    n = min(n_pred, n_true)
    if n_pred != n_true:
        print(f"    {pair_label}: length mismatch pred={n_pred} vs true={n_true}, using first {n}")
    pred = pred_ca[:n]
    true = true_ca[:n]

    rmsd, dists = kabsch_rmsd(pred, true)
    gdt = gdt_ts(dists)
    ldt = lddt_score(pred, true)

    return ComparisonResult(
        protein=name,
        pair=pair_label,
        n_residues=n,
        rmsd=rmsd,
        gdt=gdt,
        lddt=ldt,
        median_dist=float(np.median(dists)),
        max_dist=float(dists.max()),
    )


def write_fasta(sequence: str, name: str, path: Path) -> Path:
    """Write a single-chain FASTA file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f">protein|name={name}\n{sequence}\n")
    return path


def run_protein(
    protein: TestProtein,
    weights_dir: Path,
    work_dir: Path,
    *,
    num_samples: int,
    num_steps: int,
    skip_reference: bool = False,
) -> list[ComparisonResult]:
    """Run all comparisons for one protein."""
    print(f"\n{'=' * 70}")
    print(f"  {protein.name}: {protein.description}")
    print(f"  Sequence ({len(protein.sequence)} residues): {protein.sequence[:50]}...")
    print(f"{'=' * 70}")

    results = []
    protein_dir = work_dir / protein.pdb_id

    # 1. PDB ground truth
    print(f"\n  [1/3] Fetching PDB ground truth ({protein.pdb_id})...")
    try:
        pdb_ca = fetch_pdb_ca_coords(protein.pdb_id, protein.chain)
        print(f"    Got {len(pdb_ca)} Cα atoms from PDB")
    except Exception as e:
        print(f"    FAILED to fetch PDB: {e}")
        pdb_ca = None

    # 2. MLX prediction
    print(f"\n  [2/3] Running MLX prediction...")
    fasta_path = write_fasta(protein.sequence, protein.name, protein_dir / "input.fasta")
    try:
        mlx_ca = predict_mlx(
            fasta_path, weights_dir, protein_dir,
            num_samples=num_samples, num_steps=num_steps,
        )
        print(f"    Got {len(mlx_ca)} Cα atoms from MLX")
    except Exception as e:
        print(f"    FAILED: {e}")
        import traceback; traceback.print_exc()
        mlx_ca = None

    # 3. Reference (MPS) prediction
    ref_ca = None
    if not skip_reference:
        print(f"\n  [3/3] Running MPS reference prediction...")
        try:
            ref_ca = predict_reference(
                fasta_path, protein_dir,
                num_samples=num_samples, num_steps=num_steps,
            )
            print(f"    Got {len(ref_ca)} Cα atoms from MPS reference")
        except Exception as e:
            print(f"    FAILED: {e}")
            import traceback; traceback.print_exc()
            ref_ca = None
    else:
        print(f"\n  [3/3] Skipping MPS reference (--skip-reference)")

    # 4. Comparisons
    print(f"\n  Computing metrics...")
    if mlx_ca is not None and pdb_ca is not None:
        r = compare_structures(protein.name, "MLX vs PDB", mlx_ca, pdb_ca)
        if r:
            results.append(r)

    if ref_ca is not None and pdb_ca is not None:
        r = compare_structures(protein.name, "MPS vs PDB", ref_ca, pdb_ca)
        if r:
            results.append(r)

    if mlx_ca is not None and ref_ca is not None:
        r = compare_structures(protein.name, "MLX vs MPS", mlx_ca, ref_ca)
        if r:
            results.append(r)

    return results


def print_results_table(results: list[ComparisonResult]) -> None:
    """Print a formatted summary table."""
    print(f"\n{'=' * 100}")
    print("STRUCTURAL VALIDATION SUMMARY")
    print(f"{'=' * 100}")
    print(f"  {'Protein':<22} {'Comparison':<14} {'N':>4} {'RMSD':>8} {'GDT-TS':>8} "
          f"{'lDDT':>8} {'Median':>8} {'Max':>8}")
    print(f"  {'─' * 22} {'─' * 14} {'─' * 4} {'─' * 8} {'─' * 8} "
          f"{'─' * 8} {'─' * 8} {'─' * 8}")

    for r in results:
        print(f"  {r.protein:<22} {r.pair:<14} {r.n_residues:>4} "
              f"{r.rmsd:>7.2f}Å {r.gdt:>7.1%} "
              f"{r.lddt:>7.1%} {r.median_dist:>7.2f}Å {r.max_dist:>7.2f}Å")

    print()
    print("  Key:")
    print("    RMSD  = Kabsch-aligned root-mean-square deviation of Cα atoms")
    print("    GDT-TS = avg fraction of Cα within 1/2/4/8 Å thresholds")
    print("    lDDT  = local distance difference test (Cα-only approximation)")
    print("    Median/Max = per-residue Cα distance after alignment")

    # Summary interpretation
    if results:
        mlx_pdb = [r for r in results if r.pair == "MLX vs PDB"]
        ref_pdb = [r for r in results if r.pair == "MPS vs PDB"]
        mlx_ref = [r for r in results if r.pair == "MLX vs MPS"]

        if mlx_pdb and ref_pdb:
            avg_mlx = np.mean([r.rmsd for r in mlx_pdb])
            avg_ref = np.mean([r.rmsd for r in ref_pdb])
            print(f"\n  Average RMSD vs PDB:  MLX={avg_mlx:.2f}Å  MPS={avg_ref:.2f}Å")
            if mlx_ref:
                avg_cross = np.mean([r.rmsd for r in mlx_ref])
                print(f"  Average RMSD MLX vs MPS: {avg_cross:.2f}Å")

            if abs(avg_mlx - avg_ref) < 2.0:
                print("  → MLX and MPS produce structures of comparable quality")
            elif avg_mlx < avg_ref + 2.0:
                print("  → MLX structures are reasonable (within 2Å of reference quality)")
            else:
                print("  → MLX structures show significant degradation — investigate further")


def main() -> None:
    parser = argparse.ArgumentParser(description="Structural validation: MLX vs MPS vs PDB")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Diffusion samples per protein (default: 1 for speed)")
    parser.add_argument("--num-steps", type=int, default=200,
                        help="Diffusion steps (default: 200)")
    parser.add_argument("--skip-reference", action="store_true",
                        help="Skip MPS reference (only compare MLX vs PDB)")
    parser.add_argument("--proteins", type=str, nargs="*", default=None,
                        help="Run only specific proteins by PDB ID (e.g., 1L2Y 1CRN)")
    args = parser.parse_args()

    proteins = TEST_PROTEINS
    if args.proteins:
        ids = {p.upper() for p in args.proteins}
        proteins = [p for p in TEST_PROTEINS if p.pdb_id.upper() in ids]
        if not proteins:
            print(f"No matching proteins for {args.proteins}")
            print(f"Available: {[p.pdb_id for p in TEST_PROTEINS]}")
            return

    all_results = []
    with tempfile.TemporaryDirectory(prefix="chai_structural_") as tmpdir:
        work_dir = Path(tmpdir)
        for protein in proteins:
            results = run_protein(
                protein,
                args.weights_dir,
                work_dir,
                num_samples=args.num_samples,
                num_steps=args.num_steps,
                skip_reference=args.skip_reference,
            )
            all_results.extend(results)
            gc.collect()

    print_results_table(all_results)


if __name__ == "__main__":
    main()
