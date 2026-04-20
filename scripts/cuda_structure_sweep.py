"""MLX vs CUDA structural comparison sweep.

Consumes the reference outputs written by
``modal run -m cuda_harness.run_reference`` (a per-target, per-seed tree of
``pred.model_idx_*.cif`` + ``scores.model_idx_*.npz`` + ``manifest.json``)
and runs the MLX pipeline locally on the same sequences/seeds.

For every ``(target, seed, sample_idx)`` it computes:

* **Cα RMSD** (Kabsch-aligned) — MLX vs CUDA, and optionally both
  against the experimental PDB structure.
* **GDT-TS** and a Cα-only **lDDT** for MLX-vs-CUDA.
* **Aggregate score gap** (MLX ``aggregate_score`` − CUDA
  ``aggregate_score``).
* **pTM / ipTM gap**, and the root-mean-square **PAE** difference.

The result is a Markdown-friendly table plus a CSV dump so the numbers
feed cleanly into the status doc and any reporting.

This is the harness we use to answer "how close is our MLX port to
CUDA?" now that MPS can no longer run reference inference on 16 GB
MacBooks without OOMing.

Usage
-----

::

    # on Modal first:
    modal run -m cuda_harness.run_reference \\
        --targets 1L2Y,1VII,1CRN,1UBQ \\
        --seeds 0,42,123 \\
        --output-dir /tmp/chai_mlx_cuda/reference

    # then locally:
    python scripts/cuda_structure_sweep.py \\
        --weights-dir weights \\
        --reference-dir /tmp/chai_mlx_cuda/reference \\
        --mlx-output-dir /tmp/chai_mlx_cuda/mlx \\
        --mlx-dtypes reference \\
        --compare-pdb \\
        --csv /tmp/chai_mlx_cuda/structure_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import tempfile
import urllib.request
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists():
    sys.path.insert(0, str(LOCAL_CHAI_LAB))
# Allow ``python scripts/cuda_structure_sweep.py`` from the repo root to find
# ``cuda_harness`` without a separate ``pip install -e .`` step.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cuda_harness.modal_common import DEFAULT_TARGETS, Target  # noqa: E402


# ---------------------------------------------------------------------------
# CIF / PDB helpers
# ---------------------------------------------------------------------------


def _ca_from_cif(cif_path: Path) -> np.ndarray:
    """All Cα atoms across every protein chain, flat (N, 3) array."""
    chain_arrs = _ca_by_chain(cif_path)
    if not chain_arrs:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(list(chain_arrs.values()), axis=0)


# Nucleic-acid backbone atoms in chai-lab's CIF output.  P is the most
# robust choice for a single-point-per-residue representation, matching
# how Cα represents proteins.
_DNA_RNA_BACKBONE_ATOMS = ("P",)

# Residue names chai-lab emits for proteins, DNA, RNA.  Used to decide
# per-chain which backbone rule applies.
_PROTEIN_RESNAMES = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL", "UNK",
    }
)
_NUCLEIC_RESNAMES = frozenset(
    {"DA", "DT", "DG", "DC", "DU", "DX", "A", "U", "G", "C", "N"}
)


def _ca_by_chain(cif_path: Path) -> dict[str, np.ndarray]:
    """Cα coordinates bucketed by chain ID.  Empty for ligand-only chains."""
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    out: dict[str, list[tuple[float, float, float]]] = {}
    for model in structure:
        for chain in model:
            acc = out.setdefault(chain.name, [])
            for residue in chain:
                for atom in residue:
                    if atom.name.strip() == "CA":
                        acc.append((atom.pos.x, atom.pos.y, atom.pos.z))
    return {k: np.array(v, dtype=np.float64) for k, v in out.items() if v}


def _nucleic_backbone_by_chain(cif_path: Path) -> dict[str, np.ndarray]:
    """Phosphate (P) coordinates bucketed by DNA/RNA chain."""
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    out: dict[str, list[tuple[float, float, float]]] = {}
    for model in structure:
        for chain in model:
            has_nucleic = any(
                residue.name.strip() in _NUCLEIC_RESNAMES for residue in chain
            )
            if not has_nucleic:
                continue
            acc = out.setdefault(chain.name, [])
            for residue in chain:
                if residue.name.strip() not in _NUCLEIC_RESNAMES:
                    continue
                for atom in residue:
                    if atom.name.strip() in _DNA_RNA_BACKBONE_ATOMS:
                        acc.append((atom.pos.x, atom.pos.y, atom.pos.z))
    return {k: np.array(v, dtype=np.float64) for k, v in out.items() if v}


def _ligand_heavy_by_chain(cif_path: Path) -> dict[str, np.ndarray]:
    """All non-H atoms from chains that contain no polymer residues.

    chai-lab emits ligand entities as their own chain with HETATM records;
    we identify them as chains that contain zero protein/nucleic residue
    names.  Hydrogens are excluded so the RMSD is comparable to standard
    ligand-docking metrics.
    """
    import gemmi

    structure = gemmi.read_structure(str(cif_path))
    out: dict[str, list[tuple[float, float, float]]] = {}
    for model in structure:
        for chain in model:
            names = {residue.name.strip() for residue in chain}
            if names & (_PROTEIN_RESNAMES | _NUCLEIC_RESNAMES):
                continue
            acc: list[tuple[float, float, float]] = []
            for residue in chain:
                for atom in residue:
                    if atom.element.name.strip().upper() == "H":
                        continue
                    acc.append((atom.pos.x, atom.pos.y, atom.pos.z))
            if acc:
                out[chain.name] = np.array(acc, dtype=np.float64)
    return out


def _ca_from_pdb(pdb_id: str, chain_id: str) -> np.ndarray:
    from Bio.PDB import PDBParser  # type: ignore[import-not-found]

    cache = REPO_ROOT / ".pdb_cache"
    cache.mkdir(exist_ok=True)
    pdb_path = cache / f"{pdb_id.upper()}.pdb"
    if not pdb_path.exists():
        urllib.request.urlretrieve(
            f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", pdb_path
        )
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    chain = structure[0][chain_id]
    coords: list[tuple[float, float, float]] = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        if "CA" in residue:
            coords.append(tuple(residue["CA"].get_vector().get_array()))
    return np.array(coords, dtype=np.float64)


# ---------------------------------------------------------------------------
# Structural metrics
# ---------------------------------------------------------------------------


def _kabsch(pred: np.ndarray, ref: np.ndarray) -> tuple[float, np.ndarray]:
    assert pred.shape == ref.shape, f"{pred.shape} vs {ref.shape}"
    pred_c = pred - pred.mean(axis=0, keepdims=True)
    ref_c = ref - ref.mean(axis=0, keepdims=True)
    h = pred_c.T @ ref_c
    u, _s, vt = np.linalg.svd(h)
    d = np.linalg.det(vt.T @ u.T)
    sign = np.diag([1.0, 1.0, np.sign(d)])
    r = vt.T @ sign @ u.T
    pred_aligned = pred_c @ r.T
    dists = np.sqrt(((pred_aligned - ref_c) ** 2).sum(axis=-1))
    rmsd = float(np.sqrt((dists**2).mean()))
    return rmsd, dists


def _gdt_ts(dists: np.ndarray) -> float:
    n = len(dists)
    if n == 0:
        return 0.0
    thresholds = (1.0, 2.0, 4.0, 8.0)
    return float(np.mean([np.sum(dists < t) / n for t in thresholds]))


def _lddt(pred: np.ndarray, ref: np.ndarray, cutoff: float = 15.0) -> float:
    n = len(pred)
    if n < 2:
        return 0.0
    ref_d = np.sqrt(((ref[:, None] - ref[None, :]) ** 2).sum(-1))
    pred_d = np.sqrt(((pred[:, None] - pred[None, :]) ** 2).sum(-1))
    mask = (ref_d < cutoff) & (np.eye(n) == 0)
    if mask.sum() == 0:
        return 0.0
    diff = np.abs(pred_d - ref_d)
    thresholds = (0.5, 1.0, 2.0, 4.0)
    return float(np.mean([float((diff[mask] < t).mean()) for t in thresholds]))


def _align_length(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, bool]:
    n = min(len(a), len(b))
    truncated = len(a) != len(b)
    return a[:n], b[:n], n, truncated


def _pair_chains(
    mlx_chains: dict[str, np.ndarray],
    cuda_chains: dict[str, np.ndarray],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Pair MLX and CUDA chains.

    First tries name-based matching.  If the two sides use different
    chain naming (chai-lab's ``fasta_names_as_cif_chains=True`` emits
    entity names, while MLX's ``save_to_cif`` currently emits A/B/...),
    falls back to length-based positional matching: chains are sorted
    by residue count and matched in order so chain assignments don't
    drift.
    """
    common = set(mlx_chains) & set(cuda_chains)
    if common:
        return [(k, mlx_chains[k], cuda_chains[k]) for k in sorted(common)]

    if len(mlx_chains) != len(cuda_chains):
        return []

    mlx_sorted = sorted(mlx_chains.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    cuda_sorted = sorted(cuda_chains.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return [
        (f"{mk}~{ck}", marr, carr)
        for (mk, marr), (ck, carr) in zip(mlx_sorted, cuda_sorted)
    ]


def _per_chain_rmsd(
    mlx_chains: dict[str, np.ndarray],
    cuda_chains: dict[str, np.ndarray],
) -> dict[str, float]:
    """Kabsch-aligned RMSD between matching chains."""
    out: dict[str, float] = {}
    for key, mlx_arr, cuda_arr in _pair_chains(mlx_chains, cuda_chains):
        a, b, n, _ = _align_length(mlx_arr, cuda_arr)
        if n < 3:
            continue
        rmsd, _ = _kabsch(a, b)
        out[key] = rmsd
    return out


def _interface_rmsd(
    mlx_chains: dict[str, np.ndarray],
    cuda_chains: dict[str, np.ndarray],
    *,
    cutoff: float = 10.0,
) -> float | None:
    """Interface Cα RMSD: residues on chain A within ``cutoff`` of chain B,
    plus residues on chain B within ``cutoff`` of chain A, aligned jointly.
    """
    paired = _pair_chains(mlx_chains, cuda_chains)
    if len(paired) < 2:
        return None

    (_, a_mlx, a_cuda), (_, b_mlx, b_cuda) = paired[0], paired[1]
    n_a = min(len(a_mlx), len(a_cuda))
    n_b = min(len(b_mlx), len(b_cuda))
    if n_a < 3 or n_b < 3:
        return None
    a_mlx, a_cuda = a_mlx[:n_a], a_cuda[:n_a]
    b_mlx, b_cuda = b_mlx[:n_b], b_cuda[:n_b]

    d_mlx = np.linalg.norm(a_mlx[:, None, :] - b_mlx[None, :, :], axis=-1)
    in_iface_a = (d_mlx.min(axis=1) < cutoff)
    in_iface_b = (d_mlx.min(axis=0) < cutoff)
    if not in_iface_a.any() or not in_iface_b.any():
        return None

    mlx_joint = np.concatenate([a_mlx[in_iface_a], b_mlx[in_iface_b]], axis=0)
    cuda_joint = np.concatenate([a_cuda[in_iface_a], b_cuda[in_iface_b]], axis=0)
    rmsd, _ = _kabsch(mlx_joint, cuda_joint)
    return rmsd


def _classify_target(target: Target) -> str:
    """Coarse label for per-target metric selection.

    Order of precedence: dna > multimer > ligand > monomer.  A DNA multimer
    is still labelled ``dna`` because that drives backbone-atom (P) RMSD
    selection; the iPTM/per-chain columns pick up the multimer information
    independently.
    """
    if "dna" in target.kinds or "rna" in target.kinds:
        return "dna"
    if "multimer" in target.kinds:
        return "multimer"
    if "ligand" in target.kinds:
        return "ligand"
    return "monomer"


# ---------------------------------------------------------------------------
# MLX inference
# ---------------------------------------------------------------------------


def _mlx_run(
    weights_dir: Path,
    target: Target,
    seed: int,
    *,
    feature_dir: Path,
    out_dir: Path,
    dtype: str,
    num_steps: int,
    num_recycles: int,
    num_samples: int,
    esm_backend: str = "off",
) -> dict:
    import torch
    import mlx.core as mx

    from chai_lab.chai1 import Collate, feature_factory, make_all_atom_feature_context
    from chai_lab.data.io.cif_utils import get_chain_letter, save_to_cif

    from chai_mlx import ChaiMLX
    from chai_mlx.data.featurize import featurize_fasta

    fasta_path = feature_dir / f"{target.name}.fasta"
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(target.to_fasta())

    ref_ctx = make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=feature_dir / "ref_features",
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
        output_dir=feature_dir / "mlx_features",
        esm_backend=esm_backend,
        use_msa_server=False,
        use_templates_server=False,
    )

    mx.random.seed(seed)
    model = ChaiMLX.from_pretrained(weights_dir, strict=False, compute_dtype=dtype)
    result = model.run_inference(
        ctx,
        recycles=num_recycles,
        num_samples=num_samples,
        num_steps=num_steps,
    )
    coords = np.array(result.coords.astype(mx.float32))[0]  # [ds, A, 3]
    agg = np.array(result.ranking.aggregate_score.astype(mx.float32)).reshape(-1)
    ptm = np.array(result.ranking.ptm.astype(mx.float32)).reshape(-1)
    iptm = np.array(result.ranking.iptm.astype(mx.float32)).reshape(-1)

    # Write CIFs
    out_dir.mkdir(parents=True, exist_ok=True)
    cif_paths: list[Path] = []
    for ds in range(coords.shape[0]):
        path = out_dir / f"pred.model_idx_{ds}.cif"
        save_to_cif(
            coords=torch.from_numpy(coords[ds][None]).float(),
            bfactors=None,
            output_batch=output_batch,
            write_path=path,
            asym_entity_names=asym_entity_names,
        )
        cif_paths.append(path)

    del model, result, ctx, ref_ctx
    gc.collect()
    mx.clear_cache()

    return {
        "cif_paths": cif_paths,
        "aggregate_score": agg,
        "ptm": ptm,
        "iptm": iptm,
    }


# ---------------------------------------------------------------------------
# Row
# ---------------------------------------------------------------------------


@dataclass
class ComparisonRow:
    target: str
    kind: str
    seed: int
    sample_idx: int
    mlx_dtype: str
    n_residues: int
    truncated: bool
    rmsd_mlx_vs_cuda: float
    gdt_mlx_vs_cuda: float
    lddt_mlx_vs_cuda: float
    max_ca_err_mlx_vs_cuda: float
    rmsd_mlx_vs_pdb: float | None
    rmsd_cuda_vs_pdb: float | None
    agg_mlx: float
    agg_cuda: float
    agg_gap: float
    ptm_mlx: float
    ptm_cuda: float
    iptm_mlx: float
    iptm_cuda: float
    # Per-kind extras; ``None`` when the target does not exercise that metric.
    per_chain_rmsd: dict[str, float] | None = None
    interface_rmsd_mlx_vs_cuda: float | None = None
    ligand_heavy_rmsd_mlx_vs_cuda: float | None = None
    dna_backbone_rmsd_mlx_vs_cuda: float | None = None
    has_clashes_mlx: bool | None = None
    has_clashes_cuda: bool | None = None


def _pretty_print(rows: list[ComparisonRow]) -> None:
    def _f(x: float | None, width: int = 6, spec: str = ".2f", suffix: str = "Å") -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return f"{'--':>{width + len(suffix)}}"
        return f"{x:>{width}{spec}}{suffix}"

    print("\n" + "=" * 132)
    print(
        f"  {'target':<11} {'kind':<9} {'seed':>5} {'idx':>3} {'dtype':>8} {'N':>4} "
        f"{'rmsd':>8} {'gdt':>6} {'lddt':>6} {'ifRMSD':>8} {'ligRMSD':>8} "
        f"{'agg_mlx':>8} {'agg_cuda':>9} {'Δagg':>8}"
    )
    print("  " + "-" * 130)
    for r in rows:
        print(
            f"  {r.target:<11} {r.kind:<9} {r.seed:>5} {r.sample_idx:>3} {r.mlx_dtype:>8} "
            f"{r.n_residues:>4} "
            f"{_f(r.rmsd_mlx_vs_cuda, 7)} "
            f"{r.gdt_mlx_vs_cuda:>5.1%} {r.lddt_mlx_vs_cuda:>5.1%} "
            f"{_f(r.interface_rmsd_mlx_vs_cuda, 7)} "
            f"{_f(r.ligand_heavy_rmsd_mlx_vs_cuda, 7)} "
            f"{r.agg_mlx:>8.4f} {r.agg_cuda:>9.4f} {r.agg_gap:>+7.4f}"
        )


def _nanmean(values: list[float | None]) -> float:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def _aggregate_summary(rows: list[ComparisonRow]) -> dict:
    if not rows:
        return {}

    def _bucket(filtered: list[ComparisonRow]) -> dict:
        rmsds = np.array(
            [r.rmsd_mlx_vs_cuda for r in filtered
             if not np.isnan(r.rmsd_mlx_vs_cuda)]
        )
        return {
            "n_samples": len(filtered),
            "mean_rmsd_mlx_vs_cuda": float(rmsds.mean()) if rmsds.size else float("nan"),
            "median_rmsd_mlx_vs_cuda": float(np.median(rmsds)) if rmsds.size else float("nan"),
            "p90_rmsd_mlx_vs_cuda": float(np.percentile(rmsds, 90)) if rmsds.size else float("nan"),
            "max_rmsd_mlx_vs_cuda": float(rmsds.max()) if rmsds.size else float("nan"),
            "mean_gdt_mlx_vs_cuda": _nanmean([r.gdt_mlx_vs_cuda for r in filtered]),
            "mean_lddt_mlx_vs_cuda": _nanmean([r.lddt_mlx_vs_cuda for r in filtered]),
            "mean_interface_rmsd": _nanmean([r.interface_rmsd_mlx_vs_cuda for r in filtered]),
            "mean_ligand_heavy_rmsd": _nanmean([r.ligand_heavy_rmsd_mlx_vs_cuda for r in filtered]),
            "mean_dna_backbone_rmsd": _nanmean([r.dna_backbone_rmsd_mlx_vs_cuda for r in filtered]),
            "mean_agg_gap": _nanmean([r.agg_gap for r in filtered]),
            "std_agg_gap": float(
                np.std([r.agg_gap for r in filtered if not np.isnan(r.agg_gap)])
            ) if filtered else float("nan"),
        }

    per_dtype: dict[str, dict] = {}
    for dtype in sorted({r.mlx_dtype for r in rows}):
        by_dtype = [r for r in rows if r.mlx_dtype == dtype]
        bucket = {"all": _bucket(by_dtype), "per_kind": {}}
        for kind in sorted({r.kind for r in by_dtype}):
            bucket["per_kind"][kind] = _bucket(
                [r for r in by_dtype if r.kind == kind]
            )
        per_dtype[dtype] = bucket
    return per_dtype


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_cuda_scores(path: Path) -> tuple[float, float, float, bool]:
    with np.load(path, allow_pickle=False) as f:
        agg = float(f["aggregate_score"].item())
        ptm = float(f["ptm"].item())
        iptm = float(f["iptm"].item())
        clashes = bool(f["has_inter_chain_clashes"].any()) if "has_inter_chain_clashes" in f.files else False
    return agg, ptm, iptm, clashes


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Directory produced by run_reference.py (contains <target>/seed_<n>/...)",
    )
    parser.add_argument(
        "--mlx-output-dir",
        type=Path,
        default=Path("/tmp/chai_mlx_cuda/mlx"),
        help="Directory to write local MLX CIF outputs",
    )
    parser.add_argument("--feature-dir", type=Path, default=Path("/tmp/chai_mlx_cuda/mlx_features"))
    parser.add_argument(
        "--mlx-dtypes",
        nargs="+",
        default=["reference"],
        choices=["reference", "float32"],
    )
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-recycles", type=int, default=3)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Diffusion samples MLX should emit per seed",
    )
    parser.add_argument(
        "--compare-pdb",
        action="store_true",
        help="Also compare both MLX and CUDA against experimental PDB Cα coords",
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument(
        "--skip-mlx",
        action="store_true",
        help=(
            "Force cache-only comparison: require MLX CIFs and scores.json to "
            "already exist under --mlx-output-dir; skip any (target, seed) "
            "that is missing them."
        ),
    )
    parser.add_argument(
        "--force-rerun-mlx",
        action="store_true",
        help=(
            "Re-run MLX inference even when the --mlx-output-dir already has a "
            "complete set of CIFs + scores.json for a given (target, seed). "
            "By default we reuse any fully-cached run to avoid paying a 5+ min "
            "MLX roll-out per seed."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.reference_dir.exists():
        raise FileNotFoundError(f"reference dir does not exist: {args.reference_dir}")
    args.mlx_output_dir.mkdir(parents=True, exist_ok=True)
    args.feature_dir.mkdir(parents=True, exist_ok=True)

    # Discover targets/seeds from the reference tree.
    plan: list[tuple[str, int]] = []
    for target_dir in sorted(p for p in args.reference_dir.iterdir() if p.is_dir()):
        name = target_dir.name
        if name not in DEFAULT_TARGETS:
            print(f"[warn] unknown target {name!r}, skipping")
            continue
        for seed_dir in sorted(p for p in target_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")):
            try:
                seed = int(seed_dir.name.split("_", 1)[1])
            except ValueError:
                continue
            if not any(seed_dir.glob("pred.model_idx_*.cif")):
                print(f"[warn] no CUDA CIFs at {seed_dir}")
                continue
            plan.append((name, seed))

    if not plan:
        print(f"[error] no reference runs found under {args.reference_dir}")
        return

    rows: list[ComparisonRow] = []

    for name, seed in plan:
        target = DEFAULT_TARGETS[name]
        kind = _classify_target(target)
        seed_ref_dir = args.reference_dir / name / f"seed_{seed}"
        cuda_cifs = sorted(seed_ref_dir.glob("pred.model_idx_*.cif"))
        cuda_scores = [_load_cuda_scores(seed_ref_dir / f"scores.model_idx_{i}.npz")
                       for i in range(len(cuda_cifs))]

        # Optional PDB ground truth (Cα only; not meaningful for ligand-only
        # or DNA-only targets).
        pdb_ca = None
        if args.compare_pdb and target.n_protein_residues > 0:
            try:
                pdb_ca = _ca_from_pdb(name, "A")
            except Exception as exc:
                warnings.warn(f"{name}: could not fetch PDB: {exc}")

        for dtype in args.mlx_dtypes:
            mlx_dir = args.mlx_output_dir / dtype / name / f"seed_{seed}"
            cached_cifs = sorted(mlx_dir.glob("pred.model_idx_*.cif"))
            cached_scores = mlx_dir / "scores.json"
            has_complete_cache = (
                len(cached_cifs) >= args.num_samples and cached_scores.exists()
            )
            # --skip-mlx forces cache-only; without it, we still reuse a full
            # per-(target,seed) cache automatically to avoid re-running slow
            # 200-step MLX inference on already-done seeds.
            if args.skip_mlx or (has_complete_cache and not args.force_rerun_mlx):
                if args.skip_mlx and not has_complete_cache:
                    print(f"[skip-mlx] nothing to compare; skipping {name} seed={seed} dtype={dtype}")
                    continue
                label = "skip-mlx" if args.skip_mlx else "reuse-cache"
                print(f"[{label}] reusing {mlx_dir}")
                mlx_cifs = cached_cifs
                with cached_scores.open() as fh:
                    cached = json.load(fh)
                mlx_agg = np.array(cached["aggregate_score"], dtype=np.float32)
                mlx_ptm = np.array(cached["ptm"], dtype=np.float32)
                mlx_iptm = np.array(cached["iptm"], dtype=np.float32)
            else:
                print(f"[mlx] {name} seed={seed} dtype={dtype}")
                mlx_result = _mlx_run(
                    weights_dir=args.weights_dir,
                    target=target,
                    seed=seed,
                    feature_dir=args.feature_dir / name,
                    out_dir=mlx_dir,
                    dtype=dtype,
                    num_steps=args.num_steps,
                    num_recycles=args.num_recycles,
                    num_samples=args.num_samples,
                    esm_backend=getattr(args, "esm_backend", "off"),
                )
                mlx_cifs = mlx_result["cif_paths"]
                mlx_agg = mlx_result["aggregate_score"]
                mlx_ptm = mlx_result["ptm"]
                mlx_iptm = mlx_result["iptm"]
                (mlx_dir / "scores.json").write_text(
                    json.dumps(
                        {
                            "aggregate_score": mlx_agg.tolist(),
                            "ptm": mlx_ptm.tolist(),
                            "iptm": mlx_iptm.tolist(),
                        },
                        indent=2,
                    )
                )

            n_samples = min(len(mlx_cifs), len(cuda_cifs))
            for sample_idx in range(n_samples):
                mlx_ca_by_chain = _ca_by_chain(mlx_cifs[sample_idx])
                cuda_ca_by_chain = _ca_by_chain(cuda_cifs[sample_idx])
                mlx_dna_by_chain = _nucleic_backbone_by_chain(mlx_cifs[sample_idx])
                cuda_dna_by_chain = _nucleic_backbone_by_chain(cuda_cifs[sample_idx])
                mlx_lig_by_chain = _ligand_heavy_by_chain(mlx_cifs[sample_idx])
                cuda_lig_by_chain = _ligand_heavy_by_chain(cuda_cifs[sample_idx])

                # Headline: Cα for protein-bearing targets, P backbone for
                # DNA/RNA-only targets, ligand heavy atoms otherwise.
                if kind == "dna":
                    headline_mlx = (
                        np.concatenate(list(mlx_dna_by_chain.values()), axis=0)
                        if mlx_dna_by_chain else np.zeros((0, 3))
                    )
                    headline_cuda = (
                        np.concatenate(list(cuda_dna_by_chain.values()), axis=0)
                        if cuda_dna_by_chain else np.zeros((0, 3))
                    )
                elif target.n_protein_residues > 0:
                    headline_mlx = (
                        np.concatenate(list(mlx_ca_by_chain.values()), axis=0)
                        if mlx_ca_by_chain else np.zeros((0, 3))
                    )
                    headline_cuda = (
                        np.concatenate(list(cuda_ca_by_chain.values()), axis=0)
                        if cuda_ca_by_chain else np.zeros((0, 3))
                    )
                else:
                    headline_mlx = (
                        np.concatenate(list(mlx_lig_by_chain.values()), axis=0)
                        if mlx_lig_by_chain else np.zeros((0, 3))
                    )
                    headline_cuda = (
                        np.concatenate(list(cuda_lig_by_chain.values()), axis=0)
                        if cuda_lig_by_chain else np.zeros((0, 3))
                    )

                a, b, n, truncated = _align_length(headline_mlx, headline_cuda)
                if n >= 3:
                    rmsd, dists = _kabsch(a, b)
                    gdt = _gdt_ts(dists)
                    lddt = _lddt(a, b)
                    max_err = float(dists.max())
                else:
                    rmsd = float("nan")
                    gdt = float("nan")
                    lddt = float("nan")
                    max_err = float("nan")

                rmsd_mlx_pdb = None
                rmsd_cuda_pdb = None
                if pdb_ca is not None and n >= 3:
                    n_pdb = min(len(pdb_ca), len(a))
                    rmsd_mlx_pdb, _ = _kabsch(a[:n_pdb], pdb_ca[:n_pdb])
                    rmsd_cuda_pdb, _ = _kabsch(b[:n_pdb], pdb_ca[:n_pdb])

                # Per-kind extras: always computed when the target has the
                # relevant atoms.
                per_chain_dict = _per_chain_rmsd(mlx_ca_by_chain, cuda_ca_by_chain)
                interface_rmsd = (
                    _interface_rmsd(mlx_ca_by_chain, cuda_ca_by_chain)
                    if target.is_multimer and target.n_protein_residues > 0
                    else None
                )
                ligand_rmsd = None
                if mlx_lig_by_chain and cuda_lig_by_chain:
                    lig_mlx = np.concatenate(list(mlx_lig_by_chain.values()), axis=0)
                    lig_cuda = np.concatenate(list(cuda_lig_by_chain.values()), axis=0)
                    la, lb, ln, _ = _align_length(lig_mlx, lig_cuda)
                    if ln >= 3:
                        ligand_rmsd, _ = _kabsch(la, lb)
                dna_rmsd = None
                if mlx_dna_by_chain and cuda_dna_by_chain:
                    dna_mlx = np.concatenate(list(mlx_dna_by_chain.values()), axis=0)
                    dna_cuda = np.concatenate(list(cuda_dna_by_chain.values()), axis=0)
                    da, db, dn, _ = _align_length(dna_mlx, dna_cuda)
                    if dn >= 3:
                        dna_rmsd, _ = _kabsch(da, db)

                agg_mlx = float(mlx_agg[sample_idx]) if sample_idx < len(mlx_agg) else float("nan")
                agg_cuda = cuda_scores[sample_idx][0]
                ptm_mlx = float(mlx_ptm[sample_idx]) if sample_idx < len(mlx_ptm) else float("nan")
                ptm_cuda = cuda_scores[sample_idx][1]
                iptm_mlx = float(mlx_iptm[sample_idx]) if sample_idx < len(mlx_iptm) else float("nan")
                iptm_cuda = cuda_scores[sample_idx][2]
                has_clashes_cuda = cuda_scores[sample_idx][3]

                rows.append(
                    ComparisonRow(
                        target=name,
                        kind=kind,
                        seed=seed,
                        sample_idx=sample_idx,
                        mlx_dtype=dtype,
                        n_residues=n,
                        truncated=truncated,
                        rmsd_mlx_vs_cuda=rmsd,
                        gdt_mlx_vs_cuda=gdt,
                        lddt_mlx_vs_cuda=lddt,
                        max_ca_err_mlx_vs_cuda=max_err,
                        rmsd_mlx_vs_pdb=rmsd_mlx_pdb,
                        rmsd_cuda_vs_pdb=rmsd_cuda_pdb,
                        agg_mlx=agg_mlx,
                        agg_cuda=agg_cuda,
                        agg_gap=agg_mlx - agg_cuda,
                        ptm_mlx=ptm_mlx,
                        ptm_cuda=ptm_cuda,
                        iptm_mlx=iptm_mlx,
                        iptm_cuda=iptm_cuda,
                        per_chain_rmsd=per_chain_dict or None,
                        interface_rmsd_mlx_vs_cuda=interface_rmsd,
                        ligand_heavy_rmsd_mlx_vs_cuda=ligand_rmsd,
                        dna_backbone_rmsd_mlx_vs_cuda=dna_rmsd,
                        has_clashes_cuda=has_clashes_cuda,
                    )
                )

    _pretty_print(rows)
    summary = _aggregate_summary(rows)
    print("\nPer-dtype aggregate (MLX vs CUDA):")
    for dtype, bucket in summary.items():
        all_metrics = bucket["all"]
        print(f"  {dtype}: n={all_metrics['n_samples']}")
        for key, value in all_metrics.items():
            if key == "n_samples":
                continue
            if isinstance(value, float) and np.isnan(value):
                print(f"    {key:>32}: --")
            else:
                print(f"    {key:>32}: {value:+.4f}")
        for kind, kmetrics in bucket["per_kind"].items():
            print(f"    [{kind}] n={kmetrics['n_samples']}")
            for key in ("mean_rmsd_mlx_vs_cuda", "mean_interface_rmsd",
                        "mean_ligand_heavy_rmsd", "mean_dna_backbone_rmsd"):
                val = kmetrics[key]
                if isinstance(val, float) and np.isnan(val):
                    continue
                print(f"      {key:>30}: {val:+.4f}")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=list(asdict(rows[0]).keys()) if rows else []
            )
            writer.writeheader()
            for r in rows:
                record = asdict(r)
                # DictWriter can't serialize dict fields; fold them to JSON.
                if record.get("per_chain_rmsd") is not None:
                    record["per_chain_rmsd"] = json.dumps(record["per_chain_rmsd"])
                writer.writerow(record)
        print(f"[save] rows -> {args.csv}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                {
                    "reference_dir": str(args.reference_dir),
                    "mlx_output_dir": str(args.mlx_output_dir),
                    "per_dtype": summary,
                    "n_rows": len(rows),
                    "rows": [asdict(r) for r in rows],
                },
                indent=2,
                default=str,
            )
        )
        print(f"[save] summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
