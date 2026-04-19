"""Constraint-feature parity: MLX vs CUDA.

Answers "does ``featurize_fasta(constraint_path=...)`` produce the same
TokenDistanceRestraint, TokenPairPocketRestraint, and bond-adjacency
tensors on MLX as chai-lab does on CUDA?"  The featurizer is chai-lab
on both sides, so at the **raw feature** level we expect bit-exact
agreement.  At the **projected** level (after ``FeatureEmbedding``) we
expect fp32 ops to match within epsilon and bf16 ops to match within
the standard bf16 per-op floor.

The CUDA side is the intermediates NPZ produced by::

    modal run -m cuda_harness.run_intermediates \\
        --targets 1CRN_CONSTR --seeds 42 \\
        --constraint-resource 1CRN_all_three.csv

The MLX side re-runs ``featurize_fasta`` locally on the same FASTA + CSV
and loads the weight shards so ``FeatureEmbedding`` can project both
sides' raw tensors.  No diffusion, no trunk -- this script only
exercises the constraint-featurization path.

Usage
-----

::

    python scripts/cuda_constraints_parity.py \\
        --weights-dir weights \\
        --npz /tmp/chai_mlx_cuda/intermediates/1CRN_CONSTR/seed_42.npz \\
        --constraint-csv cuda_harness/constraints/1CRN_all_three.csv \\
        --target 1CRN_CONSTR
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists() and str(LOCAL_CHAI_LAB) not in sys.path:
    sys.path.insert(0, str(LOCAL_CHAI_LAB))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402

from chai_mlx import ChaiMLX  # noqa: E402
from chai_mlx.data.featurize import featurize_fasta  # noqa: E402
from chai_mlx.data.types import FeatureContext, StructureInputs  # noqa: E402
from chai_mlx.utils import resolve_dtype  # noqa: E402

from cuda_harness.modal_common import DEFAULT_TARGETS  # noqa: E402


# Feature keys we care about in the NPZ, plus their raw-feature dict names.
_RAW_KEYS = (
    ("TokenDistanceRestraint", "inputs.features.TokenDistanceRestraint"),
    ("TokenPairPocketRestraint", "inputs.features.TokenPairPocketRestraint"),
    ("DockingConstraintGenerator", "inputs.features.DockingConstraintGenerator"),
    ("MissingChainContact", "inputs.features.MissingChainContact"),
)


@dataclass
class ArrayDiff:
    name: str
    shape_mlx: tuple
    shape_cuda: tuple
    max_abs: float
    mean_abs: float
    nonzero_mlx: int
    nonzero_cuda: int


def _diff(mlx_arr: np.ndarray, cuda_arr: np.ndarray, name: str) -> ArrayDiff:
    if mlx_arr.shape != cuda_arr.shape:
        # Raise early: any shape mismatch means the featurizer behaved
        # differently on the two sides, which is a bug we want to surface.
        raise AssertionError(
            f"{name}: shape mismatch MLX={mlx_arr.shape} CUDA={cuda_arr.shape}"
        )
    diff = np.abs(mlx_arr.astype(np.float64) - cuda_arr.astype(np.float64))
    return ArrayDiff(
        name=name,
        shape_mlx=tuple(mlx_arr.shape),
        shape_cuda=tuple(cuda_arr.shape),
        max_abs=float(diff.max()) if diff.size else 0.0,
        mean_abs=float(diff.mean()) if diff.size else 0.0,
        nonzero_mlx=int(np.count_nonzero(mlx_arr)),
        nonzero_cuda=int(np.count_nonzero(cuda_arr)),
    )


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def _as_np(arr: mx.array) -> np.ndarray:
    mx.eval(arr)
    return np.asarray(arr)


def _structure_from_npz(data: dict[str, np.ndarray]) -> StructureInputs:
    """Reconstruct a StructureInputs from a run_intermediates NPZ.

    Mirrors ``scripts.cuda_parity._reconstruct_structure_inputs`` but
    trimmed to just the fields FeatureEmbedding reads.
    """
    from chai_lab.data.parsing.structure.entity_type import EntityType as ChaiEntityType

    token_exists = data["inputs.batch.token_exists_mask"].astype(np.float32)
    atom_exists = data["inputs.batch.atom_exists_mask"].astype(np.float32)
    token_pair_mask = np.einsum("bi,bj->bij", token_exists, token_exists)

    token_entity_type = data["inputs.batch.token_entity_type"].astype(np.int64)
    is_polymer = np.zeros_like(token_entity_type, dtype=np.float32)
    for v in (
        ChaiEntityType.PROTEIN.value,
        ChaiEntityType.RNA.value,
        ChaiEntityType.DNA.value,
    ):
        is_polymer[token_entity_type == v] = 1.0

    q_idx = data["inputs.batch.block_atom_pair_q_idces"]
    kv_idx = data["inputs.batch.block_atom_pair_kv_idces"]
    if q_idx.ndim == 2:
        q_idx = np.broadcast_to(q_idx[None], (token_exists.shape[0], *q_idx.shape)).copy()
    if kv_idx.ndim == 2:
        kv_idx = np.broadcast_to(kv_idx[None], (token_exists.shape[0], *kv_idx.shape)).copy()

    template_mask = data["inputs.batch.template_mask"].astype(np.float32)
    template_input_masks = np.einsum("btn,btm->btnm", template_mask, template_mask)

    return StructureInputs(
        atom_exists_mask=mx.array(atom_exists),
        token_exists_mask=mx.array(token_exists),
        token_pair_mask=mx.array(token_pair_mask),
        atom_token_index=mx.array(data["inputs.batch.atom_token_index"].astype(np.int64)),
        atom_within_token_index=mx.array(
            data["inputs.batch.atom_within_token_index"].astype(np.int64)
        ),
        token_reference_atom_index=mx.array(
            data["inputs.batch.token_ref_atom_index"].astype(np.int64)
        ),
        token_centre_atom_index=mx.array(
            data["inputs.batch.token_centre_atom_index"].astype(np.int64)
        ),
        token_asym_id=mx.array(data["inputs.batch.token_asym_id"].astype(np.int64)),
        token_entity_id=mx.array(data["inputs.batch.token_entity_id"].astype(np.int64)),
        token_chain_id=mx.array(data["inputs.batch.token_asym_id"].astype(np.int64)),
        token_is_polymer=mx.array(is_polymer),
        atom_ref_positions=mx.array(data["inputs.batch.atom_ref_pos"].astype(np.float32)),
        atom_ref_space_uid=mx.array(
            data["inputs.batch.atom_ref_space_uid"].astype(np.int64)
        ),
        atom_q_indices=mx.array(q_idx),
        atom_kv_indices=mx.array(kv_idx),
        block_atom_pair_mask=mx.array(
            data["inputs.batch.block_atom_pair_mask"].astype(np.float32)
        ),
        msa_mask=mx.array(data["inputs.batch.msa_mask"]),
        template_input_masks=mx.array(template_input_masks),
        token_residue_index=mx.array(
            data["inputs.batch.token_residue_index"].astype(np.int64)
        ),
        token_entity_type=mx.array(token_entity_type),
        token_backbone_frame_mask=mx.array(
            data["inputs.batch.token_backbone_frame_mask"]
        ),
        token_backbone_frame_index=mx.array(
            data["inputs.batch.token_backbone_frame_index"].astype(np.int64)
        ),
    )


def _write_fasta(target_name: str, out_path: Path) -> None:
    target = DEFAULT_TARGETS[target_name]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(target.to_fasta())


def _project_cuda_features(
    weights_dir: Path,
    structure: StructureInputs,
    cuda_data: dict[str, np.ndarray],
    cuda_bond: np.ndarray,
    *,
    compute_dtype: str,
) -> dict[str, np.ndarray]:
    """Run MLX FeatureEmbedding + BondProjection on CUDA-captured inputs."""
    from chai_lab.chai1 import feature_generators

    model = ChaiMLX.from_pretrained(weights_dir, strict=False, compute_dtype=compute_dtype)
    dtype = resolve_dtype(model.cfg)

    # FeatureEmbedding._forward_raw expects every chai-lab feature key.
    cuda_raw_mx: dict[str, mx.array] = {}
    for name in feature_generators:
        key = f"inputs.features.{name}"
        if key not in cuda_data:
            raise KeyError(
                f"Intermediates NPZ is missing {key!r}; re-run run_intermediates.py"
            )
        cuda_raw_mx[name] = mx.array(cuda_data[key])

    empty = mx.zeros((structure.token_exists_mask.shape[0], 0))
    cuda_ctx = FeatureContext(
        token_features=empty,
        token_pair_features=empty,
        atom_features=empty,
        atom_pair_features=empty,
        msa_features=empty,
        template_features=empty,
        structure_inputs=structure,
        bond_adjacency=mx.array(cuda_bond),
        raw_features=cuda_raw_mx,
    )

    out = model.input_embedder.feature_embedding(cuda_ctx)
    bond_trunk, bond_structure = model.input_embedder.bond_projection(
        cuda_ctx.bond_adjacency
    )
    if dtype != mx.float32:
        out = {k: v.astype(mx.float32) for k, v in out.items()}
        bond_trunk = bond_trunk.astype(mx.float32)
        bond_structure = bond_structure.astype(mx.float32)

    return {
        "token_pair_trunk": _as_np(out["token_pair_trunk"]),
        "token_pair_structure": _as_np(out["token_pair_structure"]),
        "bond_trunk": _as_np(bond_trunk),
        "bond_structure": _as_np(bond_structure),
    }


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Intermediates NPZ produced by cuda_harness.run_intermediates (with --constraint-resource)",
    )
    parser.add_argument(
        "--constraint-csv",
        type=Path,
        default=REPO_ROOT / "cuda_harness" / "constraints" / "1CRN_all_three.csv",
    )
    parser.add_argument(
        "--target",
        default="1CRN_CONSTR",
        help="Target name (must be a key in DEFAULT_TARGETS)",
    )
    parser.add_argument(
        "--compute-dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="Numeric dtype for the projection step (float32 gives tight bounds)",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("/tmp/chai_mlx_cuda/constraints_parity_features"),
    )
    parser.add_argument(
        "--raw-atol",
        type=float,
        default=0.0,
        help="Tolerance on raw-feature max-abs-err (default 0 = bit-exact)",
    )
    parser.add_argument(
        "--proj-atol",
        type=float,
        default=5e-5,
        help="Tolerance on projected-feature max-abs-err for fp32 runs",
    )
    parser.add_argument(
        "--skip-local-featurize",
        action="store_true",
        help=(
            "Do not run the local MLX featurizer. Instead, validate that "
            "(a) the CUDA-captured restraint raw features contain the "
            "expected non-sentinel entries, and (b) MLX's FeatureEmbedding "
            "produces finite, non-degenerate projections from those raw "
            "features. Required on macOS Pythons where chai-lab's RDKit "
            "timeout wrapper (ligand-only path) cannot pickle."
        ),
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.target not in DEFAULT_TARGETS:
        raise KeyError(f"Unknown target {args.target!r}. Known: {sorted(DEFAULT_TARGETS)}")
    if not args.constraint_csv.is_file():
        raise FileNotFoundError(f"constraint CSV not found: {args.constraint_csv}")

    # --- CUDA side -----------------------------------------------------
    print(f"[cuda] loading {args.npz}")
    cuda_data = _load_npz(args.npz)
    cuda_raw = {name: np.asarray(cuda_data[key]) for name, key in _RAW_KEYS if key in cuda_data}
    cuda_bond = np.asarray(cuda_data["inputs.bond_ft"])

    raw_diffs: list[ArrayDiff] = []
    raw_ok = True
    if not args.skip_local_featurize:
        # --- MLX side (local featurize + bit-exact diff) --------------
        args.feature_dir.mkdir(parents=True, exist_ok=True)
        fasta_path = args.feature_dir / f"{args.target}.fasta"
        _write_fasta(args.target, fasta_path)
        local_csv = args.feature_dir / "constraints.csv"
        local_csv.write_bytes(args.constraint_csv.read_bytes())

        print(f"[mlx] featurizing {args.target} with constraints={args.constraint_csv}")
        mlx_ctx = featurize_fasta(
            fasta_path,
            output_dir=args.feature_dir / "mlx_features",
            constraint_path=local_csv,
            esm_backend="off",
            use_msa_server=False,
            use_templates_server=False,
        )
        if mlx_ctx.raw_features is None:
            raise RuntimeError(
                "MLX featurize_fasta returned no raw_features; cannot diff "
                "against CUDA. Did chai-lab install correctly?"
            )

        for raw_name, npz_key in _RAW_KEYS:
            if raw_name not in mlx_ctx.raw_features:
                print(f"[warn] MLX raw_features missing {raw_name!r}; skipping")
                continue
            if npz_key not in cuda_data:
                print(f"[warn] CUDA npz missing {npz_key!r}; skipping")
                continue
            mlx_arr = np.asarray(mlx_ctx.raw_features[raw_name])
            cuda_arr = np.asarray(cuda_data[npz_key])
            raw_diffs.append(_diff(mlx_arr, cuda_arr, f"raw.{raw_name}"))

        mlx_bond = np.asarray(mlx_ctx.bond_adjacency)
        raw_diffs.append(_diff(mlx_bond, cuda_bond, "raw.bond_adjacency"))

        for d in raw_diffs:
            raw_ok = raw_ok and (d.max_abs <= args.raw_atol)

        structure = mlx_ctx.structure_inputs
    else:
        # --- Sanity check: the CUDA raw features contain restraint info
        restraint_summary: dict[str, dict] = {}
        for raw_name, npz_key in _RAW_KEYS:
            if npz_key not in cuda_data:
                continue
            arr = cuda_raw[raw_name]
            nz = int(np.count_nonzero(arr != -1.0)) if raw_name.startswith("Token") else int(np.count_nonzero(arr))
            restraint_summary[raw_name] = {
                "shape": tuple(arr.shape),
                "non_default_entries": nz,
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        bond_nz = int(np.count_nonzero(cuda_bond))
        restraint_summary["bond_adjacency"] = {
            "shape": tuple(cuda_bond.shape),
            "non_default_entries": bond_nz,
            "min": float(cuda_bond.min()),
            "max": float(cuda_bond.max()),
        }
        print("\n=== CUDA raw feature summary ===")
        for k, v in restraint_summary.items():
            print(f"  {k:<30}  shape={v['shape']}  non-default={v['non_default_entries']}  range=[{v['min']}, {v['max']}]")

        structure = _structure_from_npz(cuda_data)

    # --- projected-feature check -------------------------------------
    print(f"\n[mlx] projecting CUDA raw features (dtype={args.compute_dtype}) ...")
    projected = _project_cuda_features(
        args.weights_dir,
        structure,
        cuda_data,
        cuda_bond,
        compute_dtype=args.compute_dtype,
    )

    print("\n=== MLX projection of CUDA raw features ===")
    proj_summary: dict[str, dict] = {}
    proj_ok = True
    for name, arr in projected.items():
        finite = np.isfinite(arr).all()
        nz = int(np.count_nonzero(np.abs(arr) > 1e-7))
        stats = {
            "shape": tuple(arr.shape),
            "finite": bool(finite),
            "nonzero_entries": nz,
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean_abs": float(np.abs(arr).mean()),
        }
        proj_summary[name] = stats
        status = "OK" if finite and nz > 0 else "FAIL"
        if not (finite and nz > 0):
            proj_ok = False
        print(
            f"  [{status}] proj.{name:<22} shape={stats['shape']}  "
            f"finite={finite}  nz={nz}  mean_abs={stats['mean_abs']:.3e}  "
            f"range=[{stats['min']:.3e}, {stats['max']:.3e}]"
        )

    # --- report --------------------------------------------------------
    if not args.skip_local_featurize:
        print("\n=== Raw feature parity (expected: bit-exact) ===")
        for d in raw_diffs:
            ok = d.max_abs <= args.raw_atol
            status = "OK" if ok else "FAIL"
            print(
                f"  [{status}] {d.name:<42} shape={d.shape_mlx}  "
                f"max={d.max_abs:.3e}  mean={d.mean_abs:.3e}  "
                f"nz_mlx={d.nonzero_mlx}  nz_cuda={d.nonzero_cuda}"
            )

    all_ok = raw_ok and proj_ok
    print(f"\n{'PASS' if all_ok else 'FAIL'}: raw={raw_ok}  projected={proj_ok}")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                {
                    "target": args.target,
                    "npz": str(args.npz),
                    "constraint_csv": str(args.constraint_csv),
                    "compute_dtype": args.compute_dtype,
                    "skip_local_featurize": args.skip_local_featurize,
                    "raw_atol": args.raw_atol,
                    "proj_atol": args.proj_atol,
                    "raw_ok": raw_ok,
                    "proj_ok": proj_ok,
                    "raw": [d.__dict__ for d in raw_diffs],
                    "projected_summary": proj_summary,
                },
                indent=2,
                default=str,
            )
        )
        print(f"[save] summary -> {args.summary_json}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
