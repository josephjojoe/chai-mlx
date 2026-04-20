"""Isolate how much of MLX-vs-CUDA structural drift comes from the trunk vs the diffusion loop.

Given a CUDA intermediates NPZ from
``modal run -m cuda_harness.run_intermediates``, this script runs the
full 200-step MLX diffusion sampler under **two separate conditions**:

1. **trunk_from_mlx**: MLX runs everything itself — the trunk, the
   diffusion sampler, and the confidence head. This is the normal
   inference path.

2. **trunk_from_cuda**: MLX uses the *CUDA-captured* ``single_trunk`` and
   ``pair_trunk`` tensors to build the diffusion cache, and then runs
   the full MLX diffusion sampler on top. The only drift here is
   whatever the MLX diffusion module itself introduces relative to
   CUDA once both sides have identical conditioning.

The delta between these two numbers helps separate how much of the
structural disagreement comes from trunk drift propagating forward versus
how much is intrinsic to the diffusion loop once both sides have
identical conditioning.

Usage
-----

::

    python scripts/cuda_mlx_diffusion_isolation.py \\
        --weights-dir weights \\
        --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \\
        --cuda-reference-dir /tmp/chai_mlx_cuda/reference/1L2Y/seed_42 \\
        --mlx-seed 42

Note: extending this analysis to multiple seeds requires one captured NPZ
per seed.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mlx.core as mx
import numpy as np

from chai_mlx import ChaiMLX
from chai_mlx.utils import resolve_dtype

from cuda_parity import (  # type: ignore[import-not-found]
    _as_mx,
    _load_npz,
    _read_manifest,
    _reconstruct_embedding_outputs,
    _reconstruct_feature_context,
    _reconstruct_trunk_outputs,
    _tensor_to_numpy,
)


N_DIFFUSION_SAMPLES = 5


@dataclass
class SampleRMSD:
    sample_idx: int
    rmsd_trunk_from_mlx: float
    rmsd_trunk_from_cuda: float


def _ca_from_cif(path: Path) -> np.ndarray:
    """Return representative backbone atoms for the structure.

    For protein chains: Cα. For DNA/RNA chains: phosphate (P). If the
    structure has both (e.g. a protein+RNA complex), both are
    concatenated in chain order. This generalisation lets the
    diffusion-isolation harness run on nucleic-acid-only targets
    (1BNA, 2KOC) that have no Cα atoms.
    """
    import gemmi

    st = gemmi.read_structure(str(path))
    coords = []
    for chain in st[0]:
        for res in chain:
            for atom in res:
                if atom.name in ("CA", "P"):
                    p = atom.pos
                    coords.append([p.x, p.y, p.z])
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


def _token_center_ca(
    atom_pos: np.ndarray, structure_inputs, token_exists_mask: np.ndarray
) -> np.ndarray:
    """Extract Cα coordinates using ``token_centre_atom_index``.

    Avoids having to round-trip through CIF.  Mirrors what chai-lab's
    CIF writer ultimately does when extracting Cα-per-token for scoring.
    """
    idx = np.asarray(structure_inputs.token_centre_atom_index, dtype=np.int64)[0]
    mask = np.asarray(token_exists_mask, dtype=bool)[0]
    ca = atom_pos[idx]
    return ca[mask]


def _run_mlx_diffusion(
    model: ChaiMLX,
    trunk_out,
    structure,
    *,
    num_samples: int,
    num_steps: int | None,
    seed: int,
) -> mx.array:
    """Run ``diffusion_step`` num_steps times starting from ``init_noise``."""
    mx.random.seed(seed)
    cache = model.prepare_diffusion_cache(trunk_out)
    mx.eval(
        cache.s_static,
        cache.z_cond,
        cache.blocked_pair_base,
        cache.atom_cond,
        cache.atom_single_cond,
        *cache.pair_biases,
    )

    batch_size = trunk_out.single_trunk.shape[0]
    coords = model.init_noise(batch_size, num_samples, structure)
    for sigma_curr, sigma_next, gamma in model.schedule(num_steps=num_steps):
        coords = model.diffusion_step(cache, coords, sigma_curr, sigma_next, gamma)
        mx.eval(coords)
    return coords


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="CUDA intermediates NPZ (from run_intermediates.py)",
    )
    parser.add_argument(
        "--cuda-reference-dir",
        type=Path,
        required=True,
        help="Directory containing CUDA CIFs, e.g. /tmp/chai_mlx_cuda/reference/1L2Y/seed_42",
    )
    parser.add_argument(
        "--compute-dtype", default=None, choices=["reference", "float32"]
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override num diffusion steps (default: use NPZ manifest value)",
    )
    parser.add_argument("--mlx-seed", type=int, default=42)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    data = _load_npz(args.npz)
    manifest = _read_manifest(data)
    target = manifest.get("target", "?")
    seed = int(manifest.get("seed", 42))
    num_steps = args.num_steps or int(manifest.get("num_steps", 200))
    print(f"[load] {args.npz}")
    print(
        f"  target={target} seed={seed} num_steps={num_steps} "
        f"gpu={manifest.get('gpu_name', '?')}"
    )

    print(f"[load] model weights from {args.weights_dir}")
    model = ChaiMLX.from_pretrained(
        args.weights_dir, strict=False, compute_dtype=args.compute_dtype
    )
    dtype = resolve_dtype(model.cfg)
    print(f"  compute_dtype={'float32' if dtype == mx.float32 else 'reference'}")

    ctx = _reconstruct_feature_context(data)
    cuda_emb = _reconstruct_embedding_outputs(data, ctx.structure_inputs, dtype=dtype)

    # Reference CUDA Cα coordinates (from CIFs).
    cuda_cif_paths = sorted(args.cuda_reference_dir.glob("pred.model_idx_*.cif"))
    if len(cuda_cif_paths) < N_DIFFUSION_SAMPLES:
        raise FileNotFoundError(
            f"Expected {N_DIFFUSION_SAMPLES} CUDA CIFs under {args.cuda_reference_dir}, "
            f"found {len(cuda_cif_paths)}"
        )
    cuda_ca = [_ca_from_cif(p) for p in cuda_cif_paths]

    # --- Condition 1: MLX runs the trunk itself (same as full inference) ---
    print("\n[1] MLX trunk + MLX diffusion (full MLX inference path)")
    trunk_mlx = model.trunk(cuda_emb, recycles=int(manifest.get("num_recycles", 3)))
    mx.eval(trunk_mlx.single_trunk, trunk_mlx.pair_trunk)
    coords_mlx_mlx = _run_mlx_diffusion(
        model,
        trunk_mlx,
        ctx.structure_inputs,
        num_samples=N_DIFFUSION_SAMPLES,
        num_steps=num_steps,
        seed=args.mlx_seed,
    )
    coords_mlx_mlx_np = _tensor_to_numpy(coords_mlx_mlx)
    # shape: (batch=1, samples=5, atoms, 3)
    coords_mlx_mlx_np = coords_mlx_mlx_np.reshape(
        coords_mlx_mlx_np.shape[0] * coords_mlx_mlx_np.shape[1],
        coords_mlx_mlx_np.shape[2],
        coords_mlx_mlx_np.shape[3],
    )

    # Free the MLX trunk output now that we've sampled from it.
    del trunk_mlx
    gc.collect()
    mx.clear_cache()

    # --- Condition 2: MLX receives CUDA trunk tensors, runs MLX diffusion ---
    print("[2] CUDA trunk + MLX diffusion (trunk-isolation path)")
    trunk_from_cuda = _reconstruct_trunk_outputs(data, cuda_emb, dtype=dtype)
    coords_cuda_mlx = _run_mlx_diffusion(
        model,
        trunk_from_cuda,
        ctx.structure_inputs,
        num_samples=N_DIFFUSION_SAMPLES,
        num_steps=num_steps,
        seed=args.mlx_seed,
    )
    coords_cuda_mlx_np = _tensor_to_numpy(coords_cuda_mlx)
    coords_cuda_mlx_np = coords_cuda_mlx_np.reshape(
        coords_cuda_mlx_np.shape[0] * coords_cuda_mlx_np.shape[1],
        coords_cuda_mlx_np.shape[2],
        coords_cuda_mlx_np.shape[3],
    )

    # Extract Cα from each MLX sample using token_centre_atom_index.
    token_mask = np.asarray(ctx.structure_inputs.token_exists_mask)
    ca_mlx_mlx = [
        _token_center_ca(coords_mlx_mlx_np[s], ctx.structure_inputs, token_mask)
        for s in range(N_DIFFUSION_SAMPLES)
    ]
    ca_cuda_mlx = [
        _token_center_ca(coords_cuda_mlx_np[s], ctx.structure_inputs, token_mask)
        for s in range(N_DIFFUSION_SAMPLES)
    ]

    # Report per-sample RMSD for both conditions.
    print(
        f"\n{'sample':>6}  {'trunk_from_mlx Cα RMSD':>24}  "
        f"{'trunk_from_cuda Cα RMSD':>25}  {'Δ':>8}"
    )
    rows: list[SampleRMSD] = []
    for s in range(N_DIFFUSION_SAMPLES):
        r1 = _kabsch_rmsd(ca_mlx_mlx[s], cuda_ca[s])
        r2 = _kabsch_rmsd(ca_cuda_mlx[s], cuda_ca[s])
        rows.append(SampleRMSD(sample_idx=s, rmsd_trunk_from_mlx=r1, rmsd_trunk_from_cuda=r2))
        print(f"  {s:>6}  {r1:>24.4f}  {r2:>25.4f}  {r1 - r2:>+8.4f}")

    r1_vals = np.array([r.rmsd_trunk_from_mlx for r in rows])
    r2_vals = np.array([r.rmsd_trunk_from_cuda for r in rows])

    print("\n" + "=" * 70)
    print(f"Summary  (target={target} seed={seed})")
    print("=" * 70)
    print(
        f"  trunk_from_mlx  Cα RMSD: mean={r1_vals.mean():.4f} Å  "
        f"median={np.median(r1_vals):.4f} Å  max={r1_vals.max():.4f} Å"
    )
    print(
        f"  trunk_from_cuda Cα RMSD: mean={r2_vals.mean():.4f} Å  "
        f"median={np.median(r2_vals):.4f} Å  max={r2_vals.max():.4f} Å"
    )
    print(
        f"  Δ (trunk drift contribution): mean={(r1_vals - r2_vals).mean():+.4f} Å"
    )
    print()
    print("Interpretation:")
    print(
        "  If Δ is near zero, trunk drift is not the dominant driver of the current"
    )
    print("  structural gap; the diffusion loop itself is contributing most of it.")
    print(
        "  If Δ is large, trunk drift is cascading into diffusion and accounting"
    )
    print(
        "  for most of the gap; feeding MLX the CUDA trunk tightens the structural match."
    )

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "npz": str(args.npz),
            "cuda_reference_dir": str(args.cuda_reference_dir),
            "compute_dtype": "float32" if dtype == mx.float32 else "reference",
            "manifest": manifest,
            "mlx_seed": args.mlx_seed,
            "num_steps": num_steps,
            "samples": [asdict(r) for r in rows],
            "summary": {
                "rmsd_trunk_from_mlx_mean": float(r1_vals.mean()),
                "rmsd_trunk_from_mlx_max": float(r1_vals.max()),
                "rmsd_trunk_from_cuda_mean": float(r2_vals.mean()),
                "rmsd_trunk_from_cuda_max": float(r2_vals.max()),
                "delta_mean": float((r1_vals - r2_vals).mean()),
            },
        }
        args.summary_json.write_text(json.dumps(payload, indent=2))
        print(f"\n[save] summary -> {args.summary_json}")


if __name__ == "__main__":
    main()
