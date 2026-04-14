"""CIF-decoded end-to-end seed sweep for chai-lab vs MLX.

This harness measures consecutive C-alpha spacing by parsing written CIF files,
not by reading raw padded atom tensors. It is intended to answer two questions:

1. Is the residual MLX-vs-chai-lab gap stable across seeds?
2. Does switching MLX from bf16 to fp32 materially reduce that gap?
"""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists():
    sys.path.insert(0, str(LOCAL_CHAI_LAB))

import gemmi
import mlx.core as mx

from chai_lab.chai1 import Collate, feature_factory, make_all_atom_feature_context, run_folding_on_context
from chai_lab.data.io.cif_utils import get_chain_letter, save_to_cif

from chai_mlx import ChaiMLX
from chai_mlx.data.featurize import featurize_fasta


def _write_1l2y_fasta(directory: Path) -> Path:
    fasta = directory / "1L2Y.fasta"
    fasta.write_text(">protein|name=1L2Y\nNLYIQWLKDGGPSSGRPPPS\n")
    return fasta


def _extract_ca_from_cif(cif_path: Path) -> np.ndarray:
    ca_coords = []
    structure = gemmi.read_structure(str(cif_path))
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.name.strip() == "CA":
                        ca_coords.append((atom.pos.x, atom.pos.y, atom.pos.z))
    return np.array(ca_coords, dtype=np.float64)


def _ca_spacing_from_cif(cif_path: Path) -> tuple[float, float, int]:
    ca = _extract_ca_from_cif(cif_path)
    if len(ca) < 2:
        return float("nan"), float("nan"), int(len(ca))
    d = np.sqrt(np.sum(np.diff(ca, axis=0) ** 2, axis=-1))
    return float(np.median(d)), float(np.mean(d)), int(len(ca))


def _asym_entity_names(feature_context) -> dict[int, str]:
    return {i: get_chain_letter(i) for i, _ in enumerate(feature_context.chains, start=1)}


@dataclass
class RunResult:
    framework: str
    seed: int
    median_ca: float
    mean_ca: float
    n_ca: int
    cif_path: Path


def _run_reference(
    feature_context,
    *,
    output_dir: Path,
    seed: int,
    device: torch.device,
    recycles: int,
    num_steps: int,
) -> RunResult:
    candidates = run_folding_on_context(
        feature_context,
        output_dir=output_dir,
        num_trunk_recycles=recycles,
        num_diffn_timesteps=num_steps,
        num_diffn_samples=1,
        seed=seed,
        device=device,
        low_memory=False,
    )
    cif_path = candidates.cif_paths[0]
    median_ca, mean_ca, n_ca = _ca_spacing_from_cif(cif_path)
    return RunResult(
        framework="chai-lab",
        seed=seed,
        median_ca=median_ca,
        mean_ca=mean_ca,
        n_ca=n_ca,
        cif_path=cif_path,
    )


def _run_mlx(
    model: ChaiMLX,
    ctx,
    output_batch: dict,
    asym_entity_names: dict[int, str],
    *,
    output_dir: Path,
    seed: int,
    recycles: int,
    num_steps: int,
) -> RunResult:
    mx.random.seed(seed)
    result = model.fold(ctx, recycles=recycles, num_samples=1, num_steps=num_steps)
    coords = np.array(result.coords.astype(mx.float32))[0, 0]
    cif_path = output_dir / "pred.model_idx_0.cif"
    save_to_cif(
        coords=torch.from_numpy(coords[None]).float(),
        bfactors=None,
        output_batch=output_batch,
        write_path=cif_path,
        asym_entity_names=asym_entity_names,
    )
    median_ca, mean_ca, n_ca = _ca_spacing_from_cif(cif_path)
    del result
    gc.collect()
    mx.clear_cache()
    return RunResult(
        framework=model.cfg.compute_dtype,
        seed=seed,
        median_ca=median_ca,
        mean_ca=mean_ca,
        n_ca=n_ca,
        cif_path=cif_path,
    )


def _print_seed_table(
    seeds: list[int],
    ref_results: dict[int, RunResult],
    mlx_results: dict[str, dict[int, RunResult]],
) -> None:
    print("\nPer-seed CIF-decoded Cα medians (Å)")
    header = f"{'seed':>6}  {'chai-lab':>10}"
    for dtype in mlx_results:
        header += f"  {dtype:>10}  {'gap':>8}"
    print(header)
    for seed in seeds:
        line = f"{seed:6d}  {ref_results[seed].median_ca:10.4f}"
        for dtype in mlx_results:
            med = mlx_results[dtype][seed].median_ca
            gap = med - ref_results[seed].median_ca
            line += f"  {med:10.4f}  {gap:8.4f}"
        print(line)


def _print_summary(
    seeds: list[int],
    ref_results: dict[int, RunResult],
    mlx_results: dict[str, dict[int, RunResult]],
) -> None:
    ref_medians = np.array([ref_results[s].median_ca for s in seeds], dtype=np.float64)
    print("\nSummary")
    print(
        f"  chai-lab median Cα: mean={ref_medians.mean():.4f} "
        f"std={ref_medians.std(ddof=0):.4f} min={ref_medians.min():.4f} max={ref_medians.max():.4f}"
    )
    for dtype, per_seed in mlx_results.items():
        medians = np.array([per_seed[s].median_ca for s in seeds], dtype=np.float64)
        gaps = medians - ref_medians
        print(
            f"  {dtype} median Cα: mean={medians.mean():.4f} "
            f"std={medians.std(ddof=0):.4f} min={medians.min():.4f} max={medians.max():.4f}"
        )
        print(
            f"  {dtype} gap vs chai-lab: mean={gaps.mean():.4f} "
            f"std={gaps.std(ddof=0):.4f} min={gaps.min():.4f} max={gaps.max():.4f}"
        )


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Seed sweep with CIF-decoded Cα metrics")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/chai_mlx_seed_sweep"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 123, 999])
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--mlx-dtypes", nargs="+", default=["bfloat16", "float32"])
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument(
        "--torch-device",
        default="mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chai_mlx_seed_sweep_") as tmpdir:
        tmpdir = Path(tmpdir)
        fasta_path = args.fasta if args.fasta is not None else _write_1l2y_fasta(tmpdir)
        device = torch.device(args.torch_device)

        print(f"Building chai-lab feature context on {device}...")
        ref_feature_context = make_all_atom_feature_context(
            fasta_file=fasta_path,
            output_dir=args.work_dir / "ref_features",
            use_esm_embeddings=False,
            use_msa_server=False,
            use_templates_server=False,
            esm_device=device,
        )
        collator = Collate(feature_factory=feature_factory, num_key_atoms=128, num_query_atoms=32)
        output_batch = collator([ref_feature_context])["inputs"]
        asym_entity_names = _asym_entity_names(ref_feature_context)

        print("Building MLX feature context...")
        mlx_ctx = featurize_fasta(
            fasta_path,
            output_dir=args.work_dir / "mlx_features",
            use_esm_embeddings=False,
            use_msa_server=False,
            use_templates_server=False,
        )

        ref_results: dict[int, RunResult] = {}
        for seed in args.seeds:
            if args.skip_reference:
                cif_path = args.work_dir / "chai_lab" / f"seed_{seed}" / "pred.model_idx_0.cif"
                if not cif_path.exists():
                    raise FileNotFoundError(
                        f"--skip-reference requested but missing chai-lab CIF for seed {seed}: {cif_path}"
                    )
                median_ca, mean_ca, n_ca = _ca_spacing_from_cif(cif_path)
                ref_results[seed] = RunResult(
                    framework="chai-lab",
                    seed=seed,
                    median_ca=median_ca,
                    mean_ca=mean_ca,
                    n_ca=n_ca,
                    cif_path=cif_path,
                )
                print(
                    f"\n[chai-lab reused] seed={seed} "
                    f"median_ca={median_ca:.4f}Å mean_ca={mean_ca:.4f}Å n_ca={n_ca}"
                )
            else:
                print(f"\n[chai-lab] seed={seed}")
                ref_results[seed] = _run_reference(
                    ref_feature_context,
                    output_dir=args.work_dir / "chai_lab" / f"seed_{seed}",
                    seed=seed,
                    device=device,
                    recycles=args.recycles,
                    num_steps=args.num_steps,
                )
                print(
                    f"  median_ca={ref_results[seed].median_ca:.4f}Å "
                    f"mean_ca={ref_results[seed].mean_ca:.4f}Å n_ca={ref_results[seed].n_ca}"
                )

        models: dict[str, ChaiMLX] = {}
        mlx_results: dict[str, dict[int, RunResult]] = {}
        for dtype in args.mlx_dtypes:
            print(f"\nLoading MLX model ({dtype})...")
            models[dtype] = ChaiMLX.from_pretrained(args.weights_dir, strict=False, compute_dtype=dtype)
            mlx_results[dtype] = {}
            for seed in args.seeds:
                print(f"[MLX {dtype}] seed={seed}")
                mlx_results[dtype][seed] = _run_mlx(
                    models[dtype],
                    mlx_ctx,
                    output_batch,
                    asym_entity_names,
                    output_dir=args.work_dir / "mlx" / dtype / f"seed_{seed}",
                    seed=seed,
                    recycles=args.recycles,
                    num_steps=args.num_steps,
                )
                gap = mlx_results[dtype][seed].median_ca - ref_results[seed].median_ca
                print(
                    f"  median_ca={mlx_results[dtype][seed].median_ca:.4f}Å "
                    f"gap={gap:.4f}Å"
                )

        _print_seed_table(args.seeds, ref_results, mlx_results)
        _print_summary(args.seeds, ref_results, mlx_results)


if __name__ == "__main__":
    main()
