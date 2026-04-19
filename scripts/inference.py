"""End-to-end Chai-MLX inference runner for arbitrary FASTA inputs.

This is the one-stop CLI for "point chai-mlx at a FASTA and get CIFs".  It
exposes the full :func:`chai_mlx.data.featurize.featurize_fasta` surface
(constraints, offline MSA, online MSA server, offline templates, online
templates server, all four ESM backends), runs the production inference
pipeline, and writes the same per-sample CIF + scores + manifest set that
``scripts/run_mlx_sweep.py`` produces for the hardcoded target slate.

Output layout under ``--output-dir``::

    <output-dir>/
        input.fasta                 copy of the input FASTA (for reference)
        pred.model_idx_0.cif         one CIF per diffusion sample
        pred.model_idx_1.cif
        ...
        scores.json                  aggregate_score / ptm / iptm per sample
        manifest.json                dtype, recycles, steps, wall clock, etc.
        coords.npz                   optional: raw coords + scores (--save-npz)

Examples
--------

Fully offline run (matches every validation target in HANDOFF.md §1)::

    python scripts/inference.py \
        --weights-dir weights \
        --fasta input.fasta \
        --output-dir out/my_run

With constraints::

    python scripts/inference.py \
        --weights-dir weights \
        --fasta input.fasta \
        --constraint-path constraints.csv \
        --output-dir out/my_run

Using pre-computed MLX ESM embeddings (recommended on Apple silicon)::

    python scripts/precompute_esm_mlx.py --cache-dir esm_cache --fasta input.fasta
    python scripts/inference.py \
        --weights-dir weights \
        --fasta input.fasta \
        --esm-backend mlx_cache --esm-cache-dir esm_cache \
        --output-dir out/my_run

Online MSA + templates via the ColabFold API (plumbed, not validated —
see HANDOFF.md §1)::

    python scripts/inference.py \
        --weights-dir weights \
        --fasta input.fasta \
        --use-msa-server --use-templates-server \
        --output-dir out/my_run

Offline MSA directory + offline templates (plumbed, not validated)::

    python scripts/inference.py \
        --weights-dir weights \
        --fasta input.fasta \
        --msa-directory my_msas/ --templates-path my_templates.m8 \
        --output-dir out/my_run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from chai_mlx import ChaiMLX, featurize_fasta


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chai-MLX end-to-end inference for arbitrary FASTA inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    core = parser.add_argument_group("core inputs")
    core.add_argument("--weights-dir", type=Path, required=True,
                      help="Directory containing config.json + safetensors, or a HF repo id.")
    core.add_argument("--fasta", type=Path, required=True,
                      help="Input FASTA file. Header format: '>kind|name=SHORT' where "
                           "kind ∈ {protein, ligand, dna, rna, glycan} and SHORT ≤ 4 chars.")
    core.add_argument("--output-dir", type=Path, required=True,
                      help="Directory for CIFs, scores.json, manifest.json (and optional npz).")

    model = parser.add_argument_group("model / sampling")
    model.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    model.add_argument("--recycles", type=int, default=3)
    model.add_argument("--num-steps", type=int, default=200)
    model.add_argument("--num-samples", type=int, default=5)
    model.add_argument("--seed", type=int, default=42)
    model.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False,
                       help="Use the debug inference path and retain full intermediates.")

    constraints = parser.add_argument_group("constraints")
    constraints.add_argument("--constraint-path", type=Path, default=None,
                             help="Chai-lab constraint CSV (contact + pocket + covalent-bond "
                                  "restraints). See HANDOFF.md §1.6 for the tested schema.")

    msa = parser.add_argument_group("MSA (mutually exclusive modes)")
    msa.add_argument("--msa-directory", type=Path, default=None,
                     help="Pre-computed MSA directory in chai-lab's a3m layout. Passed "
                          "through to chai-lab's loader. (Plumbed, not yet validated end-"
                          "to-end in this repo; see HANDOFF.md §1.)")
    msa.add_argument("--use-msa-server", action=argparse.BooleanOptionalAction, default=False,
                     help="Fetch MSAs online via the ColabFold API. Mutually exclusive with "
                          "--msa-directory. (Plumbed via chai-lab; not yet validated here.)")
    msa.add_argument("--msa-server-url", type=str, default="https://api.colabfold.com",
                     help="ColabFold MSA server endpoint.")

    templates = parser.add_argument_group("templates (mutually exclusive modes)")
    templates.add_argument("--templates-path", type=Path, default=None,
                           help="Pre-computed templates m8 file. Mutually exclusive with "
                                "--use-templates-server. (Plumbed, not validated here.)")
    templates.add_argument("--use-templates-server", action=argparse.BooleanOptionalAction,
                           default=False,
                           help="Fetch templates online (via the MSA server). Requires "
                                "--use-msa-server. (Plumbed, not validated here.)")

    esm = parser.add_argument_group("ESM-2 embeddings")
    esm.add_argument("--esm-backend", choices=["off", "chai", "mlx", "mlx_cache"],
                     default="off",
                     help="'off' (default): zero-fill ESM features. 'chai': chai-lab's "
                          "traced CUDA fp16 checkpoint (CUDA host only). 'mlx': esm-mlx "
                          "in-process (16 GB+ RAM, not recommended on 16 GB Macs). "
                          "'mlx_cache': load pre-computed embeddings from --esm-cache-dir.")
    esm.add_argument("--esm-cache-dir", type=Path, default=None,
                     help="Directory of pre-computed <sha1(seq)>.npy files (produced by "
                          "scripts/precompute_esm_mlx.py). Required with --esm-backend mlx_cache.")

    output = parser.add_argument_group("output")
    output.add_argument("--save-npz", type=Path, default=None,
                        help="Optional path to also dump raw coords + scores as .npz "
                             "(CIFs are always produced).")
    output.add_argument("--skip-cif", action=argparse.BooleanOptionalAction, default=False,
                        help="Skip CIF output entirely (scores + optional npz only).")
    output.add_argument("--feature-dir", type=Path, default=None,
                        help="Where chai-lab's featurizer may write intermediate artifacts "
                             "(MSA/template caches, etc.). Defaults to <output-dir>/_features.")

    args = parser.parse_args()

    if args.esm_backend == "mlx_cache" and args.esm_cache_dir is None:
        parser.error("--esm-backend mlx_cache requires --esm-cache-dir")
    if args.use_msa_server and args.msa_directory is not None:
        parser.error("--use-msa-server and --msa-directory are mutually exclusive")
    if args.use_templates_server and args.templates_path is not None:
        parser.error("--use-templates-server and --templates-path are mutually exclusive")
    if args.use_templates_server and not args.use_msa_server:
        parser.error("--use-templates-server requires --use-msa-server (chai-lab's "
                     "templates-server path is driven by the MSA server pipeline)")
    if not args.fasta.exists():
        parser.error(f"--fasta {args.fasta} does not exist")

    return args


def _save_cifs(
    *,
    coords_np: "Any",
    output_dir: Path,
    fasta_path: Path,
    feature_dir: Path,
) -> list[Path]:
    """Write one chai-lab-format CIF per diffusion sample.

    We rebuild a minimal reference ``chai-lab`` feature context with
    ``entity_name_as_subchain=True`` (matching :func:`featurize_fasta`) so
    the per-chain asym labels line up with the FASTA entity names. The
    reference context is only used to drive ``save_to_cif``'s atom /
    residue bookkeeping, which depends solely on the structural layout of
    the FASTA -- not on MSAs, templates, or constraint restraints. So we
    skip those knobs for the ref pass, which keeps this path fast and
    immune to the caller's MSA / template configuration.
    """

    import torch

    # Install chai-lab's RDKit-timeout workaround before importing its
    # featurizer, so ligand targets don't hit the macOS closure pickle
    # bug (HANDOFF.md §5.1).
    from chai_mlx.data._rdkit_timeout_patch import apply_rdkit_timeout_patch
    apply_rdkit_timeout_patch()

    from chai_lab.chai1 import Collate, feature_factory, make_all_atom_feature_context
    from chai_lab.data.io.cif_utils import get_chain_letter, save_to_cif

    ref_feature_dir = feature_dir / "ref_features"
    ref_ctx = make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=ref_feature_dir,
        entity_name_as_subchain=True,
        use_esm_embeddings=False,
        use_msa_server=False,
        use_templates_server=False,
        esm_device=torch.device("cpu"),
    )
    collator = Collate(feature_factory=feature_factory, num_key_atoms=128, num_query_atoms=32)
    output_batch = collator([ref_ctx])["inputs"]
    asym_entity_names = {
        i: get_chain_letter(i) for i, _ in enumerate(ref_ctx.chains, start=1)
    }

    # ``coords_np`` has shape ``(B, S, A, 3)``.  We take B=0 and write one
    # CIF per sample.  ``save_to_cif`` expects ``(1, A, 3)`` tensors.
    output_dir.mkdir(parents=True, exist_ok=True)
    n_samples = coords_np.shape[1]
    cif_paths: list[Path] = []
    for s in range(n_samples):
        cif_path = output_dir / f"pred.model_idx_{s}.cif"
        save_to_cif(
            coords=torch.from_numpy(coords_np[:, s]),
            output_batch=output_batch,
            write_path=cif_path,
            asym_entity_names=asym_entity_names,
        )
        cif_paths.append(cif_path)
    return cif_paths


def _scores_to_dict(ranking: "Any") -> dict[str, list[float]]:
    """Flatten MLX ranking outputs to JSON-serialisable per-sample lists."""
    import numpy as np

    def _as_list(arr: "Any") -> list[float]:
        return np.array(arr.astype(mx.float32)).reshape(-1).tolist()

    out: dict[str, list[float]] = {
        "aggregate_score": _as_list(ranking.aggregate_score),
        "ptm": _as_list(ranking.ptm),
        "iptm": _as_list(ranking.iptm),
        "has_inter_chain_clashes": _as_list(ranking.has_inter_chain_clashes),
    }
    if ranking.complex_plddt is not None:
        out["complex_plddt"] = _as_list(ranking.complex_plddt)
    if ranking.total_clashes is not None:
        out["total_clashes"] = _as_list(ranking.total_clashes)
    if ranking.total_inter_chain_clashes is not None:
        out["total_inter_chain_clashes"] = _as_list(ranking.total_inter_chain_clashes)
    return out


def main() -> None:
    args = _parse_args()

    mx.random.seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.feature_dir or (output_dir / "_features")
    feature_dir.mkdir(parents=True, exist_ok=True)

    fasta_copy = output_dir / "input.fasta"
    try:
        shutil.copyfile(args.fasta, fasta_copy)
    except shutil.SameFileError:
        pass

    print(f"[inference] loading model from {args.weights_dir} (dtype={args.dtype}) ...",
          flush=True)
    t_load = time.perf_counter()
    model = ChaiMLX.from_pretrained(
        args.weights_dir, strict=False, compute_dtype=args.dtype,
    )
    t_load = time.perf_counter() - t_load
    print(f"[inference] model loaded in {t_load:.1f}s", flush=True)

    print(
        f"[inference] featurizing {args.fasta} "
        f"(esm={args.esm_backend}, constraints={bool(args.constraint_path)}, "
        f"msa={'server' if args.use_msa_server else ('dir' if args.msa_directory else 'none')}, "
        f"templates={'server' if args.use_templates_server else ('file' if args.templates_path else 'none')}) ...",
        flush=True,
    )
    ctx = featurize_fasta(
        args.fasta,
        output_dir=feature_dir / "mlx_features",
        constraint_path=args.constraint_path,
        msa_directory=args.msa_directory,
        use_msa_server=args.use_msa_server,
        msa_server_url=args.msa_server_url,
        use_templates_server=args.use_templates_server,
        templates_path=args.templates_path,
        esm_backend=args.esm_backend,
        esm_cache_dir=args.esm_cache_dir,
    )

    print(
        f"[inference] running inference (recycles={args.recycles}, steps={args.num_steps}, "
        f"samples={args.num_samples}, debug={args.debug}) ...",
        flush=True,
    )
    t_inf = time.perf_counter()
    if args.debug:
        result = model.run_inference_debug(
            ctx,
            recycles=args.recycles,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
        )
    else:
        result = model.run_inference(
            ctx,
            recycles=args.recycles,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
        )
    wall = time.perf_counter() - t_inf
    print(f"[inference] inference done in {wall:.1f}s", flush=True)

    import numpy as np

    coords_mx = result.coords.astype(mx.float32)
    coords_np = np.array(coords_mx)
    scores_json = _scores_to_dict(result.ranking)

    agg = np.array(result.ranking.aggregate_score.astype(mx.float32)).reshape(-1)
    best_idx = int(agg.argmax())
    best_score = float(agg[best_idx])
    print(f"[inference] coords shape={coords_np.shape}, best sample={best_idx} "
          f"(aggregate_score={best_score:.4f})", flush=True)

    cif_paths: list[Path] = []
    if not args.skip_cif:
        print("[inference] writing CIFs ...", flush=True)
        t_cif = time.perf_counter()
        try:
            cif_paths = _save_cifs(
                coords_np=coords_np,
                output_dir=output_dir,
                fasta_path=args.fasta,
                feature_dir=feature_dir,
            )
        except Exception as exc:  # pragma: no cover - surfaced to the user
            print(
                f"[inference] CIF export failed: {type(exc).__name__}: {exc}\n"
                "[inference] falling back to --save-npz only; "
                "re-run with --skip-cif to silence this error.",
                file=sys.stderr,
                flush=True,
            )
            cif_paths = []
        else:
            print(f"[inference] wrote {len(cif_paths)} CIF(s) in "
                  f"{time.perf_counter() - t_cif:.1f}s", flush=True)

    scores_path = output_dir / "scores.json"
    scores_path.write_text(json.dumps(scores_json, indent=2))
    print(f"[inference] wrote scores -> {scores_path}", flush=True)

    manifest = {
        "fasta": str(args.fasta),
        "output_dir": str(output_dir),
        "weights_dir": str(args.weights_dir),
        "dtype": args.dtype,
        "seed": args.seed,
        "num_recycles": args.recycles,
        "num_steps": args.num_steps,
        "num_samples": args.num_samples,
        "esm_backend": args.esm_backend,
        "esm_cache_dir": str(args.esm_cache_dir) if args.esm_cache_dir else None,
        "constraint_path": str(args.constraint_path) if args.constraint_path else None,
        "msa_directory": str(args.msa_directory) if args.msa_directory else None,
        "use_msa_server": bool(args.use_msa_server),
        "msa_server_url": args.msa_server_url if args.use_msa_server else None,
        "templates_path": str(args.templates_path) if args.templates_path else None,
        "use_templates_server": bool(args.use_templates_server),
        "best_sample_index": best_idx,
        "best_aggregate_score": best_score,
        "wall_seconds": wall,
        "weights_load_seconds": t_load,
        "cif_paths": [str(p) for p in cif_paths],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[inference] wrote manifest -> {manifest_path}", flush=True)

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            coords=coords_np,
            aggregate_score=np.array(result.ranking.aggregate_score.astype(mx.float32)),
            ptm=np.array(result.ranking.ptm.astype(mx.float32)),
            iptm=np.array(result.ranking.iptm.astype(mx.float32)),
            best_index=best_idx,
            best_score=best_score,
        )
        print(f"[inference] wrote raw tensors -> {args.save_npz}", flush=True)

    print(f"[inference] done -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
