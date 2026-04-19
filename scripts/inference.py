"""Simple end-to-end inference runner for Chai MLX.

This script is intentionally thin: it calls either the production inference API
or the debug API, depending on ``--debug``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from chai_mlx import ChaiMLX, featurize_fasta


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chai MLX inference from FASTA")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--fasta", type=Path, required=True)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("/tmp/chai_mlx_inference_features"),
        help="Directory for chai-lab featurization artifacts",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=200)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use debug inference path and retain full intermediates",
    )
    parser.add_argument(
        "--esm-backend",
        choices=["off", "chai", "mlx", "mlx_cache"],
        default="off",
        help="ESM-2 embedding source: 'off' (zero-fill, default), 'chai' "
        "(chai-lab CUDA), 'mlx' (esm-mlx in-process; requires [esm] extra), "
        "'mlx_cache' (load pre-computed embeddings from --esm-cache-dir)",
    )
    parser.add_argument(
        "--esm-cache-dir",
        type=Path,
        default=None,
        help="Directory of pre-computed ESM-MLX embeddings (only used with --esm-backend mlx_cache)",
    )
    parser.add_argument(
        "--save-npz",
        type=Path,
        default=None,
        help="Optional path to save coords/scores as .npz",
    )
    args = parser.parse_args()

    mx.random.seed(args.seed)
    args.feature_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.weights_dir} (dtype={args.dtype}) ...")
    model = ChaiMLX.from_pretrained(args.weights_dir, strict=False, compute_dtype=args.dtype)
    print("Model loaded.")

    print(f"Featurizing FASTA (esm_backend={args.esm_backend}) ...")
    ctx = featurize_fasta(
        args.fasta,
        output_dir=args.feature_dir,
        esm_backend=args.esm_backend,
        esm_cache_dir=args.esm_cache_dir,
        use_msa_server=False,
        use_templates_server=False,
    )

    print(
        f"Running inference (recycles={args.recycles}, steps={args.num_steps}, "
        f"samples={args.num_samples}, debug={args.debug}) ..."
    )
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

    scores = result.ranking.aggregate_score.astype(mx.float32)
    scores_np = scores.reshape(-1)
    best_idx = int(mx.argmax(scores_np).item())
    best_score = float(scores_np[best_idx].item())

    coords = result.coords.astype(mx.float32)
    print(f"Done. coords={tuple(coords.shape)}, scores={tuple(scores.shape)}")
    print(f"Best sample index: {best_idx} (score={best_score:.4f})")

    if args.save_npz is not None:
        import numpy as np

        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.save_npz,
            coords=np.array(coords),
            aggregate_score=np.array(scores),
            best_index=best_idx,
            best_score=best_score,
        )
        print(f"Saved outputs to {args.save_npz}")


if __name__ == "__main__":
    main()
