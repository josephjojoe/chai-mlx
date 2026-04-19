"""Pre-compute ESM-MLX embeddings for every protein target and cache them.

Why this exists: ESM-2 3B + chai-mlx cannot both fit in 16 GB of unified
memory comfortably.  Loading ESM-2 3B into a subprocess, writing the
per-sequence fp32 embeddings to disk, then letting the Python process
exit (freeing the 6 GB of weights) is the cleanest way to keep the
subsequent chai-mlx inference from paging.

Output schema::

    <cache-dir>/<sha1-of-sequence>.npy     # float32 (L, 2560)
    <cache-dir>/manifest.json              # maps target_name -> [{entity_name, sequence, sha1}]

The adapter this pairs with reads the sha1 from a sequence and loads
the .npy back in without needing the ESM weights present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for submodule in ("chai-lab", "esm-mlx"):
    p = REPO_ROOT / submodule
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _sha1(sequence: str) -> str:
    return hashlib.sha1(sequence.encode()).hexdigest()[:16]


def _collect_unique_proteins(target_names: list[str]) -> dict[str, list[dict]]:
    from cuda_harness.modal_common import DEFAULT_TARGETS

    plan: dict[str, list[dict]] = {}
    for name in target_names:
        if name not in DEFAULT_TARGETS:
            raise KeyError(f"Unknown target {name!r}")
        target = DEFAULT_TARGETS[name]
        entries: list[dict] = []
        for rec in target.records:
            if rec.kind != "protein":
                continue
            entries.append(
                {
                    "entity_name": rec.name,
                    "sequence": rec.sequence,
                    "sha1": _sha1(rec.sequence),
                }
            )
        if entries:
            plan[name] = entries
    return plan


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--targets",
        default="1L2Y,1VII,1CRN,1UBQ,1BRS,1FKB,7TIM,1UBQ_ESM,1CRN_CONSTR",
        help="Comma-separated target names. Non-protein entries are skipped.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/chai_mlx_cuda/esm_mlx_cache"),
    )
    parser.add_argument(
        "--model",
        default="esm2_t36_3B_UR50D",
        help="ESM-2 model name (see esm_mlx.MODEL_CONFIGS).",
    )
    args = parser.parse_args(argv)

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    plan = _collect_unique_proteins(targets)
    unique_sha_to_seq: dict[str, tuple[str, str]] = {}
    for name, entries in plan.items():
        for e in entries:
            unique_sha_to_seq[e["sha1"]] = (e["entity_name"], e["sequence"])

    print(f"[precompute] {len(plan)} target(s); "
          f"{len(unique_sha_to_seq)} unique protein sequence(s)")

    # Filter to those not already cached.
    to_run = [
        (sha, seq)
        for sha, (_name, seq) in unique_sha_to_seq.items()
        if not (args.cache_dir / f"{sha}.npy").is_file()
    ]
    if not to_run:
        print("[precompute] every sequence is already cached; skipping model load")
    else:
        print(f"[precompute] loading {args.model} (this downloads ~6 GB on first run)")
        import mlx.core as mx
        from esm_mlx import ESM2, Tokenizer

        model = ESM2.from_pretrained(args.model)
        tokenizer = Tokenizer()

        for i, (sha, seq) in enumerate(to_run, start=1):
            print(f"[precompute] [{i}/{len(to_run)}] sha={sha} L={len(seq)}")
            tokens = tokenizer.encode(seq)
            out = model(tokens, repr_layers=[model.num_layers])
            last_hidden = out["representations"][model.num_layers]
            trimmed = last_hidden[0, 1:-1]
            mx.eval(trimmed)
            emb = np.asarray(trimmed).astype(np.float32, copy=False)
            assert emb.shape == (len(seq), 2560), (
                f"bad shape {emb.shape} for L={len(seq)}"
            )
            np.save(args.cache_dir / f"{sha}.npy", emb)
            # drop per-step intermediates
            del tokens, out, last_hidden, trimmed, emb
            mx.clear_cache()

    manifest = {
        "model": args.model,
        "targets": plan,
    }
    manifest_path = args.cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[precompute] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
