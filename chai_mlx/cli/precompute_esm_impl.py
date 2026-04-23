"""Pre-compute ESM-MLX embeddings for protein records in a FASTA file.

This CLI exists for the ``esm_backend="mlx_cache"`` workflow on
lower-memory Macs: run ESM-2 3B in a separate process, write one
``<sha1>.npy`` per unique protein sequence, then run
``chai-mlx-infer --esm-backend mlx_cache`` without keeping the ESM
weights resident.

Output schema::

    <cache-dir>/<sha1-of-sequence>.npy     # float32 (L, 2560)
    <cache-dir>/manifest.json              # input FASTA + cached records
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha1(sequence: str) -> str:
    return hashlib.sha1(sequence.encode()).hexdigest()[:16]


def _collect_unique_proteins_from_fasta(fasta_path: Path) -> dict[str, list[dict]]:
    """Parse a FASTA and return one cache plan entry per protein record."""
    from chai_mlx.data.fasta import parse_fasta_records

    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    records = parse_fasta_records(fasta_path)
    entries: list[dict] = []
    for rec in records:
        if rec.kind != "protein" or not rec.name or not rec.sequence:
            continue
        seq = rec.sequence.upper()
        entries.append(
            {
                "entity_name": rec.name,
                "sequence": seq,
                "sha1": _sha1(seq),
            }
        )

    if not entries:
        return {}
    return {fasta_path.name: entries}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fasta",
        type=Path,
        required=True,
        help="Path to a chai-lab-format FASTA. Every '>protein|name=...' "
             "record is cached; non-protein records are skipped.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/chai_mlx_esm_cache"),
    )
    parser.add_argument(
        "--model",
        default="esm2_t36_3B_UR50D",
        help="ESM-2 model name (see esm_mlx.MODEL_CONFIGS).",
    )
    args = parser.parse_args(argv)

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    plan = _collect_unique_proteins_from_fasta(args.fasta)
    source_label = f"fasta {args.fasta}"

    unique_sha_to_seq: dict[str, tuple[str, str]] = {}
    for _name, entries in plan.items():
        for e in entries:
            unique_sha_to_seq[e["sha1"]] = (e["entity_name"], e["sequence"])

    print(f"[precompute] {source_label}; "
          f"{len(unique_sha_to_seq)} unique protein sequence(s)")
    if not unique_sha_to_seq:
        print("[precompute] no protein sequences found; nothing to cache")
        manifest = {
            "model": args.model,
            "fasta": str(args.fasta),
            "records": plan,
        }
        (args.cache_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return

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
        try:
            from esm_mlx import ESM2, Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "chai-mlx-precompute-esm requires the esm_mlx package. "
                "Install with:\n"
                "    pip install -e '.[esm]'\n"
                "    pip install \"chai-mlx[esm] @ git+https://github.com/josephjojoe/chai-mlx\""
            ) from exc

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
        "fasta": str(args.fasta),
        "records": plan,
    }
    manifest_path = args.cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[precompute] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
