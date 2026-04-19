"""Pre-compute ESM-MLX embeddings for every protein target and cache them.

Why this exists: ESM-2 3B + chai-mlx cannot both fit in 16 GB of unified
memory comfortably.  Loading ESM-2 3B into a subprocess, writing the
per-sequence fp32 embeddings to disk, then letting the Python process
exit (freeing the 6 GB of weights) is the cleanest way to keep the
subsequent chai-mlx inference from paging.

Two input modes:

* ``--targets <names>`` (default) — looks names up in
  ``cuda_harness.modal_common.DEFAULT_TARGETS`` and caches every
  protein record found there.  Used by the validation sweep driver.
* ``--fasta <path>`` — parses an arbitrary chai-lab FASTA
  (``>kind|name=SHORT`` headers) and caches every protein record in
  it.  Intended for arbitrary user inputs driven by
  ``scripts/inference.py``.

Output schema::

    <cache-dir>/<sha1-of-sequence>.npy     # float32 (L, 2560)
    <cache-dir>/manifest.json              # maps source_name -> [{entity_name, sequence, sha1}]

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


def _collect_unique_proteins_from_fasta(fasta_path: Path) -> dict[str, list[dict]]:
    """Parse a chai-lab FASTA and return the same plan shape as
    :func:`_collect_unique_proteins`, keyed by the FASTA filename.

    Only ``>protein|name=...`` records are cached; ligand / dna / rna /
    glycan records are silently skipped (they don't need ESM embeddings).
    Header parsing matches chai-lab's ``read_inputs`` but is kept
    deliberately light-weight so this script doesn't have to import
    chai-lab's featurizer just to tokenise a header line.
    """
    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")

    entries: list[dict] = []
    current_kind: str | None = None
    current_name: str | None = None
    current_seq: list[str] = []

    def _flush() -> None:
        nonlocal current_kind, current_name, current_seq
        if current_kind == "protein" and current_name and current_seq:
            seq = "".join(current_seq).strip().upper()
            if seq:
                entries.append(
                    {
                        "entity_name": current_name,
                        "sequence": seq,
                        "sha1": _sha1(seq),
                    }
                )
        current_kind = None
        current_name = None
        current_seq = []

    for line in fasta_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            _flush()
            header = line[1:]
            if "|" not in header:
                # Non-standard header; ignore (chai-lab would raise, but we
                # just skip so users can still cache the rest of the file).
                continue
            kind, _, rest = header.partition("|")
            name_part = ""
            for kv in rest.split("|"):
                k, _, v = kv.partition("=")
                if k.strip() == "name":
                    name_part = v.strip()
                    break
            current_kind = kind.strip().lower()
            current_name = name_part or None
        else:
            current_seq.append(line)
    _flush()

    if not entries:
        return {}
    return {fasta_path.name: entries}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--targets",
        default=None,
        help="Comma-separated target names from cuda_harness.modal_common."
             "DEFAULT_TARGETS. Non-protein entries are skipped.",
    )
    mode.add_argument(
        "--fasta",
        type=Path,
        default=None,
        help="Path to a chai-lab-format FASTA. Every '>protein|name=...' "
             "record is cached; non-protein records are skipped.",
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

    if args.targets is None and args.fasta is None:
        # Default to the full validation slate so existing
        # `python scripts/precompute_esm_mlx.py` invocations keep working.
        args.targets = "1L2Y,1VII,1CRN,1UBQ,1BRS,1FKB,7TIM,1UBQ_ESM,1CRN_CONSTR"

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.fasta is not None:
        plan = _collect_unique_proteins_from_fasta(args.fasta)
        source_label = f"fasta {args.fasta}"
    else:
        targets = [t.strip() for t in args.targets.split(",") if t.strip()]
        plan = _collect_unique_proteins(targets)
        source_label = f"{len(plan)} target(s)"

    unique_sha_to_seq: dict[str, tuple[str, str]] = {}
    for _name, entries in plan.items():
        for e in entries:
            unique_sha_to_seq[e["sha1"]] = (e["entity_name"], e["sequence"])

    print(f"[precompute] {source_label}; "
          f"{len(unique_sha_to_seq)} unique protein sequence(s)")
    if not unique_sha_to_seq:
        print("[precompute] no protein sequences found; nothing to cache")
        manifest = {"model": args.model, "targets": plan}
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
