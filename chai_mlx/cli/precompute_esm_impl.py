"""Pre-compute ESM-MLX embeddings for every protein target and cache them.

Canonical implementation of the ``chai-mlx-precompute-esm`` console
script declared in ``pyproject.toml`` and of the legacy
``scripts/precompute_esm_mlx.py`` forwarder. Logic lives here so the
binary works after a plain ``pip install chai-mlx``, without needing
the ``scripts/`` directory on disk.

Why this exists: ESM-2 3B + chai-mlx cannot both fit in 16 GB of unified
memory comfortably. Loading ESM-2 3B into a subprocess, writing the
per-sequence fp32 embeddings to disk, then letting the Python process
exit (freeing the 6 GB of weights) is the cleanest way to keep the
subsequent chai-mlx inference from paging.

Two input modes:

* ``--targets <names>`` (default) — looks names up in
  ``cuda_harness.modal_common.DEFAULT_TARGETS`` and caches every
  protein record found there. Used by the validation sweep driver.
* ``--fasta <path>`` — parses an arbitrary chai-lab FASTA
  (``>kind|name=SHORT`` headers) and caches every protein record in
  it. Intended for arbitrary user inputs driven by
  :mod:`chai_mlx.cli.infer`.

Output schema::

    <cache-dir>/<sha1-of-sequence>.npy     # float32 (L, 2560)
    <cache-dir>/manifest.json              # maps source_name -> [{entity_name, sequence, sha1}]

Note on UBQG: the UBQG target in DEFAULT_TARGETS reuses 1UBQ's
ubiquitin sequence verbatim, so its sha1 is identical to 1UBQ's. The
cache produced by pre-computing 1UBQ therefore also satisfies
``esm_backend="mlx_cache"`` lookups for UBQG -- UBQG does not need its
own entry in the default target slate and does not need to be passed
separately. If the two sequences ever diverge, add UBQG to
:data:`_DEFAULT_TARGET_SLATE` explicitly.

The adapter this pairs with reads the sha1 from a sequence and loads
the .npy back in without needing the ESM weights present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


# Superset of the target slate we regularly pre-compute embeddings
# against. Each name is validated against the current
# ``DEFAULT_TARGETS`` at invocation time, so dropping / renaming a
# target in ``modal_common.py`` does not silently break the default
# ``chai-mlx-precompute-esm`` run.
_DEFAULT_TARGET_SLATE: tuple[str, ...] = (
    "1L2Y",
    "1VII",
    "1CRN",
    "1UBQ",
    "1BRS",
    "1FKB",
    "7TIM",
    "1UBQ_ESM",
    "1CRN_CONSTR",
    "UBQG",
    "LYSM",
)


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


def _default_target_list() -> list[str]:
    """Return the default target slate, filtered to what exists today.

    Keeps the default invocation resilient to DEFAULT_TARGETS renames:
    if a name in :data:`_DEFAULT_TARGET_SLATE` has been removed, the
    default run continues with the remaining valid entries rather than
    crashing on the first unknown name.
    """
    from cuda_harness.modal_common import DEFAULT_TARGETS

    return [name for name in _DEFAULT_TARGET_SLATE if name in DEFAULT_TARGETS]


def _collect_unique_proteins_from_fasta(fasta_path: Path) -> dict[str, list[dict]]:
    """Parse a chai-lab FASTA and return the same plan shape as
    :func:`_collect_unique_proteins`, keyed by the FASTA filename.

    Only ``>protein|name=...`` records are cached; ligand / dna / rna /
    glycan records are silently skipped (they don't need ESM embeddings).
    Delegates to :func:`chai_mlx.data.fasta.parse_fasta_records` so the
    parser is shared with :func:`chai_mlx.cli.infer.main`.
    """
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--targets",
        default=None,
        help="Comma-separated target names from cuda_harness.modal_common."
             "DEFAULT_TARGETS. Non-protein entries are skipped. Defaults to "
             "the full validation slate when both --targets and --fasta are "
             "omitted.",
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

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.fasta is not None:
        plan = _collect_unique_proteins_from_fasta(args.fasta)
        source_label = f"fasta {args.fasta}"
    else:
        if args.targets is None:
            targets = _default_target_list()
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
        try:
            from esm_mlx import ESM2, Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "chai-mlx-precompute-esm requires the esm_mlx package. "
                "Install with:\n    pip install 'chai-mlx[esm]'"
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
        "targets": plan,
    }
    manifest_path = args.cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[precompute] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
