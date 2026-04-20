"""Produce a unified vs-PDB summary with best-of-5 + mean-of-5 columns.

Reads the four authoritative CSVs under ``/tmp/chai_mlx_cuda/findings/``
and writes ``findings/summary_primary.json`` along with a
Markdown-ready table on stdout. Best-of-5 is the primary metric because
MLX and CUDA use different RNGs on the diffusion noise, so pairing
``sample_idx=k`` across sides compares arbitrary members of each side's
5-sample distribution. Best-of-5 on each side answers the meaningful
question ("can this side produce a high-quality fold at all?") and is
invariant under that RNG-ordering non-identity.

Usage::

    python3 scripts/summarise_vs_pdb.py

Outputs under ``/tmp/chai_mlx_cuda/findings/``:

* ``summary_primary.json`` — per-axis (no-ESM / ESM-on / RNA / MSA)
  summary with ``{mlx,cuda}_{mean,best}_A`` per target.

This script is a reporting layer; the underlying numbers come from
``scripts/compare_vs_pdb.py``. If any of the input CSVs are missing
(e.g. the RNA sweep hasn't been run), the corresponding axis is
skipped with a warning.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

FINDINGS = Path("/tmp/chai_mlx_cuda/findings")


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open() as fh:
        return list(csv.DictReader(fh))


def _summarise(path: Path, label: str, esm: bool) -> dict:
    rows = _read_rows(path)
    if not rows:
        print(f"[skip] {path} not found", file=sys.stderr)
        return {}
    out: dict[str, dict] = {}
    by_target: dict[str, list[dict]] = {}
    for r in rows:
        by_target.setdefault(r["target"], []).append(r)
    for target, entries in by_target.items():
        mlx = [float(e["mlx_vs_pdb_A"]) for e in entries if e.get("mlx_vs_pdb_A")]
        cuda = [float(e["cuda_vs_pdb_A"]) for e in entries if e.get("cuda_vs_pdb_A")]
        first = entries[0]
        n = 0
        try:
            n = int(first.get("pdb_lengths_picked", "[0]").strip("[]").split(",")[0])
        except ValueError:
            pass
        out[target] = {
            "label": label,
            "esm": esm,
            "n_samples": len(entries),
            "n": n,
            "mlx_mean_A": float(np.mean(mlx)) if mlx else None,
            "mlx_best_A": float(np.min(mlx)) if mlx else None,
            "cuda_mean_A": float(np.mean(cuda)) if cuda else None,
            "cuda_best_A": float(np.min(cuda)) if cuda else None,
        }
    return out


def main() -> None:
    primary = {
        "no_esm": _summarise(FINDINGS / "vs_pdb.csv", "no-ESM", False),
        "esm_on": _summarise(FINDINGS / "vs_pdb_esm.csv", "ESM-on", True),
        "rna": _summarise(FINDINGS / "rna_vs_pdb.csv", "RNA (2KOC)", False),
        "msa_smoke": _summarise(
            FINDINGS / "msa_smoke_vs_pdb.csv",
            "1L2Y ESM+MSA+templates",
            True,
        ),
    }

    out_path = FINDINGS / "summary_primary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(primary, indent=2))
    print(f"[save] {out_path}")

    for axis_key, axis in primary.items():
        if not axis:
            continue
        print(f"\n== {axis_key} ({next(iter(axis.values()))['label']}) ==\n")
        print(
            f"  {'target':<12} {'N':>4} "
            f"{'MLX mean':>10} {'MLX best':>10} "
            f"{'CUDA mean':>10} {'CUDA best':>10} {'best-δ':>8}  verdict"
        )
        for target, r in axis.items():
            if r["mlx_best_A"] is None or r["cuda_best_A"] is None:
                continue
            d = r["mlx_best_A"] - r["cuda_best_A"]
            verdict = ""
            if d < -0.05:
                verdict = "MLX wins (best-of-5)"
            elif d > 0.05:
                verdict = "CUDA wins (best-of-5)"
            else:
                verdict = "tie"
            print(
                f"  {target:<12} {r['n']:>4} "
                f"{r['mlx_mean_A']:>9.2f}Å {r['mlx_best_A']:>9.2f}Å "
                f"{r['cuda_mean_A']:>9.2f}Å {r['cuda_best_A']:>9.2f}Å "
                f"{d:>+7.2f}Å  {verdict}"
            )


if __name__ == "__main__":
    main()
