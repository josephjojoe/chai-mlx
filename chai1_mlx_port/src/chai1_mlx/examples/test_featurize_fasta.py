"""Integration test for featurize_fasta → FeatureEmbedding.

Exercises the full FASTA → raw features → encoding → projection pipeline
and verifies output dimensions match the model config.

Requires ``chai-lab`` and ``torch`` to be installed::

    pip install .[featurize]
    python -m chai1_mlx.examples.test_featurize_fasta --fasta /path/to/input.fasta

If no --fasta is provided, a minimal synthetic FASTA is created in a temp dir.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import mlx.core as mx


def _write_minimal_fasta(directory: Path) -> Path:
    fasta = directory / "test.fasta"
    fasta.write_text(
        ">protein|name=test\nMKFLILFNILVSTLSFSSAQA\n"
    )
    return fasta


def run_test(fasta_path: Path | None = None) -> None:
    from ..config import Chai1Config
    from ..embeddings import FeatureEmbedding
    from ..featurize import featurize_fasta

    cfg = Chai1Config()

    with tempfile.TemporaryDirectory(prefix="chai_mlx_test_") as tmpdir:
        tmpdir = Path(tmpdir)
        if fasta_path is None:
            fasta_path = _write_minimal_fasta(tmpdir)

        print(f"Featurizing {fasta_path} ...")
        ctx = featurize_fasta(fasta_path, output_dir=tmpdir)

    assert ctx.raw_features is not None, "raw_features should be populated"
    print(f"  raw_features keys: {sorted(ctx.raw_features.keys())}")

    fe = FeatureEmbedding(cfg)
    feats = fe(ctx)

    expected = {
        "token_single": ("B", "N", cfg.hidden.token_single),
        "token_pair_trunk": ("B", "N", "N", cfg.hidden.token_pair),
        "token_pair_structure": ("B", "N", "N", cfg.hidden.token_pair),
        "atom_single_trunk": ("B", "A", cfg.hidden.atom_single),
        "atom_single_structure": ("B", "A", cfg.hidden.atom_single),
        "atom_pair_trunk": ("B", "blocks", "32", "128", cfg.hidden.atom_pair),
        "atom_pair_structure": ("B", "blocks", "32", "128", cfg.hidden.atom_pair),
        "msa": ("B", "M", "N", cfg.hidden.msa),
        "templates": ("B", "T", "N", "N", cfg.hidden.template_pair),
    }

    all_ok = True
    for key, expected_shape in expected.items():
        arr = feats[key]
        hidden_dim = expected_shape[-1]
        actual_dim = arr.shape[-1]
        ok = actual_dim == hidden_dim
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] {key}: shape={arr.shape}, expected last dim={hidden_dim}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\nAll dimension checks passed.")
    else:
        raise AssertionError("Some dimension checks failed — see above.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Integration test for featurize_fasta")
    parser.add_argument("--fasta", type=Path, default=None)
    args = parser.parse_args()
    run_test(fasta_path=args.fasta)


if __name__ == "__main__":
    main()
