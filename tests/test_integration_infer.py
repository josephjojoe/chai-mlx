"""Slow end-to-end integration test for ``chai-mlx-infer``.

Opt-in via the ``CHAI_MLX_RUN_SLOW=1`` environment variable. Runs the
full FASTA-to-CIF pipeline on a small target with the default production
inference knobs so the test exercises the real code path rather than a
minimal smoke configuration.

What it exercises:

* Weight resolution through ``ChaiMLX.from_pretrained`` (either a
  pre-populated ``./weights`` directory, or the HuggingFace Hub
  fallback when the env points at the repo id).
* Chai-lab featurization, MLX trunk + diffusion + confidence, ranking.
* CIF output via :func:`chai_lab.data.io.cif_utils.save_to_cif` with
  per-atom pLDDT B-factors.
* ``scores.json`` + ``manifest.json`` shape and content.
* ``scores.model_idx_0.npz`` per-sample NPZ (chai-lab parity).

This intentionally duplicates what ``tests/test_inference_cli.py``
cannot: actually running the model. It is kept opt-in because the
combination of weight download and full inference is too expensive for
default local pytest runs.

Gating: requires ``CHAI_MLX_RUN_SLOW=1``. When local weights are
absent AND the interpreter can't reach the HuggingFace Hub (offline
CI, sandboxed runners), we additionally require ``HF_HUB_TOKEN`` so
the test doesn't hang on a 600 s DNS timeout -- an authenticated
network is a positive signal the caller really intends to download
weights.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


# A small single-chain target that fits comfortably in one crop.
_TRP_CAGE_FASTA = ">protein|name=T1L2\nNLYIQWLKDGGPSSGRPPPS\n"

_SLOW_ENABLED = os.environ.get("CHAI_MLX_RUN_SLOW", "").lower() in ("1", "true", "yes")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_INFER = _REPO_ROOT / "scripts" / "inference.py"
_DEFAULT_WEIGHTS = _REPO_ROOT / "weights"


def _weights_path() -> str:
    """Resolve the weights argument for the integration run.

    Prefer a pre-populated ``./weights`` directory (the checked-in
    convention) since that avoids a 1.2 GB HF download on every CI
    invocation. Fall back to the HF repo id if no local weights exist,
    which triggers ``huggingface_hub.snapshot_download`` in
    :func:`chai_mlx.model.core.load_pretrained_config`.
    """
    if _DEFAULT_WEIGHTS.is_dir() and any(_DEFAULT_WEIGHTS.glob("*.safetensors")):
        return str(_DEFAULT_WEIGHTS)
    return "josephjojoe/chai-mlx"


def _weights_available_locally() -> bool:
    return _DEFAULT_WEIGHTS.is_dir() and any(_DEFAULT_WEIGHTS.glob("*.safetensors"))


def _skip_reason() -> str | None:
    """Return a reason string if the test should be skipped, else None.

    ``CHAI_MLX_RUN_SLOW`` is the primary opt-in gate. When local
    weights are absent we additionally require ``HF_HUB_TOKEN`` as a
    positive signal that the caller's environment can actually reach
    huggingface.co -- otherwise an offline CI runner hangs on a
    600 s DNS timeout instead of skipping cleanly.
    """
    if not _SLOW_ENABLED:
        return "opt-in; set CHAI_MLX_RUN_SLOW=1 to run real inference end-to-end"
    if not _weights_available_locally() and not os.environ.get("HF_HUB_TOKEN"):
        return (
            "no local weights under ./weights/ and HF_HUB_TOKEN is not set; "
            "skipping to avoid a 600s hang on offline CI. Either pre-populate "
            "./weights/ or export HF_HUB_TOKEN to enable the HF fallback."
        )
    return None


@pytest.mark.slow
def test_inference_script_end_to_end_trpcage() -> None:
    reason = _skip_reason()
    if reason is not None:
        pytest.skip(reason)

    with tempfile.TemporaryDirectory(prefix="chai_mlx_integration_") as tmp:
        tmp = Path(tmp)
        fasta_path = tmp / "trpcage.fasta"
        fasta_path.write_text(_TRP_CAGE_FASTA)

        output_dir = tmp / "out"

        cmd = [
            sys.executable,
            str(_SCRIPTS_INFER),
            "--weights-dir", _weights_path(),
            "--fasta", str(fasta_path),
            "--output-dir", str(output_dir),
            # Use the normal inference-scale knobs rather than a tiny
            # smoke setup so the end-to-end path is exercised honestly.
            "--recycles", "3",
            "--num-steps", "200",
            "--num-samples", "1",
            "--esm-backend", "off",
            "--dtype", "reference",
            "--seed", "42",
        ]
        # 10 min is plenty for a 20-residue target on M-series / H100;
        # CI environments without a GPU may take longer on cold start,
        # hence the generous ceiling.
        subprocess.run(cmd, cwd=_REPO_ROOT, check=True, timeout=600)

        cif_path = output_dir / "pred.model_idx_0.cif"
        scores_path = output_dir / "scores.json"
        npz_path = output_dir / "scores.model_idx_0.npz"
        manifest_path = output_dir / "manifest.json"

        assert cif_path.is_file(), f"missing CIF at {cif_path}"
        assert scores_path.is_file(), f"missing scores.json at {scores_path}"
        assert npz_path.is_file(), f"missing per-sample npz at {npz_path}"
        assert manifest_path.is_file(), f"missing manifest.json at {manifest_path}"

        scores = json.loads(scores_path.read_text())
        assert "aggregate_score" in scores
        assert len(scores["aggregate_score"]) == 1, (
            f"expected 1 sample in aggregate_score, got {scores['aggregate_score']}"
        )

        # Per-sample npz sidecar (chai-lab parity)
        import numpy as np
        npz = np.load(npz_path)
        assert "aggregate_score" in npz.files
        assert "ptm" in npz.files
        assert "iptm" in npz.files

        manifest = json.loads(manifest_path.read_text())
        assert manifest["num_samples"] == 1
        assert manifest["esm_backend"] == "off"
        assert manifest["num_steps"] == 200
        assert manifest["cif_paths"] == [str(cif_path)]

        # Basic CIF content check: pLDDT B-factors should be non-trivial.
        # ``save_to_cif`` writes a line per atom;
        # we look for any ATOM record with a non-1.0 biso value.
        cif_text = cif_path.read_text()
        atom_lines = [ln for ln in cif_text.splitlines() if ln.startswith("ATOM")]
        assert atom_lines, "CIF has no ATOM records"
        # The biso column is the penultimate numeric-looking field in
        # chai-lab's modelcif output; we just confirm it's not stuck at
        # 1.000 on every atom (i.e. pLDDTs flowed through), and that the
        # values live in a plausible 0-100 range with some variance.
        bisos = []
        for ln in atom_lines[:200]:
            parts = ln.split()
            # penultimate field in the ATOM record format
            try:
                bisos.append(float(parts[-2]))
            except (IndexError, ValueError):
                pass
        assert bisos, "could not parse any biso values out of the CIF"
        assert any(b != 1.0 for b in bisos), (
            "pLDDT B-factors appear flat (all 1.000); per-atom pLDDT "
            "was not forwarded to save_to_cif."
        )
        # pLDDTs are 0-100; at 200 steps we expect some real spread.
        # Tolerate wide range because Trp-cage at no-ESM can have low-
        # confidence tails, but a completely flat distribution is a
        # real regression signal.
        assert min(bisos) >= 0.0 and max(bisos) <= 100.0, (
            f"pLDDT B-factors out of [0, 100]: min={min(bisos)}, max={max(bisos)}"
        )
        assert max(bisos) - min(bisos) > 1.0, (
            f"pLDDT B-factor spread is <1.0 ({min(bisos)}-{max(bisos)}); "
            "per-atom pLDDT distribution looks degenerate."
        )
