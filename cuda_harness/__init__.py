"""Chai-MLX ↔ CUDA comparison harnesses running on Modal.

The modules in this package set up a shared Modal app (``chai-mlx-cuda``)
with a reusable image and persistent Volumes for weights and outputs.
Each harness is its own ``modal run`` entry point:

* ``modal run -m cuda_harness.run_reference`` — end-to-end chai-lab CUDA
  inference that writes CIF + scores to local disk.
* ``modal run -m cuda_harness.run_intermediates`` — same flow but also
  bundles every per-module boundary tensor (embedding, trunk recycles,
  diffusion schedule/snapshots, confidence logits) into a single NPZ.
* ``modal run -m cuda_harness.bench_throughput`` — per-module CUDA timings
  with warmup, for side-by-side comparison with the local MLX harness
  (``scripts/mlx_throughput.py``).

The once-off weight sync lives in :mod:`cuda_harness.modal_common`::

    modal run -m cuda_harness.modal_common::download_inference_dependencies

The local-side follow-ups that consume these outputs live under
``scripts/cuda_*``.
"""

from cuda_harness.modal_common import (
    DEFAULT_TARGETS,
    MODELS_DIR,
    OUTPUTS_DIR,
    FastaRecord,
    Target,
    app,
    chai_model_volume,
    chai_outputs_volume,
    download_inference_dependencies,
    fasta_for,
    filter_targets,
    image,
)

__all__ = [
    "DEFAULT_TARGETS",
    "FastaRecord",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "Target",
    "app",
    "chai_model_volume",
    "chai_outputs_volume",
    "download_inference_dependencies",
    "fasta_for",
    "filter_targets",
    "image",
]
