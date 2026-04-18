"""Shared Modal infrastructure for chai-mlx CUDA comparison harnesses.

This module defines:

* ``app`` — the shared Modal app (name: ``chai-mlx-cuda``).
* ``image`` — the container image with ``chai_lab``, ``torch``, ``gemmi``,
  ``biopython``, ``numpy`` pinned to match the local environment semantics.
* ``chai_model_volume`` / ``chai_outputs_volume`` — persistent Modal Volumes
  for weights and run outputs (mirrors Modal's official chai-1 example pattern
  so we don't re-download the ~10 GB of weights on every run).
* ``download_inference_dependencies`` — Modal Function that populates the
  weights volume from Chai Discovery's CDN. Runs once; subsequent harness runs
  reuse the cached weights.

All of the per-harness files in this directory (``run_reference.py``,
``run_intermediates.py``, ``bench_throughput.py``) import from here so that
the image, weights, and outputs storage are shared across harnesses.

Usage
-----

Prime the weights cache (only needed once per Modal workspace)::

    modal run cuda_harness.modal_common::download_inference_dependencies

Then any harness in this directory can run via ``modal run``.
"""

from __future__ import annotations

from pathlib import Path

import modal

MINUTES = 60

app = modal.App(name="chai-mlx-cuda")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # uv needs git on PATH to install a VCS spec; debian_slim doesn't ship it.
    .apt_install("git")
    .uv_pip_install(
        # Install chai-lab from the same upstream commit our local checkout
        # is pinned at (post-v0.6.1 main, with PR #360's `_component_moved_to`
        # caching helper that our intermediates harness depends on, and
        # PR #415's gemmi-0.7 support). Modal's canonical example pins 0.5.0
        # but that predates both PRs and the published 0.6.1 wheel does not
        # include #360 either.
        "chai_lab @ git+https://github.com/chaidiscovery/chai-lab@61036259c98222160963cb780750e354876ce485",
        "huggingface-hub==0.36.0",
        "numpy>=1.26,<2",
    )
    .uv_pip_install(
        "torch==2.7.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
)

chai_model_volume = modal.Volume.from_name(
    "chai-mlx-weights",
    create_if_missing=True,
)
MODELS_DIR = Path("/models/chai1")

chai_outputs_volume = modal.Volume.from_name(
    "chai-mlx-cuda-outputs",
    create_if_missing=True,
)
OUTPUTS_DIR = Path("/outputs")

image = image.env(
    {
        "CHAI_DOWNLOADS_DIR": str(MODELS_DIR),
        "HF_XET_HIGH_PERFORMANCE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        # chai-lab's TorchScript trunk is memory-hungry in the masked_fill /
        # einsum paths around the MSA pair-weighted averaging op. Switching
        # to expandable segments keeps fragmentation from tipping us over
        # the 80 GB H100 limit on model_size=256 crops.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
)


INFERENCE_DEPENDENCIES = (
    "conformers_v1.apkl",
    "models_v2/trunk.pt",
    "models_v2/token_embedder.pt",
    "models_v2/feature_embedding.pt",
    "models_v2/diffusion_module.pt",
    "models_v2/confidence_head.pt",
    "models_v2/bond_loss_input_proj.pt",
)


@app.function(
    volumes={MODELS_DIR: chai_model_volume},
    image=image,
    timeout=30 * MINUTES,
)
async def download_inference_dependencies(force: bool = False) -> list[str]:
    """Populate the shared weights volume from Chai's CDN.

    Mirrors Modal's own example so we stay compatible with their caching.
    Runs concurrently to minimize wall clock.  Returns the list of files it
    touched so the caller can log what happened.
    """
    import asyncio

    import aiohttp

    base_url = "https://chaiassets.com/chai1-inference-depencencies/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        )
    }

    downloaded: list[str] = []

    async def _download(session: aiohttp.ClientSession, dep: str) -> None:
        local_path = MODELS_DIR / dep
        if not force and local_path.exists():
            return
        local_path.parent.mkdir(parents=True, exist_ok=True)
        url = base_url + dep
        print(f"[download] {dep}")
        async with session.get(url) as response:
            response.raise_for_status()
            with open(local_path, "wb") as fh:
                while chunk := await response.content.read(1 << 20):
                    fh.write(chunk)
        downloaded.append(dep)

    async with aiohttp.ClientSession(headers=headers) as session:
        await asyncio.gather(*(_download(session, dep) for dep in INFERENCE_DEPENDENCIES))

    await chai_model_volume.commit.aio()
    return downloaded


# Default test sequences used by the harnesses.
DEFAULT_TARGETS: dict[str, str] = {
    # 20-residue miniprotein — the local parity baseline.
    "1L2Y": "NLYIQWLKDGGPSSGRPPPS",
    # 35-residue villin headpiece — classic small fold, mixed secondary structure.
    "1VII": "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
    # 46-residue crambin — solved to 0.54 Å, good ground-truth target.
    "1CRN": "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN",
    # 76-residue ubiquitin — solved at 1.8 Å (PDB 1UBQ), mixed α/β fold, a
    # classic stress test for mid-size monomers.
    "1UBQ": (
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    ),
}


def fasta_for(name: str, sequence: str) -> str:
    return f">protein|name={name}\n{sequence}\n"
