"""Minimal Modal smoke test: run chai_lab.chai1.run_inference on 1CRN.

Purpose: verify chai-lab itself runs end-to-end on our pinned image with the
pre-downloaded weights. If this works but ``run_intermediates`` OOMs, the bug
is in how we reproduce the pipeline manually.
"""

from __future__ import annotations

import modal

from cuda_harness.modal_common import (
    DEFAULT_TARGETS,
    MINUTES,
    MODELS_DIR,
    OUTPUTS_DIR,
    app,
    chai_model_volume,
    chai_outputs_volume,
    image,
)


@app.function(
    timeout=15 * MINUTES,
    gpu="H100",
    volumes={MODELS_DIR: chai_model_volume, OUTPUTS_DIR: chai_outputs_volume},
    image=image,
)
def smoke() -> dict:
    from pathlib import Path
    import time

    import torch
    from chai_lab import chai1

    fasta_path = Path("/tmp/smoke.fasta")
    fasta_path.write_text(DEFAULT_TARGETS["1CRN"].to_fasta().strip() + "\n")
    output_dir = OUTPUTS_DIR / "smoke" / "1CRN"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Clear any leftover so run_inference doesn't complain about non-empty dir
    for child in sorted(output_dir.rglob("*"), reverse=True):
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            try:
                child.rmdir()
            except OSError:
                pass

    t0 = time.perf_counter()
    chai1.run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        use_esm_embeddings=False,
        use_msa_server=False,
        use_templates_server=False,
        num_trunk_recycles=3,
        num_diffn_timesteps=10,
        num_diffn_samples=1,
        seed=42,
        device="cuda:0",
        low_memory=True,
    )
    elapsed = time.perf_counter() - t0
    return {"ok": True, "elapsed_seconds": elapsed, "gpu": torch.cuda.get_device_name(0)}


@app.local_entrypoint()
def smoke_entry() -> None:
    print("[modal] launching smoke")
    result = smoke.remote()
    print(f"[modal] smoke result: {result}")
