#!/usr/bin/env python3
"""Memory-efficient TorchScript → safetensors conversion.

Processes one component at a time so peak RAM stays under ~2 GB
(the trunk.pt file is ~604 MB on disk, ~1.2 GB as float32 arrays).

Usage::

    python3 scripts/convert_weights.py \\
        --pt-dir ../chai-lab/downloads/models_v2/ \\
        --out-dir weights/

Produces per-component .safetensors files and a merged index.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

COMPONENTS = [
    "feature_embedding",
    "bond_loss_input_proj",
    "token_embedder",
    "trunk",
    "diffusion_module",
    "confidence_head",
]


def convert_one_component(
    pt_path: Path,
    component: str,
    out_path: Path,
) -> dict[str, str]:
    """Convert a single .pt → .safetensors, returning the weight_map entries.

    Loads TorchScript, extracts numpy arrays, applies name mapping,
    writes safetensors, then frees everything.
    """
    from safetensors.numpy import save_file

    from chai_mlx.io.weights.name_map import build_rename_map, reshape_einsum_weight

    print(f"\n{'='*60}")
    print(f"  Processing: {component}")
    print(f"  Source:     {pt_path}")
    print(f"  Output:     {out_path}")
    print(f"{'='*60}")

    # Step 1: Load TorchScript and extract numpy arrays
    import torch
    mod = torch.jit.load(str(pt_path), map_location="cpu")
    raw: dict[str, np.ndarray] = {}
    for name, param in mod.named_parameters():
        raw[name] = param.detach().numpy().copy()
    for name, buf in mod.named_buffers():
        if name not in raw:
            raw[name] = buf.detach().numpy().copy()
    # Free the TorchScript module immediately
    del mod
    gc.collect()

    print(f"  Extracted {len(raw)} parameters from TorchScript")
    total_bytes = sum(v.nbytes for v in raw.values())
    print(f"  Total size: {total_bytes / 1024**2:.1f} MB")

    # Step 2: Apply name mapping and reshape einsum weights
    rmap = build_rename_map(component)
    renamed: dict[str, np.ndarray] = {}
    unmapped: list[str] = []
    reshaped_count = 0
    for old_key, arr in raw.items():
        new_key = rmap.get(old_key)
        if new_key is not None:
            orig_shape = arr.shape
            arr = reshape_einsum_weight(new_key, arr)
            if arr.shape != orig_shape:
                reshaped_count += 1
                print(f"    Reshaped {new_key}: {orig_shape} -> {arr.shape}")
            renamed[new_key] = arr
        else:
            unmapped.append(old_key)

    if unmapped:
        print(f"  WARNING: {len(unmapped)} unmapped keys:")
        for k in unmapped[:10]:
            print(f"    - {k}")
        if len(unmapped) > 10:
            print(f"    ... and {len(unmapped) - 10} more")

    # Free raw dict
    del raw
    gc.collect()

    mapped_keys = sorted(renamed.keys())
    print(f"  Mapped {len(mapped_keys)} parameters to MLX names")
    if reshaped_count:
        print(f"  Reshaped {reshaped_count} einsum weights to nn.Linear format")

    # Step 3: Write safetensors
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(renamed, str(out_path))
    file_size = out_path.stat().st_size
    print(f"  Wrote {file_size / 1024**2:.1f} MB to {out_path.name}")

    # Build weight_map entries
    weight_map = {k: out_path.name for k in mapped_keys}

    # Free
    del renamed
    gc.collect()

    return weight_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Chai-1 TorchScript weights to safetensors")
    parser.add_argument("--pt-dir", type=Path, required=True, help="Directory with .pt files")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()

    pt_dir = args.pt_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    full_weight_map: dict[str, str] = {}
    total_size = 0

    for component in COMPONENTS:
        pt_path = pt_dir / f"{component}.pt"
        if not pt_path.exists():
            print(f"SKIP: {pt_path} not found")
            continue

        shard_name = f"model-{component}.safetensors"
        out_path = out_dir / shard_name

        wmap = convert_one_component(pt_path, component, out_path)
        full_weight_map.update(wmap)
        total_size += out_path.stat().st_size

        # Force GC between components
        gc.collect()

    # Write the index file
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": full_weight_map,
    }
    index_path = out_dir / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote index: {index_path}")
    print(f"Total weight_map entries: {len(full_weight_map)}")

    # Write config.json
    from chai_mlx.config import ChaiConfig
    cfg = ChaiConfig()
    config_data = asdict(cfg)
    # Convert tuples to lists for JSON
    def _convert(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj
    config_data = _convert(config_data)
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(config_data, indent=2) + "\n")
    print(f"Wrote config: {config_path}")

    print(f"\nDone! Total safetensors size: {total_size / 1024**2:.1f} MB")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
