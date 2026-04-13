"""Convert TorchScript weight files directly to safetensors.

This is the memory-aware direct conversion path for real upstream Chai-1
artifacts. It processes one component at a time so large modules like
``trunk.pt`` do not require loading the entire model into RAM at once.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from chai_mlx.config import ChaiConfig
from chai_mlx.io.weights.name_map import build_rename_map, reshape_einsum_weight

COMPONENTS = [
    "feature_embedding",
    "bond_loss_input_proj",
    "token_embedder",
    "trunk",
    "diffusion_module",
    "confidence_head",
]


def _serialize_config(cfg: ChaiConfig) -> dict:
    config_data = asdict(cfg)

    def _convert(obj):
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    return _convert(config_data)


def convert_torchscript_component(
    pt_path: Path,
    component: str,
    out_path: Path,
    *,
    allow_unmapped: bool = False,
) -> dict[str, str]:
    """Convert a single ``.pt`` file to one safetensors shard."""
    from safetensors.numpy import save_file
    import torch

    print(f"\n{'=' * 60}")
    print(f"  Processing: {component}")
    print(f"  Source:     {pt_path}")
    print(f"  Output:     {out_path}")
    print(f"{'=' * 60}")

    mod = torch.jit.load(str(pt_path), map_location="cpu")
    raw: dict[str, np.ndarray] = {}
    for name, param in mod.named_parameters():
        raw[name] = param.detach().numpy().copy()
    for name, buf in mod.named_buffers():
        if name not in raw:
            raw[name] = buf.detach().numpy().copy()
    del mod
    gc.collect()

    print(f"  Extracted {len(raw)} parameters from TorchScript")
    total_bytes = sum(v.nbytes for v in raw.values())
    print(f"  Total size: {total_bytes / 1024**2:.1f} MB")

    rename_map = build_rename_map(component)
    renamed: dict[str, np.ndarray] = {}
    unmapped: list[str] = []
    reshaped_count = 0
    for old_key, arr in raw.items():
        new_key = rename_map.get(old_key)
        if new_key is None:
            unmapped.append(old_key)
            continue
        orig_shape = arr.shape
        arr = reshape_einsum_weight(new_key, arr)
        if arr.shape != orig_shape:
            reshaped_count += 1
            print(f"    Reshaped {new_key}: {orig_shape} -> {arr.shape}")
        renamed[new_key] = arr

    if unmapped:
        preview = ", ".join(unmapped[:10])
        if len(unmapped) > 10:
            preview += f", ... and {len(unmapped) - 10} more"
        if not allow_unmapped:
            raise ValueError(
                f"{component}: found {len(unmapped)} unmapped TorchScript keys: {preview}"
            )
        print(f"  WARNING: {len(unmapped)} unmapped keys:")
        for key in unmapped[:10]:
            print(f"    - {key}")
        if len(unmapped) > 10:
            print(f"    ... and {len(unmapped) - 10} more")

    del raw
    gc.collect()

    mapped_keys = sorted(renamed.keys())
    print(f"  Mapped {len(mapped_keys)} parameters to MLX names")
    if reshaped_count:
        print(f"  Reshaped {reshaped_count} einsum weights to nn.Linear format")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(renamed, str(out_path))
    file_size = out_path.stat().st_size
    print(f"  Wrote {file_size / 1024**2:.1f} MB to {out_path.name}")

    del renamed
    gc.collect()
    return {key: out_path.name for key in mapped_keys}


def convert_torchscript_dir_to_safetensors(
    pt_dir: Path,
    output_dir: Path,
    *,
    cfg: ChaiConfig | None = None,
    allow_unmapped: bool = False,
) -> Path:
    """Convert a directory of component ``.pt`` files to safetensors shards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = cfg or ChaiConfig()

    full_weight_map: dict[str, str] = {}
    total_size = 0

    for component in COMPONENTS:
        pt_path = pt_dir / f"{component}.pt"
        if not pt_path.exists():
            print(f"SKIP: {pt_path} not found")
            continue

        out_path = output_dir / f"model-{component}.safetensors"
        weight_map = convert_torchscript_component(
            pt_path,
            component,
            out_path,
            allow_unmapped=allow_unmapped,
        )
        full_weight_map.update(weight_map)
        total_size += out_path.stat().st_size
        gc.collect()

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": full_weight_map,
    }
    index_path = output_dir / "model.safetensors.index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote index: {index_path}")
    print(f"Total weight_map entries: {len(full_weight_map)}")

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(_serialize_config(cfg), indent=2) + "\n")
    print(f"Wrote config: {config_path}")

    print(f"\nDone! Total safetensors size: {total_size / 1024**2:.1f} MB")
    print(f"Output directory: {output_dir}")
    return output_dir


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert Chai-1 TorchScript weights directly to safetensors"
    )
    parser.add_argument("--pt-dir", type=Path, required=True, help="Directory with .pt files")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--allow-unmapped",
        action="store_true",
        help="Write shards even when TorchScript keys do not map to MLX params",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    convert_torchscript_dir_to_safetensors(
        args.pt_dir,
        args.out_dir,
        allow_unmapped=args.allow_unmapped,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
