"""Convert exported NPZ weight files to safetensors with MLX-native parameter names.

Usage::

    # Convert a directory of NPZ files (one per TorchScript component)
    python -m chai1_mlx.weights.convert_to_safetensors npz_dir/ output_dir/

The output directory will contain:
  - ``model.safetensors`` (or sharded files with an index)
  - ``config.json``  — serialized ``Chai1Config``
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import Chai1Config
from .name_map import build_rename_map, rename_state_dict, reshape_einsum_weight

_COMPONENT_ORDER = [
    "feature_embedding",
    "bond_loss_input_proj",
    "token_embedder",
    "trunk",
    "diffusion_module",
    "confidence_head",
]

_SHARD_MAX_BYTES = 5 * 1024 ** 3  # 5 GiB per shard


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return dict(f)


def convert_npz_dir_to_safetensors(
    npz_dir: Path,
    output_dir: Path,
    *,
    cfg: Chai1Config | None = None,
) -> Path:
    """Load all component NPZ files, rename, merge, and write safetensors.

    Returns the path to the output directory.
    """
    try:
        from safetensors.numpy import save_file as _save_np
    except ImportError:
        raise ImportError(
            "safetensors is required for conversion. "
            "Install with: pip install safetensors"
        )

    cfg = cfg or Chai1Config()
    output_dir.mkdir(parents=True, exist_ok=True)

    merged: dict[str, np.ndarray] = {}
    for comp in _COMPONENT_ORDER:
        npz_path = npz_dir / f"{comp}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Expected {npz_path}")
        raw = _load_npz(npz_path)
        rmap = build_rename_map(comp)
        renamed = rename_state_dict(raw, rmap)

        unmapped = [k for k in renamed if k.startswith("__unmapped__.")]
        if unmapped:
            import warnings
            warnings.warn(
                f"{comp}: {len(unmapped)} unmapped keys: "
                + ", ".join(unmapped[:5])
                + ("..." if len(unmapped) > 5 else ""),
                stacklevel=2,
            )

        for k, v in renamed.items():
            if not k.startswith("__unmapped__."):
                merged[k] = reshape_einsum_weight(k, v)

    total_bytes = sum(v.nbytes for v in merged.values())

    if total_bytes <= _SHARD_MAX_BYTES:
        out_path = output_dir / "model.safetensors"
        _save_np(merged, str(out_path))
    else:
        _write_sharded(merged, output_dir)

    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2, default=str) + "\n")

    return output_dir


def _write_sharded(merged: dict[str, np.ndarray], output_dir: Path) -> None:
    from safetensors.numpy import save_file as _save_np

    index: dict[str, dict] = {"metadata": {"total_size": 0}, "weight_map": {}}
    shard_num = 0
    shard: dict[str, np.ndarray] = {}
    shard_bytes = 0

    for key in sorted(merged):
        arr = merged[key]
        if shard_bytes + arr.nbytes > _SHARD_MAX_BYTES and shard:
            shard_num += 1
            fname = f"model-{shard_num:05d}-of-NNNNN.safetensors"
            _save_np(shard, str(output_dir / fname))
            for k in shard:
                index["weight_map"][k] = fname
            shard = {}
            shard_bytes = 0
        shard[key] = arr
        shard_bytes += arr.nbytes

    if shard:
        shard_num += 1
        fname = f"model-{shard_num:05d}-of-NNNNN.safetensors"
        _save_np(shard, str(output_dir / fname))
        for k in shard:
            index["weight_map"][k] = fname

    total = shard_num
    renamed_map: dict[str, str] = {}
    for old_fname in set(index["weight_map"].values()):
        new_fname = old_fname.replace("NNNNN", f"{total:05d}")
        old_path = output_dir / old_fname
        new_path = output_dir / new_fname
        if old_path.exists():
            old_path.rename(new_path)
        renamed_map[old_fname] = new_fname

    index["weight_map"] = {
        k: renamed_map.get(v, v) for k, v in index["weight_map"].items()
    }
    total_size = sum(merged[k].nbytes for k in merged)
    index["metadata"]["total_size"] = total_size

    idx_path = output_dir / "model.safetensors.index.json"
    idx_path.write_text(json.dumps(index, indent=2) + "\n")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert Chai-1 NPZ weights to safetensors"
    )
    parser.add_argument("npz_dir", type=Path, help="Directory with per-component .npz files")
    parser.add_argument("output_dir", type=Path, help="Output directory for safetensors")
    args = parser.parse_args(list(argv) if argv is not None else None)
    convert_npz_dir_to_safetensors(args.npz_dir, args.output_dir)
    print(f"Wrote safetensors to {args.output_dir}")


if __name__ == "__main__":  # pragma: no cover
    main()
