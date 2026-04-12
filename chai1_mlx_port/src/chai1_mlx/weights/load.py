from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn


def load_component_npz(module: nn.Module, path: str | Path, *, strict: bool = True) -> nn.Module:
    path = Path(path)
    return module.load_weights(str(path), strict=strict)


def load_model_weights(
    module: nn.Module,
    paths: Iterable[str | Path],
    *,
    strict: bool = False,
) -> nn.Module:
    for path in paths:
        module.load_weights(str(path), strict=strict)
    return module


def load_safetensors(
    module: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
) -> nn.Module:
    """Load weights from a safetensors file or directory containing sharded safetensors."""
    path = Path(path)
    if path.is_dir():
        index_path = path / "model.safetensors.index.json"
        single = path / "model.safetensors"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            for sf in shard_files:
                module.load_weights(str(path / sf), strict=False)
        elif single.exists():
            module.load_weights(str(single), strict=strict)
        else:
            raise FileNotFoundError(f"No safetensors found in {path}")
    else:
        module.load_weights(str(path), strict=strict)
    return module
