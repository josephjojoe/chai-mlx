from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def _get_param_keys(module: nn.Module) -> set[str]:
    """Walk module parameters and return all dotted key paths."""
    keys: list[str] = []

    def _walk(obj: object, prefix: str) -> None:
        if isinstance(obj, mx.array):
            keys.append(prefix)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                _walk(v, f"{prefix}.{i}" if prefix else str(i))

    _walk(module.parameters(), "")
    return set(keys)


def load_safetensors(
    module: nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
) -> nn.Module:
    """Load weights from a safetensors file or directory containing sharded safetensors.

    When *strict* is True and sharded loading is used, a post-load check verifies
    that every model parameter was covered by the loaded weight files.
    """
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
            if strict:
                loaded_keys = set(index["weight_map"].keys())
                model_keys = _get_param_keys(module)
                missing = model_keys - loaded_keys
                extra = loaded_keys - model_keys
                if missing or extra:
                    parts: list[str] = []
                    if missing:
                        parts.append(
                            f"{len(missing)} model params not in safetensors: "
                            + ", ".join(sorted(missing)[:5])
                            + ("..." if len(missing) > 5 else "")
                        )
                    if extra:
                        parts.append(
                            f"{len(extra)} safetensors keys not in model: "
                            + ", ".join(sorted(extra)[:5])
                            + ("..." if len(extra) > 5 else "")
                        )
                    raise ValueError(
                        "Weight loading mismatch after sharded load. " + "; ".join(parts)
                    )
        elif single.exists():
            module.load_weights(str(single), strict=strict)
        else:
            raise FileNotFoundError(f"No safetensors found in {path}")
    else:
        module.load_weights(str(path), strict=strict)
    return module
