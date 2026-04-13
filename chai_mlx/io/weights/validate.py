"""Shape and dtype validation for loaded weights against the MLX module tree."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn

from chai_mlx.model.api import ChaiMLX
from chai_mlx.config import ChaiConfig


def _flat_params(module: nn.Module) -> dict[str, mx.array]:
    """Flatten the module parameter tree into dotted-key → array."""
    out: dict[str, mx.array] = {}
    for k, v in module.parameters().items():
        if isinstance(v, mx.array):
            out[k] = v
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                out[f"{k}.{k2}"] = v2
    # MLX parameters() already returns fully-qualified paths; use trainable_parameters
    # or leaf_modules if the above doesn't work for nested structures.
    return out


def validate_weights(
    model: ChaiMLX,
    weights_path: str | Path,
    *,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Load weights into *model* and report any shape/key mismatches.

    Returns a list of issue strings (empty if all OK).
    """
    weights_path = Path(weights_path)
    issues: list[str] = []

    try:
        model.load_weights(str(weights_path), strict=strict)
        if verbose:
            print(f"load_weights succeeded (strict={strict})")
    except Exception as e:
        issues.append(f"load_weights failed: {e}")
        return issues

    return issues


def validate_shapes(
    model: ChaiMLX,
    weights: dict[str, mx.array],
    *,
    verbose: bool = True,
) -> list[str]:
    """Compare shapes of *weights* dict against *model* parameters.

    *weights* keys should use full ``ChaiMLX`` parameter paths.
    """
    issues: list[str] = []

    model_params: dict[str, tuple] = {}
    for name, arr in _iter_params(model):
        model_params[name] = tuple(arr.shape)

    for key, arr in weights.items():
        w_shape = tuple(arr.shape)
        if key not in model_params:
            issues.append(f"EXTRA weight key not in model: {key}  shape={w_shape}")
            continue
        m_shape = model_params[key]
        if w_shape != m_shape:
            issues.append(
                f"SHAPE MISMATCH {key}: weight={w_shape} model={m_shape}"
            )

    for key in model_params:
        if key not in weights:
            issues.append(f"MISSING weight for model param: {key}")

    if verbose:
        if issues:
            for iss in issues:
                print(f"  [!] {iss}")
        else:
            print(f"  All {len(model_params)} parameters matched.")

    return issues


def _iter_params(module: nn.Module, prefix: str = "") -> list[tuple[str, mx.array]]:
    """Recursively yield (dotted_name, array) for all parameters."""
    results = []
    items = module.parameters()
    _walk(items, prefix, results)
    return results


def _walk(obj, prefix: str, results: list) -> None:
    if isinstance(obj, mx.array):
        results.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, f"{prefix}.{k}" if prefix else k, results)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk(v, f"{prefix}.{i}" if prefix else str(i), results)
