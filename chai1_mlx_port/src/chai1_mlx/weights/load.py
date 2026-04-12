from __future__ import annotations

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
