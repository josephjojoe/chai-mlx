from __future__ import annotations

import json
from pathlib import Path

import pytest

import mlx.core as mx

from chai_mlx.io.weights.load import load_safetensors
from chai_mlx.io.weights.validate import validate_shapes, validate_weights


class FakeModule:
    def __init__(self, params: dict[str, object], *, fail_on_load: bool = False) -> None:
        self._params = params
        self.fail_on_load = fail_on_load
        self.loaded_paths: list[str] = []

    def parameters(self) -> dict[str, object]:
        return self._params

    def load_weights(self, path: str, strict: bool = True):
        self.loaded_paths.append(f"{path}|strict={strict}")
        if self.fail_on_load:
            raise RuntimeError("boom")
        return self


def _write_index(directory: Path, weight_map: dict[str, str]) -> None:
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    for shard_name in set(weight_map.values()):
        (directory / shard_name).write_text("")


def test_load_safetensors_strict_sharded_loads_all_shards(tmp_path: Path) -> None:
    _write_index(
        tmp_path,
        {
            "layer.weight": "shard-1.safetensors",
            "layer.bias": "shard-2.safetensors",
        },
    )
    module = FakeModule(
        {"layer": {"weight": mx.zeros((2, 2)), "bias": mx.zeros((2,))}}
    )

    result = load_safetensors(module, tmp_path, strict=True)

    assert result is module
    assert module.loaded_paths == [
        f"{tmp_path / 'shard-1.safetensors'}|strict=False",
        f"{tmp_path / 'shard-2.safetensors'}|strict=False",
    ]


def test_load_safetensors_strict_sharded_rejects_missing_model_keys(tmp_path: Path) -> None:
    _write_index(tmp_path, {"layer.weight": "shard-1.safetensors"})
    module = FakeModule(
        {"layer": {"weight": mx.zeros((2, 2)), "bias": mx.zeros((2,))}}
    )

    with pytest.raises(ValueError, match="model params not in safetensors"):
        load_safetensors(module, tmp_path, strict=True)


def test_validate_weights_reports_load_failures(tmp_path: Path) -> None:
    module = FakeModule({}, fail_on_load=True)

    issues = validate_weights(module, tmp_path / "weights.safetensors", verbose=False)

    assert issues == ["load_weights failed: boom"]


def test_validate_shapes_reports_missing_extra_and_mismatched_keys() -> None:
    module = FakeModule(
        {
            "layer": {
                "weight": mx.zeros((2, 2), dtype=mx.float32),
                "bias": mx.zeros((2,), dtype=mx.float32),
            }
        }
    )
    weights = {
        "layer.weight": mx.zeros((3, 2), dtype=mx.float32),
        "extra.weight": mx.zeros((1,), dtype=mx.float32),
    }

    issues = validate_shapes(module, weights, verbose=False)

    assert any("SHAPE MISMATCH layer.weight" in issue for issue in issues)
    assert any("EXTRA weight key not in model: extra.weight" in issue for issue in issues)
    assert any("MISSING weight for model param: layer.bias" in issue for issue in issues)
