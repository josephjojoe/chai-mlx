"""Detailed trunk block trace against TorchScript via graph-probed outputs.

This harness does not rely on TorchScript hooks. Instead it clones
``trunk.forward_<crop_size>``, registers the pairformer block outputs as
explicit graph outputs, and compares them against the MLX trunk trace when both
consume the same reference embedding tensors.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists():
    sys.path.insert(0, str(LOCAL_CHAI_LAB))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from chai_lab.chai1 import _component_moved_to  # type: ignore[import-not-found]

from chai_mlx.data.types import EmbeddingOutputs, StructureInputs
from chai_mlx.io.weights.load import _get_param_keys
from chai_mlx.model.core import _preserve_fp32_param_keys, load_pretrained_config
from chai_mlx.model.trunk import Trunk
from chai_mlx.utils import resolve_dtype
from layer_parity import capture_trunk
from stage_isolation_parity import _ref

_SINGLE_NAME_RE = re.compile(r"^token_single_repr(?P<idx>\d+)?\.1$")
_PAIR_NAME_RE = re.compile(r"^z(?P<idx>\d+)\.1$")
_PRE_PAIRFORMER_OUTPUTS = (
    ("input0.1", "pre_pairformer.single_after_recycle"),
    ("input.1", "pre_pairformer.pair_after_recycle"),
    ("token_pair_repr.1", "pre_pairformer.pair_after_template"),
    ("msa_repr.1", "pre_pairformer.msa_input"),
    ("msa_repr0.1", "pre_pairformer.msa_iter_0"),
    ("token_pair_repr0.1", "pre_pairformer.pair_iter_0"),
    ("msa_repr1.1", "pre_pairformer.msa_iter_1"),
    ("token_pair_repr1.1", "pre_pairformer.pair_iter_1"),
    ("msa_repr2.1", "pre_pairformer.msa_iter_2"),
    ("token_pair_repr2.1", "pre_pairformer.pair_iter_2"),
)
_TRACE_REF_KEYS = (
    "embedding.outputs.single_initial",
    "embedding.outputs.pair_initial",
    "embedding.outputs.msa_input",
    "embedding.outputs.template_input",
)
_TRACE_SELF_CHECK_KEYS = (
    "trunk.outputs.single_trunk",
    "trunk.outputs.pair_trunk",
)


class _TraceTrunkModel(nn.Module):
    def __init__(self, cfg, *, pairformer_block_limit: int | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk_module = Trunk(cfg)
        if pairformer_block_limit is not None:
            self.trunk_module.pairformer_stack.blocks = self.trunk_module.pairformer_stack.blocks[
                :pairformer_block_limit
            ]


def _load_reference_subset(
    path: Path,
    *,
    include_self_check: bool,
) -> dict[str, np.ndarray]:
    keys = list(_TRACE_REF_KEYS)
    if include_self_check:
        keys.extend(_TRACE_SELF_CHECK_KEYS)
    with np.load(path, allow_pickle=False) as f:
        return {key: f[key] for key in keys if key in f}


def _load_trunk_model(
    weights_dir: Path,
    *,
    compute_dtype: str,
    pairformer_block_limit: int | None = None,
) -> _TraceTrunkModel:
    path, cfg = load_pretrained_config(weights_dir, compute_dtype=compute_dtype)
    model = _TraceTrunkModel(cfg, pairformer_block_limit=pairformer_block_limit)
    dtype = resolve_dtype(cfg)
    _load_trace_weights(model, path, dtype=dtype, strict=False)
    return model


def _load_trace_weights(
    module: nn.Module,
    path: Path,
    *,
    dtype: mx.Dtype,
    strict: bool,
) -> nn.Module:
    target_keys = _get_param_keys(module)
    preserve_fp32 = _preserve_fp32_param_keys(module)

    def _cast_loaded_value(key: str, value: np.ndarray) -> mx.array:
        arr = mx.array(value)
        if dtype != mx.float32 and key not in preserve_fp32 and arr.dtype != dtype:
            arr = arr.astype(dtype)
        return arr

    def _load_single_file(file_path: Path, keys: list[str]) -> None:
        if not keys:
            return
        with safe_open(str(file_path), framework="np") as f:
            pairs = [(key, _cast_loaded_value(key, f.get_tensor(key))) for key in keys]
        module.load_weights(pairs, strict=False)
        del pairs
        _clear_accelerator_memory()

    if path.is_dir():
        index_path = path / "model.safetensors.index.json"
        single = path / "model.safetensors"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            weight_map: dict[str, str] = index["weight_map"]
            available = set(weight_map)
            missing = target_keys - available
            shard_to_keys: dict[str, list[str]] = {}
            for key in sorted(target_keys & available):
                shard_to_keys.setdefault(weight_map[key], []).append(key)
            for shard, keys in shard_to_keys.items():
                _load_single_file(path / shard, keys)
            if strict and missing:
                raise ValueError(
                    "Weight loading mismatch after filtered sharded load. "
                    f"{len(missing)} model params not in safetensors: "
                    + ", ".join(sorted(missing)[:5])
                    + ("..." if len(missing) > 5 else "")
                )
        elif single.exists():
            with safe_open(str(single), framework="np") as f:
                available = set(f.keys())
            missing = target_keys - available
            _load_single_file(single, sorted(target_keys & available))
            if strict and missing:
                raise ValueError(
                    "Weight loading mismatch after filtered load. "
                    f"{len(missing)} model params not in safetensors: "
                    + ", ".join(sorted(missing)[:5])
                    + ("..." if len(missing) > 5 else "")
                )
        else:
            raise FileNotFoundError(f"No safetensors found in {path}")
    else:
        with safe_open(str(path), framework="np") as f:
            available = set(f.keys())
        missing = target_keys - available
        _load_single_file(path, sorted(target_keys & available))
        if strict and missing:
            raise ValueError(
                "Weight loading mismatch after filtered load. "
                f"{len(missing)} model params not in safetensors: "
                + ", ".join(sorted(missing)[:5])
                + ("..." if len(missing) > 5 else "")
            )
    return module


def _mx_np(x: mx.array) -> np.ndarray:
    if x.dtype == mx.bfloat16:
        x = x.astype(mx.float32)
    mx.eval(x)
    return np.array(x, copy=False)


def _pt_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().float().numpy()


def _compare_stats(diff: np.ndarray) -> tuple[float, float, float]:
    if diff.size == 0:
        return 0.0, 0.0, 0.0
    return float(diff.max()), float(diff.mean()), float(np.quantile(diff, 0.99))


def _trace_mask(name: str, structure: StructureInputs, shape: tuple[int, ...]) -> np.ndarray | None:
    token_mask = np.array(structure.token_exists_mask, copy=False).astype(bool)
    token_pair_mask = np.array(structure.token_pair_mask, copy=False).astype(bool)
    msa_mask = np.array(structure.msa_mask, copy=False).astype(bool)
    if name.startswith("pre_pairformer.single_"):
        return np.broadcast_to(token_mask[..., None], shape)
    if name.startswith("pre_pairformer.pair_"):
        return np.broadcast_to(token_pair_mask[..., None], shape)
    if name.startswith("pre_pairformer.msa_"):
        return np.broadcast_to(msa_mask[..., None], shape)
    if name.endswith(".single") or name.endswith("single_trunk"):
        return np.broadcast_to(token_mask[..., None], shape)
    if name.endswith(".pair") or name.endswith("pair_trunk"):
        return np.broadcast_to(token_pair_mask[..., None], shape)
    return None


def _block_sort_key(name: str) -> tuple[int, int, int]:
    recycle = int(name.split("trunk.recycle_")[1].split(".")[0])
    block = int(name.split("block_")[1].split(".")[0])
    kind = 0 if name.endswith(".single") else 1
    return recycle, block, kind


def _compare_trace(
    ref_trace: OrderedDict[str, np.ndarray],
    mlx_trace: OrderedDict[str, np.ndarray],
    structure: StructureInputs,
    *,
    jump_threshold: float,
) -> tuple[int, tuple[str, float, float, float] | None]:
    failures = 0
    first_jump = None
    for name, ref in ref_trace.items():
        got = mlx_trace.get(name)
        if got is None:
            print(f"[MISSING] {name}")
            failures += 1
            continue
        if ref.shape != got.shape:
            print(f"[SHAPE] {name}: ref={ref.shape} mlx={got.shape}")
            failures += 1
            continue
        diff = np.abs(ref.astype(np.float32) - got.astype(np.float32))
        full_max, full_mean, full_p99 = _compare_stats(diff)
        mask = _trace_mask(name, structure, diff.shape)
        cmp_max, cmp_mean, cmp_p99 = full_max, full_mean, full_p99
        if mask is not None and np.any(mask):
            cmp_max, cmp_mean, cmp_p99 = _compare_stats(diff[mask])
        print(
            f"{name:<52} max={cmp_max:9.4e}  mean={cmp_mean:9.4e}  "
            f"p99={cmp_p99:9.4e}  full_max={full_max:9.4e}  shape={ref.shape}"
        )
        if first_jump is None and cmp_p99 >= jump_threshold:
            first_jump = (name, cmp_max, cmp_mean, cmp_p99)

    extra = sorted(set(mlx_trace) - set(ref_trace))
    if extra:
        print(f"[INFO] extra MLX checkpoints: {len(extra)}")
        for name in extra[:20]:
            print(f"  {name}")

    if first_jump is None:
        print(f"\nNo checkpoint exceeded threshold {jump_threshold:.3g}.")
    else:
        name, cmp_max, cmp_mean, cmp_p99 = first_jump
        print(
            f"\nFirst checkpoint over threshold {jump_threshold:.3g}: "
            f"{name} (max={cmp_max:.4e}, mean={cmp_mean:.4e}, p99={cmp_p99:.4e})"
        )
    return failures, first_jump


def _torch_tensor(
    value: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
    is_mask: bool = False,
) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.to(device)
    else:
        tensor = torch.from_numpy(np.asarray(value)).to(device)
    if is_mask:
        return tensor.bool()
    return tensor.to(dtype if dtype is not None else tensor.dtype)


def _single_sort_key(name: str) -> int:
    match = _SINGLE_NAME_RE.match(name)
    if match is None:
        raise ValueError(f"not a trunk single name: {name}")
    idx = match.group("idx")
    return -1 if idx is None else int(idx)


def _pair_sort_key(name: str) -> int:
    match = _PAIR_NAME_RE.match(name)
    if match is None:
        raise ValueError(f"not a trunk pair name: {name}")
    return int(match.group("idx"))


def _clear_accelerator_memory() -> None:
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass
    if getattr(torch, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _build_probe_function(
    trunk_module,
    *,
    crop_size: int,
    block_start: int,
    block_end: int,
    trace_scope: str,
) -> tuple[torch._C.ScriptFunction, list[str]]:
    graph = trunk_module._c._get_method(f"forward_{crop_size}").graph.copy()
    orig_output = list(graph.outputs())[0]
    final_unpack = graph.create("prim::TupleUnpack", [orig_output], 2)
    graph.appendNode(final_unpack)
    final_single, final_pair = list(final_unpack.outputs())

    values: dict[str, torch._C.Value] = {}
    for node in graph.nodes():
        for output in node.outputs():
            values[output.debugName()] = output

    while len(list(graph.outputs())):
        graph.eraseOutput(0)

    ordered_names: list[str] = []
    if trace_scope == "pre_pairformer":
        missing = [name for name, _ in _PRE_PAIRFORMER_OUTPUTS if name not in values]
        if missing:
            raise ValueError(f"missing pre-pairformer graph values: {missing}")
        for graph_name, out_name in _PRE_PAIRFORMER_OUTPUTS:
            graph.registerOutput(values[graph_name])
            ordered_names.append(out_name)
    elif trace_scope == "pairformer":
        single_names = sorted(
            [name for name in values if _SINGLE_NAME_RE.match(name) and _single_sort_key(name) <= 45],
            key=_single_sort_key,
        )
        pair_names = sorted(
            [
                name
                for name in values
                if _PAIR_NAME_RE.match(name)
                and 22 <= _pair_sort_key(name) <= 114
                and _pair_sort_key(name) % 2 == 0
            ],
            key=_pair_sort_key,
        )

        if len(single_names) != 47:
            raise ValueError(f"expected 47 pairformer single outputs, found {len(single_names)}: {single_names}")
        if len(pair_names) != 47:
            raise ValueError(f"expected 47 pairformer pair outputs, found {len(pair_names)}: {pair_names}")
        if not (0 <= block_start <= block_end <= 47):
            raise ValueError(f"invalid block range {block_start}..{block_end}; expected 0 <= start <= end <= 47")

        for block_idx, (single_name, pair_name) in enumerate(zip(single_names, pair_names)):
            if block_start <= block_idx <= block_end:
                graph.registerOutput(values[single_name])
                graph.registerOutput(values[pair_name])
                ordered_names.extend(
                    [
                        f"pairformer.block_{block_idx}.single",
                        f"pairformer.block_{block_idx}.pair",
                    ]
                )
        if block_start <= 47 <= block_end:
            graph.registerOutput(final_single)
            graph.registerOutput(final_pair)
            ordered_names.extend(
                [
                    "pairformer.block_47.single",
                    "pairformer.block_47.pair",
                ]
            )
    else:
        raise ValueError(f"unknown trace_scope: {trace_scope}")
    # Always return the final trunk outputs for recycle chaining.
    graph.registerOutput(final_single)
    graph.registerOutput(final_pair)

    graph.makeMultiOutputIntoTuple()
    fn = torch._C._create_function_from_graph(f"trunk_probe_{crop_size}", graph)
    return fn, ordered_names


def _reference_inputs(
    ref: dict[str, np.ndarray],
    structure: StructureInputs,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "single_init": _torch_tensor(ref["embedding.outputs.single_initial"], device=device, dtype=torch.bfloat16),
        "pair_init": _torch_tensor(ref["embedding.outputs.pair_initial"], device=device, dtype=torch.bfloat16),
        "msa_input": _torch_tensor(ref["embedding.outputs.msa_input"], device=device, dtype=torch.bfloat16),
        "template_input": _torch_tensor(ref["embedding.outputs.template_input"], device=device, dtype=torch.bfloat16),
        "msa_mask": _torch_tensor(np.array(structure.msa_mask), device=device, is_mask=True),
        "template_input_masks": _torch_tensor(np.array(structure.template_input_masks), device=device, is_mask=True),
        "token_single_mask": _torch_tensor(np.array(structure.token_exists_mask), device=device, is_mask=True),
        "token_pair_mask": _torch_tensor(np.array(structure.token_pair_mask), device=device, is_mask=True),
    }


def _load_trace_structure(input_npz: Path) -> StructureInputs:
    with np.load(input_npz, allow_pickle=False) as f:
        token_exists_mask = mx.array(f["structure_inputs.token_exists_mask"])
        token_pair_mask = mx.array(f["structure_inputs.token_pair_mask"])
        msa_mask = mx.array(f["structure_inputs.msa_mask"])
        template_input_masks = mx.array(f["structure_inputs.template_input_masks"])
    batch_size, n_tokens = token_exists_mask.shape
    return StructureInputs(
        atom_exists_mask=mx.zeros((batch_size, 0), dtype=mx.float32),
        token_exists_mask=token_exists_mask,
        token_pair_mask=token_pair_mask,
        atom_token_index=mx.zeros((batch_size, 0), dtype=mx.int32),
        atom_within_token_index=mx.zeros((batch_size, 0), dtype=mx.int32),
        token_reference_atom_index=mx.zeros((batch_size, n_tokens), dtype=mx.int32),
        token_asym_id=mx.zeros((batch_size, n_tokens), dtype=mx.int32),
        token_entity_id=mx.zeros((batch_size, n_tokens), dtype=mx.int32),
        token_chain_id=mx.zeros((batch_size, n_tokens), dtype=mx.int32),
        token_is_polymer=mx.ones((batch_size, n_tokens), dtype=mx.float32),
        token_centre_atom_index=mx.zeros((batch_size, n_tokens), dtype=mx.int32),
        msa_mask=msa_mask,
        template_input_masks=template_input_masks,
    )


def _reconstruct_trace_embeddings(
    ref: dict[str, np.ndarray],
    structure: StructureInputs,
    *,
    dtype: mx.Dtype,
) -> EmbeddingOutputs:
    empty_f = mx.zeros((0,), dtype=dtype)
    return EmbeddingOutputs(
        token_single_input=empty_f,
        token_pair_input=empty_f,
        token_pair_structure_input=empty_f,
        atom_single_input=empty_f,
        atom_single_structure_input=empty_f,
        atom_pair_input=empty_f,
        atom_pair_structure_input=empty_f,
        msa_input=_ref(ref, "embedding.outputs.msa_input", dtype),
        template_input=_ref(ref, "embedding.outputs.template_input", dtype),
        single_initial=_ref(ref, "embedding.outputs.single_initial", dtype),
        single_structure=empty_f,
        pair_initial=_ref(ref, "embedding.outputs.pair_initial", dtype),
        pair_structure=empty_f,
        structure_inputs=structure,
    )


def _run_reference_trace(
    ref: dict[str, np.ndarray],
    structure: StructureInputs,
    *,
    recycles: int,
    device: torch.device,
    block_start: int,
    block_end: int,
    trace_scope: str,
) -> OrderedDict[str, np.ndarray]:
    crop_size = int(ref["embedding.outputs.single_initial"].shape[1])
    inputs = _reference_inputs(ref, structure, device=device)
    trace: OrderedDict[str, np.ndarray] = OrderedDict()

    with torch.inference_mode():
        with _component_moved_to("trunk.pt", device) as trunk:
            probe_fn, ordered_names = _build_probe_function(
                trunk.jit_module,
                crop_size=crop_size,
                block_start=block_start,
                block_end=block_end,
                trace_scope=trace_scope,
            )

            single_prev = inputs["single_init"]
            pair_prev = inputs["pair_init"]

            for recycle_idx in range(recycles):
                probe_out = probe_fn(
                    trunk.jit_module,
                    inputs["single_init"],
                    inputs["pair_init"],
                    single_prev,
                    pair_prev,
                    inputs["msa_input"],
                    inputs["msa_mask"],
                    inputs["template_input"],
                    inputs["template_input_masks"],
                    inputs["token_single_mask"],
                    inputs["token_pair_mask"],
                )
                if len(probe_out) != len(ordered_names) + 2:
                    raise ValueError(
                        f"expected {len(ordered_names) + 2} probe outputs, got {len(probe_out)}"
                    )

                for name, value in zip(ordered_names, probe_out[: len(ordered_names)]):
                    trace[f"trunk.recycle_{recycle_idx}.{name}"] = _pt_np(value)
                single_prev = probe_out[-2]
                pair_prev = probe_out[-1]

    if block_end >= 47:
        trace["trunk.outputs.single_trunk"] = _pt_np(single_prev)
        trace["trunk.outputs.pair_trunk"] = _pt_np(pair_prev)
    return trace


def _reference_self_check(ref: dict[str, np.ndarray], trace: OrderedDict[str, np.ndarray]) -> None:
    for key in ("trunk.outputs.single_trunk", "trunk.outputs.pair_trunk"):
        if key not in ref or key not in trace:
            continue
        diff = np.abs(ref[key].astype(np.float32) - trace[key].astype(np.float32))
        max_diff, mean_diff, p99_diff = _compare_stats(diff)
        print(
            f"Reference self-check {key}: max={max_diff:.4e} "
            f"mean={mean_diff:.4e} p99={p99_diff:.4e}"
        )


def _write_trace_npz(path: Path, trace: OrderedDict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **trace)


def _load_trace_npz(path: Path) -> OrderedDict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as f:
        return OrderedDict((key, f[key]) for key in f.files)


def _build_mlx_trace(
    model: _TraceTrunkModel,
    ref: dict[str, np.ndarray],
    structure: StructureInputs,
    *,
    recycles: int,
    block_start: int,
    block_end: int,
) -> OrderedDict[str, np.ndarray]:
    dtype = mx.float32 if model.cfg.compute_dtype == "float32" else mx.bfloat16
    emb = _reconstruct_trace_embeddings(ref, structure, dtype=dtype)
    tensors: dict[str, np.ndarray] = {}
    stop_after_block = block_end if recycles == 1 and block_end < 47 else None
    trunk_out = capture_trunk(
        model,
        emb,
        recycles=recycles,
        tensors=tensors,
        capture_detail="pairformer",
        pairformer_block_start=block_start,
        pairformer_block_end=block_end,
        stop_after_pairformer_block=stop_after_block,
        record_outputs=False,
    )

    trace = OrderedDict()
    block_keys = sorted(
        [
            key
            for key in tensors
            if key.startswith("trunk.recycle_")
            and ".pairformer.block_" in key
            and (key.endswith(".single") or key.endswith(".pair"))
        ],
        key=_block_sort_key,
    )
    for key in block_keys:
        trace[key] = tensors[key]
    if block_end >= 47:
        trace["trunk.outputs.single_trunk"] = _mx_np(trunk_out.single_trunk)
        trace["trunk.outputs.pair_trunk"] = _mx_np(trunk_out.pair_trunk)
    return trace


def _build_mlx_pre_pairformer_trace(
    model: _TraceTrunkModel,
    ref: dict[str, np.ndarray],
    structure: StructureInputs,
) -> OrderedDict[str, np.ndarray]:
    dtype = mx.float32 if model.cfg.compute_dtype == "float32" else mx.bfloat16
    emb = _reconstruct_trace_embeddings(ref, structure, dtype=dtype)
    trace: OrderedDict[str, np.ndarray] = OrderedDict()

    single = emb.single_initial + model.trunk_module.token_single_recycle_proj(emb.single_initial)
    pair = emb.pair_initial + model.trunk_module.token_pair_recycle_proj(emb.pair_initial)
    trace["trunk.recycle_0.pre_pairformer.single_after_recycle"] = _mx_np(single)
    trace["trunk.recycle_0.pre_pairformer.pair_after_recycle"] = _mx_np(pair)

    pair = model.trunk_module.template_embedder(
        pair,
        emb.template_input,
        template_input_masks=structure.template_input_masks,
        token_pair_mask=structure.token_pair_mask,
    )
    trace["trunk.recycle_0.pre_pairformer.pair_after_template"] = _mx_np(pair)

    msa = emb.msa_input
    if msa.shape[1] > 0:
        msa = msa + model.trunk_module.msa_module.linear_s2m(single)[:, None, :, :]
    trace["trunk.recycle_0.pre_pairformer.msa_input"] = _mx_np(msa)

    msa_module = model.trunk_module.msa_module
    for i in range(len(msa_module.outer_product_mean)):
        pair = pair + msa_module.outer_product_mean[i](msa, msa_mask=structure.msa_mask)
        if i < len(msa_module.msa_transition):
            msa = msa + msa_module.msa_transition[i](msa)
            msa = msa + msa_module.msa_pair_weighted_averaging[i](
                msa,
                pair,
                token_pair_mask=structure.token_pair_mask,
                msa_mask=structure.msa_mask,
            )
            trace[f"trunk.recycle_0.pre_pairformer.msa_iter_{i}"] = _mx_np(msa)
        pair_transition_out = msa_module.pair_transition[i](pair)
        pair = msa_module.triangular_multiplication[i](pair, pair_mask=structure.token_pair_mask) + pair_transition_out
        pair = msa_module.triangular_attention[i](pair, pair_mask=structure.token_pair_mask)
        if i < len(msa_module.msa_transition):
            trace[f"trunk.recycle_0.pre_pairformer.pair_iter_{i}"] = _mx_np(pair)
    return trace


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detailed trunk block trace via TorchScript graph probe")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"])
    parser.add_argument("--block-start", type=int, default=0)
    parser.add_argument("--block-end", type=int, default=47)
    parser.add_argument("--reference-trace-npz", type=Path, default=None)
    parser.add_argument("--write-reference-trace", type=Path, default=None)
    parser.add_argument("--skip-mlx", action="store_true")
    parser.add_argument(
        "--trace-scope",
        choices=("pairformer", "pre_pairformer"),
        default="pairformer",
        help="Compare pairformer block outputs or pre-pairformer trunk checkpoints",
    )
    parser.add_argument(
        "--mlx-device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Run the MLX side on GPU or CPU",
    )
    parser.add_argument(
        "--torch-device",
        default="mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--jump-threshold", type=float, default=1e-3)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.trace_scope == "pre_pairformer" and args.recycles != 1:
        raise ValueError("pre_pairformer trace scope currently requires --recycles 1")

    mx.set_default_device(mx.Device(getattr(mx.DeviceType, args.mlx_device), 0))
    print(f"Using MLX device: {mx.default_device()}")

    ref = _load_reference_subset(
        args.reference_npz,
        include_self_check=args.reference_trace_npz is None and args.block_end >= 47,
    )
    structure = _load_trace_structure(args.input_npz)

    if args.reference_trace_npz is not None:
        ref_trace = _load_trace_npz(args.reference_trace_npz)
        print(f"Loaded reference trace from {args.reference_trace_npz}")
    else:
        torch_device = torch.device(args.torch_device)
        print(f"Building TorchScript reference trace on {torch_device}...")
        _clear_accelerator_memory()
        ref_trace = _run_reference_trace(
            ref,
            structure,
            recycles=args.recycles,
            device=torch_device,
            block_start=args.block_start,
            block_end=args.block_end,
            trace_scope=args.trace_scope,
        )
        _reference_self_check(ref, ref_trace)
        if args.write_reference_trace is not None:
            _write_trace_npz(args.write_reference_trace, ref_trace)
            print(f"Wrote reference trace to {args.write_reference_trace}")
        _clear_accelerator_memory()

    if args.skip_mlx:
        return

    for dtype in args.dtypes:
        print("\n" + "=" * 80)
        print(f"MLX trunk trace dtype={dtype}")
        print("=" * 80)
        _clear_accelerator_memory()
        if args.trace_scope == "pre_pairformer":
            pairformer_block_limit = 0
        else:
            pairformer_block_limit = args.block_end + 1 if args.recycles == 1 and args.block_end < 47 else None
        model = _load_trunk_model(
            args.weights_dir,
            compute_dtype=dtype,
            pairformer_block_limit=pairformer_block_limit,
        )
        if args.trace_scope == "pre_pairformer":
            mlx_trace = _build_mlx_pre_pairformer_trace(model, ref, structure)
        else:
            mlx_trace = _build_mlx_trace(
                model,
                ref,
                structure,
                recycles=args.recycles,
                block_start=args.block_start,
                block_end=args.block_end,
            )
        _compare_trace(ref_trace, mlx_trace, structure, jump_threshold=args.jump_threshold)
        del mlx_trace
        del model
        _clear_accelerator_memory()


if __name__ == "__main__":
    main()
