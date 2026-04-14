"""Detailed trunk block trace against TorchScript via graph-probed outputs.

This harness does not rely on TorchScript hooks. Instead it clones
``trunk.forward_<crop_size>``, registers the pairformer block outputs as
explicit graph outputs, and compares them against the MLX trunk trace when both
consume the same reference embedding tensors.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists():
    sys.path.insert(0, str(LOCAL_CHAI_LAB))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from chai_lab.chai1 import _component_moved_to  # type: ignore[import-not-found]

from chai_mlx import ChaiMLX
from chai_mlx.data.types import StructureInputs
from layer_parity import _npz_dict, capture_trunk, load_feature_context
from stage_isolation_parity import reconstruct_embedding_outputs

_SINGLE_NAME_RE = re.compile(r"^token_single_repr(?P<idx>\d+)?\.1$")
_PAIR_NAME_RE = re.compile(r"^z(?P<idx>\d+)\.1$")


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


def _build_probe_function(
    trunk_module,
    *,
    crop_size: int,
) -> tuple[torch._C.ScriptFunction, list[str], list[str]]:
    graph = trunk_module._c._get_method(f"forward_{crop_size}").graph.copy()
    orig_output = list(graph.outputs())[0]
    final_unpack = graph.create("prim::TupleUnpack", [orig_output], 2)
    graph.appendNode(final_unpack)
    final_single, final_pair = list(final_unpack.outputs())

    values: dict[str, torch._C.Value] = {}
    for node in graph.nodes():
        for output in node.outputs():
            values[output.debugName()] = output

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

    while len(list(graph.outputs())):
        graph.eraseOutput(0)

    ordered_names: list[str] = []
    for block_idx, (single_name, pair_name) in enumerate(zip(single_names, pair_names)):
        graph.registerOutput(values[single_name])
        graph.registerOutput(values[pair_name])
        ordered_names.extend(
            [
                f"pairformer.block_{block_idx}.single",
                f"pairformer.block_{block_idx}.pair",
            ]
        )
    graph.registerOutput(final_single)
    graph.registerOutput(final_pair)
    ordered_names.extend(
        [
            "pairformer.block_47.single",
            "pairformer.block_47.pair",
        ]
    )

    graph.makeMultiOutputIntoTuple()
    fn = torch._C._create_function_from_graph(f"trunk_probe_{crop_size}", graph)
    return fn, single_names, ordered_names


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


def _run_reference_trace(
    ref: dict[str, np.ndarray],
    structure: StructureInputs,
    *,
    recycles: int,
    device: torch.device,
) -> OrderedDict[str, np.ndarray]:
    crop_size = int(ref["embedding.outputs.single_initial"].shape[1])
    inputs = _reference_inputs(ref, structure, device=device)
    trace: OrderedDict[str, np.ndarray] = OrderedDict()

    with _component_moved_to("trunk.pt", device) as trunk:
        probe_fn, _, ordered_names = _build_probe_function(trunk.jit_module, crop_size=crop_size)

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
            if len(probe_out) != len(ordered_names):
                raise ValueError(f"expected {len(ordered_names)} probe outputs, got {len(probe_out)}")

            for name, value in zip(ordered_names, probe_out):
                trace[f"trunk.recycle_{recycle_idx}.{name}"] = _pt_np(value)
            single_prev = probe_out[-2]
            pair_prev = probe_out[-1]

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


def _build_mlx_trace(
    model: ChaiMLX,
    ref: dict[str, np.ndarray],
    structure: StructureInputs,
    *,
    recycles: int,
) -> OrderedDict[str, np.ndarray]:
    dtype = mx.float32 if model.cfg.compute_dtype == "float32" else mx.bfloat16
    emb = reconstruct_embedding_outputs(ref, structure, dtype=dtype)
    tensors: dict[str, np.ndarray] = {}
    trunk_out = capture_trunk(model, emb, recycles=recycles, tensors=tensors)

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
    trace["trunk.outputs.single_trunk"] = _mx_np(trunk_out.single_trunk)
    trace["trunk.outputs.pair_trunk"] = _mx_np(trunk_out.pair_trunk)
    return trace


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detailed trunk block trace via TorchScript graph probe")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--dtypes", nargs="+", default=["bfloat16", "float32"])
    parser.add_argument(
        "--torch-device",
        default="mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu",
    )
    parser.add_argument("--jump-threshold", type=float, default=1e-3)
    args = parser.parse_args(list(argv) if argv is not None else None)

    ctx, _ = load_feature_context(args.input_npz)
    ref = _npz_dict(args.reference_npz)
    structure = ctx.structure_inputs

    torch_device = torch.device(args.torch_device)
    print(f"Building TorchScript reference trace on {torch_device}...")
    ref_trace = _run_reference_trace(ref, structure, recycles=args.recycles, device=torch_device)
    _reference_self_check(ref, ref_trace)

    for dtype in args.dtypes:
        print("\n" + "=" * 80)
        print(f"MLX trunk trace dtype={dtype}")
        print("=" * 80)
        model = ChaiMLX.from_pretrained(args.weights_dir, strict=False, compute_dtype=dtype)
        mlx_trace = _build_mlx_trace(model, ref, structure, recycles=args.recycles)
        _compare_trace(ref_trace, mlx_trace, structure, jump_threshold=args.jump_threshold)


if __name__ == "__main__":
    main()
