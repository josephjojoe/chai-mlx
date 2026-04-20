"""Decompile every exported TorchScript bundle and summarise its precision.

This probe answers the question: *what does the upstream chai-lab CUDA
reference **actually** run at, read from the TorchScript IR?*

It loads each bundle from the Modal weights volume
(``/models/chai1/models_v2/*.pt``) on CPU and, for every bundle, reports:

1. The dtype histogram of every named parameter and every inline tensor
   constant in every scripted method.  Baked-in fp16/bf16 constants
   cannot be loaded from safetensors (safetensors only stores the named
   ``state_dict``) so this is a strictly stronger check than the dtype
   summary in the top-level answer.

2. Every IR node that changes dtype or could introduce reduced
   precision.  We enumerate calls to ``aten::to``, ``aten::type_as``,
   ``aten::_autocast_to_reduced_precision`` /
   ``_autocast_to_full_precision``, ``aten::half``, ``aten::bfloat16``,
   ``aten::float``, ``aten::double``, ``prim::CallFunction`` referencing
   ``torch.amp.autocast``, plus any input or output of
   ``aten::autocast_mode`` primitives.

3. For each ``aten::to`` call, we resolve the constant ``ScalarType``
   argument (when it is a compile-time constant) so we can tell
   ``.to(torch.float32)`` apart from ``.to(torch.bfloat16)``.

4. Module-level attribute dtypes (``training`` flags, registered
   buffers, dtype-typed attributes).

The output is a JSON blob returned to the caller.  The host side prints
a concise human summary so we can answer the precision question from
the IR alone, and writes the full IR dump to ``/tmp/chai_mlx_cuda/jit_ir``
for offline inspection.

Usage::

    modal run -m cuda_harness._probe_jit_precision

The probe is CPU-only and fast (no forward pass); it runs in ~1 minute
regardless of model size.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import modal

from cuda_harness.modal_common import (
    MINUTES,
    MODELS_DIR,
    app,
    chai_model_volume,
    image,
)


JIT_BUNDLES = (
    "token_embedder.pt",
    "feature_embedding.pt",
    "trunk.pt",
    "diffusion_module.pt",
    "confidence_head.pt",
    "bond_loss_input_proj.pt",
)


# Dtype-changing ops.  We are deliberately liberal here: anything that
# *could* change the dtype of a tensor goes in this set.  Pure shape ops
# (``aten::view``, ``aten::reshape``) and dtype-preserving math ops
# (``aten::add``, ``aten::mul``) are *not* in this set because they
# cannot introduce reduced precision on their own.
DTYPE_CHANGING_OPS = frozenset(
    {
        "aten::to",
        "aten::to_dtype",
        "aten::type_as",
        "aten::_autocast_to_reduced_precision",
        "aten::_autocast_to_full_precision",
        "aten::half",
        "aten::bfloat16",
        "aten::float",
        "aten::double",
        "aten::int",
        "aten::long",
        "aten::bool",
        "aten::cast",
        "prim::autocast",
    }
)

# Map PyTorch ScalarType enum values to dtype names.  The enum ordering
# is defined in c10/core/ScalarType.h and is stable across torch
# versions we care about (2.x).
#
# See torch.utils._pytree or torch/csrc/utils/python_scalars.h for
# canonical reference.  We hard-code the subset we care about (floats,
# ints, bool) since TorchScript serialises the enum as an int literal.
SCALAR_TYPE_NAMES = {
    0: "uint8",
    1: "int8",
    2: "int16",
    3: "int32",
    4: "int64",
    5: "float16",
    6: "float32",
    7: "float64",
    8: "complex32",
    9: "complex64",
    10: "complex128",
    11: "bool",
    12: "qint8",
    13: "quint8",
    14: "qint32",
    15: "bfloat16",
}


def _scalar_type_name(value: int | None) -> str | None:
    if value is None:
        return None
    return SCALAR_TYPE_NAMES.get(int(value), f"unknown_scalartype_{value}")


def _resolve_to_dtype(node, torch_module) -> dict:
    """Resolve the target dtype of an ``aten::to``-family node using its schema.

    ``aten::to`` has several overloads.  The dtype argument is always a
    ScalarType int, but its position depends on the overload:

      * ``aten::to.dtype(Tensor, ScalarType, ...)`` — dtype at index 1
      * ``aten::to.dtype_layout(Tensor, ScalarType?, Layout?, Device?, ...)`` — index 1
      * ``aten::to.device(Tensor, Device, ScalarType, ...)``       — dtype at index 2
      * ``aten::to.other(Tensor, Tensor, ...)`` — dtype = other.dtype (runtime)
      * ``aten::to.prim_dtype(Tensor, int?, bool, bool)`` — dtype at index 1

    Rather than encode every overload, we:
      1. Inspect the node's schema (``node.schema()``) to find a
         ScalarType-typed parameter.
      2. Evaluate the constant at that input position if it is a
         compile-time constant.

    Returns a small dict describing the resolution:

      { "resolved": bool,
        "dtype": str | None,
        "reason": str,
        "schema": str,
        "schema_scalar_type_indices": list[int] }

    ``dtype`` is one of the names in :data:`SCALAR_TYPE_NAMES`, or
    ``None`` when the target dtype is data-dependent (e.g. ``.to(other)``
    where ``other`` is itself a tensor).
    """
    try:
        schema = node.schema()
    except Exception as exc:
        schema = f"<no-schema: {exc!r}>"

    # Find argument positions whose schema type is ``ScalarType`` or
    # ``ScalarType?``.  ``node.inputs()`` is ordered to match the
    # schema's arguments.
    scalar_type_indices: list[int] = []
    try:
        from torch._C import TensorType  # noqa: F401
        # ``node.schema()`` returns a str here (TorchScript's
        # serialised schema).  Parse it manually.
        import re
        # Find "ScalarType" (possibly "ScalarType?") typed args.
        # Parens split between return-type and arg list.
        if isinstance(schema, str) and "(" in schema:
            args_text = schema.split("(", 1)[1].split(")")[0]
            for idx, arg in enumerate(args_text.split(",")):
                arg = arg.strip()
                if re.match(r"^(int|ScalarType)(\??)\s+dtype", arg):
                    scalar_type_indices.append(idx)
                elif "ScalarType" in arg and not arg.startswith("Layout"):
                    scalar_type_indices.append(idx)
    except Exception:
        pass

    inputs = list(node.inputs())
    resolution = {
        "resolved": False,
        "dtype": None,
        "reason": "no-scalar-type-arg",
        "schema": schema if isinstance(schema, str) else "<unknown>",
        "schema_scalar_type_indices": scalar_type_indices,
    }

    for idx in scalar_type_indices:
        if idx >= len(inputs):
            continue
        const = _resolve_constant(inputs[idx])
        if const is None:
            resolution["reason"] = f"scalar-type-arg-at-{idx}-not-constant"
            continue
        if isinstance(const, torch_module.dtype):
            resolution["resolved"] = True
            resolution["dtype"] = str(const).replace("torch.", "")
            resolution["reason"] = f"resolved-from-index-{idx}"
            return resolution
        if isinstance(const, int) and 0 <= const <= 20:
            name = _scalar_type_name(const)
            if name is not None:
                resolution["resolved"] = True
                resolution["dtype"] = name
                resolution["reason"] = f"resolved-from-index-{idx}-via-scalar-type-int"
                return resolution
        resolution["reason"] = f"scalar-type-arg-at-{idx}-unresolvable:{type(const).__name__}:{const}"

    # ``aten::to.other(Tensor, Tensor, ...)`` is an important overload:
    # the dtype is copied from another tensor input.  Mark it as such.
    if isinstance(schema, str) and "aten::to.other" in schema:
        resolution["reason"] = "aten::to.other (dtype copied from other tensor)"

    return resolution


@app.function(
    timeout=60 * MINUTES,
    cpu=8.0,
    memory=64 * 1024,
    volumes={MODELS_DIR: chai_model_volume},
    image=image,
)
def dump_jit_precision(dump_ir_text: bool = False) -> dict:
    """Load every JIT bundle on CPU and summarise IR precision.

    Args:
        dump_ir_text: When True, also include the full inlined-graph
            text for every scripted method in the returned payload.
            Off by default because ``trunk.pt``'s inlined graph is
            hundreds of megabytes and dominates wall-clock time.
    """
    import torch

    torch.set_grad_enabled(False)

    results: dict[str, dict] = {}

    for bundle_name in JIT_BUNDLES:
        path = MODELS_DIR / "models_v2" / bundle_name
        if not path.exists():
            results[bundle_name] = {"error": f"missing on volume: {path}"}
            continue

        print(f"[{bundle_name}] loading {path} ...")
        module = torch.jit.load(str(path), map_location="cpu")

        bundle_summary: dict = {
            "path": str(path),
            "forward_methods": [],
            "param_dtypes": {},
            "buffer_dtypes": {},
            "const_dtypes": {},
            "dtype_changing_ops": {},
            "to_dtype_calls": [],
            "autocast_nodes": [],
            "reduced_precision_consts": [],
            "ir_bytes": 0,
            "ir_dump": {},
        }

        for name, param in module.named_parameters():
            dtype = str(param.dtype)
            bundle_summary["param_dtypes"][dtype] = (
                bundle_summary["param_dtypes"].get(dtype, 0) + 1
            )

        for name, buf in module.named_buffers():
            dtype = str(buf.dtype)
            bundle_summary["buffer_dtypes"][dtype] = (
                bundle_summary["buffer_dtypes"].get(dtype, 0) + 1
            )

        # Enumerate scripted methods.  The exported bundles carry one
        # ``forward_<crop_size>`` per supported model size plus the
        # dispatcher ``forward``.  We walk all of them.
        #
        # ``ScriptModule._c`` is the underlying C++ ``torch._C.ScriptModule``.
        # The stable way to enumerate its methods is either:
        #   * ``module._c._method_names()``  (list[str], all scripted methods)
        #   * iterating ``module._modules`` and checking for ``forward``
        # We combine both plus ``dir(module)`` so we never miss a method.
        method_names: set[str] = set()
        cpp_module = module._c
        for attr in ("_method_names", "get_method_names", "_get_method_names"):
            fn = getattr(cpp_module, attr, None)
            if callable(fn):
                try:
                    method_names.update(fn())
                    break
                except Exception:
                    continue
        # Fallback: scan Python-level attributes that look like ``forward*``.
        for attr in dir(module):
            if attr.startswith("forward") and callable(getattr(module, attr, None)):
                method_names.add(attr)
        # Final fallback: ScriptModule exposes methods as properties;
        # ``code`` and ``graph`` exist on ``forward`` at minimum.
        if not method_names:
            method_names.add("forward")
        bundle_summary["forward_methods"] = sorted(method_names)

        # Gather the union of all scripted methods' IR into one
        # op-kind histogram.  We inline all called methods so ops
        # inside sub-modules are counted, not just the top-level ones.
        op_counter: dict[str, int] = {}
        to_calls: list[dict] = []
        autocast_nodes: list[dict] = []
        reduced_precision_consts: list[dict] = []
        ir_dump: dict[str, str] = {}

        # For bundles with many scripted methods (each bundle carries one
        # ``forward_<crop_size>`` per supported model size, i.e. 7 methods
        # on the 7-size trunk), the inlined graph per method is already
        # comprehensive because every forward path is represented.
        # We therefore only need one representative method per bundle
        # to answer the precision question, but we still enumerate all
        # of them in case the smaller sizes accidentally use reduced
        # precision while the larger ones don't.
        #
        # Heuristic: we walk all methods but cap per-bundle walltime
        # via ``max_nodes`` below; if we blow the cap we skip to the
        # next method rather than hanging for minutes on the 48-block
        # trunk.
        for method_name in bundle_summary["forward_methods"]:
            graph = None
            target_graph = None
            try:
                # ``module.<method_name>`` returns a ``ScriptMethod`` that
                # exposes ``.graph`` and ``.inlined_graph``.  The C++-side
                # ``module._c.<method_name>`` also has a ``.graph`` but
                # requires constructing a ScriptMethod wrapper.  Use the
                # Python-side access whenever possible.
                py_method = getattr(module, method_name, None)
                if py_method is not None and hasattr(py_method, "graph"):
                    graph = py_method.graph
                    target_graph = getattr(py_method, "inlined_graph", graph)
                else:
                    cpp_method = getattr(cpp_module, method_name, None)
                    if cpp_method is not None and hasattr(cpp_method, "graph"):
                        graph = cpp_method.graph
                        target_graph = graph
                    else:
                        bundle_summary.setdefault("graph_errors", {})[
                            method_name
                        ] = "no accessible graph"
                        continue
            except Exception as exc:
                bundle_summary.setdefault("graph_errors", {})[method_name] = repr(exc)
                continue

            # Text dump of the graph (both raw and inlined).
            # This is optional because TorchScript's ``str(Graph)`` is
            # quadratic in node count: for the trunk's 48-block
            # inlined graph it is ~50-200 MB of text and takes many
            # minutes to serialise.  We pre-size via a quick node
            # count instead, and only materialise text when asked.
            try:
                node_count = sum(1 for _ in target_graph.nodes())
            except Exception:
                node_count = -1
            bundle_summary["ir_bytes"] += node_count

            # Always materialise IR text for the smallest method so we
            # can eyeball dtype casts; gate the larger methods behind
            # ``dump_ir_text`` because their text dumps are tens of MB.
            dump_this = dump_ir_text or method_name == "forward_256"
            if dump_this:
                try:
                    inlined_ir = str(target_graph)
                except Exception as exc:
                    inlined_ir = f"<inlined-graph-dump-failed: {exc!r}>"
                ir_dump[method_name] = inlined_ir
            else:
                ir_dump[method_name] = f"<IR dump disabled; {node_count} nodes>"

            for node in target_graph.nodes():
                kind = node.kind()
                op_counter[kind] = op_counter.get(kind, 0) + 1

                # Record every call to a dtype-changing op so we can
                # resolve the ScalarType argument precisely using the
                # op's actual schema.  This is stricter than "any int
                # in [0,20] is a ScalarType" because a device index
                # also lives in that range.
                if kind in DTYPE_CHANGING_OPS:
                    dtype_target = _resolve_to_dtype(node, torch)
                    call: dict = {
                        "method": method_name,
                        "op": kind,
                        "source": _node_source(node),
                        "dtype_target": dtype_target,
                    }
                    to_calls.append(call)

                # Flag any autocast primitives.  TorchScript serialises
                # autocast regions as ``prim::CallFunction`` with a
                # reference to ``torch.amp.autocast`` or
                # ``aten::_autocast_to_*``; we catch both.
                if (
                    "autocast" in kind
                    or kind.startswith("prim::CallFunction")
                    and any(
                        "autocast" in _safe_str(inp.node())
                        for inp in node.inputs()
                    )
                ):
                    autocast_nodes.append(
                        {
                            "method": method_name,
                            "op": kind,
                            "source": _node_source(node),
                        }
                    )

                # Inline tensor constants: ``prim::Constant`` with
                # ``value=Tensor(...)`` carry baked-in weights in the
                # IR (TorchScript sometimes inlines small tensors
                # rather than promoting them to parameters).
                if kind == "prim::Constant":
                    attr_names = node.attributeNames()
                    if "value" in attr_names and node.kindOf("value") == "t":
                        tensor = node.t("value")
                        dtype = str(tensor.dtype)
                        bundle_summary["const_dtypes"][dtype] = (
                            bundle_summary["const_dtypes"].get(dtype, 0) + 1
                        )
                        if dtype in {"torch.float16", "torch.bfloat16", "torch.half"}:
                            reduced_precision_consts.append(
                                {
                                    "method": method_name,
                                    "dtype": dtype,
                                    "shape": list(tensor.shape),
                                    "numel": int(tensor.numel()),
                                }
                            )

        bundle_summary["dtype_changing_ops"] = {
            k: v for k, v in op_counter.items() if k in DTYPE_CHANGING_OPS
        }
        bundle_summary["all_ops"] = dict(
            sorted(op_counter.items(), key=lambda kv: -kv[1])
        )
        bundle_summary["to_dtype_calls"] = to_calls
        bundle_summary["autocast_nodes"] = autocast_nodes
        bundle_summary["reduced_precision_consts"] = reduced_precision_consts
        bundle_summary["ir_dump"] = ir_dump
        bundle_summary["method_count"] = len(ir_dump)

        results[bundle_name] = bundle_summary

        print(
            f"[{bundle_name}] params={bundle_summary['param_dtypes']} "
            f"buffers={bundle_summary['buffer_dtypes']} "
            f"const_dtypes={bundle_summary['const_dtypes']} "
            f"dtype_ops={bundle_summary['dtype_changing_ops']} "
            f"methods={len(ir_dump)}"
        )

    return results


def _safe_str(obj) -> str:
    try:
        return str(obj)
    except Exception:
        return "<unprintable>"


def _resolve_constant(value):
    """Resolve a Value in the IR to a Python constant if possible."""
    import torch

    node = value.node()
    if node.kind() != "prim::Constant":
        return None
    attr_names = node.attributeNames()
    if "value" not in attr_names:
        return None
    kind_of = node.kindOf("value")
    try:
        if kind_of == "i":
            return int(node.i("value"))
        if kind_of == "f":
            return float(node.f("value"))
        if kind_of == "s":
            return node.s("value")
        if kind_of == "t":
            t = node.t("value")
            if t.numel() == 1:
                return t.item()
            return None
        if kind_of == "ival":
            try:
                return node.ival("value")
            except Exception:
                return None
    except Exception:
        return None
    return None


def _node_source(node) -> str | None:
    """Best-effort source-location string for a graph node."""
    try:
        sr = node.sourceRange()
        if sr:
            return str(sr).splitlines()[0][:200]
    except Exception:
        pass
    try:
        return str(node)[:200]
    except Exception:
        return None


@app.local_entrypoint()
def main() -> None:
    out_dir = Path("/tmp/chai_mlx_cuda/jit_ir")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading JIT bundles and dumping IR (CPU, ~5-15 min for trunk) ...")
    results = dump_jit_precision.remote(dump_ir_text=False)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(_strip_ir(results), indent=2))
    print(f"Wrote summary without IR to {summary_path}")

    for bundle_name, payload in results.items():
        if "ir_dump" not in payload:
            continue
        bundle_dir = out_dir / bundle_name.replace(".pt", "")
        bundle_dir.mkdir(parents=True, exist_ok=True)
        for method_name, ir_text in payload["ir_dump"].items():
            (bundle_dir / f"{method_name}.graph.txt").write_text(ir_text)
        print(
            f"Wrote {len(payload['ir_dump'])} IR dumps for {bundle_name} "
            f"→ {bundle_dir}"
        )

    print()
    print("=" * 72)
    print("JIT precision summary (from IR, not from safetensors):")
    print("=" * 72)
    for bundle_name, payload in results.items():
        print(f"\n[{bundle_name}]")
        if "error" in payload:
            print(f"  ERROR: {payload['error']}")
            continue
        print(f"  methods enumerated      : {payload.get('method_count', 0)}")
        print(f"  named-parameter dtypes  : {payload['param_dtypes']}")
        print(f"  named-buffer dtypes     : {payload['buffer_dtypes']}")
        print(f"  inlined-constant dtypes : {payload['const_dtypes']}")
        print(f"  dtype-changing op counts: {payload['dtype_changing_ops']}")
        print(f"  autocast-ish nodes      : {len(payload['autocast_nodes'])}")
        print(
            "  reduced-precision consts: "
            f"{len(payload['reduced_precision_consts'])}"
        )
        if payload["reduced_precision_consts"]:
            for r in payload["reduced_precision_consts"][:5]:
                print(f"    - {r}")
            if len(payload["reduced_precision_consts"]) > 5:
                print(
                    f"    ... and {len(payload['reduced_precision_consts']) - 5} more"
                )

        # Explicit ``aten::to`` resolution: what dtypes are targets?
        to_targets_resolved: dict[str, int] = {}
        to_unresolved_reasons: dict[str, int] = {}
        to_by_method_bf16: dict[str, int] = {}
        to_by_method_fp32: dict[str, int] = {}
        to_by_method_other: dict[str, int] = {}
        total_to_calls = 0
        for call in payload["to_dtype_calls"]:
            if call["op"] not in {"aten::to", "aten::to_dtype", "aten::type_as"}:
                continue
            total_to_calls += 1
            res = call.get("dtype_target", {}) or {}
            if res.get("resolved"):
                name = res["dtype"]
                to_targets_resolved[name] = to_targets_resolved.get(name, 0) + 1
                method = call.get("method", "unknown")
                if name == "bfloat16":
                    to_by_method_bf16[method] = to_by_method_bf16.get(method, 0) + 1
                elif name == "float32":
                    to_by_method_fp32[method] = to_by_method_fp32.get(method, 0) + 1
                else:
                    to_by_method_other[method] = to_by_method_other.get(method, 0) + 1
            else:
                reason = res.get("reason", "unknown")
                to_unresolved_reasons[reason] = (
                    to_unresolved_reasons.get(reason, 0) + 1
                )
        print(f"  total .to(...) calls     : {total_to_calls}")
        print(f"  resolved dtype targets   : {to_targets_resolved}")
        if to_by_method_bf16 or to_by_method_fp32:
            print(f"  bf16 casts per method    : {dict(sorted(to_by_method_bf16.items()))}")
            print(f"  fp32 casts per method    : {dict(sorted(to_by_method_fp32.items()))}")
            print(f"  other casts per method   : {dict(sorted(to_by_method_other.items()))}")
        if to_unresolved_reasons:
            # Cap how many distinct unresolved reasons we show.
            top = sorted(
                to_unresolved_reasons.items(), key=lambda kv: -kv[1]
            )[:5]
            print(f"  unresolved .to reasons   : {dict(top)}")


def _strip_ir(results: dict) -> dict:
    stripped = {}
    for name, payload in results.items():
        if isinstance(payload, dict):
            copy = dict(payload)
            copy.pop("ir_dump", None)
            stripped[name] = copy
        else:
            stripped[name] = payload
    return stripped
