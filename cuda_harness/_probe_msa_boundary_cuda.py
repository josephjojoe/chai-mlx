"""CUDA-side boundary probe: capture ``(s, z)`` after ``msa_module`` but
before ``pairformer_stack`` in chai-lab's scripted trunk.

Companion to ``cuda_harness/_probe_recycle_mlx.py``. This probe helps
localise trunk drift by capturing the scripted CUDA boundaries around
``msa_module`` and ``pairformer_stack`` on the same input. On recycle 0
the template path is a literal no-op for the standard monomer setup and
the previous-state projections start from zero, so the post-MSA boundary
is a useful place to compare the two implementations directly. The
*direct* test
— diff the MLX post-MSA pair tensor against the CUDA post-MSA pair
tensor — requires extracting CUDA's intermediate state out of
``trunk.pt``, which is a single TorchScript bundle.

This probe does exactly that. We load ``trunk.pt`` on an H100,
run recycle 0 on the exact same 1L2Y batch that
``cuda_harness/run_intermediates.py`` produced (we in fact reuse its
NPZ), then call the bundle's exposed submodules step-by-step so we
can capture ``(s, z)`` at every boundary:

  (a) ``post_proj``         — after ``token_{single,pair}_recycle_proj``
  (b) ``post_template``     — after ``template_embedder``
  (c) ``post_msa``          — after ``msa_module``
  (d) ``post_pairformer``   — after ``pairformer_stack`` (= full trunk output)

We write these into an NPZ matching the MLX side's layout
(``post_proj_{single,pair}``, ``post_template_pair``, ``post_msa_pair``,
``post_pairformer_{single,pair}``) so the two files can be diffed
directly with a single numpy script.

Extra diagnostic
----------------

Before running the boundary capture, we enumerate ``trunk.pt``'s named
submodule tree and print a ``summary`` JSON. This confirms the scripted
bundle actually exposes ``msa_module`` / ``pairformer_stack`` as
callable submodules (rather than having them inlined at export time). If
they're not exposed as expected we fall back to a forward-hook-based
capture on whatever modules we can find.

Usage
-----

::

    # 1. produce the MLX tensors
    python3 cuda_harness/_probe_recycle_mlx.py \\
        --npz /tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz \\
        --weights-dir weights

    # 2. run this probe on Modal
    modal run -m cuda_harness._probe_msa_boundary_cuda

    # 3. diff post-MSA tensors locally
    python3 cuda_harness/_probe_msa_boundary_compare.py  # (created alongside)

Writes
------

* ``/tmp/chai_mlx_cuda/msa_boundary_probe/cuda_recycle_0_fp32.npz``
* ``/tmp/chai_mlx_cuda/msa_boundary_probe/trunk_submodule_tree.json``
"""
from __future__ import annotations

import io
import json
from pathlib import Path

from cuda_harness.modal_common import (
    MINUTES,
    MODELS_DIR,
    app,
    chai_model_volume,
    image,
)


OUT_DIR = Path("/tmp/chai_mlx_cuda/msa_boundary_probe")
OUT_DIR.mkdir(parents=True, exist_ok=True)


@app.function(
    timeout=20 * MINUTES,
    gpu="H100",
    volumes={MODELS_DIR: chai_model_volume},
    image=image,
)
def cuda_msa_boundary_probe(intermediates_npz_bytes: bytes) -> dict[str, bytes]:
    """Load ``trunk.pt``, run recycle 0, capture per-stage ``(s, z)``.

    Returns a dict with keys:
        ``cuda_recycle_0_fp32.npz``  — tensors in the MLX-side layout.
        ``trunk_submodule_tree.json`` — named-submodule dump of trunk.pt.
    """
    import numpy as np
    import torch

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")

    intermediates = np.load(io.BytesIO(intermediates_npz_bytes), allow_pickle=False)

    def _get(name: str) -> np.ndarray:
        return intermediates[name]

    token_single_initial = torch.from_numpy(_get("embedding.token_single_initial")).to(device, dtype=torch.float32)
    token_pair_initial = torch.from_numpy(_get("embedding.token_pair_initial")).to(device, dtype=torch.float32)
    msa_input_feats = torch.from_numpy(_get("embedding.msa")).to(device, dtype=torch.float32)
    template_input_feats = torch.from_numpy(_get("embedding.templates")).to(device, dtype=torch.float32)

    token_single_mask = torch.from_numpy(_get("inputs.batch.token_exists_mask")).to(device)
    msa_mask = torch.from_numpy(_get("inputs.batch.msa_mask")).to(device)
    template_mask = torch.from_numpy(_get("inputs.batch.template_mask")).to(device)

    # token_pair_mask = und_self(token_exists_mask, "b i, b j -> b i j"); template_input_masks = und_self(template_mask, ...)
    token_pair_mask = token_single_mask[..., :, None] & token_single_mask[..., None, :]
    template_input_masks = template_mask[..., :, None] & template_mask[..., None, :]

    _, _, model_size = msa_mask.shape
    crop_size = int(model_size)

    print(f"[probe] tensors loaded  n_tokens_mask={int(token_single_mask.sum().item())}  model_size={crop_size}")
    print(f"        token_single_initial  {tuple(token_single_initial.shape)}  {token_single_initial.dtype}")
    print(f"        token_pair_initial    {tuple(token_pair_initial.shape)}")
    print(f"        msa_input_feats       {tuple(msa_input_feats.shape)}")
    print(f"        template_input_feats  {tuple(template_input_feats.shape)}")

    trunk_path = MODELS_DIR / "models_v2" / "trunk.pt"
    print(f"[probe] loading {trunk_path}")
    trunk = torch.jit.load(str(trunk_path), map_location=device)
    trunk.eval()

    tree_summary: dict[str, object] = {}

    def _scripted_methods(module) -> list[str]:
        out = []
        try:
            for meth in module._c._get_methods():
                out.append(meth.name)
        except Exception:
            pass
        return out

    def _walk(module, prefix: str, depth: int, out: list[dict]) -> None:
        kind = type(module).__name__
        try:
            named_children = list(module.named_children())
        except Exception:
            named_children = []
        out.append({
            "path": prefix or "<root>",
            "class": kind,
            "children": [n for n, _ in named_children],
            "scripted_methods": _scripted_methods(module),
            "has_forward": hasattr(module, "forward"),
            "depth": depth,
        })
        if depth >= 2:
            return
        for name, child in named_children:
            _walk(child, f"{prefix}.{name}" if prefix else name, depth + 1, out)

    tree_nodes: list[dict] = []
    _walk(trunk, "", 0, tree_nodes)
    tree_summary["trunk_tree"] = tree_nodes

    top_children = {name: type(child).__name__ for name, child in trunk.named_children()}
    print(f"[probe] trunk top-level children: {top_children}")
    for name in ("template_embedder", "msa_module", "pairformer_stack", "token_single_recycle_proj", "token_pair_recycle_proj"):
        sub = getattr(trunk, name, None)
        if sub is not None:
            methods = _scripted_methods(sub)
            print(f"        {name:30s} methods={methods}")

    expected = {"token_single_recycle_proj", "token_pair_recycle_proj", "template_embedder", "msa_module", "pairformer_stack"}
    found = set(top_children.keys()) & expected
    missing = expected - found
    tree_summary["expected_children"] = sorted(expected)
    tree_summary["found_children"] = sorted(found)
    tree_summary["missing_children"] = sorted(missing)

    if missing:
        print(f"[probe] WARNING: expected submodules missing from trunk.pt: {missing}")
        print(f"        top_children actually present: {sorted(top_children.keys())}")

    dump: dict[str, np.ndarray] = {}

    if not missing:
        print("[probe] all five trunk submodules exposed; running boundary capture")

        trunk = trunk.to(device)

        prev_single = token_single_initial
        prev_pair = token_pair_initial

        single = token_single_initial + trunk.token_single_recycle_proj(prev_single)
        pair = token_pair_initial + trunk.token_pair_recycle_proj(prev_pair)
        torch.cuda.synchronize()
        dump["post_proj_single"] = single.detach().cpu().float().numpy()
        dump["post_proj_pair"] = pair.detach().cpu().float().numpy()
        print(f"  post_proj      single {tuple(single.shape)}  max_abs={single.abs().max().item():.3f}")
        print(f"                 pair   {tuple(pair.shape)}  max_abs={pair.abs().max().item():.3f}")

        try:
            pair = trunk.template_embedder(
                pair,
                template_input_feats,
                template_input_masks,
                token_pair_mask,
                crop_size,
            )
        except Exception as e_positional:
            try:
                pair = trunk.template_embedder(
                    pair=pair,
                    template_input_feats=template_input_feats,
                    template_input_masks=template_input_masks,
                    token_pair_mask=token_pair_mask,
                    crop_size=crop_size,
                )
            except Exception as e_keyword:
                print(f"[probe] template_embedder positional call failed: {e_positional}")
                print(f"[probe] template_embedder keyword call failed: {e_keyword}")
                raise
        torch.cuda.synchronize()
        dump["post_template_pair"] = pair.detach().cpu().float().numpy()
        print(f"  post_template  pair   {tuple(pair.shape)}  max_abs={pair.abs().max().item():.3f}")

        try:
            pair = trunk.msa_module(
                single,
                pair,
                msa_input_feats,
                token_pair_mask,
                msa_mask,
                crop_size,
            )
        except Exception as e_positional:
            try:
                pair = trunk.msa_module(
                    token_single_trunk_repr=single,
                    token_pair_trunk_repr=pair,
                    msa_input_feats=msa_input_feats,
                    token_pair_mask=token_pair_mask,
                    msa_mask=msa_mask,
                    crop_size=crop_size,
                )
            except Exception as e_keyword:
                print(f"[probe] msa_module positional call failed: {e_positional}")
                print(f"[probe] msa_module keyword call failed: {e_keyword}")
                raise
        torch.cuda.synchronize()
        dump["post_msa_pair"] = pair.detach().cpu().float().numpy()
        print(f"  post_msa       pair   {tuple(pair.shape)}  max_abs={pair.abs().max().item():.3f}")

        try:
            single_out, pair_out = trunk.pairformer_stack(
                single,
                pair,
                token_pair_mask,
                token_single_mask,
                crop_size,
            )
        except Exception as e_positional:
            try:
                single_out, pair_out = trunk.pairformer_stack(
                    token_single_trunk_repr=single,
                    token_pair_trunk_repr=pair,
                    token_pair_mask=token_pair_mask,
                    token_single_mask=token_single_mask,
                    crop_size=crop_size,
                )
            except Exception as e_keyword:
                print(f"[probe] pairformer_stack positional call failed: {e_positional}")
                print(f"[probe] pairformer_stack keyword call failed: {e_keyword}")
                raise
        torch.cuda.synchronize()
        dump["post_pairformer_single"] = single_out.detach().cpu().float().numpy()
        dump["post_pairformer_pair"] = pair_out.detach().cpu().float().numpy()
        print(f"  post_pairfrmr  single {tuple(single_out.shape)}  max_abs={single_out.abs().max().item():.3f}")
        print(f"                 pair   {tuple(pair_out.shape)}  max_abs={pair_out.abs().max().item():.3f}")

    else:
        print("[probe] submodules not exposed; skipping boundary capture — fall back needed")

    npz_buf = io.BytesIO()
    np.savez_compressed(npz_buf, **dump)
    tree_json_bytes = json.dumps(tree_summary, indent=2).encode()

    return {
        "cuda_recycle_0_fp32.npz": npz_buf.getvalue(),
        "trunk_submodule_tree.json": tree_json_bytes,
    }


@app.local_entrypoint()
def probe_msa_boundary(
    intermediates_npz: str = "/tmp/chai_mlx_cuda/intermediates/1L2Y/seed_42.npz",
) -> None:
    src = Path(intermediates_npz)
    if not src.is_file():
        raise FileNotFoundError(f"Intermediates NPZ not found: {src}")
    print(f"[probe] uploading intermediates NPZ ({src.stat().st_size / (1 << 20):.1f} MB)")

    result = cuda_msa_boundary_probe.remote(src.read_bytes())

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, blob in result.items():
        dst = out_dir / name
        dst.write_bytes(blob)
        print(f"[probe] wrote {len(blob) / (1 << 20):.2f} MB -> {dst}")

    tree_path = out_dir / "trunk_submodule_tree.json"
    if tree_path.is_file():
        summary = json.loads(tree_path.read_text())
        print(f"[probe] found submodules: {summary.get('found_children')}")
        if summary.get("missing_children"):
            print(f"[probe] MISSING: {summary['missing_children']}")
        print(f"[probe] top-level nodes:")
        for node in summary.get("trunk_tree", []):
            if node.get("depth", 99) <= 1:
                print(f"          {node['path']:50s}  {node['class']}  children={node['children']}")
