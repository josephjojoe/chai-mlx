"""Modal probe: inspect which methods are actually callable on the scripted
``trunk.pt`` submodules.

Goal: we need to call ``pairformer_stack``, ``msa_module``, ``template_embedder``,
``pairformer_stack.blocks[0]`` on CUDA from the loaded ``trunk.pt``, but they
don't expose a bare ``forward`` attribute. This probe walks the script bundle
and prints every scripted method, every callable attribute, and the graph
``.code`` for a few named submodules so we can see what interface to invoke.
"""

from __future__ import annotations

from cuda_harness.modal_common import (
    MINUTES,
    MODELS_DIR,
    app,
    chai_model_volume,
    image,
)


@app.function(
    timeout=10 * MINUTES,
    gpu="H100",
    volumes={MODELS_DIR: chai_model_volume},
    image=image,
)
def trunk_methods_probe() -> dict:
    import torch

    torch.set_grad_enabled(False)
    trunk_path = MODELS_DIR / "models_v2" / "trunk.pt"
    print(f"Loading {trunk_path}")
    trunk = torch.jit.load(str(trunk_path), map_location="cpu")
    trunk.eval()

    paths = [
        "",
        "token_single_recycle_proj",
        "token_pair_recycle_proj",
        "template_embedder",
        "msa_module",
        "pairformer_stack",
        "pairformer_stack.blocks",
        "pairformer_stack.blocks.0",
        "pairformer_stack.blocks.0.transition_pair",
        "pairformer_stack.blocks.0.triangle_multiplication",
        "pairformer_stack.blocks.0.triangle_attention",
        "pairformer_stack.blocks.0.attention_pair_bias",
    ]

    report = {}
    for dotted in paths:
        mod = trunk
        if dotted:
            try:
                for part in dotted.split("."):
                    mod = getattr(mod, part) if not part.isdigit() else mod[int(part)]
            except Exception as exc:
                report[dotted or "<root>"] = {"error": f"could not traverse: {exc!r}"}
                continue

        entry: dict = {"type": type(mod).__name__}

        # Scripted method names via the C++ side.
        try:
            methods = [m.name for m in mod._c._get_methods()]
            entry["scripted_methods"] = methods
        except Exception:
            entry["scripted_methods"] = None

        # Attribute directory (non-dunder).
        entry["dir_filtered"] = [a for a in dir(mod) if not a.startswith("_")][:40]

        # Named children.
        try:
            entry["named_children"] = [n for n, _ in mod.named_children()]
        except Exception:
            entry["named_children"] = None

        # If this module has a `.code`, dump its first 600 chars.
        try:
            code = getattr(mod, "code", None)
            if isinstance(code, str):
                entry["code_preview"] = code[:600]
        except Exception:
            pass

        # Check if callable.
        entry["is_callable"] = callable(mod)

        # Enumerate scripted methods callable on the container.
        if hasattr(mod, "__call__"):
            # Signature of __call__ if we can get it.
            try:
                entry["call_schema"] = str(mod._c.qualified_name)
            except Exception:
                pass

        print(f"\n--- {dotted or '<root>'} ---")
        for k, v in entry.items():
            if k == "code_preview":
                print(f"  code_preview (first 600 chars):")
                print("    " + (v or "").replace("\n", "\n    "))
            else:
                print(f"  {k}: {v}")

        report[dotted or "<root>"] = entry

    return report


@app.local_entrypoint()
def main() -> None:
    trunk_methods_probe.remote()
