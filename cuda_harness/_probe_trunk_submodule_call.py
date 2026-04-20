"""Modal probe: determine exactly how to call trunk.pt's scripted submodules.

The top-level ``trunk`` and its named children all have ``forward`` in their
``dir(...)`` listing, yet ``trunk.template_embedder(...)`` raises
``AttributeError: 'RecursiveScriptModule' object has no attribute 'forward'``
under ``_call_impl``. That is because ``__call__`` routes through the nn.Module
``_call_impl`` → ``self.forward`` path, and TorchScript's __getattr__ does not
synthesise a bound ``forward`` attribute for `RecursiveScriptModule` the same
way regular `nn.Module` does.

This probe enumerates **several alternative call paths** and reports which
ones succeed:

1. ``module.forward(...)``
2. ``getattr(module, "forward")(...)``
3. ``module._c.forward(...)``
4. ``module.graph_for(...)``
5. ``module(input)`` direct

All on ``trunk.token_single_recycle_proj`` (known to succeed in prior probes,
so this is a positive control) and on ``trunk.template_embedder``.
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
def trunk_submodule_call_probe() -> None:
    import numpy as np
    import torch

    torch.set_grad_enabled(False)
    trunk_path = MODELS_DIR / "models_v2" / "trunk.pt"
    print(f"Loading {trunk_path}")
    trunk = torch.jit.load(str(trunk_path), map_location="cuda:0")
    trunk.eval()

    # Tiny dummy inputs at the exact shapes each submodule expects.
    B, N = 1, 256
    token_single = torch.randn(B, N, 384, device="cuda:0")
    token_pair = torch.randn(B, N, N, 256, device="cuda:0")
    token_single_mask = torch.ones(B, N, dtype=torch.bool, device="cuda:0")
    token_pair_mask = torch.ones(B, N, N, dtype=torch.bool, device="cuda:0")
    msa_feats = torch.randn(B, 16384, N, 64, device="cuda:0")
    msa_mask = torch.ones(B, 16384, N, dtype=torch.bool, device="cuda:0")
    template_feats = torch.randn(B, 4, N, N, 64, device="cuda:0")
    template_mask = torch.ones(B, 4, N, N, dtype=torch.bool, device="cuda:0")
    crop = 256

    def _try(label: str, fn):
        print(f"\n>>> {label}")
        try:
            out = fn()
            if isinstance(out, tuple):
                for i, t in enumerate(out):
                    print(f"    result[{i}]: shape={tuple(t.shape) if isinstance(t, torch.Tensor) else '?'} dtype={t.dtype if isinstance(t, torch.Tensor) else type(t)}")
            elif isinstance(out, torch.Tensor):
                print(f"    result: shape={tuple(out.shape)} dtype={out.dtype} max_abs={out.abs().max().item():.4f}")
            else:
                print(f"    result type: {type(out).__name__}")
            return True
        except Exception as exc:
            print(f"    FAIL: {type(exc).__name__}: {str(exc)[:200]}")
            return False

    rp = trunk.token_single_recycle_proj
    te = trunk.template_embedder
    mm = trunk.msa_module
    ps = trunk.pairformer_stack

    # --- token_single_recycle_proj (Sequential[norm, linear]) ---
    print("\n================ token_single_recycle_proj ================")
    _try("rp(token_single)", lambda: rp(token_single))
    _try("rp.forward(token_single)", lambda: rp.forward(token_single))
    _try("getattr(rp,'forward')(token_single)", lambda: getattr(rp, "forward")(token_single))
    _try("rp._c.forward(token_single)", lambda: rp._c.forward(token_single))

    # --- template_embedder ---
    print("\n================ template_embedder ================")
    _try(
        "te(pair, template, mask_t, mask_p, crop)",
        lambda: te(token_pair, template_feats, template_mask, token_pair_mask, crop),
    )
    _try(
        "te.forward(pair, template, mask_t, mask_p, crop)",
        lambda: te.forward(token_pair, template_feats, template_mask, token_pair_mask, crop),
    )
    _try(
        "te.forward_256(pair, template, mask_t, mask_p)",
        lambda: te.forward_256(token_pair, template_feats, template_mask, token_pair_mask),
    )
    # List scripted method names on _c.
    try:
        meth_list = [m.name for m in te._c._get_method_names()] if hasattr(te._c, '_get_method_names') else []
        print(f"[template_embedder ._c methods]: {meth_list}")
    except Exception:
        pass
    try:
        # Iterate hasattr for forward_{CROP}
        for cs in (256, 384, 512, 768, 1024, 1536, 2048):
            print(f"  has te.forward_{cs}: {hasattr(te, f'forward_{cs}')}")
    except Exception:
        pass

    # --- msa_module ---
    print("\n================ msa_module ================")
    _try(
        "mm.forward_256(single, pair, msa, mask_p, mask_m)",
        lambda: mm.forward_256(token_single, token_pair, msa_feats, token_pair_mask, msa_mask),
    )
    for cs in (256, 384, 512, 768, 1024, 1536, 2048):
        print(f"  has mm.forward_{cs}: {hasattr(mm, f'forward_{cs}')}")

    # --- pairformer_stack ---
    print("\n================ pairformer_stack ================")
    _try(
        "ps.forward_256(single, pair, mask_p, mask_s)",
        lambda: ps.forward_256(token_single, token_pair, token_pair_mask, token_single_mask),
    )
    for cs in (256, 384, 512, 768, 1024, 1536, 2048):
        print(f"  has ps.forward_{cs}: {hasattr(ps, f'forward_{cs}')}")

    # --- pairformer_stack.blocks[0] ---
    print("\n================ pairformer_stack.blocks.0 ================")
    # Try several access patterns.
    block_0 = None
    for access_fn, label in [
        (lambda: getattr(ps.blocks, "0"), "getattr(ps.blocks,'0')"),
        (lambda: ps.blocks.__getitem__("0"), 'ps.blocks["0"]'),
        (lambda: ps.blocks.__getitem__(0),   "ps.blocks[0]"),
        (lambda: list(ps.blocks.children())[0], "list(ps.blocks.children())[0]"),
    ]:
        try:
            block_0 = access_fn()
            print(f"  ✓ {label} → {type(block_0).__name__}")
            break
        except Exception as exc:
            print(f"  ✗ {label}: {type(exc).__name__}: {str(exc)[:100]}")
    if block_0 is None:
        print("  could not access block_0 at all")
        return
    print(f"  block_0 type: {type(block_0).__name__}")
    # Check for forward_<crop>.
    for cs in (256, 384, 512, 768, 1024, 1536, 2048):
        print(f"  has block_0.forward_{cs}: {hasattr(block_0, f'forward_{cs}')}")
    _try(
        "block_0(single, pair, mask_p, mask_s)",
        lambda: block_0(token_single, token_pair, token_pair_mask, token_single_mask),
    )
    _try(
        "block_0.forward(single, pair, mask_p, mask_s)",
        lambda: block_0.forward(token_single, token_pair, token_pair_mask, token_single_mask),
    )
    _try(
        "block_0.forward_256(single, pair, mask_p, mask_s)",
        lambda: getattr(block_0, "forward_256")(token_single, token_pair, token_pair_mask, token_single_mask),
    )


@app.local_entrypoint()
def main() -> None:
    trunk_submodule_call_probe.remote()
