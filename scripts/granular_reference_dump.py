"""Generate per-block reference tensors from TorchScript for granular parity.

Extends ``chai_lab_reference_dump.py`` with per-pairformer-block trunk captures
and per-stage diffusion captures using forward hooks on JIT submodules.

Usage::

    python scripts/granular_reference_dump.py \
        --fasta path/to/input.fasta \
        --input-npz /tmp/chai_mlx_input.npz \
        --reference-npz /tmp/chai_mlx_granular_reference.npz

The input-npz is identical to what ``chai_lab_reference_dump.py`` produces,
so ``stage_isolation_parity.py`` and ``layer_parity.py`` both work with it.

The reference-npz adds keys:

- ``trunk.recycle_0.pairformer.block_{i}.single`` / ``.pair`` for i in 0..47
- ``denoise.encoder_tokens``, ``denoise.encoder_atom_repr``
- ``denoise.transformer.block_{i}.token_repr`` for i in 0..15
- ``denoise.pos_updates``

If JIT forward hooks fail (some TorchScript versions don't support them),
the script falls back to the same outputs as ``chai_lab_reference_dump.py``
and logs a warning.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists():
    sys.path.insert(0, str(LOCAL_CHAI_LAB))

from chai_lab.chai1 import (  # type: ignore[import-not-found]
    Collate,
    TokenBondRestraint,
    _component_moved_to,
    feature_factory,
    make_all_atom_feature_context,
)
from chai_lab.utils.tensor_utils import move_data_to_device, set_seed, und_self  # type: ignore[import-not-found]

from chai_mlx.data.featurize import _batch_to_feature_context, featurize_fasta
from chai_mlx.data.types import FeatureContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    return np.asarray(tensor)


def _record(store: dict[str, np.ndarray], key: str, value: torch.Tensor) -> None:
    store[key] = _to_numpy(value)


def _save_ctx_npz(path: Path, ctx: FeatureContext, coords: np.ndarray, sigma: np.ndarray) -> None:
    payload: dict[str, np.ndarray] = {
        "token_features": _to_numpy(ctx.token_features),
        "token_pair_features": _to_numpy(ctx.token_pair_features),
        "atom_features": _to_numpy(ctx.atom_features),
        "atom_pair_features": _to_numpy(ctx.atom_pair_features),
        "msa_features": _to_numpy(ctx.msa_features),
        "template_features": _to_numpy(ctx.template_features),
        "coords": coords.astype(np.float32),
        "sigma": sigma.astype(np.float32),
    }
    if ctx.bond_adjacency is not None:
        payload["bond_adjacency"] = _to_numpy(ctx.bond_adjacency)
    if ctx.raw_features is not None:
        for key, value in ctx.raw_features.items():
            payload[f"raw_features.{key}"] = _to_numpy(value)
    for field_name, value in vars(ctx.structure_inputs).items():
        if value is not None:
            payload[f"structure_inputs.{field_name}"] = _to_numpy(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


# ---------------------------------------------------------------------------
# Forward hook infrastructure
# ---------------------------------------------------------------------------

class HookCapture:
    """Register forward hooks on JIT submodules and capture outputs."""

    def __init__(self) -> None:
        self.captured: dict[str, np.ndarray] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(self, key: str):
        def hook(module, inp, output):
            if isinstance(output, tuple):
                for i, t in enumerate(output):
                    if isinstance(t, torch.Tensor):
                        self.captured[f"{key}.{i}"] = _to_numpy(t)
            elif isinstance(output, torch.Tensor):
                self.captured[key] = _to_numpy(output)
        return hook

    def register(self, jit_module: torch.jit.ScriptModule, pattern: str) -> int:
        """Hook submodules whose names match *pattern*.  Returns hook count."""
        regex = re.compile(pattern)
        count = 0
        try:
            for name, module in jit_module.named_modules():
                if regex.search(name):
                    h = module.register_forward_hook(self._make_hook(name))
                    self._hooks.append(h)
                    count += 1
        except Exception as exc:
            print(f"[WARN] Could not enumerate/hook JIT submodules: {exc}")
        return count

    def remove_all(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _discover_modules(jit: torch.jit.ScriptModule) -> list[str]:
    """List all named submodules in a JIT module."""
    names = []
    try:
        for name, _ in jit.named_modules():
            names.append(name)
    except Exception:
        pass
    return names


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _write_minimal_fasta(directory: Path) -> Path:
    fasta = directory / "test.fasta"
    fasta.write_text(">protein|name=test\nMKFLILFNILVSTLSFSSAQA\n")
    return fasta


@torch.no_grad()
def generate_granular_reference(
    *,
    fasta_path: Path,
    working_dir: Path,
    input_npz: Path,
    reference_npz: Path,
    num_trunk_recycles: int,
    sigma_value: float,
    seed: int,
    device: str,
) -> None:
    set_seed([seed])
    device = torch.device(device)

    print(f"[*] Building feature context on {device}")
    feature_context = make_all_atom_feature_context(
        fasta_file=fasta_path,
        output_dir=working_dir / "chai_lab",
        use_esm_embeddings=False,
        use_msa_server=False,
        use_templates_server=False,
        esm_device=device,
    )

    collator = Collate(
        feature_factory=feature_factory,
        num_key_atoms=128,
        num_query_atoms=32,
    )
    batch = collator([feature_context])
    features = {name: feature for name, feature in batch["features"].items()}
    inputs = batch["inputs"]
    block_indices_h = inputs["block_atom_pair_q_idces"]
    block_indices_w = inputs["block_atom_pair_kv_idces"]
    atom_single_mask = inputs["atom_exists_mask"]
    atom_token_indices = inputs["atom_token_index"].long()
    token_single_mask = inputs["token_exists_mask"]
    token_pair_mask = und_self(token_single_mask, "b i, b j -> b i j")
    token_reference_atom_index = inputs["token_ref_atom_index"]
    atom_within_token_index = inputs["atom_within_token_index"]
    msa_mask = inputs["msa_mask"]
    template_input_masks = und_self(
        inputs["template_mask"], "b t n1, b t n2 -> b t n1 n2"
    )
    block_atom_pair_mask = inputs["block_atom_pair_mask"]
    _, _, model_size = msa_mask.shape

    bond_ft = TokenBondRestraint().generate(batch=batch).data
    batch_ctx = _batch_to_feature_context(batch, bond_ft)

    public_ctx = featurize_fasta(
        fasta_path,
        output_dir=working_dir / "mlx",
        use_esm_embeddings=False,
        use_msa_server=False,
        use_templates_server=False,
    )

    tensors: dict[str, np.ndarray] = {}

    # ----- Feature embedding (same as basic dump) -----
    print("[*] Running feature embedding")
    with _component_moved_to("feature_embedding.pt", device) as feature_embedding:
        embedded_features = feature_embedding.forward(
            crop_size=model_size,
            move_to_device=device,
            return_on_cpu=False,
            **features,
        )

    token_single_input_feats = embedded_features["TOKEN"]
    token_pair_input_feats, token_pair_structure_input_feats = embedded_features["TOKEN_PAIR"].chunk(2, dim=-1)
    atom_single_input_feats, atom_single_structure_input_feats = embedded_features["ATOM"].chunk(2, dim=-1)
    block_atom_pair_input_feats, block_atom_pair_structure_input_feats = embedded_features["ATOM_PAIR"].chunk(2, dim=-1)
    template_input_feats = embedded_features["TEMPLATES"]
    msa_input_feats = embedded_features["MSA"]

    _record(tensors, "embedding.features.token_single", token_single_input_feats)
    _record(tensors, "embedding.features.token_pair_trunk", token_pair_input_feats)
    _record(tensors, "embedding.features.token_pair_structure", token_pair_structure_input_feats)
    _record(tensors, "embedding.features.atom_single_trunk", atom_single_input_feats)
    _record(tensors, "embedding.features.atom_single_structure", atom_single_structure_input_feats)
    _record(tensors, "embedding.features.atom_pair_trunk", block_atom_pair_input_feats)
    _record(tensors, "embedding.features.atom_pair_structure", block_atom_pair_structure_input_feats)
    _record(tensors, "embedding.features.templates", template_input_feats)
    _record(tensors, "embedding.features.msa", msa_input_feats)

    # ----- Bond projection -----
    print("[*] Running bond projection")
    with _component_moved_to("bond_loss_input_proj.pt", device) as bond_loss_input_proj:
        trunk_bond_feat, structure_bond_feat = bond_loss_input_proj.forward(
            return_on_cpu=False,
            move_to_device=device,
            crop_size=model_size,
            input=bond_ft,
        ).chunk(2, dim=-1)
    trunk_bond_feat = trunk_bond_feat.to(token_pair_input_feats.dtype)
    structure_bond_feat = structure_bond_feat.to(token_pair_structure_input_feats.dtype)
    token_pair_input_feats = token_pair_input_feats + trunk_bond_feat
    token_pair_structure_input_feats = token_pair_structure_input_feats + structure_bond_feat
    _record(tensors, "embedding.bond_trunk", trunk_bond_feat)
    _record(tensors, "embedding.bond_structure", structure_bond_feat)

    # ----- Token embedder -----
    print("[*] Running token embedder")
    with _component_moved_to("token_embedder.pt", device) as token_input_embedder:
        (
            token_single_initial_repr,
            token_single_structure_input,
            token_pair_initial_repr,
        ) = token_input_embedder.forward(
            return_on_cpu=False,
            move_to_device=device,
            token_single_input_feats=token_single_input_feats,
            token_pair_input_feats=token_pair_input_feats,
            atom_single_input_feats=atom_single_input_feats,
            block_atom_pair_feat=block_atom_pair_input_feats,
            block_atom_pair_mask=block_atom_pair_mask,
            block_indices_h=block_indices_h,
            block_indices_w=block_indices_w,
            atom_single_mask=atom_single_mask,
            atom_token_indices=atom_token_indices,
            crop_size=model_size,
        )

    _record(tensors, "embedding.outputs.token_single_input", token_single_input_feats)
    _record(tensors, "embedding.outputs.token_pair_input", token_pair_input_feats)
    _record(tensors, "embedding.outputs.token_pair_structure_input", token_pair_structure_input_feats)
    _record(tensors, "embedding.outputs.atom_single_input", atom_single_input_feats)
    _record(tensors, "embedding.outputs.atom_single_structure_input", atom_single_structure_input_feats)
    _record(tensors, "embedding.outputs.atom_pair_input", block_atom_pair_input_feats)
    _record(tensors, "embedding.outputs.atom_pair_structure_input", block_atom_pair_structure_input_feats)
    _record(tensors, "embedding.outputs.msa_input", msa_input_feats)
    _record(tensors, "embedding.outputs.template_input", template_input_feats)
    _record(tensors, "embedding.outputs.single_initial", token_single_initial_repr)
    _record(tensors, "embedding.outputs.single_structure", token_single_structure_input)
    _record(tensors, "embedding.outputs.pair_initial", token_pair_initial_repr)
    _record(tensors, "embedding.outputs.pair_structure", token_pair_structure_input_feats)

    # ----- Trunk with per-block hooks -----
    print(f"[*] Running trunk ({num_trunk_recycles} recycle) with per-block hooks")
    token_single_trunk_repr = token_single_initial_repr
    token_pair_trunk_repr = token_pair_initial_repr

    hook_capture = HookCapture()
    for recycle_idx in range(num_trunk_recycles):
        with _component_moved_to("trunk.pt", device) as trunk:
            jit = trunk.jit_module

            # Discover submodule structure
            all_names = _discover_modules(jit)
            pf_blocks = [n for n in all_names if re.match(r"pairformer_stack\.blocks\.\d+$", n)]
            if not pf_blocks:
                pf_blocks = [n for n in all_names if re.search(r"pairformer.*\d+$", n)]
            if pf_blocks:
                print(f"  Found {len(pf_blocks)} pairformer blocks: {pf_blocks[0]} ... {pf_blocks[-1]}")

            hook_capture.captured.clear()
            n_hooks = hook_capture.register(jit, r"pairformer_stack\.blocks\.\d+$")
            if n_hooks == 0:
                n_hooks = hook_capture.register(jit, r"pairformer.*blocks\.\d+$")
            if n_hooks == 0:
                print("  [WARN] No pairformer hooks registered; per-block capture unavailable")
                print(f"  Available submodules ({len(all_names)}): {all_names[:20]}")

            token_single_trunk_repr, token_pair_trunk_repr = trunk.forward(
                move_to_device=device,
                token_single_trunk_initial_repr=token_single_initial_repr,
                token_pair_trunk_initial_repr=token_pair_initial_repr,
                token_single_trunk_repr=token_single_trunk_repr,
                token_pair_trunk_repr=token_pair_trunk_repr,
                msa_input_feats=msa_input_feats,
                msa_mask=msa_mask,
                template_input_feats=template_input_feats,
                template_input_masks=template_input_masks,
                token_single_mask=token_single_mask,
                token_pair_mask=token_pair_mask,
                crop_size=model_size,
            )

            hook_capture.remove_all()

    # Save hook captures with standardised key names
    if hook_capture.captured:
        print(f"  Captured {len(hook_capture.captured)} hook tensors")
        for hook_key, arr in hook_capture.captured.items():
            match = re.search(r"blocks\.(\d+)", hook_key)
            if match:
                idx = int(match.group(1))
                if hook_key.endswith(".0"):
                    tensors[f"trunk.recycle_{num_trunk_recycles - 1}.pairformer.block_{idx}.pair"] = arr
                elif hook_key.endswith(".1"):
                    tensors[f"trunk.recycle_{num_trunk_recycles - 1}.pairformer.block_{idx}.single"] = arr
                else:
                    tensors[f"trunk.hook.{hook_key}"] = arr
            else:
                tensors[f"trunk.hook.{hook_key}"] = arr
    else:
        print("  [INFO] No hook tensors captured (JIT hooks may not be supported)")

    _record(tensors, "trunk.outputs.single_initial", token_single_initial_repr)
    _record(tensors, "trunk.outputs.single_trunk", token_single_trunk_repr)
    _record(tensors, "trunk.outputs.single_structure", token_single_structure_input)
    _record(tensors, "trunk.outputs.pair_initial", token_pair_initial_repr)
    _record(tensors, "trunk.outputs.pair_trunk", token_pair_trunk_repr)
    _record(tensors, "trunk.outputs.pair_structure", token_pair_structure_input_feats)
    _record(tensors, "trunk.outputs.atom_single_structure_input", atom_single_structure_input_feats)
    _record(tensors, "trunk.outputs.atom_pair_structure_input", block_atom_pair_structure_input_feats)
    _record(tensors, "trunk.outputs.msa_input", msa_input_feats)
    _record(tensors, "trunk.outputs.template_input", template_input_feats)

    # ----- Diffusion with per-stage capture -----
    print("[*] Running diffusion denoise with per-stage hooks")
    static_diffusion_inputs = dict(
        token_single_initial_repr=token_single_structure_input.float(),
        token_pair_initial_repr=token_pair_structure_input_feats.float(),
        token_single_trunk_repr=token_single_trunk_repr.float(),
        token_pair_trunk_repr=token_pair_trunk_repr.float(),
        atom_single_input_feats=atom_single_structure_input_feats.float(),
        atom_block_pair_input_feats=block_atom_pair_structure_input_feats.float(),
        atom_single_mask=atom_single_mask,
        atom_block_pair_mask=block_atom_pair_mask,
        token_single_mask=token_single_mask,
        block_indices_h=block_indices_h,
        block_indices_w=block_indices_w,
        atom_token_indices=atom_token_indices,
    )
    static_diffusion_inputs = move_data_to_device(static_diffusion_inputs, device=device)

    batch_size = 1
    num_atoms = atom_single_mask.shape[1]
    coords = torch.randn(batch_size, 1, num_atoms, 3, device=device)
    sigma = torch.full((batch_size, 1), float(sigma_value), device=device)

    diff_hooks = HookCapture()
    with _component_moved_to("diffusion_module.pt", device=device) as diffusion_module:
        jit = diffusion_module.jit_module
        all_names = _discover_modules(jit)

        n_enc = diff_hooks.register(jit, r"atom_attention_encoder$")
        n_dec = diff_hooks.register(jit, r"atom_attention_decoder$")
        n_blk = diff_hooks.register(jit, r"diffusion_transformer\.blocks\.\d+$")
        print(f"  Diffusion hooks: encoder={n_enc} decoder={n_dec} transformer_blocks={n_blk}")
        if n_enc + n_dec + n_blk == 0:
            diffusion_names = [n for n in all_names if "atom" in n or "transformer" in n or "diffusion" in n]
            print(f"  [WARN] No diffusion hooks; relevant submodules: {diffusion_names[:20]}")

        denoised_pos = diffusion_module.forward(
            atom_noised_coords=coords.float(),
            noise_sigma=sigma.float(),
            crop_size=model_size,
            **static_diffusion_inputs,
        )
        diff_hooks.remove_all()

    denoised_for_compare = denoised_pos[:, None] if denoised_pos.ndim == 3 else denoised_pos
    _record(tensors, "denoise.output", denoised_for_compare)

    if diff_hooks.captured:
        print(f"  Captured {len(diff_hooks.captured)} diffusion hook tensors")
        for hook_key, arr in diff_hooks.captured.items():
            if "encoder" in hook_key:
                if hook_key.endswith(".0"):
                    tensors["denoise.encoder_tokens"] = arr
                elif hook_key.endswith(".1"):
                    tensors["denoise.encoder_atom_repr"] = arr
            elif "decoder" in hook_key:
                tensors["denoise.pos_updates"] = arr
            else:
                match = re.search(r"blocks\.(\d+)", hook_key)
                if match:
                    idx = int(match.group(1))
                    tensors[f"denoise.transformer.block_{idx}.token_repr"] = arr
                else:
                    tensors[f"denoise.hook.{hook_key}"] = arr

    # ----- Confidence head -----
    print("[*] Running confidence head")
    with _component_moved_to("confidence_head.pt", device=device) as confidence_head:
        pae_logits, pde_logits, plddt_logits = confidence_head.forward(
            move_to_device=device,
            token_single_input_repr=token_single_initial_repr,
            token_single_trunk_repr=token_single_trunk_repr,
            token_pair_trunk_repr=token_pair_trunk_repr,
            token_single_mask=token_single_mask,
            atom_single_mask=atom_single_mask,
            atom_coords=denoised_for_compare[:, 0],
            token_reference_atom_index=token_reference_atom_index,
            atom_token_index=atom_token_indices,
            atom_within_token_index=atom_within_token_index,
            crop_size=model_size,
        )
    _record(tensors, "confidence.outputs.pae_logits", pae_logits)
    _record(tensors, "confidence.outputs.pde_logits", pde_logits)
    _record(tensors, "confidence.outputs.plddt_logits", plddt_logits)

    # ----- Save -----
    input_npz.parent.mkdir(parents=True, exist_ok=True)
    reference_npz.parent.mkdir(parents=True, exist_ok=True)
    _save_ctx_npz(
        input_npz,
        public_ctx,
        coords=_to_numpy(coords),
        sigma=_to_numpy(sigma),
    )
    np.savez_compressed(reference_npz, **tensors)
    print(f"[*] Reference NPZ has {len(tensors)} tensors")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate granular per-block reference tensors from TorchScript",
    )
    parser.add_argument("--fasta", type=Path, default=None)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--num-trunk-recycles", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    with tempfile.TemporaryDirectory(prefix="chai_mlx_granular_") as tmpdir:
        tmpdir = Path(tmpdir)
        fasta_path = args.fasta if args.fasta is not None else _write_minimal_fasta(tmpdir)
        generate_granular_reference(
            fasta_path=fasta_path,
            working_dir=tmpdir,
            input_npz=args.input_npz,
            reference_npz=args.reference_npz,
            num_trunk_recycles=args.num_trunk_recycles,
            sigma_value=args.sigma,
            seed=args.seed,
            device=args.device,
        )
        print(f"[*] Wrote input NPZ to {args.input_npz}")
        print(f"[*] Wrote reference NPZ to {args.reference_npz}")


if __name__ == "__main__":
    main()
