"""Generate per-block TorchScript reference for the diffusion transformer.

Since register_forward_hook doesn't work on ScriptModules, this script
manually decomposes the TorchScript diffusion module forward pass by
calling submodules directly.

Usage::

    python scripts/ts_block_reference.py \
        --input-npz /tmp/chai_mlx_input.npz \
        --reference-npz /tmp/chai_mlx_reference.npz \
        --output /tmp/chai_mlx_diffusion_blocks.npz \
        [--zero-input]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHAI_LAB = REPO_ROOT / "chai-lab"
if LOCAL_CHAI_LAB.exists():
    sys.path.insert(0, str(LOCAL_CHAI_LAB))

from chai_lab.chai1 import _component_moved_to  # type: ignore[import-not-found]


def to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().numpy()


@torch.no_grad()
def generate_block_reference(
    input_npz: Path,
    reference_npz: Path,
    output_path: Path,
    *,
    zero_input: bool = False,
    device: str = "mps",
) -> None:
    DEVICE = torch.device(device)
    inp = dict(np.load(input_npz))
    ref = dict(np.load(reference_npz))

    coords_np = inp["coords"]
    sigma_np = inp["sigma"]
    if zero_input:
        coords_np = np.zeros_like(coords_np)

    p = "trunk.outputs"
    static_inputs = dict(
        token_single_initial_repr=torch.from_numpy(ref.get(f"{p}.single_structure", ref.get("embedding.outputs.single_structure"))).float().to(DEVICE),
        token_pair_initial_repr=torch.from_numpy(ref.get(f"{p}.pair_structure", ref.get("embedding.outputs.pair_structure"))).float().to(DEVICE),
        token_single_trunk_repr=torch.from_numpy(ref[f"{p}.single_trunk"]).float().to(DEVICE),
        token_pair_trunk_repr=torch.from_numpy(ref[f"{p}.pair_trunk"]).float().to(DEVICE),
        atom_single_input_feats=torch.from_numpy(ref[f"{p}.atom_single_structure_input"]).float().to(DEVICE),
        atom_block_pair_input_feats=torch.from_numpy(ref[f"{p}.atom_pair_structure_input"]).float().to(DEVICE),
        atom_single_mask=torch.from_numpy(inp["structure_inputs.atom_exists_mask"]).bool().to(DEVICE),
        atom_block_pair_mask=torch.from_numpy(inp["structure_inputs.block_atom_pair_mask"]).bool().to(DEVICE),
        token_single_mask=torch.from_numpy(inp["structure_inputs.token_exists_mask"]).bool().to(DEVICE),
        block_indices_h=torch.from_numpy(inp["structure_inputs.atom_q_indices"]).long().squeeze(0).to(DEVICE),
        block_indices_w=torch.from_numpy(inp["structure_inputs.atom_kv_indices"]).long().squeeze(0).to(DEVICE),
        atom_token_indices=torch.from_numpy(inp["structure_inputs.atom_token_index"]).long().to(DEVICE),
    )

    coords_t = torch.from_numpy(coords_np).float().to(DEVICE)
    sigma_t = torch.from_numpy(sigma_np).float().to(DEVICE)
    crop_size = int(static_inputs["token_single_mask"].shape[-1])

    tensors: dict[str, np.ndarray] = {}

    with _component_moved_to("diffusion_module.pt", device=DEVICE) as diffusion_module:
        jit = diffusion_module.jit_module

        # Discover module structure
        print("Discovering JIT module structure...")
        for name, mod in jit.named_modules():
            if name:
                print(f"  {name}: {type(mod).__name__}")

        # Run full forward to get the output
        print(f"\nRunning full forward (zero_input={zero_input})...")
        output = diffusion_module.forward(
            atom_noised_coords=coords_t,
            noise_sigma=sigma_t,
            crop_size=crop_size,
            **static_inputs,
        )
        out_np = to_np(output)
        if out_np.ndim == 3:
            out_np = out_np[:, None, :, :]
        tensors["output"] = out_np
        print(f"  output shape: {out_np.shape}, rms: {np.sqrt(np.mean(out_np**2)):.4f}")

        # Try to access and call individual transformer blocks
        print("\nTrying to access transformer blocks...")
        try:
            dt = jit.diffusion_transformer
            blocks = dt.blocks
            n_blocks = len(list(blocks.children()))
            print(f"  Found {n_blocks} blocks in diffusion_transformer")

            # List block attributes
            block0 = blocks[0]
            print(f"  Block 0 type: {type(block0).__name__}")
            for bname, bmod in block0.named_modules():
                if bname:
                    print(f"    block0.{bname}: {type(bmod).__name__}")
        except Exception as e:
            print(f"  Failed to access blocks: {e}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **tensors)
    print(f"\nSaved {len(tensors)} tensors to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--reference-npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zero-input", action="store_true")
    args = parser.parse_args()
    generate_block_reference(args.input_npz, args.reference_npz, args.output, zero_input=args.zero_input)


if __name__ == "__main__":
    main()
