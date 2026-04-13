"""Convert TorchScript weights, load them, and run a small MLX smoke pass.

Usage::

    python scripts/weight_loading_e2e.py \
        --torchscript-dir /path/to/pt_files \
        --out-dir /tmp/chai-mlx-weights

This script is intentionally artifact-backed and does not run under pytest.
It validates the real path we care about in production:

1. convert TorchScript shards to safetensors,
2. load them through ``ChaiMLX.from_pretrained(..., strict=True)``,
3. run a small end-to-end forward smoke pass.
"""

from __future__ import annotations

import argparse
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import mlx.core as mx

from chai_mlx import ChaiMLX, featurize
from chai_mlx.io.weights.convert_torchscript import convert_torchscript_dir_to_safetensors


def build_dummy_inputs(
    *,
    batch_size: int = 1,
    n_tokens: int = 32,
    msa_depth: int = 4,
    n_templates: int = 2,
):
    n_atoms = 23 * n_tokens
    num_blocks = n_atoms // 32
    token_exists = mx.ones((batch_size, n_tokens), dtype=mx.bool_)
    atom_exists = mx.ones((batch_size, n_atoms), dtype=mx.bool_)
    atom_token_index = mx.arange(n_atoms)[None, :] // 23
    atom_within_token_index = mx.arange(n_atoms)[None, :] % 37
    token_ref_atom_index = mx.arange(n_tokens)[None, :] * 23
    q_idx = mx.arange(n_atoms).reshape(1, num_blocks, 32)
    kv_idx = mx.clip(
        q_idx[:, :, :1] + mx.arange(128)[None, None, :] - 48,
        0,
        n_atoms - 1,
    )
    block_mask = mx.ones((batch_size, num_blocks, 32, 128), dtype=mx.bool_)

    return {
        "token_features": mx.random.normal((batch_size, n_tokens, 2638)),
        "token_pair_features": mx.random.normal((batch_size, n_tokens, n_tokens, 163)),
        "atom_features": mx.random.normal((batch_size, n_atoms, 395)),
        "atom_pair_features": mx.random.normal((batch_size, num_blocks, 32, 128, 14)),
        "msa_features": mx.random.normal((batch_size, msa_depth, n_tokens, 42)),
        "template_features": mx.random.normal(
            (batch_size, n_templates, n_tokens, n_tokens, 76)
        ),
        "bond_adjacency": mx.zeros((batch_size, n_tokens, n_tokens, 1)),
        "structure_inputs": {
            "atom_exists_mask": atom_exists,
            "token_exists_mask": token_exists,
            "token_pair_mask": token_exists[:, :, None] & token_exists[:, None, :],
            "atom_token_index": atom_token_index,
            "atom_within_token_index": atom_within_token_index,
            "token_reference_atom_index": token_ref_atom_index,
            "token_asym_id": mx.zeros((batch_size, n_tokens), dtype=mx.int32),
            "token_entity_id": mx.zeros((batch_size, n_tokens), dtype=mx.int32),
            "token_chain_id": mx.zeros((batch_size, n_tokens), dtype=mx.int32),
            "token_is_polymer": mx.ones((batch_size, n_tokens), dtype=mx.bool_),
            "bond_adjacency": mx.zeros((batch_size, n_tokens, n_tokens, 1)),
            "atom_q_indices": q_idx,
            "atom_kv_indices": kv_idx,
            "block_atom_pair_mask": block_mask,
            "msa_mask": mx.ones((batch_size, msa_depth, n_tokens), dtype=mx.float32),
            "template_input_masks": mx.ones(
                (batch_size, n_templates, n_tokens, n_tokens), dtype=mx.float32
            ),
        },
    }


def run_weight_loading_e2e(
    *,
    torchscript_dir: Path,
    out_dir: Path,
    recycles: int,
    num_steps: int,
    allow_unmapped: bool,
    skip_forward: bool,
) -> None:
    print(f"[*] Converting TorchScript weights from {torchscript_dir}")
    convert_torchscript_dir_to_safetensors(
        torchscript_dir,
        out_dir,
        allow_unmapped=allow_unmapped,
    )

    print(f"[*] Loading converted weights from {out_dir}")
    model = ChaiMLX.from_pretrained(out_dir, strict=True)

    if skip_forward:
        print("[PASS] strict load completed")
        return

    print("[*] Running end-to-end smoke pass")
    ctx = featurize(build_dummy_inputs())
    emb = model.embed_inputs(ctx)
    trunk_out = model.trunk(emb, recycles=recycles)
    cache = model.prepare_diffusion_cache(trunk_out)
    coords = model.init_noise(
        batch_size=emb.token_single_input.shape[0],
        num_samples=1,
        structure=emb.structure_inputs,
    )
    sigma_curr, sigma_next, gamma = next(model.schedule(num_steps=num_steps))
    coords = model.diffusion_step(cache, coords, sigma_curr, sigma_next, gamma)
    confidence = model.confidence(trunk_out, coords)
    ranking = model.rank_outputs(confidence, coords, emb.structure_inputs)
    mx.eval(
        coords,
        confidence.pae_logits,
        confidence.pde_logits,
        confidence.plddt_logits,
        ranking.aggregate_score,
    )

    print("[PASS] convert -> load -> diffuse -> confidence -> ranking")
    print(f"       coords: {coords.shape}")
    print(f"       pae: {confidence.pae_logits.shape}")
    print(f"       score: {ranking.aggregate_score.shape}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert TorchScript weights, load them, and run an MLX smoke pass",
    )
    parser.add_argument("--torchscript-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for converted safetensors (defaults to a temporary dir)",
    )
    parser.add_argument(
        "--allow-unmapped",
        action="store_true",
        help="Allow partial conversion instead of failing on unmapped weight keys",
    )
    parser.add_argument("--recycles", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Only validate convert + strict load, skip the smoke forward pass",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ctx = nullcontext(args.out_dir)
    else:
        ctx = tempfile.TemporaryDirectory(prefix="chai_mlx_weights_")

    with ctx as tmp:
        out_dir = args.out_dir if args.out_dir is not None else Path(tmp)
        run_weight_loading_e2e(
            torchscript_dir=args.torchscript_dir,
            out_dir=out_dir,
            recycles=args.recycles,
            num_steps=args.num_steps,
            allow_unmapped=args.allow_unmapped,
            skip_forward=args.skip_forward,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
