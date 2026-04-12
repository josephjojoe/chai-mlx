from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import Chai1Config
from .layers.pairformer import PairformerBlock
from .types import ConfidenceOutputs, TrunkOutputs
from .utils import cdist, expand_plddt_to_atoms, one_hot_binned, representative_atom_coords


class ConfidenceHead(nn.Module):
    def __init__(self, cfg: Chai1Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.single_to_pair_proj = nn.Linear(cfg.hidden.token_single, 2 * cfg.hidden.token_pair, bias=False)
        self.atom_distance_bins_projection = nn.Linear(16, cfg.hidden.token_pair, bias=False)
        self.blocks = [
            PairformerBlock(
                pair_dim=cfg.hidden.token_pair,
                single_dim=cfg.hidden.token_single,
                single_heads=cfg.pairformer.single_heads,
                single_head_dim=cfg.pairformer.single_head_dim,
                triangle_heads=cfg.confidence.triangle_heads,
                triangle_head_dim=cfg.confidence.triangle_head_dim,
                use_fused_triangle_attention=True,
                eps=cfg.layer_norm_eps,
            )
            for _ in range(cfg.confidence.num_blocks)
        ]
        self.single_output_norm = nn.LayerNorm(cfg.hidden.token_single, eps=cfg.layer_norm_eps, affine=False)
        self.pair_output_norm = nn.LayerNorm(cfg.hidden.token_pair, eps=cfg.layer_norm_eps, affine=False)
        self.plddt_projection = nn.Linear(
            cfg.hidden.token_single,
            cfg.confidence.plddt_atom_positions * cfg.confidence.plddt_bins,
            bias=False,
        )
        self.pae_projection = nn.Linear(cfg.hidden.token_pair, cfg.confidence.pair_bins, bias=False)
        self.pde_projection = nn.Linear(cfg.hidden.token_pair, cfg.confidence.pair_bins, bias=False)

    def _run_single(self, trunk: TrunkOutputs, coords: mx.array) -> ConfidenceOutputs:
        structure = trunk.structure_inputs
        single_initial = trunk.single_initial
        single_trunk = trunk.single_trunk
        pair = trunk.pair_trunk

        row, col = mx.split(self.single_to_pair_proj(single_initial), 2, axis=-1)
        pair = pair + row[:, :, None, :] + col[:, None, :, :]

        ref_coords = representative_atom_coords(coords, structure.token_reference_atom_index)
        dists = cdist(ref_coords)
        dist_bins = one_hot_binned(dists, self.cfg.confidence.distance_bin_edges)
        pair = pair + self.atom_distance_bins_projection(dist_bins)

        token_mask = structure.token_exists_mask.astype(mx.float32)

        s, z = single_trunk, pair
        for block in self.blocks:
            z, s = block(z, s, pair_mask=structure.token_pair_mask, single_mask=token_mask)
        assert s is not None

        s = s * token_mask[..., None]

        s_normed = self.single_output_norm(s)
        z_normed = self.pair_output_norm(z)

        plddt_token = self.plddt_projection(s_normed)
        plddt_logits = expand_plddt_to_atoms(
            plddt_token,
            structure.atom_token_index,
            structure.atom_within_token_index,
            self.cfg.confidence.plddt_bins,
        )
        pae_logits = self.pae_projection(z_normed)
        z_sym = z_normed + z_normed.transpose(0, 2, 1, 3)
        pde_logits = self.pde_projection(z_sym)
        return ConfidenceOutputs(
            pae_logits=pae_logits,
            pde_logits=pde_logits,
            plddt_logits=plddt_logits,
            token_single=s,
            token_pair=z,
            structure_inputs=structure,
        )

    def __call__(self, trunk: TrunkOutputs, coords: mx.array) -> ConfidenceOutputs:
        if coords.ndim == 3:
            return self._run_single(trunk, coords)
        if coords.ndim != 4:
            raise ValueError(f"coords must have shape [b, a, 3] or [b, ds, a, 3], got {coords.shape}")

        outputs = [self._run_single(trunk, coords[:, i]) for i in range(coords.shape[1])]
        return ConfidenceOutputs(
            pae_logits=mx.stack([o.pae_logits for o in outputs], axis=1),
            pde_logits=mx.stack([o.pde_logits for o in outputs], axis=1),
            plddt_logits=mx.stack([o.plddt_logits for o in outputs], axis=1),
            token_single=mx.stack([o.token_single for o in outputs], axis=1),
            token_pair=mx.stack([o.token_pair for o in outputs], axis=1),
            structure_inputs=trunk.structure_inputs,
        )
