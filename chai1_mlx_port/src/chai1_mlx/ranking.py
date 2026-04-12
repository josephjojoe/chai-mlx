from __future__ import annotations

import mlx.core as mx

from .config import Chai1Config
from .types import ConfidenceOutputs, RankingOutputs, StructureInputs
from .utils import expectation_from_logits, gather_tokens_to_atoms, pairwise_distance


def _unique_int_values(x: mx.array) -> list[int]:
    return sorted(set(x.reshape(-1).tolist()))


class Ranker:
    def __init__(self, cfg: Chai1Config) -> None:
        self.cfg = cfg

    def _tm_from_pae(self, pae: mx.array) -> mx.array:
        n = pae.shape[-1]
        d0 = 1.24 * ((max(int(n), 19) - 15) ** (1.0 / 3.0)) - 1.8
        return 1.0 / (1.0 + (pae / d0) ** 2)

    def _ptm(self, pae: mx.array) -> mx.array:
        tm = self._tm_from_pae(pae)
        return mx.max(mx.sum(tm, axis=-1), axis=-1) / pae.shape[-1]

    def _iptm(self, pae: mx.array, chain_id: mx.array) -> mx.array:
        tm = self._tm_from_pae(pae)
        scores = []
        unique_chains = _unique_int_values(chain_id)
        for chain in unique_chains:
            row_mask = chain_id == chain
            col_mask = chain_id != chain
            if not bool(mx.any(row_mask).item()) or not bool(mx.any(col_mask).item()):
                continue
            sub = tm[:, row_mask][:, :, col_mask]
            denom = max(int(mx.sum(col_mask).item()), 1)
            scores.append(mx.max(mx.sum(sub, axis=-1), axis=-1) / denom)
        if not scores:
            return mx.zeros((pae.shape[0],), dtype=mx.float32)
        return mx.max(mx.stack(scores, axis=0), axis=0)

    def _inter_chain_clashes(self, coords: mx.array, structure: StructureInputs) -> mx.array:
        atom_mask = structure.atom_exists_mask.astype(mx.bool_)
        atom_chain = gather_tokens_to_atoms(structure.token_chain_id[:, :, None], structure.atom_token_index)[..., 0]
        atom_polymer = gather_tokens_to_atoms(
            structure.token_is_polymer[:, :, None], structure.atom_token_index
        )[..., 0].astype(mx.bool_)
        dists = pairwise_distance(coords)
        valid = atom_mask[:, :, None] & atom_mask[:, None, :]
        not_self = ~mx.eye(coords.shape[1], dtype=mx.bool_)[None, :, :]
        inter_chain = atom_chain[:, :, None] != atom_chain[:, None, :]
        polymer_pairs = atom_polymer[:, :, None] & atom_polymer[:, None, :]
        clash = valid & not_self & inter_chain & polymer_pairs & (dists < 1.1)

        batch_flags = []
        unique_chains = _unique_int_values(structure.token_chain_id)
        atom_chain_counts = {
            c: mx.sum((atom_chain == c) & atom_mask, axis=-1) for c in unique_chains
        }
        for bi in range(coords.shape[0]):
            has = False
            for i, c1 in enumerate(unique_chains):
                for c2 in unique_chains[i + 1 :]:
                    pair_mask = (atom_chain[bi, :, None] == c1) & (atom_chain[bi, None, :] == c2)
                    count = int(mx.sum(clash[bi] & pair_mask).item())
                    denom = max(min(int(atom_chain_counts[c1][bi].item()), int(atom_chain_counts[c2][bi].item())), 1)
                    ratio = count / denom
                    if count > 100 or ratio > 0.5:
                        has = True
                        break
                if has:
                    break
            batch_flags.append(has)
        return mx.array(batch_flags, dtype=mx.float32)

    def _rank_single(
        self,
        conf: ConfidenceOutputs,
        coords: mx.array,
        structure: StructureInputs,
    ) -> RankingOutputs:
        plddt = expectation_from_logits(conf.plddt_logits, max_value=1.0)
        pae = expectation_from_logits(conf.pae_logits, max_value=32.0)
        pde = expectation_from_logits(conf.pde_logits, max_value=32.0)
        ptm = self._ptm(pae)
        iptm = self._iptm(pae, structure.token_asym_id[0])
        clashes = self._inter_chain_clashes(coords, structure)
        score = 0.2 * ptm + 0.8 * iptm - 100.0 * clashes
        return RankingOutputs(
            plddt=plddt,
            pae=pae,
            pde=pde,
            ptm=ptm,
            iptm=iptm,
            has_inter_chain_clashes=clashes,
            aggregate_score=score,
        )

    def __call__(
        self,
        conf: ConfidenceOutputs,
        coords: mx.array,
        structure: StructureInputs,
    ) -> RankingOutputs:
        if coords.ndim == 3:
            return self._rank_single(conf, coords, structure)
        outputs = [
            self._rank_single(
                ConfidenceOutputs(
                    pae_logits=conf.pae_logits[:, i],
                    pde_logits=conf.pde_logits[:, i],
                    plddt_logits=conf.plddt_logits[:, i],
                ),
                coords[:, i],
                structure,
            )
            for i in range(coords.shape[1])
        ]
        return RankingOutputs(
            plddt=mx.stack([o.plddt for o in outputs], axis=1),
            pae=mx.stack([o.pae for o in outputs], axis=1),
            pde=mx.stack([o.pde for o in outputs], axis=1),
            ptm=mx.stack([o.ptm for o in outputs], axis=1),
            iptm=mx.stack([o.iptm for o in outputs], axis=1),
            has_inter_chain_clashes=mx.stack([o.has_inter_chain_clashes for o in outputs], axis=1),
            aggregate_score=mx.stack([o.aggregate_score for o in outputs], axis=1),
        )
