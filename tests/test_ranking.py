"""Parity tests for :class:`chai_mlx.model.ranking.Ranker` vs chai-lab."""

from __future__ import annotations

import importlib.util

import mlx.core as mx
import numpy as np
import pytest

from chai_mlx.config import ChaiConfig
from chai_mlx.data.types import ConfidenceOutputs, StructureInputs
from chai_mlx.model.ranking import Ranker

_HAS_REF = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("chai_lab") is not None
)

pytestmark = pytest.mark.skipif(
    not _HAS_REF,
    reason="chai-lab + torch are required for ranker parity tests",
)


# Tolerances: pTM / ipTM are fp32 softmax + fp32 weighted sums, so agreement
# should be near eps-level; keep a small buffer for platform differences.
_ATOL = 1e-5
_RTOL = 1e-5


def _make_structure(
    asym: np.ndarray,
    entity_type: np.ndarray,
    residue_index: np.ndarray,
    bb_mask: np.ndarray,
    centre_idx: np.ndarray,
    atom_token: np.ndarray,
    atom_exists: np.ndarray,
    tok_exists: np.ndarray,
    is_polymer_by_token: np.ndarray,
) -> StructureInputs:
    B, N = tok_exists.shape
    return StructureInputs(
        atom_exists_mask=mx.array(atom_exists.astype(np.float32)),
        token_exists_mask=mx.array(tok_exists.astype(np.float32)),
        token_pair_mask=mx.array(np.ones((B, N, N), dtype=np.float32)),
        atom_token_index=mx.array(atom_token),
        atom_within_token_index=mx.array(np.zeros_like(atom_token)),
        token_reference_atom_index=mx.array(centre_idx),
        token_asym_id=mx.array(asym),
        token_entity_id=mx.array(asym),
        token_chain_id=mx.array(asym),
        token_is_polymer=mx.array(is_polymer_by_token.astype(np.float32)),
        token_centre_atom_index=mx.array(centre_idx),
        token_residue_index=mx.array(residue_index),
        token_entity_type=mx.array(entity_type),
        token_backbone_frame_mask=mx.array(bb_mask),
        token_backbone_frame_index=mx.array(np.zeros((B, N, 3), dtype=np.int64)),
    )


def _run_reference(
    coords: np.ndarray,
    atom_exists: np.ndarray,
    atom_token: np.ndarray,
    tok_exists: np.ndarray,
    asym: np.ndarray,
    entity_type: np.ndarray,
    residue_index: np.ndarray,
    bb_mask: np.ndarray,
    centre_idx: np.ndarray,
    pae_logits: np.ndarray,
    pde_logits: np.ndarray,
    plddt_logits: np.ndarray,
):
    import torch

    from chai_lab.ranking.frames import get_frames_and_mask
    from chai_lab.ranking.rank import rank

    tc = torch.tensor
    B = coords.shape[0]
    _, vfm = get_frames_and_mask(
        tc(coords),
        tc(asym),
        tc(residue_index),
        tc(bb_mask),
        tc(centre_idx),
        tc(tok_exists),
        tc(atom_exists),
        tc(np.zeros((B, asym.shape[1], 3), dtype=np.int64)),
        tc(atom_token),
    )
    pae_bins = torch.linspace(0.0, 32.0, 2 * pae_logits.shape[-1] + 1)[1::2]
    plddt_bins = torch.linspace(0.0, 1.0, 2 * plddt_logits.shape[-1] + 1)[1::2]
    return rank(
        atom_coords=tc(coords),
        atom_mask=tc(atom_exists),
        atom_token_index=tc(atom_token),
        token_exists_mask=tc(tok_exists),
        token_asym_id=tc(asym),
        token_entity_type=tc(entity_type),
        token_valid_frames_mask=vfm,
        lddt_logits=tc(plddt_logits),
        lddt_bin_centers=plddt_bins,
        pae_logits=tc(pae_logits),
        pae_bin_centers=pae_bins,
    ), vfm.numpy()


def _two_chain_scenario(seed: int, *, with_clash: bool = False):
    rng = np.random.default_rng(seed)
    B, N, A, BIN = 1, 12, 24, 64
    PLDDT_BINS = 50
    asym = np.array([[1] * 6 + [2] * 6], dtype=np.int64)
    entity_type = np.array([[0] * 6 + [0] * 6], dtype=np.int64)
    residue_index = np.array([[1, 2, 3, 4, 5, 6] * 2], dtype=np.int64)
    bb_mask = np.ones((B, N), dtype=bool)
    centre_idx = (np.arange(N, dtype=np.int64) * 2)[None, :]
    atom_token = np.repeat(np.arange(N, dtype=np.int64)[None, :], 2, axis=-2).reshape(
        1, -1
    )[:, :A]
    atom_exists = np.ones((B, A), dtype=bool)
    tok_exists = np.ones((B, N), dtype=bool)

    coords = (rng.standard_normal((B, A, 3)).astype(np.float32) * 5.0)
    if with_clash:
        coords[0, 0] = [0.0, 0.0, 0.0]
        coords[0, 12] = [0.3, 0.0, 0.0]
        coords[0, 13] = [0.5, 0.1, 0.0]

    pae_logits = rng.standard_normal((B, N, N, BIN)).astype(np.float32)
    pde_logits = rng.standard_normal((B, N, N, BIN)).astype(np.float32)
    plddt_logits = rng.standard_normal((B, A, PLDDT_BINS)).astype(np.float32)
    is_polymer = np.ones((B, N), dtype=bool)
    return dict(
        coords=coords,
        atom_exists=atom_exists,
        atom_token=atom_token,
        tok_exists=tok_exists,
        asym=asym,
        entity_type=entity_type,
        residue_index=residue_index,
        bb_mask=bb_mask,
        centre_idx=centre_idx,
        pae_logits=pae_logits,
        pde_logits=pde_logits,
        plddt_logits=plddt_logits,
        is_polymer=is_polymer,
    )


def _assert_close(mlx_val, ref_val, name: str):
    mlx_np = np.array(mlx_val)
    ref_np = np.asarray(ref_val)
    if mlx_np.shape != ref_np.shape:
        raise AssertionError(
            f"{name}: shape mismatch mlx={mlx_np.shape} ref={ref_np.shape}"
        )
    np.testing.assert_allclose(mlx_np, ref_np, atol=_ATOL, rtol=_RTOL, err_msg=name)


@pytest.mark.parametrize("seed,with_clash", [(0, False), (1, True), (7, False)])
def test_ranker_parity_two_chain(seed: int, with_clash: bool) -> None:
    data = _two_chain_scenario(seed, with_clash=with_clash)

    structure = _make_structure(
        asym=data["asym"],
        entity_type=data["entity_type"],
        residue_index=data["residue_index"],
        bb_mask=data["bb_mask"],
        centre_idx=data["centre_idx"],
        atom_token=data["atom_token"],
        atom_exists=data["atom_exists"],
        tok_exists=data["tok_exists"],
        is_polymer_by_token=data["is_polymer"],
    )
    conf = ConfidenceOutputs(
        pae_logits=mx.array(data["pae_logits"]),
        pde_logits=mx.array(data["pde_logits"]),
        plddt_logits=mx.array(data["plddt_logits"]),
    )
    ranker = Ranker(ChaiConfig())
    out = ranker(conf, mx.array(data["coords"]), structure)

    ref, _ = _run_reference(
        coords=data["coords"],
        atom_exists=data["atom_exists"],
        atom_token=data["atom_token"],
        tok_exists=data["tok_exists"],
        asym=data["asym"],
        entity_type=data["entity_type"],
        residue_index=data["residue_index"],
        bb_mask=data["bb_mask"],
        centre_idx=data["centre_idx"],
        pae_logits=data["pae_logits"],
        pde_logits=data["pde_logits"],
        plddt_logits=data["plddt_logits"],
    )

    _assert_close(out.ptm, ref.ptm_scores.complex_ptm.numpy(), "ptm")
    _assert_close(out.iptm, ref.ptm_scores.interface_ptm.numpy(), "iptm")
    _assert_close(
        out.per_chain_ptm, ref.ptm_scores.per_chain_ptm.numpy(), "per_chain_ptm"
    )
    _assert_close(
        out.per_chain_pair_iptm,
        ref.ptm_scores.per_chain_pair_iptm.numpy(),
        "per_chain_pair_iptm",
    )
    _assert_close(
        out.complex_plddt, ref.plddt_scores.complex_plddt.numpy(), "complex_plddt"
    )
    _assert_close(
        out.per_chain_plddt, ref.plddt_scores.per_chain_plddt.numpy(), "per_chain_plddt"
    )
    _assert_close(
        out.per_atom_plddt, ref.plddt_scores.per_atom_plddt.numpy(), "per_atom_plddt"
    )
    _assert_close(
        out.chain_chain_clashes,
        ref.clash_scores.chain_chain_clashes.numpy(),
        "chain_chain_clashes",
    )
    _assert_close(
        out.total_clashes, ref.clash_scores.total_clashes.numpy(), "total_clashes"
    )
    _assert_close(
        out.total_inter_chain_clashes,
        ref.clash_scores.total_inter_chain_clashes.numpy(),
        "total_inter_chain_clashes",
    )
    _assert_close(
        out.has_inter_chain_clashes.astype(mx.bool_),
        ref.clash_scores.has_inter_chain_clashes.numpy(),
        "has_inter_chain_clashes",
    )
    _assert_close(out.aggregate_score, ref.aggregate_score.numpy(), "aggregate_score")


def test_ranker_non_polymer_chain_no_interface_clash() -> None:
    """Chain B is a ligand (non-polymer): clashes count but
    ``has_inter_chain_clashes`` must be False because the policy requires
    BOTH chains to be polymers."""
    rng = np.random.default_rng(123)
    B, N, A, BIN = 1, 8, 16, 64
    PLDDT_BINS = 50
    asym = np.array([[1] * 4 + [2] * 4], dtype=np.int64)
    entity_type = np.array([[0] * 4 + [3] * 4], dtype=np.int64)  # second chain ligand
    residue_index = np.array([[1, 1, 2, 2, 1, 1, 2, 2]], dtype=np.int64)
    bb_mask = np.array([[True] * 4 + [False] * 4])
    centre_idx = (np.arange(N, dtype=np.int64) * 2)[None, :]
    atom_token = np.repeat(np.arange(N, dtype=np.int64)[None, :], 2, axis=-2).reshape(
        1, -1
    )[:, :A]
    atom_exists = np.ones((B, A), dtype=bool)
    tok_exists = np.ones((B, N), dtype=bool)

    coords = rng.standard_normal((B, A, 3)).astype(np.float32) * 3.0
    coords[0, 0] = [0.0, 0.0, 0.0]
    coords[0, 8] = [0.3, 0.0, 0.0]
    coords[0, 9] = [0.4, 0.0, 0.0]

    pae_logits = rng.standard_normal((B, N, N, BIN)).astype(np.float32)
    pde_logits = rng.standard_normal((B, N, N, BIN)).astype(np.float32)
    plddt_logits = rng.standard_normal((B, A, PLDDT_BINS)).astype(np.float32)

    structure = _make_structure(
        asym=asym,
        entity_type=entity_type,
        residue_index=residue_index,
        bb_mask=bb_mask,
        centre_idx=centre_idx,
        atom_token=atom_token,
        atom_exists=atom_exists,
        tok_exists=tok_exists,
        is_polymer_by_token=np.array([[1] * 4 + [0] * 4], dtype=bool),
    )
    conf = ConfidenceOutputs(
        pae_logits=mx.array(pae_logits),
        pde_logits=mx.array(pde_logits),
        plddt_logits=mx.array(plddt_logits),
    )
    ranker = Ranker(ChaiConfig())
    out = ranker(conf, mx.array(coords), structure)

    ref, _ = _run_reference(
        coords=coords,
        atom_exists=atom_exists,
        atom_token=atom_token,
        tok_exists=tok_exists,
        asym=asym,
        entity_type=entity_type,
        residue_index=residue_index,
        bb_mask=bb_mask,
        centre_idx=centre_idx,
        pae_logits=pae_logits,
        pde_logits=pde_logits,
        plddt_logits=plddt_logits,
    )

    assert bool(out.has_inter_chain_clashes.item()) is False
    assert bool(ref.clash_scores.has_inter_chain_clashes.item()) is False
    _assert_close(
        out.chain_chain_clashes,
        ref.clash_scores.chain_chain_clashes.numpy(),
        "chain_chain_clashes",
    )
    _assert_close(out.aggregate_score, ref.aggregate_score.numpy(), "aggregate_score")


def test_ranker_multi_sample_stacking() -> None:
    """``coords.ndim == 4`` path: rank each sample and stack outputs."""
    data = _two_chain_scenario(0)
    S = 3
    structure = _make_structure(
        asym=data["asym"],
        entity_type=data["entity_type"],
        residue_index=data["residue_index"],
        bb_mask=data["bb_mask"],
        centre_idx=data["centre_idx"],
        atom_token=data["atom_token"],
        atom_exists=data["atom_exists"],
        tok_exists=data["tok_exists"],
        is_polymer_by_token=data["is_polymer"],
    )
    rng = np.random.default_rng(5)
    coords_s = np.stack(
        [data["coords"][0] + rng.standard_normal(data["coords"][0].shape).astype(np.float32) * 0.01
         for _ in range(S)]
    )[None]
    pae_s = np.stack([data["pae_logits"][0] + i * 0.1 for i in range(S)])[None]
    pde_s = np.stack([data["pde_logits"][0] + i * 0.1 for i in range(S)])[None]
    plddt_s = np.stack([data["plddt_logits"][0] + i * 0.1 for i in range(S)])[None]

    conf = ConfidenceOutputs(
        pae_logits=mx.array(pae_s),
        pde_logits=mx.array(pde_s),
        plddt_logits=mx.array(plddt_s),
    )
    ranker = Ranker(ChaiConfig())
    out = ranker(conf, mx.array(coords_s), structure)

    ref_agg = []
    ref_ptm = []
    for i in range(S):
        ref, _ = _run_reference(
            coords=coords_s[:, i],
            atom_exists=data["atom_exists"],
            atom_token=data["atom_token"],
            tok_exists=data["tok_exists"],
            asym=data["asym"],
            entity_type=data["entity_type"],
            residue_index=data["residue_index"],
            bb_mask=data["bb_mask"],
            centre_idx=data["centre_idx"],
            pae_logits=pae_s[:, i],
            pde_logits=pde_s[:, i],
            plddt_logits=plddt_s[:, i],
        )
        ref_agg.append(ref.aggregate_score.numpy().reshape(-1)[0])
        ref_ptm.append(ref.ptm_scores.complex_ptm.numpy().reshape(-1)[0])

    _assert_close(out.aggregate_score.reshape(-1), np.array(ref_agg), "aggregate_score_stack")
    _assert_close(out.ptm.reshape(-1), np.array(ref_ptm), "ptm_stack")
    assert out.aggregate_score.shape == (1, S)
