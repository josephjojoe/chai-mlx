"""Test the per-token PAE / PDE / pLDDT tensors in
``scores.model_idx_*.npz``.

The CLI claims drop-in compatibility with chai-lab's
``StructureCandidates`` ``pae`` / ``pde`` / ``plddt`` fields.
This test invokes :func:`chai_mlx.cli.infer._write_per_sample_scores`
directly with synthetic logits + a structure whose masks we control,
and asserts:

* Arrays land in the npz under the exact field names chai-lab uses.
* Shapes are ``(n_tokens, n_tokens)`` for pae/pde, ``(n_tokens,)`` for
  plddt (per-sample, with the sample axis stripped by the writer).
* Padded-token positions are zero, not garbage.
* Values are inside the expected bin-center ranges (``[0, 32]`` for
  pae/pde, ``[0, 1]`` for plddt).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from chai_mlx.cli.infer import _write_per_sample_scores


@dataclass
class _FakeConf:
    pae_logits: mx.array
    pde_logits: mx.array
    plddt_logits: mx.array


@dataclass
class _FakeStructure:
    token_exists_mask: mx.array
    atom_exists_mask: mx.array
    atom_token_index: mx.array


@dataclass
class _FakeRanking:
    aggregate_score: mx.array
    ptm: mx.array
    iptm: mx.array
    has_inter_chain_clashes: mx.array
    per_chain_ptm: mx.array | None = None
    per_chain_pair_iptm: mx.array | None = None
    complex_plddt: mx.array | None = None
    per_chain_plddt: mx.array | None = None
    per_atom_plddt: mx.array | None = None
    chain_chain_clashes: mx.array | None = None
    total_clashes: mx.array | None = None
    total_inter_chain_clashes: mx.array | None = None


def _make_conf_and_structure(
    n_tokens: int = 8,
    n_live_tokens: int = 6,
    n_atoms: int = 16,
    n_samples: int = 2,
):
    """Build a tiny (confidence, structure) pair for the writer.

    ``n_live_tokens`` < ``n_tokens`` lets us verify the padded-token
    zeroing behaviour.
    """
    # pae / pde use 64 bins over [0, 32]; logits sampled from a fixed
    # seed so the test is deterministic.
    rng = np.random.default_rng(42)

    pae_logits = mx.array(
        rng.normal(size=(1, n_samples, n_tokens, n_tokens, 64)).astype(np.float32)
    )
    pde_logits = mx.array(
        rng.normal(size=(1, n_samples, n_tokens, n_tokens, 64)).astype(np.float32)
    )
    # plddt uses 50 bins over [0, 1].
    plddt_logits = mx.array(
        rng.normal(size=(1, n_samples, n_atoms, 50)).astype(np.float32)
    )

    # Structure masks: first ``n_live_tokens`` exist, rest are padding.
    token_mask = np.concatenate(
        [np.ones(n_live_tokens), np.zeros(n_tokens - n_live_tokens)]
    ).astype(np.float32)
    atom_mask = np.ones(n_atoms, dtype=np.float32)
    # Distribute atoms evenly over the live tokens.
    atom_token_index = np.repeat(
        np.arange(n_live_tokens, dtype=np.int32),
        n_atoms // n_live_tokens + 1,
    )[:n_atoms]

    structure = _FakeStructure(
        token_exists_mask=mx.array(token_mask[None, :]),
        atom_exists_mask=mx.array(atom_mask[None, :]),
        atom_token_index=mx.array(atom_token_index[None, :]),
    )

    ranking = _FakeRanking(
        aggregate_score=mx.array(np.zeros((1, n_samples), dtype=np.float32)),
        ptm=mx.array(np.ones((1, n_samples), dtype=np.float32) * 0.5),
        iptm=mx.array(np.ones((1, n_samples), dtype=np.float32) * 0.3),
        has_inter_chain_clashes=mx.array(np.zeros((1, n_samples), dtype=np.float32)),
    )

    return _FakeConf(pae_logits, pde_logits, plddt_logits), structure, ranking


def test_per_token_tensors_present_in_npz() -> None:
    conf, structure, ranking = _make_conf_and_structure(
        n_tokens=8, n_live_tokens=6, n_samples=2
    )
    with tempfile.TemporaryDirectory(prefix="chai_mlx_t1_npz_") as td:
        out = Path(td)
        _write_per_sample_scores(
            output_dir=out,
            ranking=ranking,
            confidence=conf,
            structure=structure,
            num_samples=2,
        )
        assert (out / "scores.model_idx_0.npz").is_file()
        assert (out / "scores.model_idx_1.npz").is_file()

        npz0 = np.load(out / "scores.model_idx_0.npz")
        for key in ("pae", "pde", "plddt"):
            assert key in npz0.files, (
                f"per-token {key!r} missing from scores.model_idx_0.npz"
            )


def test_per_token_tensor_shapes() -> None:
    conf, structure, ranking = _make_conf_and_structure(
        n_tokens=8, n_live_tokens=6, n_samples=2
    )
    with tempfile.TemporaryDirectory(prefix="chai_mlx_t1_npz_") as td:
        out = Path(td)
        _write_per_sample_scores(
            output_dir=out,
            ranking=ranking,
            confidence=conf,
            structure=structure,
            num_samples=2,
        )
        npz = np.load(out / "scores.model_idx_0.npz")
        assert npz["pae"].shape == (8, 8)
        assert npz["pde"].shape == (8, 8)
        assert npz["plddt"].shape == (8,)


def test_padded_token_positions_are_zero() -> None:
    """Rows / cols for tokens outside ``token_exists_mask`` must be 0."""
    conf, structure, ranking = _make_conf_and_structure(
        n_tokens=8, n_live_tokens=6, n_samples=2
    )
    with tempfile.TemporaryDirectory(prefix="chai_mlx_t1_npz_") as td:
        out = Path(td)
        _write_per_sample_scores(
            output_dir=out,
            ranking=ranking,
            confidence=conf,
            structure=structure,
            num_samples=2,
        )
        npz = np.load(out / "scores.model_idx_0.npz")
        pae, pde, plddt = npz["pae"], npz["pde"], npz["plddt"]
        # Rows/cols for the padded indices (6, 7) must be zero.
        assert np.all(pae[6:, :] == 0.0)
        assert np.all(pae[:, 6:] == 0.0)
        assert np.all(pde[6:, :] == 0.0)
        assert np.all(pde[:, 6:] == 0.0)
        # Per-token pLDDT for padded tokens must also be zero (our
        # bincount remedy over atom_token_index only covers live
        # tokens, so padded indices never get a contribution).
        assert np.all(plddt[6:] == 0.0)


def test_per_token_value_ranges() -> None:
    """pae/pde are in ``[0, 32]`` (the bin-center range); plddt is in
    ``[0, 1]``. The softmax expectation cannot escape these ranges."""
    conf, structure, ranking = _make_conf_and_structure(
        n_tokens=6, n_live_tokens=6, n_samples=1
    )
    with tempfile.TemporaryDirectory(prefix="chai_mlx_t1_npz_") as td:
        out = Path(td)
        _write_per_sample_scores(
            output_dir=out,
            ranking=ranking,
            confidence=conf,
            structure=structure,
            num_samples=1,
        )
        npz = np.load(out / "scores.model_idx_0.npz")
        # Live positions are [0, 6); slice to avoid the zero-padded tail.
        live = npz["pae"][:6, :6]
        assert live.min() >= 0.0 - 1e-5
        assert live.max() <= 32.0 + 1e-5
        live = npz["pde"][:6, :6]
        assert live.min() >= 0.0 - 1e-5
        assert live.max() <= 32.0 + 1e-5
        live = npz["plddt"][:6]
        assert live.min() >= 0.0 - 1e-5
        assert live.max() <= 1.0 + 1e-5
