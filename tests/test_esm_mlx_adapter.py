"""ESM-on-MLX adapter shape/order test.

Exercises ``chai_mlx.data.esm_mlx_adapter.build_embedding_context`` by
monkey-patching ``esm_mlx.ESM2.from_pretrained`` with a tiny fake that
returns deterministic per-residue embeddings.  We then assert the
adapter:

* produces a torch tensor of shape ``(sum(n_tokens_per_chain), 2560)``,
* honours ``EntityType`` (non-protein chains get zero rows),
* preserves the chai-lab chain ordering (chain-wise concatenation).

This is the same injection point chai-lab itself uses via
``get_esm_embedding_context``; the test does not need real 3B weights.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from tests.helpers import has_chai_lab_runtime


_HAS_CHAI_LAB = has_chai_lab_runtime(require_esm_mlx=True)

pytestmark = pytest.mark.skipif(
    not _HAS_CHAI_LAB,
    reason="esm-mlx adapter test requires chai-lab, torch, and esm_mlx "
    "(install the default package plus the [esm] extra)",
)


def _make_fake_chain(entity_type, sequence: str, num_tokens: int):
    """Build a chai-lab-ish Chain stub exposing the fields the adapter reads.

    Matches the real ``AllAtomEntityData`` / ``AllAtomStructureContext``
    shape: entity_type is the ``EntityType`` enum member (not the int
    value), and ``token_residue_index`` is a 1D int64 tensor.
    """
    import torch

    class _EntityData:
        pass

    class _StructureContext:
        pass

    ed = _EntityData()
    ed.entity_type = entity_type
    ed.sequence = sequence
    sc = _StructureContext()
    sc.num_tokens = num_tokens
    sc.token_residue_index = torch.arange(num_tokens, dtype=torch.long)

    return types.SimpleNamespace(entity_data=ed, structure_context=sc)


class _FakeEsm:
    """Minimal stand-in for :class:`esm_mlx.ESM2`.

    Produces deterministic per-residue embeddings: every row is
    ``[i * 0.01] * 2560`` so the test can verify the explosion step uses
    ``token_residue_index`` correctly.
    """

    num_layers = 36

    def __init__(self, embed_dim: int = 2560) -> None:
        self.embed_dim = embed_dim

    def __call__(self, tokens, repr_layers):
        import mlx.core as mx

        # ``tokens`` has shape (1, L+2) due to <cls> and <eos>; we produce an
        # output with matching shape and 2560 dims.
        L_plus_2 = tokens.shape[1]
        vec = mx.arange(L_plus_2, dtype=mx.float32)[None, :, None]
        emb = mx.broadcast_to(vec * 0.01, (1, L_plus_2, self.embed_dim))
        return {"representations": {self.num_layers: emb}}


def test_build_embedding_context_shape_and_zero_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    from chai_lab.data.parsing.structure.entity_type import EntityType

    from chai_mlx.data.esm_mlx_adapter import build_embedding_context

    # Patch ESM2.from_pretrained to return our fake (don't need real weights).
    import esm_mlx

    def _fake_from_pretrained(model_name: str):
        assert model_name == "esm2_t36_3B_UR50D"
        return _FakeEsm()

    monkeypatch.setattr(esm_mlx.ESM2, "from_pretrained", classmethod(
        lambda cls, model_name: _fake_from_pretrained(model_name)
    ))

    seq_a = "MKTAYIAKQRQI"   # 12 residues
    seq_b = "GVQVETIS"        # 8 residues
    chains = [
        _make_fake_chain(EntityType.PROTEIN, seq_a, num_tokens=len(seq_a)),
        _make_fake_chain(EntityType.DNA, "CGCGAATTCGCG", num_tokens=12),
        _make_fake_chain(EntityType.PROTEIN, seq_b, num_tokens=len(seq_b)),
    ]

    ctx = build_embedding_context(chains)

    expected_tokens = len(seq_a) + 12 + len(seq_b)
    assert ctx.esm_embeddings.shape == (expected_tokens, 2560), (
        f"unexpected shape {ctx.esm_embeddings.shape}"
    )
    assert ctx.esm_embeddings.dtype == torch.float32, (
        f"expected float32, got {ctx.esm_embeddings.dtype}"
    )

    emb_np = ctx.esm_embeddings.numpy()

    # First len(seq_a) rows belong to chain A.  Our fake produces
    # per-token values of (token_pos + 1) * 0.01 after the adapter strips
    # the BOS row (index 0).
    expected_a = np.arange(1, len(seq_a) + 1, dtype=np.float32) * 0.01
    np.testing.assert_allclose(emb_np[: len(seq_a), 0], expected_a, atol=1e-7)

    # DNA chain rows are zero-filled.
    dna_rows = emb_np[len(seq_a) : len(seq_a) + 12]
    assert np.all(dna_rows == 0.0), "DNA chain rows must be zero-filled"

    # Last len(seq_b) rows belong to chain B, independent of chain A.
    expected_b = np.arange(1, len(seq_b) + 1, dtype=np.float32) * 0.01
    np.testing.assert_allclose(
        emb_np[len(seq_a) + 12 :, 0], expected_b, atol=1e-7
    )
