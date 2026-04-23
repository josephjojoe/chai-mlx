"""Unit tests for the exact-length pad strategy.

``pad_strategy="exact"`` is the default behaviour of
:func:`chai_mlx.data.featurize.featurize_fasta` on MLX -- it sidesteps
chai-lab's seven static TorchScript-export buckets and pads to the
tightest shape the MLX kernels accept:

* ``n_tokens`` == the real token count (no divisibility constraint --
  the MLX trunk reads the token dimension dynamically from tensor
  shapes at call time).
* ``n_atoms`` rounded up to the next multiple of 32, the query-block
  stride of the local atom attention (see
  ``chai_lab.model.utils.get_qkv_indices_for_blocks`` and the
  ``num_blocks = a // 32`` reshape in
  ``chai_mlx/nn/layers/atom_attention.py``).

This file tests the pure-Python helpers; the chai-lab integration is
exercised in :mod:`tests.test_constraints_parse` and the
``featurize_fasta`` smoke tests.
"""

from __future__ import annotations

import pytest

from chai_mlx.data.featurize import (
    _ATOM_BLOCK_STRIDE,
    _exact_pad_size,
    _override_pad_strategy,
)
from tests.helpers import has_chai_lab_runtime


class TestExactPadSize:
    def test_stride_matches_atom_attention_block_size(self) -> None:
        # The _ATOM_BLOCK_STRIDE constant is the source of truth for the
        # atom-axis divisibility constraint. It must match the 32 that
        # chai_mlx/nn/layers/atom_attention.py uses as its block size
        # (``num_blocks = a // 32`` appears verbatim in 4 places there)
        # and the 32 that chai_lab.model.utils.get_qkv_indices_for_blocks
        # asserts. If someone re-tunes the block size in one place and
        # forgets the others, this test catches the mismatch.
        assert _ATOM_BLOCK_STRIDE == 32

    def test_exact_pad_size_preserves_token_count(self) -> None:
        # Tokens are never padded under 'exact' -- the MLX forward loop
        # reads n_tokens from the tensor shape.
        for n in (1, 7, 32, 33, 137, 256, 1000, 2048):
            n_tok, _ = _exact_pad_size(n_tokens=n, n_atoms=n * 5)
            assert n_tok == n, f"tokens padded: {n} -> {n_tok}"

    def test_exact_pad_size_rounds_atoms_to_multiple_of_32(self) -> None:
        for n_atoms in range(1, 200):
            _, padded = _exact_pad_size(n_tokens=10, n_atoms=n_atoms)
            assert padded % 32 == 0, (
                f"padded atom count {padded} (from {n_atoms}) is not a "
                f"multiple of 32"
            )
            assert padded >= n_atoms, (
                f"padded atom count {padded} is below requested {n_atoms}"
            )
            assert padded - n_atoms < 32, (
                f"padded atom count {padded} wastes more than one block "
                f"(input was {n_atoms})"
            )

    def test_exact_pad_size_never_returns_zero_atoms(self) -> None:
        # 0 atoms is a degenerate case but the helper rounds it up to
        # the block size rather than returning 0 (which would break
        # ``num_blocks = a // 32 = 0`` and produce empty attention
        # tensors).
        _, padded = _exact_pad_size(n_tokens=1, n_atoms=0)
        assert padded == 32

    def test_exact_pad_size_on_exact_multiple_is_identity(self) -> None:
        # When the atom count is already a multiple of the block stride
        # we must not inflate it -- every extra block costs memory.
        for n_atoms in (32, 64, 96, 128, 256, 512, 1024):
            _, padded = _exact_pad_size(n_tokens=10, n_atoms=n_atoms)
            assert padded == n_atoms

    def test_exact_pad_beats_bucketed_23x_formula(self) -> None:
        # Motivating case from the handoff: 137 tokens + ~1000 atoms.
        # Exact-length picks (137, 1024); bucketed picks (256, 5888).
        # This test pins the exact win at a realistic size.
        n_tok, n_atoms = _exact_pad_size(n_tokens=137, n_atoms=1000)
        assert n_tok == 137
        assert n_atoms == 1024

        # Ratio should be huge -- tokens shrink 256/137 ≈ 1.87x, atoms
        # shrink 5888/1024 ≈ 5.75x. Pair-attention is O(N^2) in tokens
        # so the compute win is ≈ 3.5x on the trunk alone.
        bucketed_tokens = 256
        bucketed_atoms = 23 * 256
        assert n_tok < bucketed_tokens
        assert n_atoms < bucketed_atoms


_HAS_CHAI_LAB = has_chai_lab_runtime()


class TestOverridePadStrategy:
    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="pad_strategy"):
            with _override_pad_strategy("wrong"):  # type: ignore[arg-type]
                pass

    def test_bucket_strategy_is_noop(self) -> None:
        # 'bucket' must not touch chai-lab state. The context manager
        # must be re-entrant and exit cleanly even if chai-lab is not
        # installed (so this test works in the slim test matrix).
        with _override_pad_strategy("bucket"):
            pass

    @pytest.mark.skipif(
        not _HAS_CHAI_LAB,
        reason="exact-strategy patching requires chai-lab to be importable",
    )
    def test_exact_strategy_patches_and_restores(self) -> None:
        from chai_lab.data.collate import utils as collate_utils

        original = collate_utils.get_pad_sizes
        with _override_pad_strategy("exact"):
            # Inside the scope, the function must be a different object.
            assert collate_utils.get_pad_sizes is not original
        # After the scope, the original must be restored -- subsequent
        # chai-lab calls from anywhere else in the process must not see
        # our patched behaviour.
        assert collate_utils.get_pad_sizes is original

    @pytest.mark.skipif(
        not _HAS_CHAI_LAB,
        reason="patched collate run requires chai-lab to be importable",
    )
    def test_exact_strategy_returns_exact_sizes(self) -> None:
        # Build a stand-in context with realistic num_tokens / num_atoms
        # and verify the patched shim computes the exact-length sizes.
        from chai_lab.data.collate import utils as collate_utils

        class _FakeContext:
            def __init__(self, n_tokens: int, n_atoms: int) -> None:
                self.num_tokens = n_tokens
                self.num_atoms = n_atoms

        with _override_pad_strategy("exact"):
            result = collate_utils.get_pad_sizes(
                [_FakeContext(n_tokens=137, n_atoms=1000)]
            )
            assert result.n_tokens == 137
            assert result.n_atoms == 1024  # ceil(1000/32)*32

    @pytest.mark.skipif(
        not _HAS_CHAI_LAB,
        reason="cross-module restore requires chai-lab to be importable",
    )
    def test_exact_strategy_restores_collate_module_binding(self) -> None:
        """Regression: ``collate.py`` binds ``get_pad_sizes`` by-name at
        import time.  An earlier version of :func:`_override_pad_strategy`
        patched :mod:`chai_lab.data.collate.utils` **before** triggering
        :mod:`chai_lab.data.collate.collate`'s first import, which made
        ``collate_mod.get_pad_sizes`` *originate* as our patched shim --
        so the ``finally`` restore put the patched function back instead
        of the real one, silently poisoning every subsequent
        ``bucket``-mode call in the same Python process.  This test
        forces the same import order pytest exposes in practice
        (``test_pad_strategy`` before ``test_constraints_parse``) and
        asserts both module-level bindings round-trip to the real
        ``chai_lab`` function after the context exits.
        """
        import sys

        # Evict any cached collate.py so we exercise the "first import
        # happens inside the context manager" ordering.  This is the
        # state pytest sees when test_pad_strategy runs first in a
        # session that has only imported ``utils`` so far.
        for mod_name in (
            "chai_lab.data.collate.collate",
            "chai_lab.data.collate",
        ):
            sys.modules.pop(mod_name, None)

        # Re-import utils only; collate.py must NOT be cached.
        from chai_lab.data.collate import utils as collate_utils
        assert "chai_lab.data.collate.collate" not in sys.modules, (
            "test setup failed to evict collate.py from the module cache"
        )
        real_get_pad_sizes = collate_utils.get_pad_sizes

        with _override_pad_strategy("exact"):
            # Context is active: utils must point at our shim, and
            # collate.py should now be imported and also point at our
            # shim.
            assert collate_utils.get_pad_sizes is not real_get_pad_sizes
            from chai_lab.data.collate import collate as collate_mod
            assert collate_mod.get_pad_sizes is not real_get_pad_sizes

        # After exit BOTH bindings must be the real chai-lab function.
        from chai_lab.data.collate import collate as collate_mod
        assert collate_utils.get_pad_sizes is real_get_pad_sizes, (
            "utils.get_pad_sizes was not restored after exact override"
        )
        assert collate_mod.get_pad_sizes is real_get_pad_sizes, (
            "collate.get_pad_sizes was not restored after exact override "
            "-- this is the bug that makes bucket mode silently pad to "
            "exact sizes for the rest of the process"
        )

    @pytest.mark.skipif(
        not _HAS_CHAI_LAB,
        reason="patched collate run requires chai-lab to be importable",
    )
    def test_exact_strategy_aggregates_max_over_batch(self) -> None:
        # A batch's pad size is driven by the largest context in it.
        from chai_lab.data.collate import utils as collate_utils

        class _FakeContext:
            def __init__(self, n_tokens: int, n_atoms: int) -> None:
                self.num_tokens = n_tokens
                self.num_atoms = n_atoms

        with _override_pad_strategy("exact"):
            result = collate_utils.get_pad_sizes(
                [
                    _FakeContext(n_tokens=50, n_atoms=400),
                    _FakeContext(n_tokens=137, n_atoms=1000),
                    _FakeContext(n_tokens=100, n_atoms=800),
                ]
            )
            assert result.n_tokens == 137
            assert result.n_atoms == 1024
