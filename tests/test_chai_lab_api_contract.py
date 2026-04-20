"""Pin the chai-lab symbols we depend on at import time.

Chai-mlx's featurizer and CLI depend on a specific set of chai-lab
APIs, some of which are private (leading underscore). This test
imports every symbol we use so any chai-lab version bump that
renames or removes one of them fails fast at collection time -- not
at 5-second-into-inference runtime.

If this test breaks, either:

1. Pin ``pyproject.toml``'s ``[featurize]`` extra to a chai-lab
   commit that still has the symbol (short-term fix), or
2. Adapt chai-mlx to the new chai-lab API (proper fix). The
   canonical call sites are listed in the docstring below each
   assertion so the adaptation is mechanical.

The test is gated on the ``[featurize]`` extra being installed
(same as every other chai-lab-dependent test) so environments
without torch + chai-lab still pass ``pytest -q``.
"""

from __future__ import annotations

import importlib.util

import pytest


_HAS_CHAI_LAB = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("chai_lab") is not None
)

pytestmark = pytest.mark.skipif(
    not _HAS_CHAI_LAB,
    reason="chai-lab API contract requires the [featurize] extra",
)


def test_chai1_public_api() -> None:
    """Public symbols used by :mod:`chai_mlx.data.featurize` and the CLI."""
    from chai_lab import chai1  # noqa: F401
    from chai_lab.chai1 import (  # noqa: F401
        Collate,
        TokenBondRestraint,
        feature_factory,
        feature_generators,
        make_all_atom_feature_context,
    )


def test_cif_utils_public_api() -> None:
    """Used by :func:`chai_mlx.cli.infer._save_cifs`."""
    from chai_lab.data.io.cif_utils import (  # noqa: F401
        get_chain_letter,
        save_to_cif,
    )


def test_entity_type_enum_public_api() -> None:
    """Used by :func:`chai_mlx.data.featurize._batch_to_feature_context`."""
    from chai_lab.data.parsing.structure.entity_type import EntityType

    # These four entity values are read by name in featurize.py; if any
    # of them disappear upstream, the polymer mask construction breaks.
    for name in ("PROTEIN", "RNA", "DNA", "MANUAL_GLYCAN"):
        assert hasattr(EntityType, name), (
            f"chai_lab.data.parsing.structure.entity_type.EntityType "
            f"no longer exposes {name!r}; featurize.py needs updating."
        )


def test_embedding_context_public_api() -> None:
    """Used by :mod:`chai_mlx.data.esm_mlx_adapter`."""
    from chai_lab.data.dataset.embeddings.embedding_context import (  # noqa: F401
        EmbeddingContext,
    )


def test_glycan_parser_private_api() -> None:
    """Used by :func:`chai_mlx.data.fasta._glycan_issues`.

    ``_glycan_string_to_sugars_and_bonds`` is private (leading
    underscore). We rely on it to pre-flight glycan SMILES-like
    strings so users see a loud error at validation time rather than
    a cryptic crash inside featurization. If chai-lab renames or
    removes it, we need to either re-pin or switch to parsing the
    string ourselves (it's a small grammar; NAG(4-1 NAG) etc.).
    """
    from chai_lab.data.parsing.glycans import (  # noqa: F401
        _glycan_string_to_sugars_and_bonds,
    )


def test_plot_msa_public_api() -> None:
    """Used by :func:`chai_mlx.data.featurize.featurize_fasta` when
    ``msa_plot_path`` is set (``--write-msa-plot`` CLI flag)."""
    import inspect

    from chai_lab.utils.plot import plot_msa

    sig = inspect.signature(plot_msa)
    expected_params = {"input_tokens", "msa_tokens", "out_fname"}
    actual_params = set(sig.parameters)
    missing = expected_params - actual_params
    assert not missing, (
        f"chai_lab.utils.plot.plot_msa no longer accepts {sorted(missing)}; "
        "featurize.py's msa_plot_path path needs updating."
    )


def test_rank_get_scores_public_api() -> None:
    """Field names referenced by chai-lab parity sidecar npz layout.

    The docstring of :func:`chai_mlx.cli.infer._write_per_sample_scores`
    claims our per-sample npz sidecars mirror chai-lab's
    ``get_scores`` output. If chai-lab renames a key in that function,
    our drop-in-compatibility claim becomes a lie.
    """
    from chai_lab.ranking.rank import get_scores  # noqa: F401
    # We do not call get_scores -- its signature requires a full
    # RankingData object. Importing it is enough to catch a rename.
