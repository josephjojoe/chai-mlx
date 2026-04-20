"""Tests for chai-1 input-space limit enforcement.

Chai-mlx mirrors chai-lab's three hard input limits via
:func:`chai_mlx.data.featurize._enforce_input_limits`:

* ``MAX_MSA_DEPTH`` (16384 at the pinned commit)
* ``MAX_NUM_TEMPLATES`` (4 at the pinned commit)
* Crop size upper bound (``max(AVAILABLE_MODEL_SIZES) = 2048``)

These tests use a lightweight stand-in feature-context so they pass
without the ``[featurize]`` extra. The helper's error messages must
mention the offending value and the limit so users can act on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from chai_mlx.data.featurize import UnsupportedInputError, _enforce_input_limits


@dataclass
class _FakeStructureCtx:
    num_tokens: int


@dataclass
class _FakeTemplateCtx:
    num_templates: int


@dataclass
class _FakeMSACtx:
    depth: int


@dataclass
class _FakeFeatureCtx:
    structure_context: _FakeStructureCtx
    template_context: _FakeTemplateCtx
    msa_context: _FakeMSACtx


def _ctx(n_tokens: int = 100, n_templates: int = 0, msa_depth: int = 0) -> _FakeFeatureCtx:
    return _FakeFeatureCtx(
        structure_context=_FakeStructureCtx(num_tokens=n_tokens),
        template_context=_FakeTemplateCtx(num_templates=n_templates),
        msa_context=_FakeMSACtx(depth=msa_depth),
    )


def test_limits_pass_on_small_input() -> None:
    # 100 tokens, no MSAs, no templates: comfortably inside every limit.
    _enforce_input_limits(_ctx(n_tokens=100))


def test_too_many_tokens_raises() -> None:
    with pytest.raises(UnsupportedInputError) as exc:
        _enforce_input_limits(_ctx(n_tokens=3000))
    msg = str(exc.value)
    assert "3000" in msg
    assert "Too many tokens" in msg
    # The message should list the supported crops so users can pick one.
    assert "256" in msg


def test_too_many_templates_raises() -> None:
    with pytest.raises(UnsupportedInputError) as exc:
        _enforce_input_limits(_ctx(n_tokens=100, n_templates=999))
    msg = str(exc.value)
    assert "999" in msg
    assert "Too many templates" in msg


def test_msa_too_deep_raises() -> None:
    with pytest.raises(UnsupportedInputError) as exc:
        _enforce_input_limits(_ctx(n_tokens=100, msa_depth=100_000))
    msg = str(exc.value)
    assert "100000" in msg
    assert "MSA too deep" in msg
    # Tip towards recycle_msa_subsample as the remedy.
    assert "recycle_msa_subsample" in msg


def test_error_inherits_from_value_error() -> None:
    """``UnsupportedInputError`` subclasses ``ValueError`` so existing
    callers that catch ``ValueError`` keep working. Match chai-lab's
    upstream contract (chai-lab's ``UnsupportedInputError`` is also a
    ``ValueError``)."""
    assert issubclass(UnsupportedInputError, ValueError)
