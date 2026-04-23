"""Regression test for the ESM-MLX RAM guard.

:func:`chai_mlx.data.featurize._warn_if_insufficient_ram_for_esm_mlx`
prints a stderr warning when ``esm_backend="mlx"`` is requested on a
host with less than 20 GiB total RAM. Without this
warning, users on 16 GB Macs silently OOM mid-inference.

This test mocks ``psutil.virtual_memory()`` to fake a small machine
and a large machine, asserting the warning fires / stays silent
accordingly. It exists to stop the guard from silently rotting if
someone refactors the module or if ``psutil`` grows an API change.
"""

from __future__ import annotations

import importlib.util
import io
import sys
import types
from contextlib import redirect_stderr

import pytest

from chai_mlx.data.featurize import _warn_if_insufficient_ram_for_esm_mlx


_HAS_PSUTIL = importlib.util.find_spec("psutil") is not None


class _FakeVM:
    def __init__(self, total: int) -> None:
        self.total = total


def _install_fake_psutil(monkeypatch: pytest.MonkeyPatch, total_bytes: int) -> None:
    """Install a fake ``psutil`` module with a controlled
    ``virtual_memory().total``."""
    fake = types.ModuleType("psutil")
    fake.virtual_memory = lambda: _FakeVM(total=total_bytes)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psutil", fake)


def test_ram_guard_fires_on_small_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    # 16 GiB total -- a canonical "this will OOM" configuration.
    _install_fake_psutil(monkeypatch, total_bytes=16 * 1024 ** 3)
    buf = io.StringIO()
    with redirect_stderr(buf):
        _warn_if_insufficient_ram_for_esm_mlx()
    output = buf.getvalue()
    assert "esm_backend='mlx'" in output, (
        "RAM guard did not fire on a 16 GiB machine; check the "
        "_ESM_MLX_MIN_RAM_BYTES threshold in featurize.py."
    )
    assert "16.0" in output or "16 " in output, (
        f"RAM guard warning did not include the machine's RAM size; "
        f"got:\n{output}"
    )


def test_ram_guard_silent_on_large_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    # 64 GiB total -- far above threshold.
    _install_fake_psutil(monkeypatch, total_bytes=64 * 1024 ** 3)
    buf = io.StringIO()
    with redirect_stderr(buf):
        _warn_if_insufficient_ram_for_esm_mlx()
    assert buf.getvalue() == "", (
        "RAM guard incorrectly fired on a 64 GiB machine; check the "
        "threshold comparison direction in featurize.py."
    )


def test_ram_guard_silent_when_psutil_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """No ``psutil`` is a soft failure -- the user still gets a Metal
    allocator abort if they OOM, but pytest should not fail."""
    # Replace psutil with a sentinel that raises ImportError on access.
    monkeypatch.setitem(sys.modules, "psutil", None)
    buf = io.StringIO()
    with redirect_stderr(buf):
        _warn_if_insufficient_ram_for_esm_mlx()
    # No assertion on output -- just that no exception was raised.
    # The body silently returns when psutil is unimportable.


@pytest.mark.skipif(not _HAS_PSUTIL, reason="psutil not installed")
def test_ram_guard_exercises_real_psutil_path() -> None:
    """Smoke test the real psutil code path: should not raise
    regardless of whether this machine triggers the warning."""
    buf = io.StringIO()
    with redirect_stderr(buf):
        _warn_if_insufficient_ram_for_esm_mlx()
    # Output is environment-dependent; we only assert we didn't crash.
