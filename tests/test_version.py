"""gridstate must expose a usable ``__version__`` — release / consumer hygiene.

A published package is expected to answer ``gridstate.__version__``; this guards
against the attribute silently disappearing from the public surface again.
"""

from __future__ import annotations

import sys
import types
from importlib.metadata import PackageNotFoundError

import pytest

import gridstate


def test_version_is_nonempty_string() -> None:
    assert isinstance(gridstate.__version__, str)
    assert gridstate.__version__


def test_version_is_not_the_uninstalled_fallback() -> None:
    # The test environment installs the package (editable), so metadata exists
    # and the bare-source-tree fallback must not be what we observe.
    assert gridstate.__version__ != "0+unknown"


def test_version_prefers_generated_module(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("gridstate._version")
    fake.__version__ = "9.8.7"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gridstate._version", fake)
    monkeypatch.setattr(gridstate, "version", lambda _name: "0.0.1")

    assert gridstate._resolve_version() == "9.8.7"


def test_version_falls_back_to_package_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``None`` in sys.modules makes the import of the generated module fail.
    monkeypatch.setitem(sys.modules, "gridstate._version", None)
    monkeypatch.setattr(gridstate, "version", lambda _name: "1.2.3")

    assert gridstate._resolve_version() == "1.2.3"


def test_version_unknown_without_module_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setitem(sys.modules, "gridstate._version", None)
    monkeypatch.setattr(gridstate, "version", missing)

    assert gridstate._resolve_version() == "0+unknown"
