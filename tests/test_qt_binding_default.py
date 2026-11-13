"""fibsem picks PySide6 for qtpy unless QT_API names a binding (FIB-1565)."""

import sys
from types import SimpleNamespace

import pytest

import fibsem


def test_pyside6_by_default(monkeypatch):
    monkeypatch.delenv("QT_API", raising=False)
    fibsem._default_qt_binding()
    assert fibsem.os.environ["QT_API"] == "pyside6"


def test_qt_api_wins(monkeypatch):
    monkeypatch.setenv("QT_API", "pyqt5")
    fibsem._default_qt_binding()
    assert fibsem.os.environ["QT_API"] == "pyqt5"


def _windows(monkeypatch, build):
    monkeypatch.delenv("QT_API", raising=False)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        sys, "getwindowsversion", lambda: SimpleNamespace(build=build), raising=False
    )


def test_windows_10_before_1809_is_warned(monkeypatch):
    """Qt 6 does not run there."""
    _windows(monkeypatch, 17134)
    with pytest.warns(UserWarning, match="build 1809"):
        fibsem._default_qt_binding()


def test_windows_10_1809_gets_pyside6(monkeypatch, recwarn):
    _windows(monkeypatch, 17763)
    fibsem._default_qt_binding()
    assert fibsem.os.environ["QT_API"] == "pyside6"
    assert not recwarn.list
