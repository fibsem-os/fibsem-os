"""A failing context-menu action warns the user without breaking the caller (FIB-407).

`_invoke_action_callback` used to reach for `napari.utils.notifications` inside a bare
`except Exception: pass`. Phase 5 repoints it at `notification_service`, and the shape of
that code means a mistake there is completely silent — a wrong module, a wrong function
name or a wrong signature would be swallowed by the same `except`, and the user would get
no warning at all while every test still passed.

So this asserts on the toast actually arriving, not on the call not raising.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5", reason="UI dependencies not installed")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.ui import notification_service  # noqa: E402
from fibsem.ui.widgets.custom_widgets import (  # noqa: E402
    ContextMenu,
    ContextMenuAction,
)


@pytest.fixture(scope="module")
def _app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def toasts():
    """Every toast emitted during the test, as (message, type, temporary)."""
    received = []
    service = notification_service._get_service()
    service.toast.connect(lambda *args: received.append(args))
    yield received


def _invoke(label, callback):
    """Drive the private handler the context menu calls when an action is chosen."""
    menu = ContextMenu.__new__(ContextMenu)  # the handler needs no Qt setup
    menu._invoke_action_callback(ContextMenuAction(label=label, callback=callback))


def test_a_failing_action_warns_the_user(_app, toasts):
    def boom():
        raise RuntimeError("no")

    _invoke("Move Patterns Here", boom)

    assert toasts, "a failing context-menu action produced no warning"
    message, kind = toasts[-1][0], toasts[-1][1]
    assert "Move Patterns Here" in message, f"warning does not name the action: {message!r}"
    assert kind == "warning", f"expected a warning, got {kind!r}"


def test_a_failing_action_does_not_propagate(_app, toasts):
    """The whole point of the try/except: one bad action must not break the caller."""
    def boom():
        raise RuntimeError("no")

    _invoke("Anything", boom)  # must not raise


def test_a_successful_action_warns_about_nothing(_app, toasts):
    calls = []
    _invoke("Fine", lambda: calls.append(1))
    assert calls == [1]
    assert not toasts, f"a successful action emitted a warning: {toasts}"


def test_an_action_without_a_callback_is_a_no_op(_app, toasts):
    _invoke("No callback", None)
    assert not toasts
