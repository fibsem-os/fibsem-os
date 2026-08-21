"""The connection dialog is offered as the session starts.

Connecting is the first thing every session does, and until the dialog existed the
only way to do it was a tab — which is why `--quickstart` was written, to skip two
clicks nobody could avoid. Offering it outright is what that flag was working
around (FIB-775).

The window cannot be constructed offscreen (napari, vispy), so the launch handler
is borrowed onto a host. What it decides is the whole of it: whether to open the
dialog at all.
"""

from __future__ import annotations

import ast
import inspect
import os
import textwrap

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (  # noqa: E402
    AutoLamellaSingleWindowUI,
    run_ui,
)


class _SystemWidgetStub:
    def __init__(self, microscope=None):
        self.microscope = microscope


class _UIStub:
    def __init__(self, microscope=None):
        self.system_widget = _SystemWidgetStub(microscope)


class _LaunchHost:
    offer_connection_at_launch = AutoLamellaSingleWindowUI.offer_connection_at_launch

    def __init__(self, chip_enabled=True, microscope=None, autolamella_ui=True):
        self._connection_chip_enabled = chip_enabled
        self.autolamella_ui = _UIStub(microscope) if autolamella_ui else None
        self.opened = 0

    def _on_connect_microscope(self):
        self.opened += 1


def test_it_opens_on_a_fresh_start():
    host = _LaunchHost()

    host.offer_connection_at_launch()

    assert host.opened == 1


def test_it_stays_shut_with_the_feature_off():
    """The shipping default, where the Connection tab is still how people connect."""
    host = _LaunchHost(chip_enabled=False)

    host.offer_connection_at_launch()

    assert host.opened == 0


def test_it_stays_shut_when_something_is_already_connected():
    """`--quickstart` got there first, and asking again would be asking twice."""
    host = _LaunchHost(microscope=object())

    host.offer_connection_at_launch()

    assert host.opened == 0


def test_it_stays_shut_before_the_panel_exists():
    host = _LaunchHost(autolamella_ui=False)

    host.offer_connection_at_launch()

    assert host.opened == 0


def test_the_quick_flags_skip_it():
    """They already do what the dialog would be asking for.

    Read off `run_ui`, which cannot be executed in a test: it builds the window and
    enters the event loop. The offer has to sit in the branch the quick flags do
    not take.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(run_ui)))

    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(n, ast.Attribute) and n.attr in ("quickstart", "quickload")
            for n in ast.walk(node.test)
        )
    ]
    assert branches, "sanity: no branch on the quick flags found in run_ui"

    def _calls_offer(nodes) -> bool:
        return any(
            isinstance(n, ast.Attribute) and n.attr == "offer_connection_at_launch"
            for node in nodes
            for n in ast.walk(node)
        )

    branch = branches[0]
    assert _calls_offer(branch.orelse), "the offer is not in the else branch"
    assert not _calls_offer(branch.body), (
        "the offer is in the quick-flag branch, so it would ask on top of a "
        "connection those flags are already making"
    )


def test_it_is_deferred_past_the_first_paint():
    """A modal opened before the window behind it has painted floats over an empty
    frame. The quick flags are deferred for the same reason, and say so."""
    source = inspect.getsource(run_ui)

    assert "QTimer.singleShot(0, window.offer_connection_at_launch)" in source
