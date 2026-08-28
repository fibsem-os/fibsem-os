"""Toggling a defect on the Experiment tab writes the experiment once, not twice.

`AutoLamellaUI.lamella_list.defect_changed` had two handlers and both called
`experiment.save()` -- one on the widget, one on the window that embeds it. Marking a
lamella defective therefore serialised the whole experiment twice, back to back: 1.7 s
of frozen GUI at 100 lamella against 0.84 s for the one write that was needed
(FIB-682). The window's handler is the one kept, because it also redraws the rows, the
cards and the name list; the widget's did nothing the window did not.

Both halves are pinned here, and the second is the one with history. FIB-564 was the
opposite failure -- a defect set from a list whose host never wired persistence, so it
lived in memory and was gone on reload. Deleting a save is exactly the shape of change
that reintroduces it, so "somebody still saves" is asserted as hard as "only once".

Split across two mechanisms because the two halves live in different places:

* the widget can be built (`AutoLamellaUI(parent_ui=None)` constructs offscreen), so
  what it does on the signal is checked by driving it;
* the window cannot -- `AutoLamellaSingleWindowUI` builds a vispy quad view and a
  napari viewer, neither of which survives `QT_QPA_PLATFORM=offscreen` -- so its half is
  read off the source, the same way `test_lamella_defect_sync.py` handles the minimap
  and the coincidence viewer.
"""
import ast
import inspect
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import DefectState, DefectType, Lamella

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def lamella(tmp_path):
    return Lamella(petname="01-test", path=str(tmp_path / "01-test"), number=1)


class TestTheWidgetNoLongerSavesItself:
    def test_a_defect_toggle_does_not_save_from_the_widget(
        self, lamella, tmp_path, monkeypatch
    ):
        """Driven, not read: the connection is what changed, so emit and count.

        `parent_ui=None` leaves the window's handler unconnected, which is precisely
        the seam wanted -- whatever this widget does on its own is all that is counted.

        The widget is given a real experiment, and that is load-bearing rather than
        set dressing. Every handler on this path opens with `if self.experiment is
        None: return`, so against a widget that has none the old duplicate save
        returns early and this passes with the bug fully reinstated -- verified by
        putting it back. Without an experiment the test asserts nothing at all.
        """
        from fibsem.applications.autolamella.structures import Experiment
        from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

        experiment = Experiment(path=str(tmp_path), name="defect-save-count")
        experiment.positions.append(lamella)

        saves = []
        monkeypatch.setattr(Experiment, "save", lambda self, **kw: saves.append(1))

        widget = AutoLamellaUI(parent_ui=None)
        try:
            widget.experiment = experiment
            lamella.defect = DefectState(state=DefectType.FAILURE)
            widget.lamella_list.defect_changed.emit(lamella)
        finally:
            widget.close()

        assert saves == [], (
            f"the widget saved {len(saves)} time(s) on defect_changed -- the window's "
            "handler already does, so this is the duplicate write (FIB-682)"
        )

    def test_the_handler_is_gone_rather_than_merely_disconnected(self):
        """A method left behind reads as a live path to anyone grepping for one."""
        from fibsem.applications.autolamella.ui import AutoLamellaUI as module

        assert not hasattr(module.AutoLamellaUI, "_on_lamella_defect_changed")


class TestTheWindowStillSaves:
    """The FIB-564 half. Removing a save must not remove *the* save."""

    def test_the_window_wires_defect_changed_to_a_handler_that_saves(self):
        from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
            AutoLamellaSingleWindowUI,
        )

        window = inspect.getsource(AutoLamellaSingleWindowUI)
        handler = inspect.getsource(
            AutoLamellaSingleWindowUI._on_lamella_defect_changed
        )

        assert "lamella_list.defect_changed.connect(" in window, (
            "the window no longer wires defect_changed -- a defect set on the "
            "Experiment tab is not persisted at all (FIB-564)"
        )
        assert "experiment.save()" in handler, (
            "the window's defect handler no longer saves -- nothing else does now"
        )

    def test_exactly_one_handler_is_wired_to_that_signal(self):
        """The property FIB-682 is about, across both files at once.

        Counted by parsing rather than by substring: the widget and the window both
        contain other `defect_changed.connect(...)` calls for their other lists (the
        card container, the workflow widget), and a text search cannot tell those from
        the one under test. The chain `autolamella_ui.lamella_list.defect_changed`
        names it exactly.
        """
        from pathlib import Path

        from fibsem.applications.autolamella.ui import AutoLamellaMainUI, AutoLamellaUI

        wired = []
        for module, prefix in (
            (AutoLamellaUI, ["self", "lamella_list", "defect_changed"]),
            (AutoLamellaMainUI, ["self", "autolamella_ui", "lamella_list", "defect_changed"]),
        ):
            tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                    continue
                if node.func.attr != "connect":
                    continue
                parts, cur = [], node.func.value
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                if list(reversed(parts)) == prefix:
                    wired.append(module.__name__)

        assert wired == ["fibsem.applications.autolamella.ui.AutoLamellaMainUI"], (
            f"the Experiment-tab list's defect_changed is wired in {wired}; it must be "
            "wired exactly once, in the window (FIB-682/FIB-564)"
        )
