"""Editing a lamella's protocol writes the experiment once editing settles, not per field.

`Experiment.save()` serialises the whole experiment -- 0.84 s at 100 lamella, on the GUI
thread -- and every committed field in the lamella protocol editor called it. Editing a
task means several fields in a row, and each one paid the full price (FIB-683).

The write is now debounced. Deliberately *only* the write: the handlers apply the edit to
the lamella immediately, as they always did, so a flush at any later point serialises the
complete current state. There is no snapshot to go stale and no ordering between edits to
preserve -- which is what makes this far cheaper than moving the write off-thread.

What can still go wrong is the write landing late or not at all, so that is what is
pinned here:

* **A burst coalesces, and still lands.** The point of the change, and it fails silently
  in both directions -- too eager and nothing is gained, never firing and edits are lost.
* **A pending write survives the experiment being swapped, and goes to the right file.**
  The sharp edge. `set_experiment` runs *after* the parent has swapped `.experiment`, so
  a flush that looked the experiment up again would write the new one and drop the old
  one's last edit. The pending write carries its own experiment for this reason.
* **Every flush point is wired.** Leaving the editor's lamella or task, hiding the panel,
  starting a run, closing the window. Miss one and an edit is lost quietly, which is a
  worse failure than the lag this fixes.
* **No handler bypasses the debounce.** Ten call sites route through `_save_experiment`;
  one calling `experiment.save()` directly would keep its 0.84 s and nobody would notice.
"""

import ast
import inspect
import os
from pathlib import Path

os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import Experiment
from fibsem.applications.autolamella.ui import (
    autolamella_lamella_protocol_editor as editor_module,
)
from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (
    _SAVE_DEBOUNCE_MS,
    AutoLamellaProtocolEditorWidget,
)

_app = QApplication.instance() or QApplication(sys.argv)

# Enough for the timer to have fired and its slot to have run, without making the suite
# wait on a fixed sleep any longer than it must.
_PAST_DEBOUNCE_MS = _SAVE_DEBOUNCE_MS + 250


@pytest.fixture(scope="module")
def connected_ui():
    """A real AutoLamellaUI on a Demo microscope -- the editor needs one to build.

    `parent_ui=None` is only stored, so the widget stands alone; the shipped default
    configuration is Demo, so `connect_to_microscope` builds a real microscope with no
    hardware in about 0.15 s. Module-scoped because that cost is per-connect, not
    per-test, and the editor is rebuilt per test anyway.
    """
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI

    ui = AutoLamellaUI(parent_ui=None)
    ui.system_widget.connect_to_microscope()
    assert ui.microscope is not None, "Demo connect failed; the editor cannot be built"
    yield ui
    ui.close()


@pytest.fixture
def experiment(tmp_path):
    return Experiment(path=str(tmp_path), name="debounce")


@pytest.fixture
def editor(connected_ui, experiment):
    connected_ui.experiment = experiment
    widget = AutoLamellaProtocolEditorWidget(parent=connected_ui)
    yield widget
    widget._save_timer.stop()  # never let one fire into a later test
    widget.close()


@pytest.fixture
def saves(monkeypatch):
    """Every `Experiment.save`, as the experiment it was called on."""
    recorded = []
    monkeypatch.setattr(Experiment, "save", lambda self, **kw: recorded.append(self))
    return recorded


class TestABurstBecomesOneWrite:
    def test_ten_edits_write_once(self, editor, saves, experiment):
        for _ in range(10):
            editor._save_experiment()

        assert saves == [], "the editor wrote before editing had settled"
        QTest.qWait(_PAST_DEBOUNCE_MS)

        assert saves == [experiment], f"expected one write, got {len(saves)}"

    def test_the_write_does_land(self, editor, saves, experiment):
        """The other direction: a debounce that never fires loses every edit."""
        editor._save_experiment()
        QTest.qWait(_PAST_DEBOUNCE_MS)

        assert saves == [experiment]

    def test_a_later_edit_restarts_the_wait(self, editor, saves):
        """Coalescing, not batching: the wait is from the *last* edit."""
        editor._save_experiment()
        QTest.qWait(_SAVE_DEBOUNCE_MS // 2)
        editor._save_experiment()
        QTest.qWait(_SAVE_DEBOUNCE_MS // 2 + 50)

        assert saves == [], "the second edit did not restart the timer"
        QTest.qWait(_PAST_DEBOUNCE_MS)
        assert len(saves) == 1

    def test_nothing_pending_writes_nothing(self, editor, saves):
        """`flush_pending_save` is called from five places and must be a no-op cold."""
        editor.flush_pending_save()
        editor.flush_pending_save()

        assert saves == []


class TestTheFlushPoints:
    def test_leaving_the_lamella_flushes(self, editor, saves, experiment):
        editor._save_experiment()
        editor._on_selected_lamella_changed()  # returns early with no lamella; still flushes

        assert saves == [experiment], "the pending edit did not land before the switch"

    def test_hiding_the_panel_flushes(self, editor, saves, experiment):
        """Driven through the event rather than through `hide()`.

        Qt only delivers a hideEvent to a widget that was actually visible, and this
        editor's parent chain is never shown here -- so `hide()` is a silent no-op and
        the test would pass with the override deleted. Sending the event is what
        exercises our half; delivering it on a tab switch is Qt's.
        """
        from PyQt5.QtGui import QHideEvent

        editor._save_experiment()
        editor.hideEvent(QHideEvent())

        assert saves == [experiment]

    def test_swapping_the_experiment_flushes_the_previous_one(
        self, editor, saves, experiment, tmp_path
    ):
        """The sharp edge, and the reason the pending write carries its experiment.

        `set_experiment` runs *after* the parent has swapped `.experiment`, so a flush
        that re-read it here would save the incoming experiment and silently drop the
        outgoing one's last edit.
        """
        editor._save_experiment()
        replacement = Experiment(path=str(tmp_path / "next"), name="next")
        editor.parent_widget.experiment = replacement

        editor.set_experiment()

        assert saves == [experiment], (
            "the flush wrote the wrong experiment -- the pending edit belonged to "
            f"{experiment.name}, not {replacement.name}"
        )

    def test_an_explicit_flush_and_the_timer_do_not_both_write(
        self, editor, saves, experiment
    ):
        """One edit is one write however the flush and the timer interleave.

        Two things stand the second write down and only one of them is load-bearing:
        `flush_pending_save` stops the timer, *and* it clears the pending experiment,
        which the timer's own call then finds empty. Deleting the `stop()` alone does
        not break this -- checked -- so what is pinned here is the behaviour, not the
        mechanism. It would still catch a refactor that dropped the empty check, which
        is the one that matters.
        """
        editor._save_experiment()
        editor.flush_pending_save()
        QTest.qWait(_PAST_DEBOUNCE_MS)

        assert saves == [experiment], f"wrote {len(saves)} times for one edit"


class TestTheWindowFlushesToo:
    """Read off the source: `AutoLamellaSingleWindowUI` builds a vispy quad view and a
    napari viewer, neither of which survives `QT_QPA_PLATFORM=offscreen`."""

    def test_a_run_flushes_before_it_starts(self):
        from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
            AutoLamellaSingleWindowUI,
        )

        source = inspect.getsource(AutoLamellaSingleWindowUI._on_run_workflow_clicked)
        flush = source.index("lamella_widget.flush_pending_save()")
        start = source.index("_start_run_workflow_thread")

        assert flush < start, (
            "the run starts before the editor's pending edit is written -- the worker "
            "thread and the debounce timer would both be writing the experiment"
        )

    def test_closing_the_window_flushes(self):
        from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
            AutoLamellaSingleWindowUI,
        )

        source = inspect.getsource(AutoLamellaSingleWindowUI.closeEvent)

        assert "lamella_widget.flush_pending_save()" in source, (
            "closing the window does not flush -- an edit made in the last "
            f"{_SAVE_DEBOUNCE_MS} ms before quitting is lost"
        )


class TestNothingBypassesIt:
    def test_no_editor_handler_saves_directly(self):
        """Ten handlers route through `_save_experiment`; one going around it keeps
        the full cost, and the only symptom is that the editor still feels slow."""
        source = Path(editor_module.__file__).read_text(encoding="utf-8")
        offenders = []
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            if node.func.attr == "save" and isinstance(node.func.value, ast.Attribute):
                if node.func.value.attr == "experiment":
                    offenders.append(node.lineno)

        assert offenders == [], (
            f"{editor_module.__file__} calls experiment.save() directly at lines "
            f"{offenders} -- that bypasses the debounce"
        )

    def test_the_editor_no_longer_rewrites_the_protocol(self):
        """A lamella's task config lives in `experiment.yaml`. `save_protocol=True`
        rewrote `protocol.yaml` on every parameter edit without it having changed.

        Parsed rather than grepped: the comment explaining why the flag went away says
        `save_protocol`, so a substring check finds it in prose and fails on a module
        that is entirely correct. That is the same trap `test_lamella_defect_sync.py`
        documents for commented-out connections -- the parser does not read comments.
        """
        passes_the_flag = [
            node.lineno
            for node in ast.walk(
                ast.parse(Path(editor_module.__file__).read_text(encoding="utf-8"))
            )
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "save"
            and any(kw.arg == "save_protocol" for kw in node.keywords)
        ]

        assert passes_the_flag == [], (
            f"the editor still passes save_protocol at lines {passes_the_flag}"
        )
