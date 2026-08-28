"""Driving the stage from an overview canvas has to say it is doing something.

Double-clicking either overview drives the stage. The beam tab said nothing at all --
no status, no toast, no cursor change -- so a double-click that started a multi-second
move was indistinguishable from one that did nothing. And because `_may_move` refusals
*do* toast, silence was the state where something was actually happening (FIB-765).

Neither tab reported a *failure*. `_move_worker` caught the exception and logged it, so
a stage that never arrived looked exactly like one that did: on the fluorescence tab the
status line read "Moving to …" for the rest of the session.

There is no progress to report, and these tests pin that too. `safe_absolute_stage_movement`
blocks and emits nothing along the way, so a determinate bar would be inventing a number.
What is available is a busy state and an outcome, and that is what is asserted here.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_click_to_move_says_so.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QObject, pyqtSignal  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.structures import FibsemStagePosition  # noqa: E402
from fibsem.ui.widgets import overview_widget as beam_module  # noqa: E402
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)


class SynchronousWorker(QObject):
    """Stands in for FunctionWorker so a test sees the outcome, not a thread.

    Mirrors the whole contract -- `started`, then `returned` or `errored`, then always
    `finished` -- and catches the exception rather than letting it escape, which is what
    the real worker does. A stub that let it propagate would make the caller's error
    handling untestable, which is how the failure path went unwritten for this long.
    """

    started = pyqtSignal()
    returned = pyqtSignal(object)
    errored = pyqtSignal(object)
    finished = pyqtSignal()

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs

    def start(self):
        self.started.emit()
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as error:
            self.errored.emit(error)
        else:
            self.returned.emit(result)
        finally:
            self.finished.emit()

    def is_alive(self):
        return False


@pytest.fixture(scope="module")
def microscope():
    import fibsem.config as fibsem_config

    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    return scope


@pytest.fixture
def widget(microscope, monkeypatch):
    monkeypatch.setattr(beam_module, "FunctionWorker", SynchronousWorker)
    w = FibsemOverviewWidget(microscope)
    w.resize(900, 700)
    yield w
    w.close()


@pytest.fixture
def said(widget, monkeypatch):
    """Everything the widget puts on screen about a move, in order.

    The status label, which is where both tabs report -- the progress bar is for a run
    with a fraction to show, and a stage move has none. Recorded rather than read back
    at the end so a message that appears and is immediately replaced is still visible
    to a test.
    """
    messages, toasts = [], []
    monkeypatch.setattr(
        widget.label_status, "setText", lambda text: messages.append(text)
    )
    monkeypatch.setattr(
        beam_module.notification_service,
        "show_toast",
        lambda text, level="info", *a, **k: toasts.append((text, level)),
    )
    return {"messages": messages, "toasts": toasts}


def _target(microscope, dx=50e-6):
    base = microscope.get_stage_position()
    return FibsemStagePosition(x=base.x + dx, y=base.y, z=base.z, r=base.r, t=base.t)


class TestASuccessfulMove:
    def test_it_says_it_is_moving(self, widget, microscope, said):
        widget.move_to(_target(microscope))

        assert said["messages"], "the beam tab said nothing at all"
        assert "Moving to" in said["messages"][0]

    def test_the_message_names_where_it_is_going(self, widget, microscope, said):
        widget.move_to(_target(microscope))

        # Not an exact string: `_describe` formats it, and this is about the message
        # carrying the destination rather than about how a position is spelled.
        assert said["messages"][0].strip() != "Moving to …"
        assert len(said["messages"][0]) > len("Moving to …")

    def test_it_shows_no_progress_bar(self, widget, microscope, said, monkeypatch):
        """`safe_absolute_stage_movement` blocks and emits nothing on the way, so there
        is no fraction to show, and this tab's own rule is that the bar carries a run
        while the label carries a thing that merely happened."""
        bar = []
        monkeypatch.setattr(
            widget.progress, "update_progress", lambda info: bar.append(info)
        )

        widget.move_to(_target(microscope))

        assert bar == [], (
            f"a progress bar was driven for a move with no fraction: {bar}"
        )

    def test_it_says_where_the_stage_got_to(self, widget, microscope, said):
        """Not just "done". Where it actually reached, read back by the worker rather
        than assumed from the target -- the same report the fluorescence tab gives."""
        widget.move_to(_target(microscope))

        assert "Moving to" not in said["messages"][-1]
        assert said["messages"][-1].startswith("At ")

    def test_it_does_not_toast_on_success(self, widget, microscope, said):
        """The refusals already toast. A toast per successful move would make the tab
        chatty about the ordinary case and bury the ones that matter."""
        widget.move_to(_target(microscope))

        assert said["toasts"] == []


class TestAFailedMove:
    @pytest.fixture
    def broken(self, widget, monkeypatch):
        def explode(*args, **kwargs):
            raise RuntimeError("stage controller said no")

        monkeypatch.setattr(widget.microscope, "safe_absolute_stage_movement", explode)

    def test_it_says_the_move_failed(self, widget, microscope, said, broken):
        widget.move_to(_target(microscope))

        assert any("Could not move" in m for m in said["messages"])

    def test_it_toasts_the_failure(self, widget, microscope, said, broken):
        """The progress bar lives in the settings column, which may be scrolled away or
        collapsed. A failed stage move should not be discoverable only by looking."""
        widget.move_to(_target(microscope))

        assert said["toasts"], "a failed stage move passed in silence"
        assert said["toasts"][-1][1] == "error"

    def test_the_failure_message_is_not_then_replaced(
        self, widget, microscope, said, broken
    ):
        """`finished` always runs, after `errored`. Reporting success there
        unconditionally would wipe the failure a fraction of a second after putting it
        up."""
        widget.move_to(_target(microscope))

        assert "Could not move" in said["messages"][-1]

    def test_it_still_reads_the_position_back(self, widget, microscope, said, broken):
        """A move that stopped part-way has left the stage somewhere. The read-back is
        in a `finally` so the marker says where it got to, not where it set off from."""
        # Built before the counter goes in: `_target` reads the stage position itself,
        # and counting that read made this pass against a version with no `finally` at
        # all -- it was counting its own setup.
        target = _target(microscope)

        reads = []
        original = widget.microscope.get_stage_position

        def counted(*args, **kwargs):
            reads.append(1)
            return original(*args, **kwargs)

        widget.microscope.get_stage_position = counted
        try:
            widget.move_to(target)
        finally:
            widget.microscope.get_stage_position = original

        assert reads, "the stage position was never re-read after a failed move"

    def test_the_next_move_starts_clean(self, widget, microscope, said):
        """The failure flag must not outlive the failure, or the move after a failed one
        would leave its own message up forever.

        The failing method is swapped by hand rather than with `monkeypatch`: that
        fixture is function-scoped and shared with `said` and `widget`, so undoing it
        mid-test would take the recorders and the worker stub with it.
        """
        original = widget.microscope.safe_absolute_stage_movement

        def explode(*args, **kwargs):
            raise RuntimeError("stage controller said no")

        widget.microscope.safe_absolute_stage_movement = explode
        try:
            widget.move_to(_target(microscope))
        finally:
            widget.microscope.safe_absolute_stage_movement = original
        assert "Could not move" in said["messages"][-1]

        widget.move_to(_target(microscope))

        assert said["messages"][-1].startswith("At ")


class TestARefusedMove:
    def test_a_refusal_says_nothing_about_moving(
        self, widget, microscope, said, monkeypatch
    ):
        """`_may_move` already toasts its own reason. A "Moving to …" that appeared and
        vanished underneath it would be the tab contradicting itself."""
        monkeypatch.setattr(widget, "_may_move", lambda: False)

        widget.move_to(_target(microscope))

        assert said["messages"] == []


# ── the fluorescence tab ─────────────────────────────────────────────────────
#
# It already said "Moving to …" and that half was right. What it never did was take the
# message down: `_move_worker` swallowed the exception, so a stage that never arrived
# left the line reading "Moving to …" for the rest of the session, and one that did
# arrive left it too.


@pytest.fixture(scope="module")
def fm_widget(qapp):
    from fibsem.ui.fm.overview_app import build_microscope
    from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget

    scope = build_microscope()
    scope.fm.objective.insert()
    w = FMOverviewWidget(scope)
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    return w


@pytest.fixture
def fm_synchronous(monkeypatch):
    from fibsem.ui.fm.widgets import fm_overview_widget as fm_module

    monkeypatch.setattr(fm_module, "FunctionWorker", SynchronousWorker)
    return fm_module


def _fm_target(fm_widget, dx=30e-6):
    base = fm_widget.microscope.get_stage_position()
    return FibsemStagePosition(x=base.x + dx, y=base.y, z=base.z, r=base.r, t=base.t)


def test_the_fluorescence_tab_takes_its_message_down(qapp, fm_widget, fm_synchronous):
    """A move that finished should not still read "Moving to …"."""
    fm_widget.move_to(_fm_target(fm_widget))
    qapp.processEvents()

    assert "Moving to" not in fm_widget.status.text()
    assert fm_widget.status.text().strip(), "it went blank instead of reporting"


def test_the_fluorescence_tab_reports_a_failed_move(
    qapp, fm_widget, fm_synchronous, monkeypatch
):
    toasts = []
    monkeypatch.setattr(
        fm_synchronous.notification_service,
        "show_toast",
        lambda text, level="info", *a, **k: toasts.append((text, level)),
    )

    def explode(*args, **kwargs):
        raise RuntimeError("stage controller said no")

    monkeypatch.setattr(fm_widget.microscope, "safe_absolute_stage_movement", explode)

    fm_widget.move_to(_fm_target(fm_widget))
    qapp.processEvents()

    assert "Could not move" in fm_widget.status.text()
    assert toasts and toasts[-1][1] == "error"


def test_both_tabs_report_a_move_the_same_way(
    qapp, widget, fm_widget, fm_synchronous, said
):
    """Same surface, same shape of message, on both tabs.

    These two drifted apart once already -- that drift *is* FIB-765, one tab reporting
    and the other silent -- and the first version of this fix drifted them again by
    putting the beam tab's message on the progress bar and the fluorescence tab's on the
    status label. Same event, two surfaces.

    The position is formatted differently by each tab and that is left alone; what is
    held here is that both say it, in the same place, in the same three shapes.
    """
    fm_before = fm_widget.status.text()

    widget.move_to(_target(widget.microscope))
    fm_widget.move_to(_fm_target(fm_widget))
    qapp.processEvents()

    beam_said = said["messages"]
    assert beam_said[0].startswith("Moving to ")
    assert beam_said[-1].startswith("At ")

    fm_said = fm_widget.status.text()
    assert fm_said != fm_before, "the fluorescence tab said nothing new"
    assert fm_said.startswith("At "), fm_said
