"""What each widget renders for a milling progress report (FIB-797).

This decode path has never had a test of any kind -- nothing in `tests/` referenced
`_on_milling_progress` or `milling_progress_signal` before this file. Three widgets each
took the same nested dict apart by hand, and the bugs that produced were all rendering
ones: a message that matched no branch and appeared nowhere, an offset applied seven
times across three files, three different defaults for the same report.

So these assert the **rendered text**, not which branch ran. Asserting on control flow
would have caught none of the above.

Each widget's decode is borrowed onto a host carrying the real Qt controls it writes to.
The widgets themselves need a microscope, a canvas and (for the main window) a whole
`QMainWindow` of tabs; the decode is a method that reads a report and writes to a
progress bar, so what it writes to is what has to be real.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (  # noqa: E402
    AutoLamellaSingleWindowUI,
)
from fibsem.applications.autolamella.ui.fluorescence_coincidence_viewer_widget import (  # noqa: E402
    FluorescenceCoincidenceViewerWidget,
)
from fibsem.milling.progress import (  # noqa: E402
    MillingMessageTracker,
    MillingProgress,
    MillingProgressStatus,
)
from fibsem.ui.widgets.milling_widget import FibsemMillingWidget2  # noqa: E402

# --------------------------------------------------------------------------------------
# One builder per producer, each shaped exactly as that producer builds it. Kept
# separate rather than collapsed into one parametrised helper: the differences between
# them -- which fields a backend tick omits, which a strategy stamps -- are the thing
# under test.
# --------------------------------------------------------------------------------------


def stage_start(stage_name="Rough Mill", current_stage=0, total_stages=3):
    """`MillingTask._mill_stage`, once per stage."""
    return MillingProgress(
        status=MillingProgressStatus.STAGE_STARTED,
        message=None,
        task_id="task-1",
        task_name="Trench",
        stage_name=stage_name,
        current_stage=current_stage,
        total_stages=total_stages,
        start_time=100.0,
    )


def backend_tick(remaining_time=30.0, estimated_time=60.0):
    """A backend's poll loop -- `microscope.py`, `simulator.py`, `tescan.py`. Carries no
    message: the backend has no idea what the strategy driving it calls itself."""
    return MillingProgress(
        status=MillingProgressStatus.STAGE_UPDATE,
        start_time=100.0,
        estimated_time=estimated_time,
        remaining_time=remaining_time,
    )


def strategy_message(stage_name="Rough Mill"):
    """`strategy/standard.py`. This is the report that used to carry no `state` at all,
    so it matched no branch in any of the three consumers and rendered nowhere."""
    return MillingProgress(
        status=MillingProgressStatus.STAGE_UPDATE,
        message=f"Running {stage_name}...",
        stage_name=stage_name,
        start_time=100.0,
        estimated_time=60.0,
    )


def task_finished():
    """`MillingTask.run`'s `finally`, once per task."""
    return MillingProgress(
        status=MillingProgressStatus.TASK_FINISHED,
        task_id="task-1",
        task_name="Trench",
    )


# --------------------------------------------------------------------------------------
# Hosts
# --------------------------------------------------------------------------------------


class _MillingWidgetHost:
    """`FibsemMillingWidget2`'s two bars, plus the real button-state method the decode
    calls on every report."""

    _on_milling_progress = FibsemMillingWidget2._on_milling_progress
    _update_button_states = FibsemMillingWidget2._update_button_states
    is_milling = FibsemMillingWidget2.is_milling

    def __init__(self):
        from PyQt5.QtWidgets import QProgressBar, QPushButton, QWidget

        self.progressBar_milling = QProgressBar()
        self.progressBar_milling_stages = QProgressBar()
        self.pushButton_run_milling = QPushButton()
        self.pushButton_stop_milling = QPushButton()
        self.pushButton_pause_milling = QPushButton()
        self._milling_thread = None
        self._has_stages = True
        self._milling_label = MillingMessageTracker()

        class _Parent:
            pass

        self.parent_widget = _Parent()
        self.parent_widget.config_widget = QWidget()

    def deleteLater(self):
        for widget in (
            self.progressBar_milling,
            self.progressBar_milling_stages,
            self.pushButton_run_milling,
            self.pushButton_stop_milling,
            self.pushButton_pause_milling,
            self.parent_widget.config_widget,
        ):
            widget.deleteLater()


class _MainWindowHost:
    """`AutoLamellaSingleWindowUI`'s status-bar progress bar."""

    _on_milling_progress = AutoLamellaSingleWindowUI._on_milling_progress

    def __init__(self):
        from PyQt5.QtWidgets import QProgressBar

        self.milling_progress_bar = QProgressBar()
        self._milling_label = MillingMessageTracker()

    def deleteLater(self):
        self.milling_progress_bar.deleteLater()


class _CoincidenceHost:
    """`FluorescenceCoincidenceViewerWidget`'s two bars and the controls its stage-start
    branch reveals."""

    _on_milling_progress = FluorescenceCoincidenceViewerWidget._on_milling_progress

    def __init__(self):
        from PyQt5.QtWidgets import (
            QAction,
            QDoubleSpinBox,
            QLabel,
            QProgressBar,
            QPushButton,
            QToolButton,
        )

        self.progressBar_stage = QProgressBar()
        self.progressBar_stages = QProgressBar()
        self.btn_milling = QPushButton()
        self.btn_pause = QToolButton()
        self.btn_supervised = QPushButton()
        self.spin_drop_threshold = QDoubleSpinBox()
        self.label_threshold_chip = QLabel()
        self._act_pause_milling = QAction("Pause Milling")
        self._act_pause_acquisition = QAction("Pause Acquisition")
        self._milling_paused = False
        self._supervised = True
        self._milling_label = MillingMessageTracker()
        self.border_states = []

    def _set_border_state(self, state):
        self.border_states.append(state)

    def deleteLater(self):
        for widget in (
            self.progressBar_stage,
            self.progressBar_stages,
            self.btn_milling,
            self.btn_pause,
            self.btn_supervised,
            self.spin_drop_threshold,
            self.label_threshold_chip,
        ):
            widget.deleteLater()


@pytest.fixture
def milling_widget(qapp):
    host = _MillingWidgetHost()
    yield host
    host.deleteLater()


@pytest.fixture
def main_window(qapp):
    host = _MainWindowHost()
    yield host
    host.deleteLater()


@pytest.fixture
def coincidence(qapp):
    host = _CoincidenceHost()
    yield host
    host.deleteLater()


# --------------------------------------------------------------------------------------
# The offset, in each of the three widgets that used to apply it by hand
# --------------------------------------------------------------------------------------


class TestTheStageCountIsOneBased:
    """`current_stage` is 0-based on the wire. `display_stage` is the one place the
    offset is applied now; it was written out at seven call sites across these three
    files, and the two spellings sat one function apart on the fluorescence runner."""

    def test_the_milling_widget_counts_from_one(self, milling_widget):
        milling_widget._on_milling_progress(
            stage_start(current_stage=0, total_stages=3)
        )
        assert (
            milling_widget.progressBar_milling_stages.format()
            == "Milling Stage: 1/3 - Rough Mill"
        )

    def test_the_main_window_counts_from_one(self, main_window):
        main_window._on_milling_progress(
            stage_start(current_stage=1, total_stages=3, stage_name="Polish")
        )
        assert (
            main_window.milling_progress_bar.toolTip() == "Milling Stage: 2/3 - Polish"
        )

    def test_the_coincidence_viewer_counts_from_one(self, coincidence):
        coincidence._on_milling_progress(stage_start(current_stage=2, total_stages=3))
        assert coincidence.progressBar_stages.format() == "Stage 3/3"


# --------------------------------------------------------------------------------------
# The strategy's message: the feature that was silently dropped
# --------------------------------------------------------------------------------------


class TestTheStrategysWordsReachTheScreen:
    """`strategy/standard.py` emits its `msg` carrying no `state`, so it matched no
    branch in any consumer. The one message a strategy sends today is precisely the one
    that was dropped -- and milling strategies are plugin-loadable, which is why this
    signal keeps a `message` at all."""

    def test_the_milling_widget_shows_it(self, milling_widget):
        milling_widget._on_milling_progress(strategy_message())
        assert "Running Rough Mill..." in milling_widget.progressBar_milling.format()

    def test_the_main_window_shows_it(self, main_window):
        main_window._on_milling_progress(strategy_message())
        assert "Running Rough Mill..." in main_window.milling_progress_bar.format()

    def test_the_coincidence_viewer_shows_it(self, coincidence):
        coincidence._on_milling_progress(strategy_message())
        assert "Running Rough Mill..." in coincidence.progressBar_stage.format()

    def test_it_survives_the_backends_messageless_ticks(self, milling_widget):
        """The stickiness rule, end to end. A *delegating* strategy emits its label once
        and then hands the loop to `microscope.run_milling()`; every tick after that
        comes from a backend that has no idea what the strategy calls itself."""
        milling_widget._on_milling_progress(strategy_message())
        milling_widget._on_milling_progress(backend_tick(remaining_time=15.0))

        rendered = milling_widget.progressBar_milling.format()
        assert "Running Rough Mill..." in rendered
        assert "remaining" in rendered

    def test_a_new_stage_drops_the_previous_stages_words(self, milling_widget):
        milling_widget._on_milling_progress(strategy_message("Rough Mill"))
        milling_widget._on_milling_progress(stage_start("Polish", 1, 3))

        rendered = milling_widget.progressBar_milling.format()
        assert "Rough Mill" not in rendered
        assert rendered == "Preparing: Polish"


# --------------------------------------------------------------------------------------
# The producers this repo does not own
# --------------------------------------------------------------------------------------


def out_of_tree_delegating_strategy():
    """A strategy built against the older contract: nested dict, no `state`, `name`
    rather than `stage_name`, and a `msg` of its own.

    Milling strategies are plugin-loadable, so this is not a historical shape that went
    away when the in-tree producers were flipped -- it is what an installed plugin still
    emits, and `psygnal` hands it to these slots unchanged.
    """
    return {
        "msg": "Running Rough Mill cycle 2...",
        "progress": {
            "started": True,
            "start_time": 100.0,
            "estimated_time": 60.0,
            "name": "Rough Mill",
        },
    }


class TestAnUnknownProducerCannotKillTheApp:
    """The consequence of getting this wrong is not a missing label.

    These slots are queued, and on PyQt5 an exception escaping a queued slot is
    `qFatal`: the process aborts with nothing written to the logfile (FIB-329). An
    `AttributeError` on the first field read would take the milling run with it.
    """

    def test_the_milling_widget_survives_and_renders(self, milling_widget):
        milling_widget._on_milling_progress(out_of_tree_delegating_strategy())
        assert (
            "Running Rough Mill cycle 2..."
            in milling_widget.progressBar_milling.format()
        )

    def test_the_main_window_survives_and_renders(self, main_window):
        main_window._on_milling_progress(out_of_tree_delegating_strategy())
        assert (
            "Running Rough Mill cycle 2..." in main_window.milling_progress_bar.format()
        )

    def test_the_coincidence_viewer_survives_and_renders(self, coincidence):
        coincidence._on_milling_progress(out_of_tree_delegating_strategy())
        assert "Running Rough Mill cycle 2..." in coincidence.progressBar_stage.format()

    @pytest.mark.parametrize(
        "payload",
        [None, "not a dict", 42, {}, {"progress": None}, {"progress": {"state": "?"}}],
    )
    def test_no_payload_at_all_raises_in_any_consumer(
        self, payload, milling_widget, main_window, coincidence
    ):
        """Nothing a producer can emit may escape as an exception -- not a malformed
        dict, not `None`, not a bare string."""
        milling_widget._on_milling_progress(payload)
        main_window._on_milling_progress(payload)
        coincidence._on_milling_progress(payload)

    def test_a_legacy_terminal_still_clears_the_bar(self, milling_widget):
        """The guard has to decode the *outcome* too, or a plugin's run leaves the
        progress bar up forever."""
        milling_widget._on_milling_progress(out_of_tree_delegating_strategy())
        milling_widget._on_milling_progress(
            {"msg": "done", "progress": {"state": "finished", "task_name": "Trench"}}
        )
        assert not milling_widget.progressBar_milling.isVisible()


# --------------------------------------------------------------------------------------
# The countdown
# --------------------------------------------------------------------------------------


class TestTheCountdown:
    def test_the_bar_tracks_the_time_left(self, milling_widget):
        milling_widget._on_milling_progress(
            backend_tick(remaining_time=15.0, estimated_time=60.0)
        )
        assert milling_widget.progressBar_milling.value() == 75
        assert "remaining" in milling_widget.progressBar_milling.format()

    def test_an_estimate_of_zero_does_not_take_the_process_down(self, milling_widget):
        """`milling_widget` divided by `estimated_time` after checking only that it was
        not `None`, unlike the other two consumers. That division runs inside a queued Qt
        slot, and on PyQt5 an exception escaping one of those is `qFatal`: the process
        aborts with nothing written to the logfile (FIB-329)."""
        milling_widget._on_milling_progress(
            backend_tick(remaining_time=0.0, estimated_time=0.0)
        )
        assert milling_widget.progressBar_milling.format()

    def test_a_stage_start_with_no_stages_does_not_take_the_process_down(
        self, milling_widget
    ):
        """Same class: `int(stage / total_stages * 100)` with a total of zero."""
        milling_widget._on_milling_progress(
            stage_start(current_stage=0, total_stages=0)
        )
        assert milling_widget.progressBar_milling_stages.format()


# --------------------------------------------------------------------------------------
# The task ending
# --------------------------------------------------------------------------------------


class TestATaskEnding:
    def test_the_milling_widget_hides_both_bars(self, milling_widget):
        milling_widget._on_milling_progress(stage_start())
        assert milling_widget.progressBar_milling.isVisible()
        milling_widget._on_milling_progress(task_finished())
        assert not milling_widget.progressBar_milling.isVisible()
        assert not milling_widget.progressBar_milling_stages.isVisible()

    def test_the_main_window_hides_its_bar(self, main_window):
        main_window._on_milling_progress(stage_start())
        main_window._on_milling_progress(task_finished())
        assert not main_window.milling_progress_bar.isVisible()

    def test_a_stage_finishing_does_not_hide_the_bar(self, milling_widget):
        """`STAGE_FINISHED` is not terminal. Treating it as one hides the bar after the
        first stage of an N-stage task, and it stays hidden for the rest of the run."""
        milling_widget._on_milling_progress(stage_start())
        milling_widget.progressBar_milling.setVisible(True)
        milling_widget._on_milling_progress(
            MillingProgress(
                MillingProgressStatus.STAGE_FINISHED, stage_name="Rough Mill"
            )
        )
        assert milling_widget.progressBar_milling.isVisible()

    def test_a_cancelled_task_ends_the_run_too(self, main_window):
        """Nothing emits `TASK_CANCELLED` yet -- the producers land in the next PRs --
        but the consumer has to be ready before they do, or a cancelled mill leaves the
        bar up for the rest of the session."""
        main_window._on_milling_progress(stage_start())
        main_window._on_milling_progress(
            MillingProgress(MillingProgressStatus.TASK_CANCELLED)
        )
        assert not main_window.milling_progress_bar.isVisible()


# --------------------------------------------------------------------------------------
# Typed reports, which is what the producers send from the next PR onwards
# --------------------------------------------------------------------------------------


class TestNothingTakesTheProcessDown:
    """Every one of these runs inside a queued Qt slot. On PyQt5 an exception escaping
    one is `qFatal`: the process aborts with nothing written to the logfile (FIB-329),
    so a degenerate report has to render badly rather than abort.

    Typed now that the payload is, which narrows the risk to field *values* -- a zero
    where a divisor is expected, a count with no total, an outcome carrying stage
    fields. Those are what a producer can still get wrong.
    """

    DEGENERATE = [
        MillingProgress(MillingProgressStatus.STAGE_STARTED),
        MillingProgress(
            MillingProgressStatus.STAGE_STARTED, total_stages=0, current_stage=0
        ),
        MillingProgress(
            MillingProgressStatus.STAGE_STARTED, current_stage=7, total_stages=2
        ),
        MillingProgress(MillingProgressStatus.STAGE_UPDATE),
        MillingProgress(
            MillingProgressStatus.STAGE_UPDATE, remaining_time=1.0, estimated_time=0.0
        ),
        MillingProgress(MillingProgressStatus.STAGE_UPDATE, estimated_time=60.0),
        MillingProgress(
            MillingProgressStatus.STAGE_UPDATE, remaining_time=99.0, estimated_time=1.0
        ),
        MillingProgress(MillingProgressStatus.STAGE_FINISHED),
        MillingProgress(MillingProgressStatus.TASK_STARTED),
        MillingProgress(MillingProgressStatus.TASK_CANCELLED, stage_name="Rough Mill"),
        MillingProgress(MillingProgressStatus.TASK_FAILED, error="the column tripped"),
    ]

    @pytest.mark.parametrize("report", DEGENERATE, ids=lambda r: r.status.value)
    def test_the_milling_widget_survives(self, milling_widget, report):
        milling_widget._on_milling_progress(report)

    @pytest.mark.parametrize("report", DEGENERATE, ids=lambda r: r.status.value)
    def test_the_main_window_survives(self, main_window, report):
        main_window._on_milling_progress(report)

    @pytest.mark.parametrize("report", DEGENERATE, ids=lambda r: r.status.value)
    def test_the_coincidence_viewer_survives(self, coincidence, report):
        coincidence._on_milling_progress(report)
