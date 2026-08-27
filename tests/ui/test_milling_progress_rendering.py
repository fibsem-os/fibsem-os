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
    MillingStatus,
)
from fibsem.ui.widgets.milling_widget import FibsemMillingWidget2  # noqa: E402

# --------------------------------------------------------------------------------------
# The payloads the producers actually build today. Written out rather than constructed
# through a helper: these are the wire format, and a helper that drifts from it is a
# suite that passes while the real payload stops decoding.
# --------------------------------------------------------------------------------------


def legacy_stage_start(stage_name="Rough Mill", current_stage=0, total_stages=3):
    """`MillingTask._mill_stage`, once per stage."""
    return {
        "msg": f"Preparing: {stage_name}",
        "progress": {
            "state": "start",
            "start_time": 100.0,
            "current_stage": current_stage,
            "total_stages": total_stages,
            "task_id": "task-1",
            "task_name": "Trench",
            "stage_name": stage_name,
        },
    }


def legacy_update(remaining_time=30.0, estimated_time=60.0):
    """A backend's poll loop -- `microscope.py`, `simulator.py`, `tescan.py`."""
    return {
        "progress": {
            "state": "update",
            "start_time": 100.0,
            "estimated_time": estimated_time,
            "remaining_time": remaining_time,
        }
    }


def legacy_strategy_message(stage_name="Rough Mill"):
    """`strategy/standard.py`. Carries **no** `state`, so it matched no branch in any of
    the three consumers and its message rendered nowhere."""
    return {
        "msg": f"Running {stage_name}...",
        "progress": {
            "started": True,
            "start_time": 100.0,
            "estimated_time": 60.0,
            "name": stage_name,
        },
    }


def legacy_task_finished():
    """`MillingTask.run`'s `finally`, once per task."""
    return {
        "msg": "Finished Milling Task: Trench. Restoring Imaging Conditions...",
        "progress": {"state": "finished", "task_id": "task-1", "task_name": "Trench"},
    }


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
            legacy_stage_start(current_stage=0, total_stages=3)
        )
        assert (
            milling_widget.progressBar_milling_stages.format()
            == "Milling Stage: 1/3 - Rough Mill"
        )

    def test_the_main_window_counts_from_one(self, main_window):
        main_window._on_milling_progress(
            legacy_stage_start(current_stage=1, total_stages=3, stage_name="Polish")
        )
        assert (
            main_window.milling_progress_bar.toolTip() == "Milling Stage: 2/3 - Polish"
        )

    def test_the_coincidence_viewer_counts_from_one(self, coincidence):
        coincidence._on_milling_progress(
            legacy_stage_start(current_stage=2, total_stages=3)
        )
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
        milling_widget._on_milling_progress(legacy_strategy_message())
        assert "Running Rough Mill..." in milling_widget.progressBar_milling.format()

    def test_the_main_window_shows_it(self, main_window):
        main_window._on_milling_progress(legacy_strategy_message())
        assert "Running Rough Mill..." in main_window.milling_progress_bar.format()

    def test_the_coincidence_viewer_shows_it(self, coincidence):
        coincidence._on_milling_progress(legacy_strategy_message())
        assert "Running Rough Mill..." in coincidence.progressBar_stage.format()

    def test_it_survives_the_backends_messageless_ticks(self, milling_widget):
        """The stickiness rule, end to end. A *delegating* strategy emits its label once
        and then hands the loop to `microscope.run_milling()`; every tick after that
        comes from a backend that has no idea what the strategy calls itself."""
        milling_widget._on_milling_progress(legacy_strategy_message())
        milling_widget._on_milling_progress(legacy_update(remaining_time=15.0))

        rendered = milling_widget.progressBar_milling.format()
        assert "Running Rough Mill..." in rendered
        assert "remaining" in rendered

    def test_a_new_stage_drops_the_previous_stages_words(self, milling_widget):
        milling_widget._on_milling_progress(legacy_strategy_message("Rough Mill"))
        milling_widget._on_milling_progress(legacy_stage_start("Polish", 1, 3))

        rendered = milling_widget.progressBar_milling.format()
        assert "Rough Mill" not in rendered
        assert rendered == "Preparing: Polish"


# --------------------------------------------------------------------------------------
# The countdown
# --------------------------------------------------------------------------------------


class TestTheCountdown:
    def test_the_bar_tracks_the_time_left(self, milling_widget):
        milling_widget._on_milling_progress(
            legacy_update(remaining_time=15.0, estimated_time=60.0)
        )
        assert milling_widget.progressBar_milling.value() == 75
        assert "remaining" in milling_widget.progressBar_milling.format()

    def test_an_estimate_of_zero_does_not_take_the_process_down(self, milling_widget):
        """`milling_widget` divided by `estimated_time` after checking only that it was
        not `None`, unlike the other two consumers. That division runs inside a queued Qt
        slot, and on PyQt5 an exception escaping one of those is `qFatal`: the process
        aborts with nothing written to the logfile (FIB-329)."""
        milling_widget._on_milling_progress(
            legacy_update(remaining_time=0.0, estimated_time=0.0)
        )
        assert milling_widget.progressBar_milling.format()

    def test_a_stage_start_with_no_stages_does_not_take_the_process_down(
        self, milling_widget
    ):
        """Same class: `int(stage / total_stages * 100)` with a total of zero."""
        milling_widget._on_milling_progress(
            legacy_stage_start(current_stage=0, total_stages=0)
        )
        assert milling_widget.progressBar_milling_stages.format()


# --------------------------------------------------------------------------------------
# The task ending
# --------------------------------------------------------------------------------------


class TestATaskEnding:
    def test_the_milling_widget_hides_both_bars(self, milling_widget):
        milling_widget._on_milling_progress(legacy_stage_start())
        assert milling_widget.progressBar_milling.isVisible()
        milling_widget._on_milling_progress(legacy_task_finished())
        assert not milling_widget.progressBar_milling.isVisible()
        assert not milling_widget.progressBar_milling_stages.isVisible()

    def test_the_main_window_hides_its_bar(self, main_window):
        main_window._on_milling_progress(legacy_stage_start())
        main_window._on_milling_progress(legacy_task_finished())
        assert not main_window.milling_progress_bar.isVisible()

    def test_a_stage_finishing_does_not_hide_the_bar(self, milling_widget):
        """`STAGE_FINISHED` is not terminal. Treating it as one hides the bar after the
        first stage of an N-stage task, and it stays hidden for the rest of the run."""
        milling_widget._on_milling_progress(legacy_stage_start())
        milling_widget.progressBar_milling.setVisible(True)
        milling_widget._on_milling_progress(
            MillingProgress(MillingStatus.STAGE_FINISHED, stage_name="Rough Mill")
        )
        assert milling_widget.progressBar_milling.isVisible()

    def test_a_cancelled_task_ends_the_run_too(self, main_window):
        """Nothing emits `TASK_CANCELLED` yet -- the producers land in the next PRs --
        but the consumer has to be ready before they do, or a cancelled mill leaves the
        bar up for the rest of the session."""
        main_window._on_milling_progress(legacy_stage_start())
        main_window._on_milling_progress(MillingProgress(MillingStatus.TASK_CANCELLED))
        assert not main_window.milling_progress_bar.isVisible()


# --------------------------------------------------------------------------------------
# Typed reports, which is what the producers send from the next PR onwards
# --------------------------------------------------------------------------------------


class TestATypedReportRendersTheSame:
    """The consumers are dual-tolerant for the length of the migration: a dict and the
    typed report it decodes to must render identically, or flipping a producer changes
    what the user sees."""

    @pytest.mark.parametrize(
        "payload",
        [legacy_stage_start(), legacy_update(), legacy_strategy_message()],
    )
    def test_a_dict_and_its_decoded_report_render_the_same(self, qapp, payload):
        from_dict = _MillingWidgetHost()
        from_typed = _MillingWidgetHost()
        try:
            from_dict._on_milling_progress(payload)
            from_typed._on_milling_progress(MillingProgress.from_payload(payload))

            assert (
                from_dict.progressBar_milling.format()
                == from_typed.progressBar_milling.format()
            )
            assert (
                from_dict.progressBar_milling_stages.format()
                == from_typed.progressBar_milling_stages.format()
            )
        finally:
            from_dict.deleteLater()
            from_typed.deleteLater()


class TestNothingTakesTheProcessDown:
    """Every one of these runs inside a queued Qt slot. On PyQt5 an exception escaping
    one is `qFatal`, so a malformed payload has to render badly rather than abort."""

    @pytest.mark.parametrize(
        "payload",
        [
            None,
            {},
            "not a dict",
            {"progress": None},
            {"progress": {"state": "start", "total_stages": 0}},
            {"progress": {"state": "update", "estimated_time": 0}},
            {"progress": {"state": "update", "milling_state": "NOT_A_STATE"}},
        ],
    )
    def test_the_milling_widget_survives(self, milling_widget, payload):
        milling_widget._on_milling_progress(payload)

    @pytest.mark.parametrize(
        "payload",
        [None, {}, "not a dict", {"progress": {"state": "start", "total_stages": 0}}],
    )
    def test_the_main_window_survives(self, main_window, payload):
        main_window._on_milling_progress(payload)

    @pytest.mark.parametrize(
        "payload",
        [None, {}, "not a dict", {"progress": {"state": "start", "total_stages": 0}}],
    )
    def test_the_coincidence_viewer_survives(self, coincidence, payload):
        coincidence._on_milling_progress(payload)
