"""Three questions about the FM, asked in one place (FIB-441, FIB-513).

    is_streaming    the live view is on
    is_acquiring    anything is running: the stream, or a claimed operation
    is_interactive  the user may drive the hardware

They are not interchangeable, and every bug here came from asking the wrong one. A
tileset moves the stage between every pair of tiles with no stream running, so
`is_streaming` reads idle for most of somebody else's run -- which is how a live stream
came to be startable mid-tileset, and how the objective panel came to stay live against
a runner that was stepping the objective itself.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QDialog

from fibsem.fm.microscope import FluorescenceMicroscope
from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def widget(qapp):
    from fibsem.ui.fm.overview_app import build_microscope

    widget = FMOverviewWidget(build_microscope())
    # `build_microscope` leaves the objective retracted, which `acquire` refuses
    # (FIB-417) before it ever reaches the busy flag this file is about. Without
    # this, the tests that assert the flag is *given back* would pass on a run that
    # never took it.
    widget.fm.objective.insert()
    return widget


@pytest.fixture()
def accept_dialog(monkeypatch):
    """Confirm the run, and count how many times the dialog was reached.

    Patched in every test that calls `acquire`, refusal cases included: the dialog is
    modal, so a guard that stopped working would hang the suite rather than fail it.
    """
    from fibsem.ui.fm.widgets import fm_overview_widget as module

    opened = []

    class _Dialog:
        def __init__(self, **kwargs):
            opened.append(kwargs)

        def exec_(self):
            return QDialog.Accepted

    monkeypatch.setattr(module, "FMOverviewConfirmationDialog", _Dialog)
    return opened


@pytest.fixture()
def no_worker(monkeypatch):
    """Stop `acquire` short of actually running a tileset.

    The FM is marked on the GUI thread before the worker starts, which is the part under
    test; running the acquisition would only add simulated stage travel to each of these.
    """
    from fibsem.ui.fm.widgets import fm_overview_widget as module

    class _Worker:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def is_alive(self):
            return True

    monkeypatch.setattr(module, "FunctionWorker", _Worker)


class TestTheThreeQuestions:
    @pytest.fixture()
    def fm(self):
        return FluorescenceMicroscope()

    def test_an_idle_microscope(self, fm):
        assert (fm.is_streaming, fm.is_acquiring, fm.is_interactive) == (
            False,
            False,
            True,
        )
        assert fm.acquiring_reason == ""

    def test_a_claimed_operation(self, fm):
        """A tileset, a z-stack, an autofocus sweep. Not a stream, and the hardware is
        being driven by whatever claimed it."""
        fm.set_acquiring(True, "overview acquisition")

        assert (fm.is_streaming, fm.is_acquiring, fm.is_interactive) == (
            False,
            True,
            False,
        )
        assert fm.acquiring_reason == "overview acquisition"

    def test_a_live_stream(self, fm):
        """The case the whole split exists for: acquiring, but the user may still focus
        and tune exposure, because watching while adjusting is the point of live view."""
        fm.start_acquisition()
        try:
            assert (fm.is_streaming, fm.is_acquiring, fm.is_interactive) == (
                True,
                True,
                True,
            )
            assert fm.acquiring_reason == "live acquisition"
        finally:
            fm.stop_acquisition()

        assert (fm.is_streaming, fm.is_acquiring, fm.is_interactive) == (
            False,
            False,
            True,
        )

    def test_finishing_gives_it_back(self, fm):
        fm.set_acquiring(True, "overview acquisition")
        fm.set_acquiring(False)

        assert fm.is_acquiring is False
        assert fm.is_interactive is True
        assert fm.acquiring_reason == ""

    def test_it_announces_itself(self, fm):
        """So a widget that did not start it can grey out, not only refuse the click."""
        seen = []
        fm.acquiring_changed.connect(seen.append)

        fm.set_acquiring(True, "overview acquisition")
        fm.set_acquiring(False)

        assert seen == [True, False]

    def test_a_stream_announces_itself_too(self, fm):
        """It does not go through `set_acquiring` -- it reports through its thread -- so
        `start_acquisition` has to raise the signal itself."""
        seen = []
        fm.acquiring_changed.connect(seen.append)

        fm.start_acquisition()
        try:
            assert seen == [True]
        finally:
            fm.stop_acquisition()
        assert seen[-1] is False


class TestTheOverviewMarksTheInstrument:
    def test_a_run_marks_the_microscope(self, widget, accept_dialog, no_worker):
        widget.acquire()

        assert widget.fm.is_acquiring is True
        assert widget.fm.acquiring_reason == "overview acquisition"

    def test_a_run_is_not_a_stream(self, widget, accept_dialog, no_worker):
        """The distinction the hazard rests on: a widget watching `is_streaming` sees an
        idle microscope right through somebody else's tileset."""
        widget.acquire()

        assert widget.fm.is_streaming is False
        assert widget.fm.is_interactive is False

    def test_the_worker_gives_it_back(self, widget, accept_dialog, no_worker):
        widget.acquire()
        widget._runner = _RunnerStub()

        widget._acquire_worker()

        assert widget.fm.is_acquiring is False

    def test_a_failed_run_gives_it_back(self, widget, accept_dialog, no_worker):
        """A run that dies holding the FM locks out every FM widget for the rest of the
        session, with nothing on screen to say why."""
        widget.acquire()
        widget._runner = _RunnerStub(error=RuntimeError("stage refused"))

        widget._acquire_worker()

        assert widget.fm.is_acquiring is False

    def test_a_cancelled_run_gives_it_back(self, widget, accept_dialog, no_worker):
        from fibsem.cancellation import OperationCancelledError

        widget.acquire()
        widget._runner = _RunnerStub(error=OperationCancelledError())

        widget._acquire_worker()

        assert widget.fm.is_acquiring is False


class TestTheOverviewRespectsTheInstrument:
    def test_it_will_not_start_while_something_else_is_running(
        self, widget, accept_dialog, no_worker
    ):
        """What `is_streaming` could not answer: a z-stack in the control widget is not
        a stream, so the old guard saw an idle microscope and started a tileset straight
        through it."""
        widget.fm.set_acquiring(True, "z-stack")

        assert widget.fm.is_streaming is False  # the flag the old guard asked
        widget.acquire()

        assert accept_dialog == [], "asked the user to confirm a run it then refused"

    def test_it_starts_once_the_instrument_is_free(self, widget, accept_dialog, no_worker):
        widget.fm.set_acquiring(True, "z-stack")
        widget.fm.set_acquiring(False)

        widget.acquire()

        assert len(accept_dialog) == 1


class TestTheAcquireButtonFollowsTheInstrument:
    def test_it_greys_out_when_something_else_starts(self, widget, qapp):
        assert widget.button_acquire.isEnabled() is True

        widget.fm.set_acquiring(True, "z-stack")
        qapp.processEvents()

        assert widget.button_acquire.isEnabled() is False

    def test_it_comes_back_when_that_finishes(self, widget, qapp):
        widget.fm.set_acquiring(True, "z-stack")
        qapp.processEvents()
        assert widget.button_acquire.isEnabled() is False, "nothing to come back from"

        widget.fm.set_acquiring(False)
        qapp.processEvents()

        assert widget.button_acquire.isEnabled() is True

    def test_a_live_stream_greys_it_out_too(self, widget, qapp):
        widget.fm.start_acquisition()
        try:
            qapp.processEvents()
            assert widget.button_acquire.isEnabled() is False
        finally:
            widget.fm.stop_acquisition()
        qapp.processEvents()
        assert widget.button_acquire.isEnabled() is True


class TestTheControlWidgetGuard:
    """`FMControlWidget`, the other half of the pair.

    Built without `__init__`. It takes a `napari.Viewer`, and constructing one under the
    offscreen platform segfaults this process (measured: SIGSEGV, no OpenGL) -- which is
    why nothing else in the suite instantiates it either. The guard itself reads one
    attribute and no widgets, so it is testable; that it is reached from each entry
    point is covered structurally below.
    """

    @staticmethod
    def _widget(fm):
        from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget

        widget = FMControlWidget.__new__(FMControlWidget)
        widget.fm = fm
        widget._acquisition_thread = None
        return widget

    @pytest.fixture()
    def fm(self):
        return FluorescenceMicroscope()

    def test_an_idle_instrument_is_not_refused(self, fm):
        assert self._widget(fm)._refuse_to_start("live acquisition") is False

    def test_an_overview_elsewhere_is_refused(self, fm):
        """The case the old per-widget union could not see."""
        fm.set_acquiring(True, "overview acquisition")

        assert fm.is_streaming is False  # the flag the old guard asked
        assert self._widget(fm)._refuse_to_start("live acquisition") is True

    def test_its_own_worker_is_still_refused(self, fm):
        """One question replacing a union must not drop what the union knew. This widget
        marks its own work through `set_acquiring`, so the microscope answers for it."""
        widget = self._widget(fm)
        fm.set_acquiring(True, "z-stack")
        widget._acquisition_thread = _AliveThread()

        assert widget._refuse_to_start("image acquisition") is True

    def test_cancel_asks_about_its_own_thread(self, fm):
        """Not the microscope: `cancel_acquisition` joins `_acquisition_thread`, so on
        somebody else's run the old `is_acquiring` union sent it to `None.join()`."""
        widget = self._widget(fm)
        fm.set_acquiring(True, "overview acquisition")

        assert widget._has_worker is False

        widget._acquisition_thread = _AliveThread()
        assert widget._has_worker is True


class TestTheControlWidgetWiring:
    """That the questions are asked where they should be.

    Structural, because the widget cannot be instantiated here. These state the
    invariants a new acquisition path would have to keep.
    """

    @staticmethod
    def _source():
        from pathlib import Path

        import fibsem.ui.widgets.fluorescence_control_widget as module

        return Path(module.__file__).read_text(encoding="utf-8")

    @classmethod
    def _functions(cls):
        import ast

        tree = ast.parse(cls._source())
        return {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }

    @staticmethod
    def _calls(node):
        import ast

        return {
            child.func.attr
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
        }

    @pytest.mark.parametrize(
        "name", ["start_acquisition", "acquire_image", "run_autofocus"]
    )
    def test_every_way_in_asks_first(self, name):
        assert "_refuse_to_start" in self._calls(self._functions()[name])

    @pytest.mark.parametrize("name", ["acquire_image", "run_autofocus"])
    def test_every_worker_is_marked_for(self, name):
        assert "set_acquiring" in self._calls(self._functions()[name])

    @pytest.mark.parametrize(
        "name", ["_image_acquistion_worker", "_autofocus_worker"]
    )
    def test_every_worker_clears_on_the_way_out(self, name):
        """In the `finally`, not the happy path: a worker that raises still holds it."""
        import ast

        node = self._functions()[name]
        cleared = any(
            "set_acquiring"
            in self._calls(ast.Module(body=block.finalbody, type_ignores=[]))
            for block in ast.walk(node)
            if isinstance(block, ast.Try)
        )
        assert cleared

    def test_it_refuses_a_pose_the_fm_cannot_see_from(self):
        """`start_acquisition` had this check and the other two did not, so an image, a
        z-stack or an autofocus sweep could be started from a beam pose with nothing but
        the button stopping it."""
        from fibsem.fm.microscope import FluorescenceMicroscope

        fm = FluorescenceMicroscope()
        fm._allow_unknown_orientations = False  # the escape hatch is on by default
        fm.valid_orientations = []  # so no pose is a valid one
        fm.parent = _StageAt("SEM")
        widget = TestTheControlWidgetGuard._widget(fm)
        widget.microscope = fm.parent

        assert fm.is_acquiring is False, "refused for the pose, not for being busy"
        assert widget._refuse_to_start("image acquisition") is True

    def test_every_control_is_set_on_every_path(self):
        """`_update_acquisition_button_states` returned early on a bad pose, setting
        three buttons and skipping the rest -- so the objective panel and the channel
        settings kept whatever they were last set to, including enabled during somebody
        else's tileset. Dead as shipped, since `ALLOW_UNKNOWN_ORIENTATIONS` answers that
        check yes unconditionally, which is why nobody noticed.
        """
        import ast

        node = self._functions()["_update_acquisition_button_states"]
        assert not [n for n in ast.walk(node) if isinstance(n, ast.Return)]

    def test_the_hardware_controls_ask_whether_the_user_may_drive(self):
        """The bug FIB-513 closes. The objective panel gated on `is_acquisition_active`
        -- this widget's own work -- so another tab's tileset left the position spinbox
        live while `FMTiledAcquisitionRunner` was stepping the objective per z-level.

        Not `any_acquisition_active` either: that includes the live stream, and focusing
        during live view is what the panel is for.
        """
        source = self._source()
        for control in ("objectiveControlWidget", "channelSettingsWidget"):
            assert f"self.{control}.setEnabled(may_touch)" in source, control
        assert "interactive = self.fm.is_interactive" in source
        assert "may_touch = posed and interactive" in source


class _StageAt:
    """The parent microscope, for the two orientation questions the FM forwards."""

    def __init__(self, orientation: str):
        self._orientation = orientation
        self.stage_is_compustage = True

    def get_stage_orientation(self, stage_position=None) -> str:
        return self._orientation


class _AliveThread:
    """A worker that has not finished."""

    def is_alive(self) -> bool:
        return True


class _RunnerStub:
    """Stands in for `FMTiledAcquisitionRunner` so a run can end three ways."""

    def __init__(self, error=None):
        self.error = error

    def run_and_stitch(self):
        if self.error is not None:
            raise self.error
        import numpy as np

        from fibsem.fm.structures import FluorescenceImage

        return FluorescenceImage(data=np.zeros((4, 4), dtype=np.uint16))
