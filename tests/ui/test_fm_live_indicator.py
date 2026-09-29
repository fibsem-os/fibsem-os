"""The FM panel shows that it is streaming, like the beam panels do (FIB-596).

Neither half of this needed new chrome. `FibsemCanvasBase.set_live_badge` gives every
canvas the green border and the "● LIVE" badge, and the quad view has kept the FM under
the view key `"fm"` alongside the two `BeamType`s since it was built -- so
`QuadViewWidget.set_live("fm", True)` already worked.

What stopped short was one type annotation and one caller.
`MicroscopeViewController.set_live` was typed `BeamType`, which read as "beams only" and
was believed, and nothing ever drove it from the FM's own state. The plumbing looked
closed from the outside, which is why the panel that streams was the one panel that
never said so.

Driven from `fm.is_streaming` rather than from a button press or this widget's own
worker: the live stream is the microscope's state, and another tab can start or stop one
(FIB-513).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

import fibsem.config as _fibsem_config
from fibsem import utils
from fibsem.structures import BeamType
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController, QuadViewWidget

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def controller():
    return MicroscopeViewController(view=QuadViewWidget())


def _badge_on(controller, canvas) -> bool:
    return bool(canvas._live_on)


class TestTheFMCanvasCanBeMarkedLive:
    def test_the_badge_and_border_reach_the_fm_panel(self, controller):
        controller.set_fm_live(True)

        assert _badge_on(controller, controller.fm_canvas)
        assert "fm" in controller.widget._live

    def test_it_clears(self, controller):
        controller.set_fm_live(True)
        controller.set_fm_live(False)

        assert not _badge_on(controller, controller.fm_canvas)
        assert "fm" not in controller.widget._live

    def test_it_does_not_touch_the_beam_panels(self, controller):
        controller.set_fm_live(True)

        assert not _badge_on(controller, controller.sem_canvas)
        assert not _badge_on(controller, controller.fib_canvas)

    def test_a_live_beam_and_a_live_fm_coexist(self, controller):
        """They are separate instruments and can both be running. The beam path clears
        its *own* previous beam on a switch (`_set_live_indicator`); nothing should
        clear the FM."""
        controller.set_live(BeamType.ELECTRON, True)
        controller.set_fm_live(True)

        assert _badge_on(controller, controller.sem_canvas)
        assert _badge_on(controller, controller.fm_canvas)

    def test_a_view_without_live_support_is_a_no_op(self):
        """The lamella editor view has no panels to border. `set_live` is guarded for
        it, and so `set_fm_live` must be too."""

        class _ViewWithoutLive:
            sem_canvas = fib_canvas = fm_canvas = None
            fm_widget = None

        controller = MicroscopeViewController.__new__(MicroscopeViewController)
        controller._widget = _ViewWithoutLive()

        controller.set_fm_live(True)  # must not raise


# The simulated compustage, which is the only Demo configuration that has an FM. Shared
# with tests/ui/test_overview_tab_host.py and test_beam_stage_projection.py so they all
# agree on what an Arctis is.
_ARCTIS_CONFIG = os.path.join(
    os.path.dirname(_fibsem_config.__file__), "config", "sim-arctis-configuration.yaml"
)


class TestTheStreamDrivesIt:
    """The half that was missing. `_update_acquisition_button_states` is the one place
    that already asks `fm.is_streaming` and re-derives the whole panel from it, so the
    chrome follows the stream however it was started."""

    @pytest.fixture
    def widget(self, controller):
        from PyQt5.QtWidgets import QWidget

        from fibsem.ui.widgets.fluorescence_control_widget import FMControlWidget

        # A simulated Arctis: `DemoMicroscope` only builds a
        # `SimulatedFluorescenceMicroscope` when `sim.is_compustage` is set, and
        # `FMControlWidget` refuses a microscope without one -- so on a plain Demo
        # session every test in this class errored in setup rather than running
        # (FIB-734).
        microscope, _ = utils.setup_session(
            manufacturer="Demo", config_path=_ARCTIS_CONFIG
        )

        class _Host(QWidget):
            """The chain `_view_controller` walks: parent -> parent_widget -> the UI
            holding `view_controller`."""

            def __init__(self):
                super().__init__()
                self.view_controller = controller
                self.parent_widget = self
                self.microscope = microscope

        host = _Host()
        w = FMControlWidget(microscope=microscope, parent=host)
        yield w
        w.close()

    def test_starting_a_stream_lights_the_panel(self, widget, controller):
        widget.fm.start_acquisition(channel_settings=widget.channel_settings)
        try:
            widget._update_acquisition_button_states()

            assert _badge_on(controller, controller.fm_canvas)
        finally:
            widget.fm.stop_acquisition()

    def test_stopping_it_clears_the_panel(self, widget, controller):
        widget.fm.start_acquisition(channel_settings=widget.channel_settings)
        widget._update_acquisition_button_states()
        assert _badge_on(controller, controller.fm_canvas), "never lit; nothing to clear"

        widget.fm.stop_acquisition()
        widget._update_acquisition_button_states()

        assert not _badge_on(controller, controller.fm_canvas)

    def test_a_still_acquisition_is_not_a_stream(self, widget, controller):
        """`is_acquiring` covers a z-stack, a tileset, an autofocus sweep. None of them
        is the live view, and the badge means the live view."""
        widget.fm.start_acquisition(channel_settings=widget.channel_settings)
        widget._update_acquisition_button_states()
        widget.fm.stop_acquisition()

        widget.fm.set_acquiring(True)  # a z-stack, say, on the same instrument
        try:
            widget._update_acquisition_button_states()

            assert widget.fm.is_acquiring, "the distinction under test is not set up"
            assert not _badge_on(controller, controller.fm_canvas)
        finally:
            widget.fm.set_acquiring(False)

    def test_a_stream_nobody_pressed_a_button_for_still_lights_it(
        self, widget, controller
    ):
        """No direct `_update_acquisition_button_states` call here. The microscope emits
        `acquiring_changed` on start and stop, this widget is subscribed, and that is
        what makes the chrome follow a stream another tab started -- the reason it reads
        the microscope's state instead of tracking its own buttons."""
        widget.fm.start_acquisition(channel_settings=widget.channel_settings)
        try:
            assert _badge_on(controller, controller.fm_canvas)
        finally:
            widget.fm.stop_acquisition()

        assert not _badge_on(controller, controller.fm_canvas)

    def test_teardown_clears_it(self, widget, controller):
        """The canvas outlives the widget, and the host tears down without stopping the
        acquisition first (`AutoLamellaUI.setup_experiment`) -- so a green border would
        be left on a panel with nothing driving it."""
        widget.fm.start_acquisition(channel_settings=widget.channel_settings)
        widget._update_acquisition_button_states()
        assert _badge_on(controller, controller.fm_canvas)

        widget._teardown_connections()

        assert not _badge_on(controller, controller.fm_canvas)
