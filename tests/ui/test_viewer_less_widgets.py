"""The image and movement widgets drive the quad-view canvas (FIB-406 / FIB-407).

Both widgets carried two display paths while the tabs migrated one at a time, selected
per instance by whether the host supplied a ``viewer`` or a ``view_controller``. Every
host is viewer-less now and the napari halves are deleted, so the tests that pinned
*which branch ran* went with them — what is left pins the canvas behaviour itself.

The failure mode worth guarding is still a silent one: get the wiring wrong and the
widget constructs, acquires, and simply draws nothing.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python tests/ui/test_viewer_less_widgets.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

# CI installs `.[test]`, not `.[ui]`, so PyQt5 is absent there. Without this the
# module-level import below turns a skip into a collection error.
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QWidget

from fibsem import utils
from fibsem.structures import BeamType, FibsemImage, FibsemRectangle
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController
from fibsem.ui.widgets.stage_control_widget import StageControlWidget

_app = QApplication.instance() or QApplication(sys.argv)

_RESOLUTION = (768, 512)  # (x, y) — non-square, so an axis mix-up shows up
_HFW = 80e-6

_microscope = None
_settings = None


def _session():
    """One Demo microscope for the whole file — setup_session is slow."""
    global _microscope, _settings
    if _microscope is None:
        _microscope, _settings = utils.setup_session(manufacturer="Demo")
    return _microscope, _settings


def _image(beam: BeamType = BeamType.ELECTRON) -> FibsemImage:
    image = FibsemImage.generate_blank_image(
        resolution=_RESOLUTION, hfw=_HFW, random=True
    )
    image.metadata.image_settings.beam_type = beam
    return image


# --- hosts -------------------------------------------------------------------


class _CanvasHost(QWidget):
    """Stands in for standalone FibsemUI: a view_controller, no viewer."""

    def __init__(self) -> None:
        super().__init__()
        self.view_controller = MicroscopeViewController(parent=self)


class _NestedCanvasHost(QWidget):
    """Stands in for AutoLamella's shape: the controller lives one level up, on
    ``parent_widget``. Both hosts must resolve, or the Microscope tab silently keeps
    drawing nothing after Phase 4b flips it."""

    def __init__(self, outer) -> None:
        super().__init__()
        self.parent_widget = outer


def _image_widget(host) -> FibsemImageSettingsWidget:
    _, settings = _session()
    widget = FibsemImageSettingsWidget(
        microscope=_session()[0], image_settings=settings.image, parent=host
    )
    host.image_widget = widget
    return widget


def _movement_widget(host) -> FibsemMovementWidget:
    """Both real hosts publish the widget as ``movement_widget``, and the image widget
    reaches back through it when interactions are toggled — so a host that omits it
    passes tests the app would fail."""
    widget = FibsemMovementWidget(microscope=_session()[0], parent=host)
    host.movement_widget = widget
    return widget


def _control_widget(host) -> StageControlWidget:
    """The half that moves the stage (FIB-783). Built through the tab rather than
    directly, because the tab is what wires it to the form and the saved positions --
    a hand-built control widget would pass tests the application would fail."""
    return _movement_widget(host).control_widget


# --- the controller is found -------------------------------------------------


def test_host_with_a_controller_resolves_it():
    host = _CanvasHost()
    widget = _image_widget(host)
    assert widget._view_controller() is host.view_controller


def test_controller_resolves_through_parent_widget():
    """AutoLamella nests the controller one level up. A resolver that only checked the
    direct parent would leave the Microscope tab blank after 4b, with no error."""
    outer = _CanvasHost()
    host = _NestedCanvasHost(outer)
    widget = _image_widget(host)
    assert widget._view_controller() is outer.view_controller


# --- images actually reach the display ---------------------------------------


def test_acquired_image_lands_on_its_own_beam_canvas():
    host = _CanvasHost()
    widget = _image_widget(host)
    sem_canvas = host.view_controller.get_canvas(BeamType.ELECTRON)
    fib_canvas = host.view_controller.get_canvas(BeamType.ION)
    assert sem_canvas._content_extent() is None, "canvas should start empty"

    widget._on_acquire(_image(BeamType.ELECTRON))
    assert sem_canvas._content_extent() is not None, (
        "SEM image did not reach the canvas"
    )
    assert fib_canvas._content_extent() is None, "SEM image leaked onto the FIB canvas"

    widget._on_acquire(_image(BeamType.ION))
    assert fib_canvas._content_extent() is not None, (
        "FIB image did not reach the canvas"
    )


def test_one_shot_acquisition_rebinds_the_image_references():
    """A workflow one-shot (post-mill inspect, spot burn) emits on the acquisition
    signals with no live stream running. The references must follow the canvas:
    an ``is_acquiring`` gate here left ``ib_image`` at the pre-mill frame, so the
    agent server's /app/images served a stale image through the inspect prompt."""
    microscope, _ = _session()
    host = _CanvasHost()
    widget = _image_widget(host)
    assert not microscope.is_acquiring, "this test pins the non-streaming path"

    sem, fib = _image(BeamType.ELECTRON), _image(BeamType.ION)
    widget._on_acquire(sem)
    widget._on_acquire(fib)
    assert widget.eb_image is sem
    assert widget.ib_image is fib


def test_no_blank_placeholder_is_seeded():
    """napari seeded two blank images at construction; the canvas shows "No image"
    instead. Seeding would put a grey rectangle up that looks like a failed acquisition."""
    host = _CanvasHost()
    _image_widget(host)
    assert host.view_controller.get_canvas(BeamType.ELECTRON)._content_extent() is None
    assert host.view_controller.get_canvas(BeamType.ION)._content_extent() is None


# --- the alignment-area contract ---------------------------------------------


def test_alignment_area_round_trips_through_the_model():
    host = _CanvasHost()
    widget = _image_widget(host)
    widget.toggle_alignment_area(FibsemRectangle(0.3, 0.3, 0.4, 0.4), editable=True)
    got = widget.get_alignment_area()
    assert got is not None
    assert (got.left, got.top, got.width, got.height) == (0.3, 0.3, 0.4, 0.4)


def test_clearing_the_alignment_area_keeps_its_value():
    """``update_alignment_area_ui`` sends "clear" and then reads ``get_alignment_area()``
    straight after. A clear that discarded the rect would hand the workflow None mid-run."""
    host = _CanvasHost()
    widget = _image_widget(host)
    widget.toggle_alignment_area(FibsemRectangle(0.3, 0.3, 0.4, 0.4), editable=True)
    widget.clear_alignment_area()
    assert widget.get_alignment_area() is not None, "clear discarded the alignment area"


def test_alignment_edit_forwards_to_the_existing_signal():
    """A canvas drag must drive the widget's existing validation path."""
    host = _CanvasHost()
    widget = _image_widget(host)
    seen = []
    widget.alignment_area_updated.connect(seen.append)
    widget.toggle_alignment_area(FibsemRectangle(0.3, 0.3, 0.4, 0.4), editable=True)
    edited = FibsemRectangle(0.1, 0.1, 0.2, 0.2)
    host.view_controller.overlay_edited.emit(BeamType.ION, "alignment", edited)
    assert seen and seen[-1] is edited, f"alignment edit not forwarded: {seen}"


def test_unrelated_overlay_edits_are_ignored():
    host = _CanvasHost()
    widget = _image_widget(host)
    seen = []
    widget.alignment_area_updated.connect(seen.append)
    widget.toggle_alignment_area(FibsemRectangle(0.3, 0.3, 0.4, 0.4), editable=True)
    host.view_controller.overlay_edited.emit(BeamType.ION, "milling", object())
    assert not seen, "a non-alignment overlay edit was forwarded as an alignment change"


# --- working-distance Shift+scroll -------------------------------------------


def _sem_beam_settings(widget):
    return widget.dual_beam_widget.sem_widget.beam_settings_widget


def test_shift_scroll_nudges_working_distance_and_plain_scroll_does_not():
    host = _CanvasHost()
    widget = _image_widget(host)
    assert len(widget._wd_scroll_connections) == 2, (
        "WD scroll not wired to both canvases"
    )

    settings_widget = _sem_beam_settings(widget)
    spinbox = settings_widget.working_distance_spinbox
    start = spinbox.value()

    host.view_controller.sem_canvas.canvas_scrolled.emit(0.0, 0.0, 1, set())
    assert spinbox.value() == start, "plain scroll changed the working distance"

    host.view_controller.sem_canvas.canvas_scrolled.emit(0.0, 0.0, 1, {"Shift"})
    assert spinbox.value() > start, "Shift+scroll did not nudge the working distance"


def test_working_distance_spinbox_resolves_the_scroll_step():
    """The step is 1 um; at 2 decimals (0.01 mm) a notch would round away to nothing."""
    from fibsem.ui.widgets.beam_settings_widget import WD_WHEEL_STEP_MM

    widget = _image_widget(_CanvasHost())
    decimals = _sem_beam_settings(widget).working_distance_spinbox.decimals()
    assert 10**-decimals <= WD_WHEEL_STEP_MM, (
        f"spinbox resolves {10**-decimals} mm but the scroll step is {WD_WHEEL_STEP_MM} mm"
    )


# --- quad-view selection <-> beam radio --------------------------------------


def test_selecting_a_canvas_checks_its_beam_radio():
    host = _CanvasHost()
    widget = _image_widget(host)
    host.view_controller.widget.set_selected(BeamType.ION)
    assert widget.dual_beam_widget.fib_radio.isChecked()
    host.view_controller.widget.set_selected(BeamType.ELECTRON)
    assert widget.dual_beam_widget.sem_radio.isChecked()


def test_checking_a_beam_radio_selects_its_canvas():
    host = _CanvasHost()
    widget = _image_widget(host)
    widget.dual_beam_widget.fib_radio.setChecked(True)
    assert host.view_controller.selected_view is BeamType.ION
    widget.dual_beam_widget.sem_radio.setChecked(True)
    assert host.view_controller.selected_view is BeamType.ELECTRON


def test_selecting_the_fm_view_leaves_the_beam_radios_alone():
    """There is no FM radio. Forcing one of the beam radios would misreport which beam
    the Acquire buttons are about to use."""
    host = _CanvasHost()
    widget = _image_widget(host)
    widget.dual_beam_widget.fib_radio.setChecked(True)
    widget._on_view_selected("fm")
    assert widget.dual_beam_widget.fib_radio.isChecked(), (
        "FM selection moved the beam radio"
    )


# --- movement: double-click input --------------------------------------------


def test_movement_binds_double_click_to_both_canvases():
    host = _CanvasHost()
    _image_widget(host)
    movement = _movement_widget(host)
    assert len(movement.control_widget._canvas_dbl_click_conns) == 2


def test_a_second_double_click_during_a_move_is_ignored():
    """Reported from the app: double-click FIB, then double-click again while the
    post-move acquisition is still running, and a SEM frame lands on the FIB canvas.

    Two overlapping stage moves each trigger their own acquisition on one microscope.
    `_toggle_interactions` disabled the *buttons* for that whole window — including the
    acquisition, which is why `move_stage_finished` returns early while
    `is_acquiring` — but nothing gated canvas input, so the click got through."""
    host = _CanvasHost()
    _image_widget(host)
    movement = _movement_widget(host)

    started = []
    movement.control_widget._stage_move_worker = lambda *a, **k: (
        started.append(a) or _NoopWorker()
    )

    assert movement.control_widget._click_to_move_available()
    movement.control_widget._on_canvas_double_click(BeamType.ION, 10.0, 10.0, set())
    assert len(started) == 1, "first double-click did not start a move"

    # the first move is in flight: interactions are off, so a second click must not fire
    assert not movement.control_widget._click_to_move_available()
    movement.control_widget._on_canvas_double_click(BeamType.ION, 20.0, 20.0, set())
    assert len(started) == 1, "a second move started while the first was still running"


def test_click_to_move_is_blocked_while_acquiring_after_a_move():
    """The exact window in the report: the move itself has finished, but the images it
    triggered have not. `move_stage_finished` leaves interactions off on purpose."""
    host = _CanvasHost()
    image_widget = _image_widget(host)
    movement = _movement_widget(host)

    movement._toggle_interactions(enable=False)
    image_widget.is_acquiring = True
    movement.control_widget.move_stage_finished()  # move done, acquisition still running

    assert not movement.control_widget._click_to_move_available(), (
        "click-to-move re-armed while the post-move acquisition was still running"
    )


class _NoopWorker:
    """Stands in for a FunctionWorker without starting a thread."""

    finished = property(lambda self: self)
    returned = property(lambda self: self)
    errored = property(lambda self: self)

    def connect(self, *a, **k):
        pass

    def start(self):
        pass


# --- teardown: the process-abort guard ---------------------------------------


def test_movement_teardown_stops_canvas_double_clicks_reaching_a_dead_widget():
    """The canvases outlive the widget (removeTab + deleteLater fires neither closeEvent
    nor close). A double-click delivered afterwards raises inside a slot, and PyQt5 turns
    that into qFatal — the process aborts rather than logging (FIB-329)."""
    host = _CanvasHost()
    _image_widget(host)
    movement = _movement_widget(host)

    calls = []
    movement._toggle_interactions = lambda *a, **k: calls.append(1)

    host.view_controller.sem_canvas.canvas_double_clicked.emit(10.0, 10.0, set())
    assert calls, "double-click never reached the widget while connected"

    movement._teardown_connections()
    calls.clear()
    host.view_controller.sem_canvas.canvas_double_clicked.emit(10.0, 10.0, set())
    host.view_controller.fib_canvas.canvas_double_clicked.emit(10.0, 10.0, set())
    assert not calls, "double-click still reached the widget after teardown"


def test_image_widget_teardown_stops_scroll_reaching_a_dead_widget():
    host = _CanvasHost()
    widget = _image_widget(host)
    spinbox = _sem_beam_settings(widget).working_distance_spinbox

    widget._teardown_connections()
    before = spinbox.value()
    host.view_controller.sem_canvas.canvas_scrolled.emit(0.0, 0.0, 1, {"Shift"})
    assert spinbox.value() == before, (
        "WD scroll still reached the widget after teardown"
    )


def test_teardown_is_idempotent():
    """Both ``closeEvent`` and the host's disconnect path call it, and either may run first."""
    host = _CanvasHost()
    widget = _image_widget(host)
    movement = _movement_widget(host)
    for _ in range(3):
        widget._teardown_connections()
        movement._teardown_connections()
    assert widget._wd_scroll_connections == []
    assert widget._view_sync_connections == []
    assert movement.control_widget._canvas_dbl_click_conns == []


# --- position readout ---------------------------------------------------------


def test_position_readout_goes_to_the_controller_info_bar():
    host = _CanvasHost()
    _image_widget(host)
    movement = _movement_widget(host)
    seen = []
    host.view_controller.update_info = lambda *a, **k: seen.append((a, k))
    movement.control_widget._update_position_readout()
    assert seen, "stage readout never reached the quad-view info bar"


def test_the_stage_position_shows_without_touching_anything():
    """``setup_connections`` ends with ``update_ui()``, which is the *only* thing that
    puts a position on the canvas before the operator moves or acquires. Truncating that
    method leaves the info bar blank on connect and everything else still looks fine —
    which is exactly how it shipped and got caught by hand."""
    host = _CanvasHost()
    _image_widget(host)
    _movement_widget(host)
    info = host.view_controller._states[host.view_controller.sem_canvas].info
    assert any(key == "stage" for key, _ in info), (
        f"no stage readout after construction; info bar holds {info}"
    )


def test_setup_connections_wires_the_whole_widget():
    """Guards the shape of the bug above rather than one symptom of it: every one of
    these is set at a different point in a ``setup_connections``, so a method truncated
    anywhere shows up here.

    Spans both halves since FIB-783 split them -- the instructions label and the
    orientation text come from ``StageControlWidget.setup_connections``, the saved
    positions from the container's. That is deliberate: this is the tab's wiring guard,
    and a tab is only wired when both are."""
    host = _CanvasHost()
    image_widget = _image_widget(host)
    movement = _movement_widget(host)

    from fibsem.ui.FibsemMovementWidget import INSTRUCTIONS_TEXT

    # early: the instructions label
    assert (
        movement.control_widget.label_movement_instructions.text() == INSTRUCTIONS_TEXT
    )
    # middle: the saved positions hand-off, and the orientation button text, which is
    # rewritten from its constructed label a little further down
    assert movement.saved_positions_widget.microscope is not None, (
        "saved positions unwired"
    )
    assert movement.control_widget.pushButton_move_to_sem_orientation.text() == (
        "Move to SEM Orientation"
    ), "orientation button text unset"
    # late: milling-angle controls
    assert movement.control_widget.doubleSpinBox_milling_angle.maximum() == 45
    assert movement.control_widget.doubleSpinBox_milling_angle.suffix(), (
        "milling angle suffix unset"
    )
    # the acquisition signal must reach the widget (drives update_ui after each acquire)
    image_widget.acquisition_progress_signal.emit({"finished": True})


# --- click-to-move must not raise --------------------------------------------


def test_milling_widget_reports_whether_it_is_milling():
    """``FibsemMovementWidget`` blocks click-to-move on ``milling_widget.is_milling``.
    That attribute lives on the embedded run controls, not on the task viewer the hosts
    actually store, so the guard raised ``AttributeError`` inside the worker and every
    double-click died there."""
    from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget

    widget = MillingTaskViewerWidget(microscope=_session()[0])
    assert widget.is_milling is False


def test_double_click_to_move_survives_a_host_that_owns_a_milling_widget():
    """``FibsemUI`` sets ``self.milling_widget``; ``AutoLamellaUI`` does not — which is
    why only the standalone app hit this. The move must be dispatched, not swallowed."""
    from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget

    host = _CanvasHost()
    image_widget = _image_widget(host)
    host.milling_widget = MillingTaskViewerWidget(microscope=_session()[0])
    movement = _movement_widget(host)

    image_widget._on_acquire(_image(BeamType.ELECTRON))
    moves = []
    movement.control_widget._execute_stage_move = lambda *a, **k: moves.append((a, k))
    # the handler resolves every guard synchronously, so no thread is involved
    movement.control_widget._on_canvas_double_click(
        BeamType.ELECTRON, 100.0, 100.0, set()
    )
    assert moves, "double-click did not dispatch a stage move"


def _main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - standalone runner
            failures += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'FAILED' if failures else 'OK'} — {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
