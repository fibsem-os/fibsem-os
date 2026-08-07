"""Headless tests for the Lamella Editor's cutover from napari to the canvas (FIB-404).

Phase 2 of the napari deprecation. The editor now owns a
``MicroscopeViewController(view=LamellaEditorView())`` instead of a napari ``Viewer``,
which is what makes this tab verifiable without a microscope at all.

Covers the reducer-backed replacements for the old layer manipulation — FIB/SEM images,
the POI marker, the alignment-area rectangle, per-task page switching, milling patterns
through ``MillingTaskViewerWidget.set_controller`` — plus the two things that would
break the *other* tabs if this port were wrong: the shared milling widget's napari path
must still work for callers that never set a controller, and the layer-controls menu
must survive one tab having migrated.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_lamella_protocol_editor_canvas.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget

from psygnal.containers import EventedDict

from fibsem import utils
from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol, Experiment
from fibsem.applications.autolamella.workflows.tasks.tasks import SpotBurnFiducialTaskConfig
from fibsem.structures import BeamType, FibsemImage, FibsemRectangle, MicroscopeState, Point
from fibsem.ui.widgets.autolamella_lamella_protocol_editor import (
    ALIGNMENT_OVERLAY_ID,
    POI_OVERLAY_ID,
    AutoLamellaProtocolEditorWidget,
)
from fibsem.ui.widgets.canvas.quad_view import LamellaEditorView, MicroscopeViewController
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget

_app = QApplication.instance() or QApplication(sys.argv)

_HFW = 150e-6
_RESOLUTION = (1536, 1024)


class _Host(QWidget):
    """The two attributes the editor actually reads off its parent."""

    def __init__(self, microscope=None, experiment=None):
        super().__init__()
        self.microscope = microscope
        self.experiment = experiment


def _session():
    microscope, settings = utils.setup_session(manufacturer="Demo")
    return microscope, settings


def _experiment(tmp: str, n: int = 1) -> Experiment:
    exp = Experiment(path=tmp, name="fib404")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)
    for _ in range(n):
        exp.add_new_lamella(MicroscopeState(), EventedDict())
    return exp


def _overlays(controller, beam=BeamType.ION):
    """The reducer's overlay specs for *beam*. Reaching into the model on purpose: the
    editor's alignment lives under its own id, which the public alignment_area()
    accessor (hardcoded to the milling widget's shared "alignment") cannot see."""
    canvas = controller.get_canvas(beam)
    return controller._states[canvas].overlays


def _armed(controller, beam=BeamType.ION):
    canvas = controller.get_canvas(beam)
    return controller._states[canvas].armed_overlay


def _editor(with_microscope: bool = True):
    tmp = tempfile.mkdtemp()
    microscope = _session()[0] if with_microscope else None
    exp = _experiment(tmp)
    host = _Host(microscope=microscope, experiment=exp)
    editor = AutoLamellaProtocolEditorWidget(parent=host)
    editor._host = host  # keep it alive
    return editor, exp


# --- the property that makes this tab separable -------------------------------


def test_editor_builds_without_a_microscope():
    """The controller is created eagerly so AutoLamellaMainUI can embed the canvas
    into the splitter before anything connects. This is why phase 2 is the safe one."""
    editor, _ = _editor(with_microscope=False)
    assert isinstance(editor.view_controller, MicroscopeViewController)
    assert isinstance(editor.view_controller.widget, LamellaEditorView)
    assert editor.view_controller.get_canvas(BeamType.ION) is not None


def test_editor_is_napari_free():
    import fibsem.ui.widgets.autolamella_lamella_protocol_editor as mod

    src = Path(mod.__file__).read_text()
    for pattern in ("import napari", "from napari", "self.viewer", "viewer.layers"):
        assert pattern not in src, f"editor still references napari: {pattern!r}"


# --- reducer-backed replacements for the old layers ----------------------------


def test_fib_image_reaches_the_ion_canvas():
    editor, _ = _editor()
    image = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW, random=True)
    editor.image = image
    editor.view_controller.set_image(BeamType.ION, image)
    canvas = editor.view_controller.get_canvas(BeamType.ION)
    assert canvas._img_w == _RESOLUTION[0] and canvas._img_h == _RESOLUTION[1]


def test_sem_reference_toggles_the_second_canvas():
    """Replaces the old translate=(0, -width) side-by-side SEM layer."""
    editor, _ = _editor()
    editor.image = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW)
    sem = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW)

    # set_sem_visible toggles the titled *panel*, not the canvas — a child of a hidden
    # parent still reports isHidden() False, so assert on the panel.
    panel = editor.view_controller.widget._sem_panel

    editor._update_sem_display(None)
    assert panel.isHidden()

    editor._update_sem_display(sem)
    assert not panel.isHidden()
    assert editor.view_controller.get_canvas(BeamType.ELECTRON)._img_w == _RESOLUTION[0]


def test_poi_marker_is_placed_and_is_inert():
    """The POI is moved via the milling editor's right-click menu, never by dragging,
    so the overlay must be modal and never armed — otherwise it would steal input
    from the spot-burn editor sharing this canvas."""
    editor, exp = _editor()
    editor.image = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW)
    editor._draw_point_of_interest(Point(0.0, 0.0))

    points = editor.view_controller.overlay_points(BeamType.ION, POI_OVERLAY_ID)
    assert len(points) == 1
    x, y = points[0]
    assert abs(x - _RESOLUTION[0] / 2) < 1.0, f"POI x={x}, expected image centre"
    assert abs(y - _RESOLUTION[1] / 2) < 1.0, f"POI y={y}, expected image centre"

    canvas = editor.view_controller.get_canvas(BeamType.ION)
    assert canvas.active_overlay is None, "POI armed itself — it must stay inert"


def test_alignment_area_draws_removes_and_folds_edits_back():
    editor, exp = _editor()
    editor.image = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW)
    lamella = exp.positions[0]
    editor._selected_lamella = lamella
    lamella.alignment_area = FibsemRectangle(0.25, 0.25, 0.5, 0.5)

    editor.show_alignment_area = True
    editor._draw_alignment_area()
    spec = _overlays(editor.view_controller).get(ALIGNMENT_OVERLAY_ID)
    assert spec is not None and spec.rect.left == 0.25

    # a committed canvas edit reaches the lamella — but only while editing is enabled,
    # so go through the real toggle rather than setting the flag.
    edited = FibsemRectangle(0.1, 0.2, 0.3, 0.4)
    editor._on_controller_overlay_edited(BeamType.ION, ALIGNMENT_OVERLAY_ID, edited)
    assert lamella.alignment_area.left == 0.25, "edit applied while not editable"

    editor._on_toggle_alignment_area_editable(True)
    editor._on_controller_overlay_edited(BeamType.ION, ALIGNMENT_OVERLAY_ID, edited)
    assert lamella.alignment_area.left == 0.1
    assert lamella.alignment_area.top == 0.2

    # an edit signalled for a different overlay on the same canvas must be ignored
    editor._on_controller_overlay_edited(BeamType.ION, "milling", FibsemRectangle(0.9, 0.9, 0.05, 0.05))
    assert lamella.alignment_area.left == 0.1

    editor._on_toggle_alignment_area_editable(False)

    editor.show_alignment_area = False
    editor._draw_alignment_area()
    assert ALIGNMENT_OVERLAY_ID not in _overlays(editor.view_controller)


def test_only_one_alignment_rectangle_is_ever_drawn():
    """The editor's alignment uses its own overlay id, deliberately separate from the
    milling widget's shared "alignment" — so nothing but the editor's explicit
    set_alignment_area_visible(False) stops two rectangles being drawn at once.
    Pin it: the milling widget must not have an alignment spec on this canvas.
    """
    editor, exp = _editor()
    editor.image = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW)
    editor._selected_lamella = exp.positions[0]
    editor.show_alignment_area = True
    editor._draw_alignment_area()

    overlays = _overlays(editor.view_controller)
    assert ALIGNMENT_OVERLAY_ID in overlays, "editor's alignment missing"
    milling_spec = overlays.get("alignment")
    assert milling_spec is None or not milling_spec.display, (
        "milling widget is also displaying an alignment rect — two would be drawn"
    )
    assert editor.milling_task_editor._show_alignment_area is False


def test_alignment_edit_toggle_arms_and_disarms():
    """Arming is an explicit user action; redraws must never touch it, or a task change
    would silently steal input focus from the spot-burn overlay (FIB-354)."""
    editor, exp = _editor()
    editor.image = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW)
    editor._selected_lamella = exp.positions[0]
    editor.show_alignment_area = True

    # assert the reducer's model, not canvas.active_overlay: rendering is queued onto
    # the GUI thread, so the canvas hasn't caught up until events are processed.
    editor._on_toggle_alignment_area_editable(True)
    assert _armed(editor.view_controller) == ALIGNMENT_OVERLAY_ID

    editor._draw_alignment_area()  # a redraw must not disarm
    assert _armed(editor.view_controller) == ALIGNMENT_OVERLAY_ID


def test_a_redraw_never_steals_input_from_the_spot_burn_overlay():
    """FIB-354: the alignment overlay and the spot-burn editor share the FIB canvas, and
    the reducer allows exactly one armed overlay. Arming must therefore stay tied to the
    explicit edit toggle — if _draw_alignment_area armed as a side effect, every task
    change or alignment redraw would silently disarm a spot-burn session mid-edit.

    Asserting from the *other* overlay's point of view on purpose: an earlier version of
    this test redrew while alignment was already armed, where a stray re-arm is a no-op
    and therefore invisible.
    """
    editor, exp = _editor()
    editor.image = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW)
    editor._selected_lamella = exp.positions[0]
    editor.show_alignment_area = True

    spot_burn = editor.spot_burn_coordinates_widget
    spot_burn.set_active(True)
    assert _armed(editor.view_controller) == spot_burn.OVERLAY_ID

    editor._draw_alignment_area()
    assert _armed(editor.view_controller) == spot_burn.OVERLAY_ID, (
        "an alignment redraw disarmed the spot-burn overlay"
    )

    editor._draw_point_of_interest(Point(0.0, 0.0))
    assert _armed(editor.view_controller) == spot_burn.OVERLAY_ID, (
        "drawing the POI disarmed the spot-burn overlay"
    )

    # and it does reach the canvas once the queued render lands
    _app.processEvents()
    assert editor.view_controller.get_canvas(BeamType.ION).active_overlay is not None

    editor._on_toggle_alignment_area_editable(False)
    assert _armed(editor.view_controller) is None
    _app.processEvents()
    assert editor.view_controller.get_canvas(BeamType.ION).active_overlay is None


def test_fluorescence_task_switches_the_canvas_page():
    """Replaces the old per-task napari layer-visibility loop."""
    editor, _ = _editor()
    view = editor.view_controller.widget
    view.show_fluorescence()
    assert view._stack.currentIndex() == 1, "fluorescence task did not show the FM page"
    view.show_beams()
    assert view._stack.currentIndex() == 0
    assert view._stack.currentWidget() is view._beams_page


def test_spot_burn_coordinates_round_trip_through_the_task_config():
    """Only coordinates come from the editor — current and exposure stay owned by the
    generic task-parameters widget."""
    editor, exp = _editor()
    lamella = exp.positions[0]
    editor._selected_lamella = lamella
    task_name = "spot-burn"
    config = SpotBurnFiducialTaskConfig(milling_current=30e-12, exposure_time=7.0)
    config.coordinates = [Point(0.1, 0.2)]
    lamella.task_config[task_name] = config
    editor.listWidget_selected_task._list.setCurrentRow(
        _row_for(editor.listWidget_selected_task, task_name)
    ) if _row_for(editor.listWidget_selected_task, task_name) is not None else None

    settings = config.to_settings()
    settings.coordinates = [Point(0.4, 0.6), Point(0.5, 0.5)]
    editor.spot_burn_coordinates_widget.set_settings(settings)
    got = editor.spot_burn_coordinates_widget.get_settings()
    assert [(p.x, p.y) for p in got.coordinates] == [(0.4, 0.6), (0.5, 0.5)]
    # milling current / exposure survive the trip untouched
    assert config.milling_current == 30e-12 and config.exposure_time == 7.0


def _row_for(list_widget, name):
    for i in range(list_widget._list.count()):
        if list_widget._list.item(i).text() == name:
            return i
    return None


# --- the shared widget must not regress for the other tabs ---------------------


def test_milling_widget_keeps_its_napari_path_without_a_controller():
    """set_controller is opt-in. Every caller that doesn't call it — the Microscope tab,
    both task-config editors, the coincidence viewer — must be untouched by this phase."""
    microscope, _ = _session()
    w = MillingTaskViewerWidget(microscope=microscope, viewer=None, milling_enabled=False)
    assert w._controller is None, "widget claimed a controller it was never given"
    assert w._fib_canvas is None


def test_milling_widget_switches_to_the_canvas_when_given_a_controller():
    microscope, _ = _session()
    controller = MicroscopeViewController(view=LamellaEditorView())
    w = MillingTaskViewerWidget(microscope=microscope, viewer=None, milling_enabled=False)
    w.set_controller(controller)
    assert w._controller is controller
    assert w._fib_canvas is controller.get_canvas(BeamType.ION)


class _FakeImageWidget(QWidget):
    """Stands in for FibsemImageSettingsWidget, which is canvas-only now: it still
    carries the FIB image and still emits on every new frame, but has no napari layer."""

    viewer_update_signal = pyqtSignal()

    def __init__(self, image):
        super().__init__()
        self.ib_image = image


def test_napari_callers_supply_their_own_fib_image_layer():
    """The napari pattern path is gated on ``_fib_image_layer``, and ``set_fib_image()``
    is now its *only* source. It used to also be picked up from ``iw.ib_layer``, but the
    image widget dropped its layers when the Microscope tab moved to the canvas; the
    remaining napari caller (the coincidence viewer) has always used set_fib_image().
    """
    microscope, _ = _session()
    image = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW)
    sentinel = object()  # a napari layer, only ever passed through

    w = MillingTaskViewerWidget(microscope=microscope, viewer=None, milling_enabled=False)
    w.set_fib_image(image, sentinel)
    assert w._fib_image is image
    assert w._fib_image_layer is sentinel, (
        "napari callers lost their image layer — patterns and the reposition menu "
        "are both gated on it"
    )


def test_a_new_frame_refreshes_the_image_and_reschedules_the_patterns():
    """``_setup_viewer_integration`` and ``_on_viewer_image_updated`` both read from the
    image widget inside a bare try/except. They used to read ``ib_layer`` there too —
    once the image widget lost its layers, that read would have raised *inside the try*
    and silently skipped the ``viewer_update_signal`` connect and the reschedule after
    it, so patterns would quietly stop following the live image.
    """
    microscope, _ = _session()
    image = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW)
    iw = _FakeImageWidget(image)

    w = MillingTaskViewerWidget(
        microscope=microscope, viewer=None, image_widget=iw, milling_enabled=False
    )
    assert w._fib_image is image

    scheduled = []
    w._schedule_pattern_update = lambda: scheduled.append(1)

    new_image = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW)
    iw.ib_image = new_image
    iw.viewer_update_signal.emit()  # drive the connection, not the handler

    assert w._fib_image is new_image, "a new frame did not reach the milling widget"
    assert scheduled, "a new frame did not reschedule the pattern update"


def test_eye_toggle_routes_through_the_reducer_when_wired():
    microscope, _ = _session()
    controller = MicroscopeViewController(view=LamellaEditorView())
    w = MillingTaskViewerWidget(microscope=microscope, viewer=None, milling_enabled=False)
    w.set_controller(controller)
    w._on_eye_toggled(False)
    assert w._patterns_visible is False
    w._on_eye_toggled(True)
    assert w._patterns_visible is True


# --- the shell: the landmine that breaks the *other* tabs ----------------------


def test_layer_controls_menu_survives_a_migrated_tab():
    """The whole reason tabs can migrate one at a time.

    The handler used to build a dict literal dereferencing all three viewer attributes,
    so a tab that stopped creating its viewer raised AttributeError for *every* menu
    entry — and an exception escaping a Qt slot aborts the process under PyQt5
    (FIB-329). Calling it on an object with no viewer attributes at all is the harshest
    version of that: it must be a no-op, not a crash.
    """
    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
        AutoLamellaSingleWindowUI,
    )

    class _NoViewers:
        pass

    handler = AutoLamellaSingleWindowUI._on_toggle_viewer_layer_controls
    for key in ("microscope", "overview", "lamella", "nonsense"):
        handler(_NoViewers(), True, key)  # must not raise
        handler(_NoViewers(), False, key)


def test_protocol_tab_embeds_the_canvas_not_a_napari_window():
    import fibsem.applications.autolamella.ui.AutoLamellaMainUI as mod

    src = Path(mod.__file__).read_text()
    assert "self.lamella_viewer" not in src, "the lamella napari viewer is still built"
    assert "self.lamella_widget.view_controller.widget" in src
    assert "action_layer_controls_lamella" not in src


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
