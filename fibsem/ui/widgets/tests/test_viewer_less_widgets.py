"""The image and movement widgets must work with OR without a napari viewer (FIB-406, P4a).

Standalone ``FibsemUI`` now runs viewer-less on the quad-view canvas while AutoLamella's
Microscope tab still runs on napari, so both widgets carry two display paths at once,
selected per instance by whether the host supplies a ``viewer`` or a ``view_controller``.
The failure mode this guards is a silent one: pick the wrong branch and the widget still
constructs, still acquires, and simply draws nothing.

The viewer-less half runs against a **real** ``MicroscopeViewController`` and real canvases.
The napari half runs against a stub viewer — napari's GL canvas cannot be constructed under
``QT_QPA_PLATFORM=offscreen`` (it aborts the process), so a stub is the only way to assert
the napari branch at all here. That is a real limit on this file: it pins *which branch runs
and what it calls*, not napari's own rendering.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python fibsem/ui/widgets/tests/test_viewer_less_widgets.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget

from fibsem import utils
from fibsem.structures import BeamType, FibsemImage, FibsemRectangle
from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
from fibsem.ui.widgets.canvas.quad_view import MicroscopeViewController

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
    image = FibsemImage.generate_blank_image(resolution=_RESOLUTION, hfw=_HFW, random=True)
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


# --- napari stubs ------------------------------------------------------------


class _StubLayer:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.translate = [0.0, 0.0]
        self.visible = True
        self.mode = "pan_zoom"
        self.metadata = {}
        self.mouse_double_click_callbacks = []
        self.mouse_drag_callbacks = []
        self.selected_data = set()
        self.events = _StubEvents()


class _StubEvents:
    def __init__(self):
        self.data = _StubSignal()


class _StubSignal:
    def __init__(self):
        self.callbacks = []

    def connect(self, fn):
        self.callbacks.append(fn)


class _StubLayerList(list):
    """napari's LayerList iterates *layers* but indexes by name — a dict subclass would
    yield names on iteration and break ``draw_scalebar_in_napari``, which reads ``.name``
    off each item."""

    def __init__(self):
        super().__init__()
        self.selection = type("S", (), {"active": None})()

    def __getitem__(self, key):
        if isinstance(key, str):
            for layer in self:
                if layer.name == key:
                    return layer
            raise KeyError(key)
        return super().__getitem__(key)

    def __contains__(self, item):
        if isinstance(item, str):
            return any(layer.name == item for layer in self)
        return any(layer is item for layer in self)

    def remove(self, item):
        for layer in list(self):
            if layer is item or layer.name == item:
                super().remove(layer)
                return


class _StubViewer:
    """The slice of napari.Viewer these widgets touch. See the module docstring for why."""

    def __init__(self):
        self.layers = _StubLayerList()
        self.title = ""
        # the stage/beam readout the movement widget writes on the napari path
        self.text_overlay = type("T", (), {"text": "", "visible": False, "position": ""})()

    def add_image(self, data, name=None, blending=None, **kw):
        layer = _StubLayer(data, name)
        self.layers.append(layer)
        return layer

    def add_shapes(self, data=None, name=None, **kw):
        layer = _StubLayer(data, name)
        self.layers.append(layer)
        return layer

    def add_points(self, data=None, name=None, **kw):
        layer = _StubLayer(data, name)
        self.layers.append(layer)
        return layer

    def reset_view(self):
        pass


class _NapariHost(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.viewer = _StubViewer()


# --- which display backend is chosen -----------------------------------------


def test_viewer_less_host_uses_the_controller():
    host = _CanvasHost()
    widget = _image_widget(host)
    assert widget.uses_napari is False
    assert widget._view_controller() is host.view_controller


def test_controller_resolves_through_parent_widget():
    """AutoLamella nests the controller one level up. A resolver that only checked the
    direct parent would leave the Microscope tab blank after 4b, with no error."""
    outer = _CanvasHost()
    host = _NestedCanvasHost(outer)
    widget = _image_widget(host)
    assert widget._view_controller() is outer.view_controller


def test_napari_host_keeps_the_layer_path():
    host = _NapariHost()
    widget = _image_widget(host)
    assert widget.uses_napari is True
    assert widget._view_controller() is None, "a viewer host must not pick up a controller"
    assert widget.eb_layer is not None and widget.ib_layer is not None
    names = {layer.name for layer in host.viewer.layers}
    assert names >= {BeamType.ELECTRON.name, BeamType.ION.name}


class _BothHost(QWidget):
    """A misconfigured host holding both. ``FibsemUI`` always builds a controller, so
    ``FibsemUI(viewer=...)`` reaches exactly this shape."""

    def __init__(self) -> None:
        super().__init__()
        self.viewer = _StubViewer()
        self.view_controller = MicroscopeViewController(parent=self)


def test_a_viewer_wins_when_a_host_offers_both():
    """The two paths must be strictly exclusive. Running both would draw every image
    twice and, worse, put the alignment area in the model while ``get_alignment_area``
    reads it back off an empty layer."""
    host = _BothHost()
    widget = _image_widget(host)
    assert widget.uses_napari is True
    assert widget._view_controller() is None, "both display paths active at once"

    movement = FibsemMovementWidget(microscope=_session()[0], parent=host)
    assert movement._view_controller() is None
    assert movement._canvas_dbl_click_conns == [], "a click here would move the stage twice"

    widget._on_acquire(_image(BeamType.ELECTRON))
    assert host.view_controller.get_canvas(BeamType.ELECTRON)._content_extent() is None


def test_scalebar_and_crosshair_buttons_are_napari_only():
    """They drive napari layers; the canvas has its own. Left visible they would be dead
    controls — hidden, not merely inert, so nothing reads as broken."""
    canvas_widget = _image_widget(_CanvasHost())
    assert not canvas_widget.btn_scalebar.isVisibleTo(canvas_widget)
    assert not canvas_widget.btn_crosshair.isVisibleTo(canvas_widget)

    napari_widget = _image_widget(_NapariHost())
    assert napari_widget.btn_scalebar.isVisibleTo(napari_widget)
    assert napari_widget.btn_crosshair.isVisibleTo(napari_widget)


# --- images actually reach the display ---------------------------------------


def test_acquired_image_lands_on_its_own_beam_canvas():
    host = _CanvasHost()
    widget = _image_widget(host)
    sem_canvas = host.view_controller.get_canvas(BeamType.ELECTRON)
    fib_canvas = host.view_controller.get_canvas(BeamType.ION)
    assert sem_canvas._content_extent() is None, "canvas should start empty"

    widget._on_acquire(_image(BeamType.ELECTRON))
    assert sem_canvas._content_extent() is not None, "SEM image did not reach the canvas"
    assert fib_canvas._content_extent() is None, "SEM image leaked onto the FIB canvas"

    widget._on_acquire(_image(BeamType.ION))
    assert fib_canvas._content_extent() is not None, "FIB image did not reach the canvas"


def test_no_blank_placeholder_is_seeded_viewer_less():
    """napari seeded two blank images at construction; the canvas shows "No image"
    instead. Seeding would put a grey rectangle up that looks like a failed acquisition."""
    host = _CanvasHost()
    _image_widget(host)
    assert host.view_controller.get_canvas(BeamType.ELECTRON)._content_extent() is None
    assert host.view_controller.get_canvas(BeamType.ION)._content_extent() is None


def test_napari_path_still_writes_to_the_layer():
    host = _NapariHost()
    widget = _image_widget(host)
    image = _image(BeamType.ELECTRON)
    widget._on_acquire(image)
    layer = host.viewer.layers[BeamType.ELECTRON.name]
    assert np.array_equal(layer.data, image.filtered_data), "layer not updated"


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
    """A canvas drag must drive the same validation path the napari layer event did."""
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
    assert len(widget._wd_scroll_connections) == 2, "WD scroll not wired to both canvases"

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
    assert 10 ** -decimals <= WD_WHEEL_STEP_MM, (
        f"spinbox resolves {10 ** -decimals} mm but the scroll step is {WD_WHEEL_STEP_MM} mm"
    )


def test_wd_scroll_is_not_wired_on_the_napari_path():
    widget = _image_widget(_NapariHost())
    assert widget._wd_scroll_connections == []


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
    assert widget.dual_beam_widget.fib_radio.isChecked(), "FM selection moved the beam radio"


# --- movement: double-click input --------------------------------------------


def test_movement_binds_double_click_to_both_canvases_viewer_less():
    host = _CanvasHost()
    _image_widget(host)
    movement = FibsemMovementWidget(microscope=_session()[0], parent=host)
    assert movement.uses_napari is False
    assert len(movement._canvas_dbl_click_conns) == 2


def test_movement_keeps_napari_layer_callbacks_with_a_viewer():
    host = _NapariHost()
    image_widget = _image_widget(host)
    movement = FibsemMovementWidget(microscope=_session()[0], parent=host)
    assert movement.uses_napari is True
    assert movement._canvas_dbl_click_conns == [], "canvas input wired on the napari path"
    assert movement._double_click in image_widget.eb_layer.mouse_double_click_callbacks
    assert movement._double_click in image_widget.ib_layer.mouse_double_click_callbacks


# --- teardown: the process-abort guard ---------------------------------------


def test_movement_teardown_stops_canvas_double_clicks_reaching_a_dead_widget():
    """The canvases outlive the widget (removeTab + deleteLater fires neither closeEvent
    nor close). A double-click delivered afterwards raises inside a slot, and PyQt5 turns
    that into qFatal — the process aborts rather than logging (FIB-329)."""
    host = _CanvasHost()
    _image_widget(host)
    movement = FibsemMovementWidget(microscope=_session()[0], parent=host)

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
    assert spinbox.value() == before, "WD scroll still reached the widget after teardown"


def test_teardown_is_idempotent():
    """Both ``closeEvent`` and the host's disconnect path call it, and either may run first."""
    host = _CanvasHost()
    widget = _image_widget(host)
    movement = FibsemMovementWidget(microscope=_session()[0], parent=host)
    for _ in range(3):
        widget._teardown_connections()
        movement._teardown_connections()
    assert widget._wd_scroll_connections == []
    assert widget._view_sync_connections == []
    assert movement._canvas_dbl_click_conns == []


def test_teardown_on_the_napari_path_does_not_raise():
    """FibsemUI calls it unconditionally on disconnect, whichever path is live."""
    host = _NapariHost()
    widget = _image_widget(host)
    movement = FibsemMovementWidget(microscope=_session()[0], parent=host)
    widget._teardown_connections()
    movement._teardown_connections()


# --- position readout ---------------------------------------------------------


def test_position_readout_goes_to_the_controller_info_bar():
    host = _CanvasHost()
    _image_widget(host)
    movement = FibsemMovementWidget(microscope=_session()[0], parent=host)
    seen = []
    host.view_controller.update_info = lambda *a, **k: seen.append((a, k))
    movement._update_position_readout()
    assert seen, "stage readout never reached the quad-view info bar"


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
