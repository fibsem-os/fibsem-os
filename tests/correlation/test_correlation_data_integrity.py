"""Data-integrity regressions in the correlation flow (FIB-315..318).

Each of these let wrong data reach the experiment or the disk silently:

* FIB-315 — the result's provenance snapshot aliased the live coordinates, so
  ``matches_inputs`` could never report stale;
* FIB-316 — an image without metadata took the auto-save down for the whole
  session, silently;
* FIB-317 — swapping the image left the result armed, committing a POI scaled by
  the previous image;
* FIB-318 — adopting new coordinates kept (and re-saved) the previous result.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    CorrelationState,
    PointType,
    PointXYZ,
    load_correlation_file,
)
from fibsem.structures import FibsemImage


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw

    monkeypatch.setattr(riw, "_ensure_lut", lambda: None)


def _fib(x=1.0, y=2.0):
    return Coordinate(point=PointXYZ(x=x, y=y, z=0.0), point_type=PointType.FIB)


def _inputs(x=1.0):
    return CorrelationInputData(fib_coordinates=[_fib(x=x)])


def _result(inp):
    return CorrelationResult(
        poi=[CorrelationPointOfInterest()], rms_error=1.0, input_data=inp
    )


def _widget():
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    return CorrelationTabWidget()


# ---------------------------------------------------------------------------
# FIB-315 — the snapshot must not alias the live coordinates
# ---------------------------------------------------------------------------


def test_legacy_result_load_copies_the_snapshot(tmp_path):
    """Loading a legacy result file must not hand the live set the same objects
    the transform's snapshot holds, or every later edit mutates history."""
    path = tmp_path / "correlation_result.json"
    _result(_inputs(x=1.0)).save(str(path))

    state = load_correlation_file(str(path))
    assert state.input_data is not state.result.input_data
    assert (
        state.input_data.fib_coordinates[0]
        is not (state.result.input_data.fib_coordinates[0])
    )

    # editing the live point must make the result stale
    state.input_data.fib_coordinates[0].point.x = 9999.0
    assert state.result.matches_inputs(state.input_data) is False


def test_adopting_a_results_inputs_copies_them(qapp):
    w = _widget()
    result = _result(_inputs(x=1.0))
    w._load_result(result, adopt_inputs=True)

    w.data.fib_coordinates[0].point.x = 9999.0
    assert result.matches_inputs(w.data) is False


# ---------------------------------------------------------------------------
# FIB-316 — an image without metadata must not kill the auto-save
# ---------------------------------------------------------------------------


def test_serialisation_survives_an_image_without_metadata():
    """A plain (non-OME) TIFF loads with metadata=None; to_dict is on the only
    persistence path, so it must not raise."""
    img = FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6)
    img.metadata = None
    data = CorrelationInputData(fib_image=img, fib_coordinates=[_fib()])

    payload = data.to_dict()  # must not raise
    assert payload["fib_image_pixel_size"] is None
    assert payload["fib_image_filename"] is None
    assert tuple(payload["fib_image_shape"]) == (200, 300)


def test_autosave_failure_is_reported_not_swallowed(qapp, tmp_path, monkeypatch):
    w = _widget()
    w.set_project_dir(str(tmp_path))

    def _boom(self, filename):
        raise OSError("disk on fire")

    monkeypatch.setattr(CorrelationState, "save", _boom)
    w.data_changed.emit(w.data)

    assert "NOT being saved" in w._lbl_status.text()
    assert "disk on fire" in w._lbl_status.text()

    # and it clears once saving works again
    monkeypatch.undo()
    w.data_changed.emit(w.data)
    assert "NOT being saved" not in w._lbl_status.text()


# ---------------------------------------------------------------------------
# FIB-317 — a different image means different coordinates
# ---------------------------------------------------------------------------


def _load_via_picker(w, monkeypatch, answer):
    """Drive a user-initiated image load, answering the confirm with `answer`."""
    from PyQt5.QtWidgets import QMessageBox

    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    monkeypatch.setattr(
        ctw.FibsemImage,
        "load",
        staticmethod(
            lambda p: FibsemImage.generate_blank_image(resolution=(768, 512), hfw=20e-6)
        ),
    )
    monkeypatch.setattr(
        ctw.QMessageBox, "question", staticmethod(lambda *a, **k: answer)
    )
    w._images_tab._load_fib("/lam/other_ib.tif")
    return QMessageBox


def test_changing_the_image_clears_coordinates_and_result(qapp, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    w = _widget()
    w.set_fib_image(
        FibsemImage.generate_blank_image(resolution=(1536, 1024), hfw=80e-6)
    )
    w.set_data(_inputs(x=1382.0))
    w._load_result(_result(w.data), adopt_inputs=False)

    _load_via_picker(w, monkeypatch, QMessageBox.Ok)

    assert w.data.fib_coordinates == []  # points belonged to the previous image
    assert w._result is None


def test_declining_the_change_keeps_both_image_and_points(qapp, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    w = _widget()
    original = FibsemImage.generate_blank_image(resolution=(1536, 1024), hfw=80e-6)
    w.set_fib_image(original)
    w.set_data(_inputs(x=1382.0))

    _load_via_picker(w, monkeypatch, QMessageBox.Cancel)

    assert w.data.fib_coordinates[0].point.x == 1382.0
    assert w._fib_image is original
    # the picker must not be left showing a file that was never loaded
    assert w._images_tab._fib_picker.current_path() == w._images_tab._fib_loaded_path


def test_programmatic_image_load_does_not_prompt(qapp, monkeypatch):
    """The lamella's images arrive this way on open — prompting there would be
    a dialog nobody asked for."""
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    asked = []
    monkeypatch.setattr(
        ctw.QMessageBox, "question", staticmethod(lambda *a, **k: asked.append(1))
    )
    w = _widget()
    w.set_data(_inputs(x=1.0))
    w.set_fib_image(FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6))

    assert asked == []
    assert w.data.fib_coordinates  # and it left them alone


def test_image_combo_is_guarded_against_the_mouse_wheel(qapp):
    """Changing the image discards the coordinates, so a scroll passing over the
    combo must not be able to trigger it.

    Asserts the guard is installed rather than delivering a synthetic wheel
    event: without the guard an event-based check *hangs* offscreen instead of
    failing, which is a useless diagnostic in CI.
    """
    from PyQt5.QtCore import Qt

    from fibsem.ui.utils import _WHEEL_GUARD_PROPERTY

    combo = _widget()._images_tab._fib_picker.combo
    assert combo.property(_WHEEL_GUARD_PROPERTY) is True
    assert combo.focusPolicy() == Qt.StrongFocus  # no wheel-focus either


# ---------------------------------------------------------------------------
# FIB-318 — new coordinates must not keep the old result
# ---------------------------------------------------------------------------


def test_adopting_a_state_without_a_result_discards_the_old_one(qapp, tmp_path):
    w = _widget()
    w.set_project_dir(str(tmp_path))
    w._load_result(_result(_inputs(x=1.0)), adopt_inputs=False)
    assert w._result is not None

    w._adopt_state(CorrelationState(input_data=_inputs(x=5.0), result=None))
    assert w._result is None

    saved = json.loads((tmp_path / "correlation.json").read_text(encoding="utf-8"))
    assert saved["result"] is None  # not re-saved against the new coordinates


def test_seeding_discards_a_previous_result(qapp):
    w = _widget()
    w._load_result(_result(_inputs(x=1.0)), adopt_inputs=False)
    w.seed_coordinates(_inputs(x=7.0))
    assert w._result is None
