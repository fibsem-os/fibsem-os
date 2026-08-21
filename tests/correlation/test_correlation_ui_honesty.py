"""UI states that misrepresent the data (FIB-321).

None of these corrupt what is stored; each shows the user something untrue, and
two of them let a wrong number reach the experiment:

* a result marked "Correction applied" when the factor is merely armed;
* a finished run marked live even though the points moved while it ran;
* a post-correction that updates the display but not ``px_m`` — the value
  Continue commits;
* an image picker still displaying a file that failed to load;
* a previous run with no coordinates silently clearing the live ones;
* the editor's "latest image" default never applying, so the *oldest* image is
  the one handed to correlation;
* legend entries surviving an image swap that removed their markers.

Headless PyQt5, offscreen.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.config import CorrelationConfig
from fibsem.correlation.history import CorrelationRun, LamellaCorrelation
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    CorrelationState,
    PointType,
    PointXYZ,
)
from fibsem.structures import FibsemImage, Point


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw

    monkeypatch.setattr(riw, "_ensure_lut", lambda: None)


def _widget():
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    return CorrelationTabWidget()


def _fib_image(resolution=(300, 200)):
    return FibsemImage.generate_blank_image(resolution=resolution, hfw=100e-6)


def _fib(x=1.0, y=2.0):
    return Coordinate(point=PointXYZ(x=x, y=y, z=0.0), point_type=PointType.FIB)


def _poi():
    return CorrelationPointOfInterest(image_px=Point(x=100.0, y=300.0))


# ---------------------------------------------------------------------------
# 1 — "applied" must mean applied, in this mode
# ---------------------------------------------------------------------------


def test_a_pre_factor_matching_a_post_result_is_still_only_armed(qapp):
    """Post-correct at ζ, then add an FM surface point and store the same ζ: the
    factors match but the result was corrected in the *other* space, so the POI
    on screen is not the one this factor describes."""
    w = _widget()
    data = CorrelationInputData(
        fm_surface_coordinate=Coordinate(
            point=PointXYZ(x=1.0, y=2.0, z=3.0), point_type=PointType.SURFACE_FM
        ),
        ri_pre_correction_factor=1.470,
    )
    result = CorrelationResult(
        poi=[_poi()],
        refractive_index_correction_factor=1.470,
        refractive_index_correction_mode="post",
    )

    w._ri_tab.set_result(result, input_data=data)

    assert "applied on the next run" in w._ri_tab._lbl_warning.text()
    assert "Correction applied" not in w._ri_tab._lbl_warning.text()


def test_a_pre_factor_matching_a_pre_result_reads_as_applied(qapp):
    """The other side of it: same mode, same factor — that one really is applied."""
    w = _widget()
    data = CorrelationInputData(
        fm_surface_coordinate=Coordinate(
            point=PointXYZ(x=1.0, y=2.0, z=3.0), point_type=PointType.SURFACE_FM
        ),
        ri_pre_correction_factor=1.470,
    )
    result = CorrelationResult(
        poi=[_poi()],
        refractive_index_correction_factor=1.470,
        refractive_index_correction_mode="pre",
    )

    w._ri_tab.set_result(result, input_data=data)

    assert "Correction applied" in w._ri_tab._lbl_warning.text()


# ---------------------------------------------------------------------------
# 2 — a finished run is judged against the points as they are now
# ---------------------------------------------------------------------------


def test_a_run_finishing_after_an_edit_is_not_live(qapp):
    """_run snapshots the data for the worker but leaves the canvases editable, so
    a fiducial dragged mid-run makes the delivered result already stale. Continue
    commits poi[0].px_m, so it must not stay armed."""
    w = _widget()
    w.set_fib_image(_fib_image())
    w.set_data(CorrelationInputData(fib_coordinates=[_fib(x=10.0)]))
    snapshot = CorrelationInputData(fib_coordinates=[_fib(x=10.0)])

    w.data.fib_coordinates[0].point.x = 999.0  # dragged while the run was in flight
    w._on_run_finished(
        CorrelationResult(poi=[_poi()], rms_error=1.0, input_data=snapshot)
    )

    assert not w._btn_continue.isEnabled()
    assert "points have changed" in w._lbl_status.text()


def test_an_undisturbed_run_is_live(qapp):
    """Guards the over-correction: a normal run must still arm Continue."""
    w = _widget()
    w.set_fib_image(_fib_image())
    w.set_data(CorrelationInputData(fib_coordinates=[_fib(x=10.0)]))

    w._on_run_finished(
        CorrelationResult(poi=[_poi()], rms_error=1.0, input_data=w.data)
    )

    assert w._btn_continue.isEnabled()


# ---------------------------------------------------------------------------
# 3 — a correction that cannot reach px_m is refused, not half-applied
# ---------------------------------------------------------------------------


def test_correction_without_image_dimensions_raises():
    """px_m is what Continue commits. Correcting only image_px moved the marker
    and recorded the factor while the committed number stayed uncorrected."""
    result = CorrelationResult(
        poi=[_poi()],
        input_data=CorrelationInputData(
            surface_coordinate=Coordinate(
                point=PointXYZ(x=1.0, y=200.0, z=0.0), point_type=PointType.SURFACE
            )
        ),
    )

    with pytest.raises(ValueError, match="px_m"):
        result.apply_refractive_index_correction(1.5)

    # and it left the result alone rather than half-correcting it
    assert result.poi[0].image_px.y == pytest.approx(300.0)
    assert result.refractive_index_correction_factor is None


# ---------------------------------------------------------------------------
# 4 — the picker must not display a file that failed to load
# ---------------------------------------------------------------------------


def test_a_file_that_fails_to_load_is_dropped_from_the_picker(qapp, monkeypatch):
    """A browsed file joins the list before anything opens it. With nothing loaded
    before, the revert (`show_path("")`) returned immediately and left the bad
    file on display as though it were the current image."""
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    w = _widget()
    tab, picker = w._images_tab, w._images_tab._fib_picker
    picker.show_path("/lam/broken_ib.tif")  # as _browse does
    monkeypatch.setattr(
        ctw.FibsemImage,
        "load",
        staticmethod(lambda p: (_ for _ in ()).throw(OSError("not a TIFF"))),
    )
    monkeypatch.setattr(ctw.QMessageBox, "warning", staticmethod(lambda *a, **k: None))

    tab._load_fib("/lam/broken_ib.tif")

    assert picker.current_path() == ""  # not claiming to show an image
    assert picker.combo.findData("/lam/broken_ib.tif") < 0  # nor offering it again


def _loads_as(path):
    """A FIB image whose metadata names the file it came from, as a saved one does."""
    image = _fib_image()
    image.metadata.image_settings.filename = path
    return image


def test_a_failed_load_falls_back_to_the_last_good_image(qapp, monkeypatch):
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    w = _widget()
    tab, picker = w._images_tab, w._images_tab._fib_picker
    monkeypatch.setattr(ctw.FibsemImage, "load", staticmethod(_loads_as))
    tab._load_fib("/lam/good_ib.tif")

    monkeypatch.setattr(
        ctw.FibsemImage,
        "load",
        staticmethod(lambda p: (_ for _ in ()).throw(OSError("not a TIFF"))),
    )
    monkeypatch.setattr(ctw.QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    picker.show_path("/lam/broken_ib.tif")
    tab._load_fib("/lam/broken_ib.tif")

    assert picker.current_path() == "/lam/good_ib.tif"
    assert picker.combo.findData("/lam/broken_ib.tif") < 0


def test_loading_an_image_does_not_list_it_twice(qapp, monkeypatch):
    """Metadata records a basename, entries are keyed on full paths — so the
    lookup missed and the bare name was added as a second entry for the same
    file. Picking that one calls FibsemImage.load on a bare name, which resolves
    against the working directory and fails."""
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    w = _widget()
    tab, picker = w._images_tab, w._images_tab._fib_picker
    path = "/lam/03_ref_MillRough_ib.tif"
    picker.set_options([path], path)
    monkeypatch.setattr(
        ctw.FibsemImage,
        "load",
        staticmethod(lambda p: _loads_as("03_ref_MillRough_ib.tif")),
    )

    tab._load_fib(path)

    assert [picker.combo.itemData(i) for i in range(picker.combo.count())] == [path]
    assert picker.current_path() == path
    assert tab._fib_loaded_path == path  # a path, not a bare name


def test_an_unknown_image_does_not_invent_an_entry(qapp):
    """An image whose metadata names a file the picker doesn't offer must leave
    the list — and the display — alone, rather than showing something it can't
    load."""
    w = _widget()
    tab, picker = w._images_tab, w._images_tab._fib_picker
    known = "/lam/03_ref_MillRough_ib.tif"
    picker.set_options([known], known)

    tab.set_fib_image(_loads_as("/somewhere/else/other_ib.tif"))

    assert [picker.combo.itemData(i) for i in range(picker.combo.count())] == [known]
    assert picker.current_path() == known


# ---------------------------------------------------------------------------
# 5 — a previous run with no coordinates has nothing to seed
# ---------------------------------------------------------------------------


def _run(name, inp):
    return CorrelationRun(
        path="/tmp/" + name, name=name, state=CorrelationState(input_data=inp)
    )


def test_an_empty_previous_run_does_not_wipe_the_coordinates(qapp, monkeypatch):
    """An empty run — or a legacy file with a null input_data — took the
    "Previous correlation" branch and replaced every point with nothing."""
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    asked = []
    monkeypatch.setattr(
        ctw.QMessageBox,
        "information",
        staticmethod(lambda *a, **k: asked.append(a[1])),
    )

    w = _widget()
    w.set_fib_image(_fib_image())
    good = CorrelationInputData(fib_coordinates=[_fib(x=77.0)])
    section = w.add_lamella_setup(
        # reversed for display: index 0 is the newest (good), index 1 the empty one
        history=LamellaCorrelation(
            runs=[
                _run("2026-07-20_09-00-00", CorrelationInputData()),
                _run("2026-07-24_14-32-00", good),
            ]
        ),
        config=CorrelationConfig(),
    )
    section.emit_current_seed()
    assert w.data.fib_coordinates[0].point.x == 77.0

    section.run_combo.setCurrentIndex(1)  # the empty run

    assert asked == ["Nothing to seed"]
    assert w.data.fib_coordinates[0].point.x == 77.0  # untouched
    assert section.run_combo.currentIndex() == 0  # and the control went back


# ---------------------------------------------------------------------------
# 6 — the editor's default image
# ---------------------------------------------------------------------------


def test_editor_defaults_to_the_newest_image():
    """The fallback was a full path from glob while the combo holds basenames, so
    setCurrentText silently no-op'd and the combo stayed on the oldest image."""
    from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (
        AutoLamellaProtocolEditorWidget as Editor,
    )

    names = ["01_ib.tif", "02_ib.tif", "03_ib.tif"]

    assert Editor._default_fib_filename(names, "", "") == "03_ib.tif"
    # a task reference image wins, then a previous selection
    assert Editor._default_fib_filename(names, "02_ib.tif", "01_ib.tif") == "02_ib.tif"
    assert Editor._default_fib_filename(names, "gone.tif", "01_ib.tif") == "01_ib.tif"
    assert Editor._default_fib_filename([], "", "") == ""


# ---------------------------------------------------------------------------
# 7 — the legend describes what is on the canvas
# ---------------------------------------------------------------------------
