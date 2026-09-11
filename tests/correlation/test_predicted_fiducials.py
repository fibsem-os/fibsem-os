"""Predicted fiducials in the correlation widget (FIB-956): one button
projects the FIB fiducials into the FM, predictions are drawn but never fed
to the fit, a drop confirms a pair, and projecting again moves only what the
user has not touched.

Images are built from the nominal-transform fixture's metadata over blank
pixels; the fits themselves are covered by the correlation util tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.structures import (  # noqa: E402
    Coordinate,
    PointProvenance,
    PointStatus,
    PointType,
    PointXYZ,
)
from fibsem.fm.structures import (  # noqa: E402
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import (  # noqa: E402
    BeamType,
    CameraImageTransform,
    FibsemHardwareGeometry,
    FibsemImage,
    FibsemImageMetadata,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)
from fibsem.ui.correlation.widgets.correlation_tab_widget import (  # noqa: E402
    CorrelationTabWidget,
    find_spot_burns,
)

FIXTURE = Path(__file__).parent / "fixtures" / "nominal_transform.json"
ENTRIES = json.loads(FIXTURE.read_text())["entries"]
ARCTIS = next(
    e
    for e in ENTRIES
    if e["system"] == "arctis" and e["fit"]["fm_slices_in_fit"] != 106
)


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw

    monkeypatch.setattr(riw, "_ensure_lut", lambda *a, **k: None, raising=False)


def _images(entry: dict, *, fm_geometry: bool):
    from fibsem.correlation.geometry import fm_geometry_for

    fib_md = FibsemImageMetadata(
        image_settings=ImageSettings(beam_type=BeamType.ION, hfw=100e-6),
        pixel_size=Point(entry["fib"]["pixel_size"], entry["fib"]["pixel_size"]),
        microscope_state=MicroscopeState(
            stage_position=FibsemStagePosition.from_dict(entry["fib"]["stage_position"])
        ),
        hardware_geometry=FibsemHardwareGeometry.from_dict(
            entry["fib"]["hardware_geometry"]
        ),
    )
    fib_md.microscope_state.ion_beam.scan_rotation = entry["fib"]["scan_rotation"]
    fib = FibsemImage(
        data=np.zeros(tuple(entry["fib"]["shape"]), dtype=np.uint8), metadata=fib_md
    )
    geometry = (
        fm_geometry_for(
            fib_md.hardware_geometry, CameraImageTransform(entry["fm"]["transform"])
        )
        if fm_geometry
        else None
    )
    fm_md = FluorescenceImageMetadata(
        acquisition_date="2026-01-01T00:00:00",
        pixel_size_x=entry["fm"]["pixel_size_x"],
        pixel_size_y=entry["fm"]["pixel_size_x"],
        pixel_size_z=entry["fm"]["pixel_size_z"],
        resolution=tuple(entry["fm"]["shape"][::-1]),
        channels=[
            FluorescenceChannelMetadata(
                name="reflection",
                excitation_wavelength=635.0,
                power=0.01,
                exposure_time=0.002,
                gain=None,
                offset=0.0,
            )
        ],
        stage_position=FibsemStagePosition.from_dict(entry["fm"]["pose"]),
        geometry=geometry,
    )
    fm = FluorescenceImage(
        data=np.zeros((1, 6, *entry["fm"]["shape"]), dtype=np.uint16), metadata=fm_md
    )
    return fib, fm


def _burns(entry: dict):
    h, w = entry["fib"]["shape"]
    return [Point(x=x / w, y=y / h) for x, y in entry["coords"]["fib"]]


@pytest.fixture
def widget(qapp):
    w = CorrelationTabWidget()
    yield w
    w.close()
    w.deleteLater()


@pytest.fixture
def loaded(widget, tmp_path):
    fib, fm = _images(ARCTIS, fm_geometry=True)
    widget.set_project_dir(str(tmp_path))
    widget.set_fib_image(fib)
    widget.set_fm_image(fm)
    return widget


def _fm(widget):
    return widget._coords_tab.fm_list.coordinates


def _fib(widget):
    return widget._coords_tab.fib_list.coordinates


# ── the button ───────────────────────────────────────────────────────────


def test_project_needs_images_geometry_and_fib_fiducials(widget, loaded, tmp_path):
    cl = loaded._coords_tab
    assert not cl.btn_project.isEnabled()
    assert "Needs FIB fiducials" in cl._predict_hint.text()
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    assert cl.btn_project.isEnabled()
    assert "each of the 7 FIB fiducials" in cl._predict_hint.text()


def test_a_stack_without_geometry_says_why_not(widget, tmp_path):
    fib, fm = _images(ARCTIS, fm_geometry=False)
    widget.set_project_dir(str(tmp_path))
    widget.set_fib_image(fib)
    widget.set_fm_image(fm)
    widget.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    cl = widget._coords_tab
    assert not cl.btn_project.isEnabled()
    assert cl._predict_hint.text().startswith("Not available:")


def test_project_adds_predictions_that_never_feed_the_fit(loaded):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    fm = _fm(loaded)
    assert len(fm) == len(_fib(loaded)) == 7
    assert all(c.status in PointStatus.TENTATIVE for c in fm)
    assert all(c.provenance == PointProvenance.PROJECTED for c in fm)
    assert sum(1 for c in fm if c.status == PointStatus.SUGGESTED) == 3
    # on screen and in the record ...
    assert len(loaded.data.fm_coordinates) == 7
    # ... but not in the fit
    assert loaded.fit_data.fib_coordinates == []
    assert loaded.fit_data.fm_coordinates == []
    assert not loaded._can_run()
    assert "7 FM predictions" in loaded._lbl_status.text()
    cl = loaded._coords_tab
    assert cl._predict_count_label.text() == "(7 predicted)"
    assert cl._fm_count_label.text() == "(0 of 7 confirmed)"
    assert cl.btn_accept_predictions.isEnabled()
    assert "3 highlighted" in cl._predict_hint.text()


def test_predictions_follow_the_geometry(loaded):
    from fibsem.correlation.geometry import nominal_transform

    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    nominal = nominal_transform(loaded._fib_image, loaded._fm_image)
    P = nominal.projection
    zan = nominal.z_anisotropy
    backs = [
        np.array([a.point.x, a.point.y])
        - P @ np.array([b.point.x, b.point.y, b.point.z * zan])
        for a, b in zip(_fib(loaded), _fm(loaded))
    ]
    # one translation for every point: the pattern was projected rigidly
    assert np.allclose(backs, backs[0], atol=1e-6)
    assert all(
        b.point.z == pytest.approx(loaded._fm_display.current_z) for b in _fm(loaded)
    )


def test_predictions_are_saved_with_their_state(loaded, tmp_path):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    saved = next(tmp_path.rglob("correlation.json"))
    raw = json.loads(saved.read_text())
    fm = raw["input_data"]["fm_coordinates"]
    assert len(fm) == 7
    assert all(c["provenance"] == PointProvenance.PROJECTED for c in fm)
    assert sum(1 for c in fm if c["status"] == PointStatus.SUGGESTED) == 3


# ── the drop and the re-projection ───────────────────────────────────────


def test_a_drop_confirms_the_pair_and_the_next_projection_moves_only_the_rest(
    loaded,
):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    fm = _fm(loaded)
    i = next(k for k, c in enumerate(fm) if c.status == PointStatus.SUGGESTED)
    before = np.array([[c.point.x, c.point.y] for c in fm])
    fm[i].point.x += 40.0
    fm[i].point.y -= 25.0
    loaded._on_canvas_moved(fm[i])
    assert fm[i].status == PointStatus.ADJUSTED
    assert fm[i].provenance == PointProvenance.USER
    # the drop alone moves nothing else
    still = np.array([[c.point.x, c.point.y] for c in fm])
    others = [k for k in range(len(fm)) if k != i]
    assert np.allclose(still[others], before[others])
    assert len(loaded.fit_data.fm_coordinates) == 1
    assert "1 placed" in loaded._coords_tab._predict_hint.text()
    assert "Project again" in loaded._coords_tab._predict_hint.text()

    loaded.project_fm_from_fib()
    fm = _fm(loaded)
    assert len(fm) == 7  # nothing appended: every FIB point has a partner
    after = np.array([[c.point.x, c.point.y] for c in fm])
    delta = after - before
    # every prediction moved by the drop's offset; the dropped point did not move
    assert np.allclose(delta[others], delta[i], atol=1e-6)
    assert np.allclose(after[i], still[i])
    assert fm[i].status == PointStatus.ADJUSTED
    assert all(fm[k].status == PointStatus.PREDICTED for k in others)
    assert "translation from 1 pair" in loaded._lbl_status.text()


def test_accept_all_keeps_provenance_and_the_status_line_says_so(loaded):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    fm = _fm(loaded)
    fm[0].point.x += 5.0
    loaded._on_canvas_moved(fm[0])
    loaded.accept_all_predictions()
    fm = _fm(loaded)
    assert all(c.status == PointStatus.CONFIRMED for c in fm[1:])
    assert all(c.provenance == PointProvenance.PROJECTED for c in fm[1:])
    assert not loaded._coords_tab.btn_accept_predictions.isEnabled()
    assert len(loaded.fit_data.fm_coordinates) == 7
    loaded._coords_tab.poi_list.coordinates = [
        Coordinate(PointXYZ(300.0, 300.0, 3.0), PointType.POI)
    ]
    loaded.data_changed.emit(loaded.data)
    assert loaded._can_run()
    assert (
        "1 pair placed by you, 6 accepted from the projection"
        in loaded._lbl_status.text()
    )


def test_rejected_pairs_leave_the_fit_but_stay_on_screen(loaded):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    fm = _fm(loaded)
    for c in fm[:4]:
        c.status = PointStatus.CONFIRMED
    fm[1].status = PointStatus.REJECTED
    assert len(loaded.fit_data.fib_coordinates) == 3
    assert len(loaded.data.fm_coordinates) == 7


def test_the_run_gate_counts_confirmed_pairs_only(loaded):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    loaded._coords_tab.poi_list.coordinates = [
        Coordinate(PointXYZ(300.0, 300.0, 3.0), PointType.POI)
    ]
    loaded.data_changed.emit(loaded.data)
    assert not loaded._can_run()
    assert "FM=0 confirmed, 7 predicted" in loaded._lbl_status.text()
    for c in _fm(loaded)[:4]:
        c.point.x += 1.0
        loaded._on_canvas_moved(c)
    assert loaded._can_run()
    assert loaded._lbl_status.text() == "Ready."


# ── the list row and the spot-burn lookup ────────────────────────────────


def test_list_rows_show_the_prediction_state(loaded):
    from fibsem.ui.correlation.widgets.coordinate_list_widget import (
        CoordinateRowWidget,
    )

    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    lw = loaded._coords_tab.fm_list
    rows = [lw._list.itemWidget(lw._list.item(i)) for i in range(lw._list.count())]
    rows = [r for r in rows if isinstance(r, CoordinateRowWidget)]
    assert len(rows) == 7
    tips = [r.fitted_icon.toolTip() for r in rows]
    assert all(t.startswith("Predicted") for t in tips)
    assert sum(1 for t in tips if "start here" in t) == 3


def test_find_spot_burns_reads_the_experiment_above_the_run(tmp_path):
    experiment = tmp_path / "AutoLamella-x"
    lamella = experiment / "03-quick-tomcat"
    run = lamella / "Correlation" / "2026-09-09_10-00"
    run.mkdir(parents=True)
    (experiment / "experiment.yaml").write_text(
        "positions:\n"
        "- petname: 03-quick-tomcat\n"
        "  task_config:\n"
        "    Spot Burn Fiducial:\n"
        "      task_type: SPOT_BURN_FIDUCIAL\n"
        "      coordinates:\n"
        "      - {x: 0.25, y: 0.5}\n"
        "      - {x: 0.75, y: 0.5}\n"
    )
    burns, reason = find_spot_burns(str(run))
    assert reason == ""
    assert [(p.x, p.y) for p in burns] == [(0.25, 0.5), (0.75, 0.5)]
    burns, reason = find_spot_burns(str(tmp_path))
    assert burns == [] and "experiment.yaml" in reason


def test_predictions_follow_the_z_slider_without_staling_anything(loaded):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    fm = _fm(loaded)
    fm[0].point.x += 5.0
    loaded._on_canvas_moved(fm[0])
    dropped = (fm[0].point.x, fm[0].point.y, fm[0].point.z)
    before = np.array([[c.point.x, c.point.y] for c in fm[1:]])
    saves = []
    loaded.data_changed.connect(lambda d: saves.append(d))
    slider = loaded._fm_display._z_slider
    slider.setValue(slider.value() + 2)
    after = np.array([[c.point.x, c.point.y] for c in fm[1:]])
    assert all(c.point.z == pytest.approx(slider.value()) for c in fm[1:])
    step = after - before
    assert np.allclose(step, step[0], atol=1e-6) and np.linalg.norm(step[0]) > 0
    # the confirmed point did not move, and nothing was announced as an edit
    assert (fm[0].point.x, fm[0].point.y, fm[0].point.z) == dropped
    assert saves == []
