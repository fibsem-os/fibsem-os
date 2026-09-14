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
    assert sum(1 for c in fm if c.suggested) == 3
    # on screen and in the record ...
    assert len(loaded.data.fm_coordinates) == 7
    # ... but not in the fit
    assert loaded.fit_data.fib_coordinates == []
    assert loaded.fit_data.fm_coordinates == []
    assert not loaded._can_run()
    assert "7 FM predictions" in loaded._lbl_status.text()
    cl = loaded._coords_tab
    assert cl._predict_count_label.text() == "7 predicted"
    assert cl._fm_count_label.text() == "7 predicted"
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
    assert all(c["status"] == PointStatus.PREDICTED for c in fm)
    assert "suggested" not in fm[0]  # a highlight, never saved


# ── the drop and the re-projection ───────────────────────────────────────


def test_a_drop_confirms_the_pair_and_the_next_projection_moves_only_the_rest(
    loaded,
):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    fm = _fm(loaded)
    i = next(k for k, c in enumerate(fm) if c.suggested)
    before = np.array([[c.point.x, c.point.y] for c in fm])
    fm[i].point.x += 40.0
    fm[i].point.y -= 25.0
    loaded._on_canvas_moved(fm[i])
    assert fm[i].status == PointStatus.PLACED
    assert fm[i].provenance == PointProvenance.PROJECTED  # where it came from
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
    assert fm[i].status == PointStatus.PLACED
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
    assert all(c.status == PointStatus.ACCEPTED for c in fm[1:])
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
        c.status = PointStatus.ACCEPTED
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
    words = [r.state_label.text() for r in rows]
    assert all(w.startswith("predicted") for w in words)
    assert sum(1 for w in words if "start here" in w) == 3


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


def test_predictions_stay_put_when_the_slider_moves(loaded):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    fm = _fm(loaded)
    before = [(c.point.x, c.point.y, c.point.z) for c in fm]
    slider = loaded._fm_display._z_slider
    slider.setValue(slider.value() + 2)
    assert [(c.point.x, c.point.y, c.point.z) for c in fm] == before


def test_project_with_nothing_to_predict_says_so(loaded):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded._coords_tab.fm_list.coordinates = [
        Coordinate(PointXYZ(10.0 * i, 20.0 * i, 3.0), PointType.FM) for i in range(7)
    ]
    before = [(c.point.x, c.point.y) for c in _fm(loaded)]
    loaded.project_fm_from_fib()
    assert loaded._lbl_status.text().startswith("Nothing to project")
    assert [(c.point.x, c.point.y) for c in _fm(loaded)] == before


def test_reject_and_reset_from_the_row_menu_change_the_fit_inputs(loaded):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    fm = _fm(loaded)
    for c in fm[:5]:
        c.point.x += 1.0
        loaded._on_canvas_moved(c)
    assert len(loaded.fit_data.fm_coordinates) == 5
    lw = loaded._coords_tab.fm_list
    # reject one placed point: out of the fit, still on screen, header says so
    lw.reject_toggled.emit(fm[0])
    assert fm[0].status == PointStatus.REJECTED
    assert len(loaded.fit_data.fm_coordinates) == 4
    assert len(loaded.data.fm_coordinates) == 7
    assert "1 removed" in loaded._coords_tab._fm_count_label.text()
    # and back
    lw.reject_toggled.emit(fm[0])
    assert fm[0].status == PointStatus.PLACED
    assert len(loaded.fit_data.fm_coordinates) == 5
    # reset a placed prediction: it is a guess again and moves back to the map
    moved = (fm[1].point.x, fm[1].point.y)
    lw.reset_requested.emit(fm[1])
    assert fm[1].status == PointStatus.PREDICTED
    assert (fm[1].point.x, fm[1].point.y) != moved
    assert len(loaded.fit_data.fm_coordinates) == 4
    # a hand-placed point (no projection behind it) cannot be reset
    hand = Coordinate(PointXYZ(5.0, 5.0, 3.0), PointType.FM)
    lw.reset_requested.emit(hand)
    assert hand.status == ""


def test_fit_settings_live_on_the_setup_tab_and_the_coordinates_tab_says_so(
    loaded,
):
    panel = loaded._coords_tab._fit_panel
    assert panel.parent() is not None
    assert loaded._images_tab.isAncestorOf(panel)
    assert not loaded._coords_tab.isAncestorOf(panel)


def _fake_seeded_result(loaded, diag: dict):
    from fibsem.correlation.structures import (
        CorrelationPointOfInterest,
        CorrelationResult,
    )

    return CorrelationResult(
        poi=[CorrelationPointOfInterest()],
        rms_error=1.0,
        input_data=loaded.fit_data,
        seed={"eulers_deg": [0, 0, 0]},
        branch_check={"selected": "nominal", "angle_to_nominal_deg": 0.4},
        diagnostics=diag,
    )


def _diag(loo, worst=None, mirror=2.0, hull=0.0, jackknife=0.1, suggested_z=None):
    return {
        "rms_um": 0.3,
        "pairs": [
            {"index": i, "residual_um": v * 0.6, "loo_error_um": v}
            for i, v in enumerate(loo)
        ],
        "mirror_ratio": mirror,
        "depth_span_um": 3.0,
        "scale_ratio": 1.0,
        "n_pairs": len(loo),
        "n_accepted": 0,
        "poi_jackknife_um": jackknife,
        "poi_hull_distance_um": hull,
        "worst": worst,
        "suggested_z": suggested_z,
    }


def test_a_seeded_run_shows_the_verdict_and_annotates_the_rows(loaded):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    for c in _fm(loaded):
        c.point.x += 1.0
        loaded._on_canvas_moved(c)
    loaded._coords_tab.poi_list.coordinates = [
        Coordinate(PointXYZ(300.0, 300.0, 3.0), PointType.POI)
    ]
    loaded.data_changed.emit(loaded.data)
    loo = [0.2, 0.3, 0.25, 0.2, 0.3, 0.2, 1.4]
    loaded._on_run_finished(
        _fake_seeded_result(loaded, _diag(loo, worst=6, suggested_z=5.0))
    )
    status = loaded._lbl_status.text()
    assert "Check FM 7." in status
    assert "1.4 µm off" in status and "slice 5" in status
    assert 'href="pair:6"' in status
    assert loaded._lbl_result.isHidden()  # the badge gave way to the line
    assert loaded._btn_continue.isEnabled()  # check, not poor
    # every FM row carries its leave-one-out error; the flagged one is amber
    from fibsem.ui.correlation.widgets.coordinate_list_widget import (
        CoordinateRowWidget,
    )

    lw = loaded._coords_tab.fm_list
    rows = [
        lw._list.itemWidget(lw._list.item(i))
        for i in range(lw._list.count())
        if isinstance(lw._list.itemWidget(lw._list.item(i)), CoordinateRowWidget)
    ]
    assert [r.state_label.text() for r in rows] == [f"{v:.1f} µm" for v in loo]
    assert "e0a030" in rows[6].state_label.styleSheet()  # WARN_COLOR
    # the link selects the pair
    loaded._on_status_link("pair:6")
    assert lw.selected_coordinate is _fm(loaded)[6]
    # the Results tab speaks the same language
    assert loaded._results_tab._lbl_worst.text().startswith("FM 7, 1.40 µm off")
    assert "determined by the fiducials" in loaded._results_tab._lbl_depth.text()
    assert loaded._results_tab._table.horizontalHeaderItem(2).text() == (
        "Left-out error (µm)"
    )
    # an edit clears the notes: they describe a run that no longer matches
    _fm(loaded)[0].point.x += 1.0
    loaded._on_canvas_moved(_fm(loaded)[0])
    assert rows[1].state_label.text() == ""


def test_a_poor_verdict_disables_continue(loaded):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    for c in _fm(loaded):
        c.point.x += 1.0
        loaded._on_canvas_moved(c)
    loaded._coords_tab.poi_list.coordinates = [
        Coordinate(PointXYZ(300.0, 300.0, 3.0), PointType.POI)
    ]
    loaded.data_changed.emit(loaded.data)
    loaded._on_run_finished(
        _fake_seeded_result(
            loaded, _diag([6.0, 3.0, 2.5, 8.0, 5.0, 15.0, 4.0], worst=5, mirror=1.1)
        )
    )
    status = loaded._lbl_status.text()
    assert "Poor fit. Do not continue." in status
    assert "cannot say which way is deeper" in status
    assert "Remove it and run again" in status
    assert not loaded._btn_continue.isEnabled()
    assert "ambiguous" in loaded._results_tab._lbl_depth.text()


def test_the_image_panes_split_by_aspect_ratio_until_the_user_drags(widget, tmp_path):
    fib, fm = _images(ARCTIS, fm_geometry=True)  # FIB 3:2 landscape, FM square
    widget.resize(1600, 900)
    widget.set_project_dir(str(tmp_path))
    widget.set_fib_image(fib)
    widget.set_fm_image(fm)
    fib_w, fm_w, side = widget._splitter.sizes()
    assert side > 0
    fib_h, fib_wpx = fib.data.shape[:2]
    fm_h, fm_wpx = fm.data.shape[-2:]
    assert fib_w / fm_w == pytest.approx((fib_wpx / fib_h) / (fm_wpx / fm_h), rel=0.05)
    # a drag makes the split the user's
    widget._on_splitter_moved(300, 1)
    widget._splitter.setSizes([300, fib_w + fm_w - 300, side])
    before = widget._splitter.sizes()
    widget.set_fm_image(fm)
    assert widget._splitter.sizes() == before
