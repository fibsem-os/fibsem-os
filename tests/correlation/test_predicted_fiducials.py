"""Predicted fiducials in the correlation widget (FIB-956): one button
projects the FIB fiducials into the FM, predictions are drawn but never fed
to the fit, a drop confirms a pair, and projecting again moves only what the
user has not touched.

Images are built from the nominal-transform fixture's metadata over blank
pixels; the fits themselves are covered by the correlation util tests.
"""

from __future__ import annotations

import copy
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
    assert cl._fm_count_label.text() == "7 predicted"
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
    assert loaded._lbl_status.text().endswith("0 of 4 pairs placed.")
    for c in _fm(loaded)[:4]:
        c.point.x += 1.0
        loaded._on_canvas_moved(c)
    assert loaded._can_run()
    assert loaded._lbl_status.text().startswith("Ready to run with 4 pairs.")


# ── the list row and the spot-burn lookup ────────────────────────────────


def test_the_projection_row_lives_in_the_fm_panel(loaded):
    """Project and Accept all are the FM list's actions and sit under it, in
    its panel; there is no Predicted Fiducials panel of their own."""
    from fibsem.ui.widgets.custom_widgets import TitledPanel

    cl = loaded._coords_tab
    titles = [p._title_label.text() for p in cl.findChildren(TitledPanel)]
    assert "Predicted Fiducials" not in titles
    assert cl._fm_panel.isAncestorOf(cl.btn_project)
    assert cl._fm_panel.isAncestorOf(cl.btn_accept_predictions)
    assert cl._fm_panel.isAncestorOf(cl._predict_hint)


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


def test_a_moved_point_is_placed_whatever_it_was(loaded):
    """Dragging a fitted or accepted point makes it the user's: the row's
    state word goes, and an accepted point becomes evidence for the map."""
    from fibsem.correlation.prediction import independent_pairs

    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    loaded.accept_all_predictions()
    fib, fm = loaded._coords_tab.fib_list.coordinates, _fm(loaded)
    assert independent_pairs(fib, fm) == []
    fm[0].point.x += 1.0
    loaded._on_canvas_moved(fm[0])
    assert fm[0].status == PointStatus.PLACED
    assert fm[0].provenance == PointProvenance.PROJECTED  # where it came from
    assert len(independent_pairs(fib, fm)) == 1

    fm[1].status = PointStatus.FITTED
    fm[1].fitted = True
    lw = loaded._coords_tab.fm_list
    lw.refresh_coordinate(fm[1])
    row = lw._list.itemWidget(lw._list.item(1))
    assert row.state_label.text() == "fitted"
    loaded._on_list_changed(loaded._point_specs[PointType.FM], fm[1], "z", 5.0)
    assert fm[1].status == PointStatus.PLACED and not fm[1].fitted
    assert row.state_label.text() == ""


def test_rows_show_z_as_a_slice_unless_fitted(loaded):
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    fm = _fm(loaded)
    lw = loaded._coords_tab.fm_list
    row = lw._list.itemWidget(lw._list.item(0))
    assert row.z_spin.decimals() == 0
    assert "." not in row.z_spin.cleanText()
    fm[0].point.z = 2.4  # within the stack: the row clamps to the axis
    fm[0].status = PointStatus.FITTED
    lw.refresh_coordinate(fm[0])
    assert row.z_spin.decimals() == 1
    assert row.z_spin.cleanText() == "2.4"


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
    # what to start from, how it fits, then what the experiment decided
    section = loaded.add_lamella_setup(spot_burns=_burns(ARCTIS))
    layout = loaded._images_tab._content_layout
    assert layout.indexOf(section) < layout.indexOf(panel)
    assert layout.indexOf(panel) < layout.indexOf(section.inherited_panel)
    assert section.inherited_panel.isVisibleTo(loaded._images_tab)


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


def test_the_verdict_names_rows_the_fit_skipped_over(loaded):
    """Rejecting FIB 2 takes row 2 out of the fit on both sides. The verdict
    still speaks in rows: its notes land on the rows that were fitted, the
    link selects the right FM point, and the Results tab names the same
    fiducial as the run bar."""
    from fibsem.ui.correlation.widgets.coordinate_list_widget import (
        CoordinateRowWidget,
    )

    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    for c in _fm(loaded):
        loaded._on_canvas_moved(c)
    loaded._coords_tab.fib_list.coordinates[1].status = PointStatus.REJECTED
    loaded.data_changed.emit(loaded.data)
    n = len(_fm(loaded))
    assert loaded._fit_rows() == [i for i in range(n) if i != 1]

    # the worker's diagnostics carry the rows; fake one the way it would
    diag = _diag([0.2, 0.3, 0.25, 0.2, 0.3, 1.4], worst=5, suggested_z=5.0)
    for p, row in zip(diag["pairs"], loaded._fit_rows()):
        p["index"] = row
    loaded._on_run_finished(_fake_seeded_result(loaded, diag))

    status = loaded._lbl_status.text()
    worst_row = loaded._fit_rows()[5]
    assert f"Check FM {worst_row + 1}." in status
    assert f'href="pair:{worst_row}"' in status
    lw = loaded._coords_tab.fm_list
    rows = [
        lw._list.itemWidget(lw._list.item(i))
        for i in range(lw._list.count())
        if isinstance(lw._list.itemWidget(lw._list.item(i)), CoordinateRowWidget)
    ]
    assert rows[1].state_label.text() != "0.3 µm"  # the skipped row has no note
    assert rows[worst_row].state_label.text() == "1.4 µm"
    loaded._on_status_link(f"pair:{worst_row}")
    assert lw.selected_coordinate is _fm(loaded)[worst_row]
    fib_list = loaded._coords_tab.fib_list
    assert fib_list.selected_coordinate is fib_list.coordinates[worst_row]
    fib_surface = loaded._point_specs[PointType.FIB].adapter._surface
    assert (
        fib_surface.picking.points.selected_coordinate()
        is fib_list.coordinates[worst_row]
    )
    assert loaded._results_tab._lbl_worst.text().startswith(f"FM {worst_row + 1},")
    assert loaded._results_tab._table.item(5, 0).text() == f"FM {worst_row + 1}"


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
    assert "poor" in loaded._btn_continue.toolTip()
    assert "ambiguous" in loaded._results_tab._lbl_depth.text()
    # a good run after it: Continue is back, without the old warning
    loaded._on_run_finished(
        _fake_seeded_result(loaded, _diag([0.2, 0.3, 0.25, 0.2, 0.3, 0.2, 0.3]))
    )
    assert loaded._btn_continue.isEnabled()
    assert loaded._btn_continue.toolTip() == ""


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


# ── the placement offset and the Method panel's projection row (FIB-979) ──


def _offset_runs(offset_um, *, rms_um=0.5, age_days=0.0):
    import time

    from fibsem.correlation.history import CorrelationRun
    from fibsem.correlation.structures import (
        CorrelationInputData,
        CorrelationResult,
        CorrelationState,
    )

    px = ARCTIS["fib"]["pixel_size"]
    result = CorrelationResult(
        placement_offset=list(offset_um),
        rms_error=rms_um * 1e-6 / px,
        updated_at=time.time() - age_days * 86400,
        input_data=CorrelationInputData(stored_fib_image_pixel_size=px),
    )
    run = CorrelationRun(path="/x/r", name="r", state=CorrelationState(result=result))
    return [("02-other, run r", run)]


def test_a_previous_runs_offset_moves_the_first_placement_and_can_be_ignored(loaded):
    from fibsem.correlation.geometry import nominal_transform

    bare = nominal_transform(loaded._fib_image, loaded._fm_image).translation
    px_um = ARCTIS["fib"]["pixel_size"] * 1e6
    cl = loaded._coords_tab

    loaded.set_prior_runs(_offset_runs([3.0, -4.0], age_days=9))
    nominal, _ = loaded._nominal_transform()
    assert np.allclose(nominal.translation, bare + np.array([3.0, -4.0]) / px_um)
    text = cl._lbl_projection.text()
    assert "calibrated placement (9 days old)" in text and 'href="ignore"' in text
    assert "02-other" not in text  # the source run is tooltip material
    assert "(+3.0, -4.0) µm, measured by 02-other" in cl._lbl_projection.toolTip()
    assert "FM px per slice along the beam" in cl._lbl_projection.toolTip()

    loaded._on_projection_link("ignore")
    nominal, _ = loaded._nominal_transform()
    assert np.allclose(nominal.translation, bare)
    text = cl._lbl_projection.text()
    assert "placed from stage metadata" in text and 'href="use"' in text
    assert "ignored for this lamella" in cl._lbl_projection.toolTip()

    loaded._on_projection_link("use")
    nominal, _ = loaded._nominal_transform()
    assert np.allclose(nominal.translation, bare + np.array([3.0, -4.0]) / px_um)


def test_ignoring_the_offset_leaves_a_live_result_live(loaded):
    """Ignore changes the next projection, not the points, so Continue stays."""
    from fibsem.correlation.structures import (
        CorrelationPointOfInterest,
        CorrelationResult,
    )

    loaded.set_prior_runs(_offset_runs([3.0, -4.0]))
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
        CorrelationResult(
            poi=[CorrelationPointOfInterest()],
            rms_error=1.0,
            input_data=copy.deepcopy(loaded.fit_data),
        )
    )
    assert loaded._btn_continue.isEnabled()
    loaded._on_projection_link("ignore")
    assert 'href="use"' in loaded._coords_tab._lbl_projection.text()
    assert loaded._btn_continue.isEnabled()


def test_a_poor_previous_run_does_not_supply_an_offset(loaded):
    loaded.set_prior_runs(_offset_runs([3.0, -4.0], rms_um=5.0))
    text = loaded._coords_tab._lbl_projection.text()
    assert text == "geometry · placed from stage metadata"


def test_a_run_records_the_offset_its_fiducials_measured(loaded):
    """Only pairs the user placed measure the offset. Predictions accepted
    where the offset-corrected projection put them would echo the previous
    offset back, so a run of accepted pairs records nothing, and accepted
    pairs beside placed ones do not dilute what the placed ones say."""
    from fibsem.correlation.structures import CorrelationResult

    loaded.set_prior_runs(_offset_runs([3.0, -4.0]))
    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    loaded.project_fm_from_fib()
    loaded.accept_all_predictions()
    loaded._run_nominal, _ = loaded._nominal_transform()
    result = CorrelationResult(input_data=loaded.fit_data)
    assert loaded._measure_placement_offset(result) is None

    # move two FM points 10 FM px in x, as a drop would: the FIB-side
    # translation moves by minus the prior's in-plane map of that shift,
    # measured from those two alone
    P = loaded._run_nominal.projection
    px_um = ARCTIS["fib"]["pixel_size"] * 1e6
    for c in _fm(loaded)[:2]:
        c.point.x += 10.0
        c.status = PointStatus.PLACED  # the user's, from here on
    result = CorrelationResult(input_data=loaded.fit_data)
    expected = np.array([3.0, -4.0]) - (P[:, :2] @ [10.0, 0.0]) * px_um
    assert np.allclose(loaded._measure_placement_offset(result), expected, atol=1e-6)

    loaded._on_run_finished(result)
    assert loaded._result.placement_offset == pytest.approx(list(expected))


def test_a_previous_run_far_from_the_geometry_is_not_used_as_the_prior(loaded):
    """One saved Arctis run was an unseeded fit 62 degrees from the geometry;
    used as a prior it would project every ring on the wrong line."""
    from fibsem.correlation.geometry import nominal_transform
    from fibsem.correlation.history import CorrelationRun
    from fibsem.correlation.structures import (
        CorrelationInputData,
        CorrelationResult,
        CorrelationState,
    )

    geometry = nominal_transform(loaded._fib_image, loaded._fm_image)
    a = np.radians(60.0)
    about_x = np.array(
        [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
    )
    px = ARCTIS["fib"]["pixel_size"]

    def runs(rotation):
        result = CorrelationResult(
            scale=geometry.scale,
            rotation_quaternion=(rotation).tolist(),
            translation=[0.0, 0.0, 0.0],
            fm_z_scale=geometry.z_anisotropy,
            input_data=CorrelationInputData(stored_fib_image_pixel_size=px),
        )
        run = CorrelationRun(path="/x", name="r", state=CorrelationState(result=result))
        return [("02-other, run r", run)]

    loaded.set_prior_runs(runs(about_x @ geometry.rotation))
    assert loaded._prior_transform() is None
    nominal, _ = loaded._nominal_transform()
    assert np.allclose(nominal.projection, geometry.projection)
    row = loaded._coords_tab._lbl_projection
    assert row.text().startswith("geometry ·")
    assert "02-other, run r not used: its rotation is 60° from the geometry" in (
        row.toolTip()
    )

    loaded.set_prior_runs(runs(geometry.rotation))
    assert loaded._prior_transform() is not None
    assert loaded._coords_tab._lbl_projection.text().startswith(
        "previous run (02-other) ·"
    )


# ── the Setup tab's Images panel (FIB-978) ───────────────────────────────


def test_a_preloaded_image_from_outside_the_lamella_shows_its_file(widget, tmp_path):
    """The standalone launcher preloads images the picker was never offered;
    the picker must still name what is on the canvas."""
    fib, fm = _images(ARCTIS, fm_geometry=True)
    fib_path = tmp_path / "ref_ib.tif"
    fib_path.write_bytes(b"")  # exists on disk; never re-read here
    fib.filepath = str(fib_path)
    widget.set_fib_image(fib)
    tab = widget._images_tab
    assert tab._fib_picker.current_path() == str(fib_path)
    assert tab._fib_loaded_path == str(fib_path)
    assert "1024 × 1536 · 65.10 nm" in tab._lbl_fib_info.text()
    # offering the lamella's (empty) list afterwards must not blank it
    widget.add_lamella_setup(spot_burns=_burns(ARCTIS))
    assert tab._fib_picker.current_path() == str(fib_path)
    # a bare name cannot be loaded again, so it is not shown (FIB-321)
    fm.filepath = "stack.ome.tiff"
    widget.set_fm_image(fm)
    assert tab._fm_picker.current_path() == ""
    assert tab._lbl_fm_px.text().startswith("1 channel · 6 slices · ")
    assert tab._lbl_fm_px.toolTip() == "reflection"


def test_the_setup_tab_has_one_images_panel_with_interpolate_in_its_header(loaded):
    from fibsem.ui.widgets.custom_widgets import TitledPanel

    titles = [
        p._title_label.text() for p in loaded._images_tab.findChildren(TitledPanel)
    ]
    assert titles.count("Images") == 1
    assert "FIB Image" not in titles and "FM Image" not in titles
    assert "Project" not in titles
    btn = loaded._images_tab._btn_interpolate
    assert btn.isEnabled()  # a 6-slice stack with a z step
    assert next(
        p
        for p in loaded._images_tab.findChildren(TitledPanel)
        if p._title_label.text() == "Images"
    ).isAncestorOf(btn)


# ── the run bar (FIB-978 §1.4) ───────────────────────────────────────────


def test_the_status_line_names_the_next_step_and_the_bar_shows_one_primary_button(
    widget,
    loaded,
):
    from fibsem.correlation.structures import (
        CorrelationPointOfInterest,
        CorrelationResult,
    )

    fresh = CorrelationTabWidget()
    assert fresh._lbl_status.text() == "Load the FIB and FM images on the Setup tab."
    fresh.close()

    status = loaded._lbl_status
    assert status.text().startswith("Seed the spot burns on the Setup tab")
    assert not loaded._btn_continue.isVisibleTo(loaded)
    assert loaded._btn_run.text() == "Run Correlation"

    loaded.seed_fib_fiducials_from_spot_burns(_burns(ARCTIS))
    assert status.text().startswith("Project FM from FIB")
    loaded.project_fm_from_fib()
    assert status.text().startswith("7 FM predictions placed")  # the action's own line
    loaded.data_changed.emit(loaded.data)  # the idle sentence comes back with an edit
    assert (
        status.text()
        == "Drag the predicted rings onto their burns: 0 of 4 pairs placed."
    )
    fm = _fm(loaded)
    for c in fm[:3]:
        c.point.x += 1.0
        loaded._on_canvas_moved(c)
    assert (
        status.text()
        == "Drag the predicted rings onto their burns: 3 of 4 pairs placed."
    )
    for c in fm[3:]:
        c.point.x += 1.0
        loaded._on_canvas_moved(c)
    assert status.text() == "Place the target on the FM image."
    # a pair rejected from the fit is out on both sides: still the target
    fm[1].status = PointStatus.REJECTED
    loaded.data_changed.emit(loaded.data)
    assert status.text() == "Place the target on the FM image."
    fm[1].status = PointStatus.PLACED
    loaded._coords_tab.poi_list.coordinates = [
        Coordinate(PointXYZ(300.0, 300.0, 3.0), PointType.POI)
    ]
    loaded.data_changed.emit(loaded.data)
    assert status.text() == "Ready to run with 7 pairs."

    # a live result: Continue is the primary, Run becomes "Run again". A real
    # run snapshots the inputs (FIB-315); so must the stand-in, or the edit
    # below mutates the snapshot and the result never reads as stale.
    loaded._on_run_finished(
        CorrelationResult(
            poi=[CorrelationPointOfInterest()],
            rms_error=1.0,
            input_data=copy.deepcopy(loaded.fit_data),
        )
    )
    assert loaded._btn_continue.isVisibleTo(loaded) and loaded._btn_continue.isEnabled()
    assert loaded._btn_run.text() == "Run again"
    # an edit makes it stale: back to one primary Run (the lists were rebuilt
    # when the result was adopted, so take the live one)
    fm = _fm(loaded)
    fm[0].point.x += 1.0
    loaded._on_canvas_moved(fm[0])
    assert not loaded._btn_continue.isVisibleTo(loaded)
    assert loaded._btn_run.text() == "Run Correlation"
    assert (
        status.text()
        == "The points changed since the last run; run again with 7 pairs."
    )
