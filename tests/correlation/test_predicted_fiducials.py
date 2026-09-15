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
    assert "FM=0 confirmed, 7 predicted" in loaded._lbl_status.text()
    for c in _fm(loaded)[:4]:
        c.point.x += 1.0
        loaded._on_canvas_moved(c)
    assert loaded._can_run()
    assert loaded._lbl_status.text() == "Ready."


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
