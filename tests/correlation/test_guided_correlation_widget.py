"""The guided correlation widget (FIB-956): the rail reads state, predictions
are guesses that never feed the fit, a drop becomes a confirmed pair and
re-projects the rest.

Images are built from the nominal-transform fixture's metadata over blank
pixels, so every local fit fails and positions stay where they were put --
which is the behaviour under test here; the fits themselves are covered by the
correlation util tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fibsem.correlation.geometry import fm_geometry_for, nominal_transform
from fibsem.correlation.structures import (
    Coordinate,
    PointProvenance,
    PointStatus,
    PointType,
    PointXYZ,
)
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import (
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
from fibsem.ui.correlation.widgets.guided_correlation_widget import (
    STEP_FIDUCIALS,
    STEP_IMAGES,
    STEP_REVIEW,
    STEP_TARGET,
    GuidedCorrelationWidget,
    find_spot_burns,
)

pytest.importorskip("PyQt5")

FIXTURE = Path(__file__).parent / "fixtures" / "nominal_transform.json"
ENTRIES = json.loads(FIXTURE.read_text())["entries"]
ARCTIS = next(
    e
    for e in ENTRIES
    if e["system"] == "arctis" and e["fit"]["fm_slices_in_fit"] != 106
)


def _images(entry: dict, *, fm_geometry: bool):
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


def _pattern(entry: dict):
    """The fixture's FIB picks as a normalised spot-burn pattern for its image."""
    h, w = entry["fib"]["shape"]
    return [Point(x=x / w, y=y / h) for x, y in entry["coords"]["fib"]]


@pytest.fixture
def widget(qapp):
    w = GuidedCorrelationWidget()
    yield w
    w.close()
    w.deleteLater()


@pytest.fixture
def loaded(widget, tmp_path):
    fib, fm = _images(ARCTIS, fm_geometry=True)
    widget.set_project_dir(str(tmp_path))
    widget.set_fib_image(fib)
    widget.set_fm_image(fm)
    widget.set_spot_burns(_pattern(ARCTIS), field_of_view=100e-6)
    return widget


# ── the rail reads state ──────────────────────────────────────────────────


def test_rail_starts_on_images_with_later_steps_blocked(widget):
    assert widget._step == STEP_IMAGES
    assert widget._rail._states[STEP_IMAGES] == "todo"
    assert widget._rail._states[STEP_FIDUCIALS] == "blocked"
    assert widget._rail._states[STEP_TARGET] == "blocked"
    assert widget._rail._states[STEP_REVIEW] == "blocked"
    assert not widget._btn_predict.isEnabled()


def test_loading_both_images_unblocks_fiducials_and_reports_readiness(loaded):
    assert loaded._rail._states[STEP_IMAGES] == "done"
    assert loaded._rail._states[STEP_FIDUCIALS] == "todo"
    assert "predictions available" in loaded._rail._subtitles[STEP_IMAGES].text()
    strip = loaded._readiness.text()
    assert "compustage" in strip
    assert "FM geometry recorded" in strip
    assert "spot burns" in strip
    assert "Predictions available" in strip
    assert loaded._btn_predict.isEnabled()
    # the stack recorded its transform, so nothing is assumed
    assert not loaded._assume_row.isVisibleTo(loaded)


def test_a_stack_without_geometry_gets_an_explicit_assumed_transform(widget, tmp_path):
    fib, fm = _images(ARCTIS, fm_geometry=False)
    widget.set_project_dir(str(tmp_path))
    widget.set_fib_image(fib)
    widget.set_fm_image(fm)
    assert widget._assume_row.isVisibleTo(widget)
    assert "assuming none" in widget._readiness.text()
    assert widget._seed_available()  # the host assumes; the library would refuse
    # the assumption is the one the seed is built on
    nominal_none, _ = widget._nominal_transform()
    widget._cmb_transform.setCurrentIndex(
        widget._cmb_transform.findData(CameraImageTransform.FLIP_XY)
    )
    nominal_xy, _ = widget._nominal_transform()
    assert "assuming flip xy" in widget._readiness.text()
    assert not np.allclose(nominal_none.projection, nominal_xy.projection)


# ── predictions ───────────────────────────────────────────────────────────


def test_predict_draws_pattern_and_projects_it_as_tentative_points(loaded):
    loaded.predict_fiducials()
    fib = loaded._coords_tab.fib_list.coordinates
    fm = loaded._coords_tab.fm_list.coordinates
    assert len(fib) == len(fm) == len(ARCTIS["coords"]["fib"])
    assert all(c.provenance == PointProvenance.PATTERN for c in fib)
    # blank pixels: the fitter can only hand back a sub-pixel shuffle, so every
    # point stays within a pixel of where the pattern put it
    h, w = ARCTIS["fib"]["shape"]
    for c, (x, y) in zip(fib, ARCTIS["coords"]["fib"]):
        assert abs(c.point.x - x) < 1.0 and abs(c.point.y - y) < 1.0
        assert c.status in ("", PointStatus.FITTED, PointStatus.FIT_FAILED)
    assert all(c.provenance == PointProvenance.PROJECTED for c in fm)
    assert all(c.status in PointStatus.TENTATIVE for c in fm)
    assert sum(1 for c in fm if c.status == PointStatus.SUGGESTED) == 3
    # tentative points never feed the fit
    assert loaded.data.fib_coordinates == []
    assert loaded.data.fm_coordinates == []
    assert not loaded._can_run()
    assert "10 predicted" in loaded._rail._subtitles[STEP_FIDUCIALS].text() or (
        "predicted" in loaded._rail._subtitles[STEP_FIDUCIALS].text()
    )


def test_predictions_follow_the_nominal_transform(loaded):
    loaded.predict_fiducials()
    fib, fm = _images(ARCTIS, fm_geometry=True)
    nominal = nominal_transform(fib, fm)
    P = nominal.projection
    z_iso = loaded._fm_display.current_z * nominal.z_anisotropy
    for a, b in zip(
        loaded._coords_tab.fib_list.coordinates, loaded._coords_tab.fm_list.coordinates
    ):
        back = P @ np.array([b.point.x, b.point.y, b.point.z * nominal.z_anisotropy])
        # the same translation for every point: the pattern was projected rigidly
        assert np.allclose(
            np.array([a.point.x, a.point.y]) - back,
            loaded._last_translation(),
            atol=1e-6,
        )
    assert z_iso == pytest.approx(
        loaded._coords_tab.fm_list.coordinates[0].point.z * nominal.z_anisotropy
    )


def test_a_drop_confirms_the_pair_and_reprojects_the_rest(loaded):
    loaded.predict_fiducials()
    fm = loaded._coords_tab.fm_list.coordinates
    suggested = [i for i, c in enumerate(fm) if c.status == PointStatus.SUGGESTED]
    before = np.array([[c.point.x, c.point.y] for c in fm])
    i = suggested[0]
    dropped = loaded._coords_tab.fm_list.coordinates[i]
    dropped.point.x += 40.0
    dropped.point.y -= 25.0
    loaded._on_canvas_moved(dropped)
    fm = loaded._coords_tab.fm_list.coordinates
    assert (
        fm[i].status == PointStatus.ADJUSTED
    )  # blank pixels: the fit cannot improve it
    assert fm[i].provenance == PointProvenance.USER
    assert len(loaded.data.fib_coordinates) == 1
    after = np.array([[c.point.x, c.point.y] for c in fm])
    moved = after - before
    # every still-tentative point moved by the same offset as the drop
    tentative = [k for k, c in enumerate(fm) if c.status in PointStatus.TENTATIVE]
    assert tentative
    assert np.allclose(moved[tentative], moved[i], atol=1e-6)
    assert "1 of 4 pairs" in loaded._rail._subtitles[STEP_FIDUCIALS].text()


def test_rejected_pairs_leave_the_fit_but_stay_on_screen(loaded):
    loaded.predict_fiducials()
    fm = loaded._coords_tab.fm_list.coordinates
    for c in fm[:4]:
        c.status = PointStatus.CONFIRMED
    fm[1].status = PointStatus.REJECTED
    loaded._coords_tab.fm_list.coordinates = fm
    assert len(loaded.data.fib_coordinates) == 3
    assert len(loaded._coords_tab.fm_list.coordinates) == len(fm)


def test_four_confirmed_pairs_and_a_poi_make_the_run_possible(loaded):
    loaded.predict_fiducials()
    fm = loaded._coords_tab.fm_list.coordinates
    for c in fm[:4]:
        c.status = PointStatus.CONFIRMED
    loaded._coords_tab.fm_list.coordinates = fm
    loaded.data_changed.emit(loaded.data)
    assert not loaded._can_run()
    assert "point of interest" in loaded._lbl_status.text()
    assert loaded._rail._states[STEP_TARGET] == "todo"
    loaded._coords_tab.poi_list.coordinates = [
        Coordinate(PointXYZ(300.0, 300.0, 3.0), PointType.POI)
    ]
    loaded.data_changed.emit(loaded.data)
    assert loaded._can_run()
    assert loaded._lbl_status.text() == "Ready."
    assert loaded._rail._states[STEP_TARGET] == "done"
    assert loaded._rail._states[STEP_REVIEW] == "todo"


def test_all_steps_toggle_moves_pages_and_back(loaded):
    assert loaded._stack.count() == 4
    loaded._rail._all.setChecked(True)
    assert loaded._stack.count() == 0
    assert loaded._all_layout.count() - 1 == 4
    loaded._rail._all.setChecked(False)
    assert loaded._stack.count() == 4
    assert loaded._all_layout.count() - 1 == 0


# ── the state fields survive the file ─────────────────────────────────────


def test_coordinate_state_round_trips_and_does_not_change_identity():
    c = Coordinate(
        PointXYZ(1.0, 2.0, 3.0),
        PointType.FM,
        status=PointStatus.PREDICTED,
        provenance=PointProvenance.PROJECTED,
    )
    back = Coordinate.from_dict(json.loads(json.dumps(c.to_dict())))
    assert back.status == PointStatus.PREDICTED
    assert back.provenance == PointProvenance.PROJECTED
    assert not back.usable
    legacy = Coordinate.from_dict(
        {"point": {"x": 1, "y": 2, "z": 3}, "point_type": "FM"}
    )
    assert legacy.status == "" and legacy.usable


# ── the spot-burn lookup ──────────────────────────────────────────────────


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
        "      reference_imaging: {field_of_view1: 0.0001}\n"
    )
    burns, fov, reason = find_spot_burns(str(run))
    assert reason == ""
    assert fov == pytest.approx(1e-4)
    assert [(p.x, p.y) for p in burns] == [(0.25, 0.5), (0.75, 0.5)]
    burns, fov, reason = find_spot_burns(str(tmp_path))
    assert burns == [] and "experiment.yaml" in reason
