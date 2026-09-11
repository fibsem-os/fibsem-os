"""The nominal FM->FIB transform built from image geometry (FIB-881).

The fixture is nine saved multi-point fits -- two Arctis (compustage), seven
METEOR (Aquilos2 offset mount) -- with the metadata their images recorded and
the fiducials they were fitted to. Every assertion here is against those fits:
the nominal transform must land on their branch and near their rotation, and a
fit seeded from it must find them. If the depth-axis sign in
``geometry._depth_column`` were wrong, every entry would fail, not one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fibsem.correlation import correlation_v2
from fibsem.correlation.geometry import (
    NominalTransform,
    NominalTransformError,
    branch_of,
    fm_geometry_for,
    nominal_transform,
    nominal_transform_from_geometry,
    rotation_angle_deg,
)
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationResult,
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

FIXTURE = Path(__file__).parent / "fixtures" / "nominal_transform.json"
ENTRIES = json.loads(FIXTURE.read_text())["entries"]
IDS = [e["name"] for e in ENTRIES]


def _nominal(entry: dict) -> NominalTransform:
    fib_geometry = FibsemHardwareGeometry.from_dict(entry["fib"]["hardware_geometry"])
    fm_geometry = fm_geometry_for(
        fib_geometry, CameraImageTransform(entry["fm"]["transform"])
    )
    return nominal_transform_from_geometry(
        fib_geometry=fib_geometry,
        fib_pose=FibsemStagePosition.from_dict(entry["fib"]["stage_position"]),
        scan_rotation=entry["fib"]["scan_rotation"],
        fib_pixel_size=entry["fib"]["pixel_size"],
        fib_shape=tuple(entry["fib"]["shape"]),
        fm_geometry=fm_geometry,
        fm_pose=FibsemStagePosition.from_dict(entry["fm"]["pose"]),
        fm_pixel_size=entry["fm"]["pixel_size_x"],
        fm_pixel_size_z=entry["fm"]["pixel_size_z"],
        fm_shape=tuple(entry["fm"]["shape"]),
    )


def _fitted_rotation(entry: dict) -> np.ndarray:
    return np.asarray(entry["fit"]["rotation"], dtype=float)


def _isotropic_coords(entry: dict, nominal: NominalTransform):
    """The fiducials with z in xy pixels, as the fit needs them."""
    fib = np.array([[x, y, 0.0] for x, y in entry["coords"]["fib"]], dtype=float)
    fm = np.array(entry["coords"]["fm"], dtype=float)
    # the 106-slice Arctis picks were made on an isotropic interpolation already
    if entry["fit"]["fm_slices_in_fit"] != 106:
        fm[:, 2] *= nominal.z_anisotropy
    return fib, fm


# ── the transform itself ──────────────────────────────────────────────────


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_rows_are_orthogonal_at_the_pixel_size_ratio(entry):
    nominal = _nominal(entry)
    gram = nominal.projection @ nominal.projection.T
    # to 1%: on the METEOR the FM pose is calibrated rather than recorded, and a
    # degree or two of pose error leaves the depth column very slightly long
    assert np.allclose(gram, nominal.scale**2 * np.eye(2), rtol=1e-2, atol=1e-3)
    assert nominal.scale == pytest.approx(
        entry["fm"]["pixel_size_x"] / entry["fib"]["pixel_size"]
    )


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_nominal_lands_on_the_fitted_branch(entry):
    nominal = _nominal(entry)
    branch, to_nominal, to_mirror = branch_of(_fitted_rotation(entry), nominal)
    assert branch == "nominal", (to_nominal, to_mirror)
    # a few degrees of holder rotation / pose uncertainty; the mirror is ~150 away
    assert to_nominal < 8.0
    assert to_mirror > 90.0


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_in_plane_axes_match_the_fit(entry):
    """Axis signs and foreshortening come out of the composed projections."""
    nominal = _nominal(entry)
    fitted = entry["fit"]["scale"] * _fitted_rotation(entry)[:2, :2]
    # Signs and rough direction: the fitted y column also carries the picks' few
    # degrees of in-plane residual, magnified by the foreshortening, so this is
    # a sign guard, not a precision one (the geodesic angle test is the latter)
    for col in range(2):
        a = nominal.projection[:, col] / np.linalg.norm(nominal.projection[:, col])
        b = fitted[:, col] / np.linalg.norm(fitted[:, col])
        assert float(a @ b) > 0.9, (col, a, b)
    fitted_fore = np.linalg.svd(fitted, compute_uv=False)
    assert nominal.foreshortening == pytest.approx(
        fitted_fore[1] / fitted_fore[0], abs=0.02
    )


def test_arctis_foreshortening_is_the_milling_angle():
    entry = next(e for e in ENTRIES if e["system"] == "arctis")
    nominal = _nominal(entry)
    assert nominal.foreshortening == pytest.approx(np.sin(np.radians(15.0)), abs=0.002)
    # identity in-plane: scan rotation pi and FLIP_XY cancel
    assert nominal.projection[0, 0] > 0 and nominal.projection[1, 1] > 0
    assert abs(nominal.projection[0, 1]) < 1e-6 and abs(nominal.projection[1, 0]) < 1e-6


def test_depth_lands_on_image_y_with_cos_milling_angle():
    entry = next(e for e in ENTRIES if e["system"] == "arctis")
    nominal = _nominal(entry)
    dz = nominal.projection[:, 2]
    assert abs(dz[0]) < 1e-6
    assert dz[1] == pytest.approx(np.cos(np.radians(15.0)) * nominal.scale, rel=1e-3)
    # one raw slice of this stack is 5.05 xy px deep
    assert nominal.dimage_dz_px_per_slice[1] == pytest.approx(
        dz[1] * nominal.z_anisotropy
    )


def test_mirror_is_a_proper_rotation_on_the_other_branch():
    nominal = _nominal(ENTRIES[0])
    mirror = nominal.mirrored()
    assert np.linalg.det(mirror.rotation) == pytest.approx(1.0)
    assert np.allclose(mirror.projection[:, :2], nominal.projection[:, :2])
    assert np.allclose(mirror.projection[:, 2], -nominal.projection[:, 2])
    assert rotation_angle_deg(nominal.rotation, mirror.rotation) > 90.0
    assert branch_of(mirror.rotation, nominal)[0] == "mirror"


def test_eulers_round_trip_through_pyto():
    from fibsem.correlation.pyto.rigid_3d import Rigid3D

    nominal = _nominal(ENTRIES[0])
    ck = Rigid3D.euler_to_ck(np.radians(nominal.eulers_deg()), mode="x")
    rebuilt = Rigid3D.make_r_ck(ck)
    assert rotation_angle_deg(rebuilt, nominal.rotation) < 1e-6


# ── seeding the solver ────────────────────────────────────────────────────


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_seeded_fit_finds_the_saved_solution(entry):
    nominal = _nominal(entry)
    fib, fm = _isotropic_coords(entry, nominal)
    rotation, scale, rms = correlation_v2._fit_from_seed(
        fm, fib, nominal.eulers_deg(), nominal.scale
    )
    assert branch_of(rotation, nominal)[0] == "nominal"
    assert rotation_angle_deg(rotation, _fitted_rotation(entry)) < 8.0
    # the saved fits were on raw slices (bar the interpolated Arctis one), so their
    # RMS is only a loose reference; what matters is that the seed converged
    assert rms < max(3 * entry["fit"]["rms"], 6.0)


@pytest.mark.parametrize("entry", ENTRIES, ids=IDS)
def test_mirror_seed_finds_the_mirror_branch(entry):
    nominal = _nominal(entry)
    fib, fm = _isotropic_coords(entry, nominal)
    mirror = nominal.mirrored()
    rotation, _, _ = correlation_v2._fit_from_seed(
        fm, fib, mirror.eulers_deg(), mirror.scale
    )
    assert branch_of(rotation, nominal)[0] == "mirror"


def test_correlate_rejects_a_malformed_seed():
    with pytest.raises(ValueError, match="rotation_init"):
        correlation_v2.correlate(
            markers_3d=np.zeros((3, 3)),
            markers_2d=np.zeros((3, 2)),
            poi_3d=np.zeros((0, 3)),
            rotation_center=(0, 0, 0),
            optimiser_params={"rotation_init": 45.0, "random_rotations": False},
        )


# ── through run_correlation_from_data ─────────────────────────────────────


def _images(entry: dict):
    """Real image objects carrying the fixture's metadata, with empty pixels."""
    fib_md = FibsemImageMetadata(
        image_settings=ImageSettings(beam_type=BeamType.ION),
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
    fm_geometry = fm_geometry_for(
        fib_md.hardware_geometry, CameraImageTransform(entry["fm"]["transform"])
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
        geometry=fm_geometry,
    )
    nz = 4
    fm = FluorescenceImage(
        data=np.zeros((1, nz, *entry["fm"]["shape"]), dtype=np.uint16), metadata=fm_md
    )
    return fib, fm


def _input_data(entry: dict) -> CorrelationInputData:
    fib, fm = _images(entry)
    return CorrelationInputData(
        fib_image=fib,
        fm_image=fm,
        fib_coordinates=[
            Coordinate(PointXYZ(x, y, 0.0), PointType.FIB)
            for x, y in entry["coords"]["fib"]
        ],
        fm_coordinates=[
            Coordinate(PointXYZ(x, y, z), PointType.FM)
            for x, y, z in entry["coords"]["fm"]
        ],
        poi_coordinates=[Coordinate(PointXYZ(500.0, 500.0, 10.0), PointType.POI)],
    )


def test_nominal_transform_reads_the_images():
    entry = next(
        e
        for e in ENTRIES
        if e["system"] == "arctis" and e["fit"]["fm_slices_in_fit"] != 106
    )
    fib, fm = _images(entry)
    nominal = nominal_transform(fib, fm)
    assert np.allclose(nominal.projection, _nominal(entry).projection)


def test_run_seeds_when_the_images_carry_geometry():
    entry = next(
        e
        for e in ENTRIES
        if e["system"] == "arctis" and e["fit"]["fm_slices_in_fit"] != 106
    )
    data = _input_data(entry)
    nominal = nominal_transform(data.fib_image, data.fm_image)
    result = run(data, nominal)
    assert result.seed is not None
    assert result.branch_check["selected"] == "nominal"
    assert result.branch_check["warning"] is None
    assert result.branch_check["rms_mirror"] > result.branch_check["rms_selected"]
    assert result.branch_check["angle_to_nominal_deg"] < 8.0
    # raw slices were scaled to xy pixels for the fit
    assert result.fm_z_scale == pytest.approx(nominal.z_anisotropy)
    assert branch_of(np.asarray(result.rotation_quaternion), nominal)[0] == "nominal"
    # depth per *slice*, back in the picked stack's units, on the fit's z column
    dz = result.dimage_dz_px_per_slice
    assert dz[1] == pytest.approx(nominal.dimage_dz_px_per_slice[1], rel=0.2)
    # and it all survives the file
    back = CorrelationResult.from_dict(json.loads(json.dumps(result.to_dict())))
    assert back.seed == result.seed and back.branch_check == result.branch_check
    assert back.fm_z_scale == result.fm_z_scale


def test_unseeded_run_reports_the_mirror_branch():
    entry = next(
        e
        for e in ENTRIES
        if e["system"] == "arctis" and e["fit"]["fm_slices_in_fit"] != 106
    )
    result = run(_input_data(entry), None)
    assert result.seed is None
    check = result.branch_check
    assert check["selected"] == "unseeded"
    assert check["rms_mirror"] is not None
    assert check["fiducial_z_span_slices"] > 0


def test_pre_correction_scales_with_the_stack():
    """The FM surface z is in slices too, so the transient RI scaling must follow."""
    entry = next(
        e
        for e in ENTRIES
        if e["system"] == "arctis" and e["fit"]["fm_slices_in_fit"] != 106
    )
    data = _input_data(entry)
    data.fm_surface_coordinate = Coordinate(
        PointXYZ(500.0, 500.0, 8.0), PointType.SURFACE_FM
    )
    data.ri_pre_correction_factor = 1.5
    plain = run(_input_data(entry), None)
    corrected = run(data, None)
    # 2 slices deeper x 1.5 = 3 slices: the POI moves 1 slice further, along +y
    dz = plain.dimage_dz_px_per_slice[1]
    shift = corrected.poi[0].image_px.y - plain.poi[0].image_px.y
    assert shift == pytest.approx(dz, rel=0.05)


def run(data: CorrelationInputData, nominal) -> CorrelationResult:
    return correlation_v2.run_correlation_from_data(data, nominal=nominal)


# ── refusing to guess ─────────────────────────────────────────────────────


def test_fm_without_geometry_raises():
    entry = next(e for e in ENTRIES if e["system"] == "arctis")
    fib, fm = _images(entry)
    fm.metadata.geometry = None
    with pytest.raises(NominalTransformError, match="geometry"):
        nominal_transform(fib, fm)


def test_fm_pose_without_rotation_or_tilt_is_completed_from_the_mounting():
    """Both mountings image at a pose the hardware fixes, so a position recorded
    without r and t (every odemis-written METEOR stack) is completed, not refused."""
    arctis = next(e for e in ENTRIES if e["system"] == "arctis")
    fib, fm = _images(arctis)
    fm.metadata.stage_position.r = None
    fm.metadata.stage_position.t = None
    nominal = nominal_transform(fib, fm)  # compustage: turned over to face the FM
    assert np.allclose(nominal.projection, _nominal(arctis).projection)

    meteor = next(e for e in ENTRIES if e["system"] == "meteor")
    fib, fm = _images(meteor)
    fm.metadata.stage_position.r = None
    fm.metadata.stage_position.t = None
    nominal = nominal_transform(fib, fm)  # offset mount: the stage's FIB orientation
    assert np.allclose(nominal.projection, _nominal(meteor).projection)
    assert branch_of(_fitted_rotation(meteor), nominal)[0] == "nominal"


def test_fm_without_any_stage_position_raises():
    entry = next(e for e in ENTRIES if e["system"] == "meteor")
    fib, fm = _images(entry)
    fm.metadata.stage_position = None
    with pytest.raises(NominalTransformError, match="stage position"):
        nominal_transform(fib, fm)


def test_fib_without_hardware_geometry_raises():
    entry = ENTRIES[0]
    fib, fm = _images(entry)
    fib.metadata.hardware_geometry = None
    with pytest.raises(NominalTransformError, match="hardware geometry"):
        nominal_transform(fib, fm)
