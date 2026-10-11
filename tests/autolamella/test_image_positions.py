"""Where a reference image was taken, against where the lamella is now (FIB-1170).

Images are real FibsemImages written to disk and read back from their headers, as the
lamella editor reads them.
"""

import math
from datetime import datetime

import numpy as np
import tifffile as tff

from fibsem.applications.autolamella.image_positions import (
    PositionMatch,
    compare_positions,
    latest_image_at,
    read_image_position,
)
from fibsem.structures import FibsemImage, FibsemStagePosition, MicroscopeState

POSE = FibsemStagePosition(
    x=100e-6, y=-50e-6, z=20e-6, r=0.0, t=math.radians(18), coordinate_system="RAW"
)


def _at(dx=0.0, dy=0.0, dz=0.0, dr=0.0, dt=0.0, **kwargs) -> FibsemStagePosition:
    position = FibsemStagePosition(
        x=POSE.x + dx,
        y=POSE.y + dy,
        z=POSE.z + dz,
        r=POSE.r + dr,
        t=POSE.t + dt,
        coordinate_system=POSE.coordinate_system,
    )
    for key, value in kwargs.items():
        setattr(position, key, value)
    return position


def _save(tmp_path, name, position=None, timestamp=1000.0) -> str:
    image = FibsemImage.generate_blank_image(resolution=[64, 48], hfw=100e-6)
    if position is not None:
        image.metadata.microscope_state = MicroscopeState(
            timestamp=timestamp, stage_position=position
        )
    return image.save(str(tmp_path / name))


def test_position_and_time_come_from_the_header(tmp_path):
    path = _save(tmp_path, "ref_Rough Milling_final_res_01_ib.tif", _at(dx=3e-6), 42.0)
    image = read_image_position(path)
    assert image.filename == "ref_Rough Milling_final_res_01_ib.tif"
    assert image.timestamp == 42.0
    assert image.stage_position.x == POSE.x + 3e-6
    assert image.stage_position.coordinate_system == "RAW"
    # the same position the full load gives
    loaded = FibsemImage.load(path).metadata.microscope_state.stage_position
    assert image.stage_position == loaded


def test_a_thermofisher_timestamp_string_is_read(tmp_path):
    """The ThermoFisher driver records AutoScript's acquisition_datetime, a string."""
    path = _save(tmp_path, "tfs_ib.tif", _at(), timestamp="09/13/2026 20:32:36")
    expected = datetime(2026, 9, 13, 20, 32, 36).timestamp()
    assert read_image_position(path).timestamp == expected


def test_an_image_without_metadata_is_unknown_not_moved(tmp_path):
    bare = str(tmp_path / "bare_ib.tif")
    tff.imwrite(bare, np.zeros((48, 64), dtype=np.uint8))
    no_state = _save(tmp_path, "no_state_ib.tif")  # metadata, but no microscope state

    for path in (bare, no_state):
        image = read_image_position(path)
        assert image.stage_position is None
        assert image.timestamp > 0  # falls back to the file's mtime
        comparison = compare_positions(image.stage_position, POSE)
        assert comparison.match is PositionMatch.UNKNOWN


def test_a_missing_file_is_unknown(tmp_path):
    image = read_image_position(str(tmp_path / "gone_ib.tif"))
    assert image.stage_position is None


def test_translation_tolerance():
    assert compare_positions(_at(dx=4e-6), POSE).match is PositionMatch.SAME
    moved = compare_positions(_at(dx=6e-6), POSE)
    assert moved.match is PositionMatch.MOVED
    assert math.isclose(moved.distance_m, 6e-6)
    # x, y and z together: 3-4-0 um is 5 um, 3-4-12 is 13
    assert math.isclose(compare_positions(_at(3e-6, 4e-6), POSE).distance_m, 5e-6)
    assert compare_positions(_at(3e-6, 4e-6, 12e-6), POSE).match is PositionMatch.MOVED


def test_tilt_and_rotation_tolerance():
    assert compare_positions(_at(dt=math.radians(0.4)), POSE).match is (
        PositionMatch.SAME
    )
    tilted = compare_positions(_at(dt=math.radians(1.0)), POSE)
    assert tilted.match is PositionMatch.MOVED
    assert tilted.describe() == "tilted 1.0°"
    assert compare_positions(_at(dr=math.radians(90)), POSE).match is (
        PositionMatch.MOVED
    )


def test_rotation_wraps():
    near_full_turn = _at(r=math.radians(359.9))
    assert compare_positions(near_full_turn, POSE).match is PositionMatch.SAME


def test_an_axis_missing_on_either_side_is_not_compared():
    assert compare_positions(_at(z=None), POSE).match is PositionMatch.SAME
    assert compare_positions(_at(r=None, t=None), POSE).match is PositionMatch.SAME
    # ... but no x or y is no position at all
    assert compare_positions(_at(x=None), POSE).match is PositionMatch.UNKNOWN
    assert compare_positions(POSE, FibsemStagePosition()).match is PositionMatch.UNKNOWN
    assert compare_positions(POSE, None).match is PositionMatch.UNKNOWN


def test_stage_frames_must_agree_when_both_are_named():
    specimen = _at(coordinate_system="SPECIMEN")
    assert compare_positions(specimen, POSE).match is PositionMatch.UNKNOWN
    assert compare_positions(_at(coordinate_system="raw"), POSE).match is (
        PositionMatch.SAME
    )
    # older files did not record the frame
    assert compare_positions(_at(coordinate_system=None), POSE).match is (
        PositionMatch.SAME
    )


def test_describe_reads_as_a_rough_distance():
    assert compare_positions(_at(dx=42.4e-6), POSE).describe() == "~42 µm"
    assert compare_positions(_at(dx=7.3e-6), POSE).describe() == "~7.3 µm"
    both = compare_positions(_at(dx=50e-6, dt=math.radians(2)), POSE)
    assert both.describe() == "~50 µm, tilted 2.0°"


def test_latest_image_at_the_pose(tmp_path):
    paths = [
        _save(tmp_path, "a_ib.tif", _at(), timestamp=10.0),
        _save(tmp_path, "b_ib.tif", _at(dx=1e-6), timestamp=30.0),
        _save(tmp_path, "c_ib.tif", _at(dx=60e-6), timestamp=50.0),  # elsewhere
        _save(tmp_path, "d_ib.tif", None),  # unknown
    ]
    images = [read_image_position(p) for p in paths]
    assert latest_image_at(images, POSE).filename == "b_ib.tif"
    assert latest_image_at(images, _at(dx=200e-6)) is None
    assert latest_image_at(images, None) is None
