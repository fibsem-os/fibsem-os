"""A position's volume milling, in one piece: every piece placed from the point
of interest (the start slice's centre) and the slicing direction."""

import pytest

from fibsem.applications.volume.position import VolumePosition
from fibsem.applications.volume.tasks.milling import (
    VolumeFIBMillingTaskConfig,
    lanes_at,
    slice_at,
    start_slice_at,
    trench_at,
)
from fibsem.structures.settings import Point

POI = Point(1e-6, 2e-6)


def _pattern(milling, index=0):
    return milling.stages[index].pattern


@pytest.fixture
def config():
    config = VolumeFIBMillingTaskConfig(slice_thickness=100e-9)
    _pattern(config.start_slice).height = 2e-6
    _pattern(config.milling).height = 100e-9
    return config


@pytest.fixture
def position():
    return VolumePosition(name="aspen", poi=Point(POI.x, POI.y), iterations=40)


def _edges(pattern):
    return (
        pattern.point.y - pattern.height / 2,
        pattern.point.y + pattern.height / 2,
    )


def test_the_start_slice_is_centred_on_the_point_of_interest(config, position):
    start = _pattern(start_slice_at(config, position))

    assert (start.point.x, start.point.y) == (POI.x, POI.y)
    assert start.scan_direction == "TopToBottom"


def test_each_slice_cuts_one_thickness_past_the_one_before(config, position):
    start = _pattern(config.start_slice)
    face = POI.y - start.height / 2

    for index in (0, 1, 39):
        cut = _pattern(slice_at(config, position, index))
        bottom, top = _edges(cut)
        assert bottom == pytest.approx(face - (index + 1) * 100e-9)
        assert top == pytest.approx(face - index * 100e-9)
        assert cut.point.x == POI.x
        # as wide and deep as the start slice
        assert (cut.width, cut.depth) == (start.width, start.depth)
    # the protocol's config is not moved
    assert _pattern(config.milling).point == Point()


def test_the_trench_is_against_the_start_slice_on_the_far_side(config, position):
    start_bottom, start_top = _edges(_pattern(start_slice_at(config, position)))
    trench = _pattern(trench_at(config, position))

    assert _edges(trench)[0] == pytest.approx(start_top)
    assert trench.point.x == POI.x


def test_a_lane_runs_down_each_side_for_the_slices_whole_run(config, position):
    start = _pattern(config.start_slice)
    lanes = lanes_at(config, position)

    assert [s.name for s in lanes.stages] == [
        "Redeposition lane · left",
        "Redeposition lane · right",
    ]
    last_bottom, _ = _edges(_pattern(slice_at(config, position, 39)))
    for stage, side in zip(lanes.stages, (-1, 1)):
        lane = stage.pattern
        bottom, top = _edges(lane)
        assert top == pytest.approx(POI.y + start.height / 2)
        assert bottom == pytest.approx(last_bottom)
        # against the slices' side
        assert lane.point.x - side * lane.width / 2 == pytest.approx(
            POI.x + side * start.width / 2
        )


def test_each_lane_is_optional(config, position):
    config.lane_left = False
    assert [s.name for s in lanes_at(config, position).stages] == [
        "Redeposition lane · right"
    ]
    config.lane_right = False
    assert lanes_at(config, position) is None


def test_flipped_slices_go_up_the_image_with_the_trench_below(config, position):
    config.flipped = True
    start = _pattern(config.start_slice)

    cut = _pattern(slice_at(config, position, 0))
    assert _edges(cut)[1] == pytest.approx(POI.y + start.height / 2 + 100e-9)
    assert cut.scan_direction == "BottomToTop"
    trench = _pattern(trench_at(config, position))
    assert _edges(trench)[1] == pytest.approx(POI.y - start.height / 2)
    assert trench.scan_direction == "BottomToTop"
