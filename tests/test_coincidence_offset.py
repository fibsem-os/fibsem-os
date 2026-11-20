"""Tests for fibsem.analysis.coincidence: SEM/FIB coincidence from features
picked in both images of a saved pair. Real `FibsemImage`s with the metadata the
coincidence tests use (tests.test_coincidence); the feature is placed in both
views through the stage projection the images record, so the answer is known
from the existing maths rather than from this module's."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from fibsem.alignment.coincidence import geometry_from_images
from fibsem.alignment.coincidence_target import measure_coincidence_at_target
from fibsem.analysis.coincidence import coincidence_offset
from fibsem.geometry.projection import BeamStageProjection, stage_to_pixel
from fibsem.structures.settings import BeamType
from tests.test_coincidence import _image_with_metadata
from tests.test_coincidence_target import PIXEL_SIZE, SHAPE, _fib, _pair, make_scene


def _blank_pair():
    data = np.zeros(SHAPE)
    return (
        _image_with_metadata(data, PIXEL_SIZE, BeamType.ELECTRON),
        _image_with_metadata(data, PIXEL_SIZE, BeamType.ION),
    )


def _one(sem, fib, sem_px, fib_px):
    """The answer for one feature, and its pair."""
    answer = coincidence_offset(sem, fib, [{"sem": sem_px, "fib": fib_px}])
    return answer, answer["pairs"][0]


def _plane(projection, feature, base):
    """Where *feature* (a stage position) sits in a view from stage *base*:
    metres from the centre, x right, y up."""
    x, y_down = projection.to_plane(feature, base)
    return np.array([x, -y_down])


@pytest.mark.parametrize(
    "dz, dy",
    [(10e-6, 0.0), (-6e-6, 0.0), (8e-6, 4e-6), (0.0, -5e-6)],
)
def test_the_vertical_move_it_answers_brings_the_feature_to_coincidence(dz, dy):
    """The feature is the point under stage position base + (0, dy, dz), put
    in both images by their projections. The move the answer's dy makes, by
    `vertical_move_delta`, must leave the SEM view as it was and put the
    feature in the FIB where the SEM offset, foreshortened, says."""
    sem, fib = _blank_pair()
    base = sem.metadata.microscope_state.stage_position
    feature = deepcopy(base)
    feature.y += dy
    feature.z += dz

    answer, pair = _one(
        sem, fib, stage_to_pixel(sem, feature), stage_to_pixel(fib, feature)
    )

    sem_view = BeamStageProjection.from_image(sem)
    fib_view = BeamStageProjection.from_image(fib)
    assert pair["sem_offset_m"] == pytest.approx(
        _plane(sem_view, feature, base), abs=1e-9
    )
    assert pair["fib_offset_m"] == pytest.approx(
        _plane(fib_view, feature, base), abs=1e-9
    )

    moved = deepcopy(base)
    for axis in ("x", "y", "z"):
        setattr(moved, axis, getattr(base, axis) + answer["stage_move_m"][axis])
    after_sem = _plane(sem_view, feature, moved)
    after_fib = _plane(fib_view, feature, moved)
    h, w = SHAPE
    lands = pair["after_fib_px"]
    landed = [(lands[0] - w / 2) * PIXEL_SIZE, (h / 2 - lands[1]) * PIXEL_SIZE]
    assert after_sem == pytest.approx(pair["sem_offset_m"], abs=1e-9)
    assert after_fib == pytest.approx(landed, abs=0.06 * PIXEL_SIZE)
    assert after_fib[1] == pytest.approx(after_sem[1] / answer["y_stretch"], abs=1e-9)
    assert answer["dx"] == pytest.approx(0.0, abs=1e-9)
    assert answer["warnings"] == []


def test_a_feature_above_where_coincidence_puts_it_answers_a_positive_dy():
    """The sign of stage_vertical_move: dy is the feature's offset up the
    FIB image from where it belongs, as the feature centred in the SEM and
    seen 7 um above the FIB centre is handed dy = +7 um."""
    sem, fib = _blank_pair()
    h, w = SHAPE
    answer, pair = _one(sem, fib, [w / 2, h / 2], [w / 2, h / 2 - 7e-6 / PIXEL_SIZE])
    assert answer["vertical_move"] == {
        "dy": pytest.approx(7e-6),
        "dx": 0.0,
        "beam_type": "ION",
        "from_pairs": ["1"],
    }
    assert "7.0 um above" in answer["direction"]
    assert pair["dy"] == pytest.approx(7e-6)
    assert pair["after_fib_px"] == pytest.approx([w / 2, h / 2])
    assert "spread_m" not in answer and "disagrees" not in pair
    geometry = geometry_from_images(sem, fib)
    assert answer["dz"] == pytest.approx(7e-6 * geometry.dz_per_dy, abs=1e-9)


def test_it_answers_what_the_target_measure_finds_given_its_match():
    """The same dy and dz as `measure_coincidence_at_target`, handed the
    point that measure found: one convention, not a second."""
    sem_data = make_scene()
    sem, fib = _pair(sem_data, _fib(sem_data, 12e-6))
    found = measure_coincidence_at_target(sem, fib, target=(20e-6, 10e-6))

    answer, _ = _one(sem, fib, found.target_px, found.found_px)

    assert answer["vertical_move"]["dy"] == pytest.approx(found.dy, abs=1e-9)
    assert answer["dz"] == pytest.approx(found.dz, abs=1e-9)
    assert answer["dx"] == pytest.approx(found.dx, abs=1e-9)
    assert answer["vertical_move"]["dy"] == pytest.approx(12e-6, abs=3 * PIXEL_SIZE)


def test_a_sideways_offset_is_a_warning_not_a_move():
    sem, fib = _blank_pair()
    h, w = SHAPE
    answer, _ = _one(sem, fib, [w / 2, h / 2], [w / 2 + 40, h / 2])
    assert answer["vertical_move"]["dy"] == pytest.approx(0.0)
    assert answer["dx"] == pytest.approx(-40 * PIXEL_SIZE)
    assert any("cannot move x" in warning for warning in answer["warnings"])


def _features(sem, fib, xs, dz=8e-6):
    """Pairs for features on one height (dz above the stage position), at
    stage x offsets *xs*: each pair's dy is the same."""
    base = sem.metadata.microscope_state.stage_position
    pairs = []
    for x in xs:
        feature = deepcopy(base)
        feature.x += x
        feature.z += dz
        pairs.append(
            {"sem": stage_to_pixel(sem, feature), "fib": stage_to_pixel(fib, feature)}
        )
    return pairs


def test_agreeing_pairs_give_their_median():
    sem, fib = _blank_pair()
    single = coincidence_offset(sem, fib, _features(sem, fib, [0.0]))
    answer = coincidence_offset(sem, fib, _features(sem, fib, [-20e-6, 0.0, 20e-6]))

    assert answer["vertical_move"]["from_pairs"] == ["1", "2", "3"]
    assert answer["vertical_move"]["dy"] == pytest.approx(
        single["vertical_move"]["dy"], abs=1e-9
    )
    assert answer["spread_m"]["dy"] == pytest.approx(0.0, abs=1e-8)
    assert not any("disagrees" in p for p in answer["pairs"])
    assert answer["warnings"] == []


def test_one_wrong_match_among_three_is_flagged_and_left_out():
    """The third FIB point is the wrong copy of a repeating feature, 15 um
    down the image: the median holds, and that pair says to check it."""
    sem, fib = _blank_pair()
    pairs = _features(sem, fib, [-20e-6, 0.0, 20e-6])
    true_dy = coincidence_offset(sem, fib, pairs)["vertical_move"]["dy"]
    x, y = pairs[2]["fib"]
    pairs[2] = {**pairs[2], "fib": (x, y + 15e-6 / PIXEL_SIZE), "label": "speck"}

    answer = coincidence_offset(sem, fib, pairs)

    assert answer["vertical_move"]["dy"] == pytest.approx(true_dy, abs=1e-9)
    assert answer["vertical_move"]["from_pairs"] == ["1", "2"]
    flagged = answer["pairs"][2]
    assert "check it is the same feature" in flagged["disagrees"]
    assert "15.0 um" in flagged["disagrees"]
    assert answer["spread_m"]["dy"] == pytest.approx(15e-6, abs=1e-8)
    assert "after_fib_px" in flagged  # where it would land, to judge by eye


def test_a_pair_whose_dx_differs_is_flagged():
    sem, fib = _blank_pair()
    pairs = _features(sem, fib, [-20e-6, 0.0, 20e-6])
    x, y = pairs[0]["fib"]
    pairs[0] = {**pairs[0], "fib": (x + 10e-6 / PIXEL_SIZE, y)}
    answer = coincidence_offset(sem, fib, pairs)
    assert "its dx is 10.0 um" in answer["pairs"][0]["disagrees"]
    assert answer["vertical_move"]["from_pairs"] == ["2", "3"]


def test_two_pairs_that_disagree_give_no_move():
    sem, fib = _blank_pair()
    pairs = _features(sem, fib, [0.0, 20e-6])
    x, y = pairs[1]["fib"]
    pairs[1] = {**pairs[1], "fib": (x, y - 12e-6 / PIXEL_SIZE)}

    answer = coincidence_offset(sem, fib, pairs)

    assert answer["vertical_move"] is None
    assert "none is chosen" in answer["direction"]
    assert [p["dy"] for p in answer["pairs"]] != [None, None]
    for pair in answer["pairs"]:
        assert "differ by 12.0 um" in pair["disagrees"]
        assert "after_fib_px" not in pair


def test_too_many_pairs_are_refused():
    sem, fib = _blank_pair()
    centre = [SHAPE[1] / 2, SHAPE[0] / 2]
    with pytest.raises(ValueError, match="1 to 10"):
        coincidence_offset(sem, fib, [{"sem": centre, "fib": centre}] * 11)


def test_a_pair_it_cannot_measure_is_refused():
    sem, fib = _blank_pair()
    centre = [SHAPE[1] / 2, SHAPE[0] / 2]

    with pytest.raises(ValueError, match="ION image; give an ELECTRON image"):
        _one(fib, sem, centre, centre)

    moved = deepcopy(fib)
    moved.metadata.microscope_state.stage_position.z += 5e-6
    answer = coincidence_offset(sem, moved, [{"sem": centre, "fib": centre}])
    assert any("z by 5.00 um" in w for w in answer["warnings"])
    assert answer["vertical_move"] is not None

    tilted = deepcopy(fib)
    tilted.metadata.microscope_state.stage_position.t += np.deg2rad(1.0)
    answer = coincidence_offset(sem, tilted, [{"sem": centre, "fib": centre}])
    assert any("t by 1.00 deg" in w for w in answer["warnings"])

    unscaled = deepcopy(fib)
    unscaled.metadata.pixel_size = None
    with pytest.raises(ValueError, match="no pixel size"):
        _one(sem, unscaled, centre, centre)

    no_geometry = deepcopy(fib)
    no_geometry.metadata.hardware_geometry = None
    with pytest.raises(ValueError, match="no coincidence geometry"):
        _one(sem, no_geometry, centre, centre)

    with pytest.raises(ValueError, match="outside"):
        _one(sem, fib, centre, [SHAPE[1] + 5, 0])


def test_the_tool_reads_saved_files_and_saves_a_picture(tmp_path):
    from fibsem.analysis.tools import ImageRefused, ImageTools

    sem_data = make_scene()
    sem, fib = _pair(sem_data, _fib(sem_data, 6e-6))
    sem.save(str(tmp_path / "site-sem.tif"))
    fib.save(str(tmp_path / "site-fib.tif"))
    tools = ImageTools(lambda: str(tmp_path))
    h, w = SHAPE
    fib_point = [w / 2, h / 2 - 6e-6 / PIXEL_SIZE]

    answer = tools.coincidence_offset(
        "site-sem.tif", [w / 2, h / 2], "site-fib.tif", fib_point
    )

    assert answer.measurement == "coincidence"
    assert answer.result["vertical_move"]["dy"] == pytest.approx(6e-6, rel=1e-3)
    assert answer.result["fib_path"] == "site-fib.tif"
    assert Path(answer.path).parts[-2:] == (
        "drawn",
        "site-sem-site-fib-coincidence.png",
    )
    assert Path(answer.path).is_file()

    # a point read off a crop's picture, converted for it
    cropped = tools.coincidence_offset(
        "site-sem.tif",
        [w / 2, h / 2],
        "site-fib.tif",
        [fib_point[0] - 300, fib_point[1] - 200],
        fib_crop=[300, 200, 200, 100],
    )
    assert cropped.result["pairs"][0]["fib_px"] == pytest.approx(fib_point, abs=0.05)

    labelled = tools.coincidence_offset(
        "site-sem.tif",
        None,
        "site-fib.tif",
        None,
        pairs=[
            {"sem": [w / 2, h / 2], "fib": fib_point, "label": "speck"},
            {"sem": [w / 2 + 50, h / 2], "fib": [fib_point[0] + 50, fib_point[1]]},
        ],
    )
    assert [p["label"] for p in labelled.result["pairs"]] == ["speck", "2"]
    assert labelled.result["vertical_move"]["from_pairs"] == ["speck", "2"]
    with pytest.raises(ImageRefused, match="not both"):
        tools.coincidence_offset(
            "site-sem.tif", [1, 1], "site-fib.tif", None, pairs=[{"sem": [1, 1]}]
        )

    with pytest.raises(ImageRefused) as swapped:
        tools.coincidence_offset("site-fib.tif", fib_point, "site-sem.tif", fib_point)
    assert swapped.value.error_type == "invalid"
