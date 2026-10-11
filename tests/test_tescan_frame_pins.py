"""Pin what Tescan's moves and projections do today, where the operator sees it.

The Tescan stage is moving onto a device, and then into fibsem's stage frame (FIB-1161,
FIB-1114). Positions in fibsem's frame are what that work changes, so nothing here is
pinned in them. Everything is pinned where it must not change:

* **the stage command**: the ``Stage.MoveTo`` a move, a click on a live view or a click
  on a saved image sends, in Tescan's own millimetres and degrees;
* **the image**: the plane offset a position projects to on a live view, and the pixel a
  saved position is drawn at on an image taken at the same or another pose;
* **the pose name** a position at each pose is classified as.

The same click must send the same command and land on the same pixel after the frame
change, so these tests run unchanged across it.

Each case runs on a TescanMicroscope connected over a recording fake of the SDK
(``tests/fixtures/tescan_sdk.py``), at SEM, FIB and MILLING, with and without shuttle
pre-tilt, both beams, scan rotation 0 and 180. The expected values are in
``tests/fixtures/tescan_frame_pins.json``, recorded from main at the commit that added
this file. A change that is meant to move one regenerates the file with

    PYTHONPATH=. python tests/test_tescan_frame_pins.py --write

and says in its pull request which values moved and why.
"""

import itertools
import json
import os
import sys
from typing import Any, Callable, Dict

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.imaging.tiling.reprojection import reproject_stage_positions_onto_image2
from fibsem.projection import BeamStageProjection
from fibsem.structures import BeamType, FibsemStagePosition
from tests.fixtures.tescan_sdk import connect, image_at

PINS_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "tescan_frame_pins.json"
)

PRETILTS = (0.0, 35.0)  # degrees
POSES = ("SEM", "FIB", "MILLING")
BEAMS = (BeamType.ELECTRON, BeamType.ION)
SCAN_ROTATIONS = (0.0, 180.0)  # degrees

# Where every case starts, in Tescan millimetres; rotation and tilt come from the pose.
BASE_XYZ = (1.2, -0.8, 5.0)
# A second position near it, as a saved position or a target would be.
OFFSET_XYZ = (0.03, -0.02, 0.01)
# An image-space offset (metres), as a click or a measured shift would be.
DX, DY = 20e-6, -15e-6


def _system(pretilt: float):
    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    system.stage.shuttle_pre_tilt = pretilt
    return system


def _go(microscope, fake, pose: str, xyz=BASE_XYZ) -> FibsemStagePosition:
    """Put the fake stage at ``xyz`` (mm) and ``pose``; return the driver's read-back."""
    orientation = microscope.get_orientation(pose)
    fake.Stage.position = [
        *xyz,
        float(np.rad2deg(orientation.r)),
        float(np.rad2deg(orientation.t)),
    ]
    return microscope.get_stage_position()


def _offset(xyz=BASE_XYZ):
    return tuple(a + b for a, b in zip(xyz, OFFSET_XYZ))


def _moves(fake) -> list:
    return fake.calls("Stage.MoveTo")


def _point(pt) -> list:
    return [float(pt.x), float(pt.y)]


# -- the cases ------------------------------------------------------------------------


def _stable_move(m, f, pose, beam):
    _go(m, f, pose)
    m.stable_move(DX, DY, beam)
    return _moves(f)


def _project_stable_move_then_move(m, f, pose, beam):
    base = _go(m, f, pose)
    m.move_stage_absolute(m.project_stable_move(DX, DY, beam, base))
    return _moves(f)


def _vertical_move(m, f, pose, beam):
    _go(m, f, pose)
    m.vertical_move(dy=DY, dx=DX if beam is BeamType.ELECTRON else 0.0, beam_type=beam)
    return _moves(f)


def _live_click(m, f, pose, beam):
    """Double-click on a live view: the plane offset back to a stage move."""
    base = _go(m, f, pose)
    projection = BeamStageProjection.from_microscope(m, beam)
    m.move_stage_absolute(projection.from_plane(DX, DY, base))
    return _moves(f)


def _live_to_plane(m, f, pose, beam):
    """Where a nearby position is drawn on a live view, as a plane offset (metres)."""
    target = _go(m, f, pose, _offset())
    base = _go(m, f, pose)
    projection = BeamStageProjection.from_microscope(m, beam)
    return [float(v) for v in projection.to_plane(target, base)]


def _image_click(m, f, pose, beam):
    """Double-click on a saved image: the click back to a stage move."""
    _go(m, f, pose)
    image = image_at(m, f, beam)
    _go(m, f, "SEM", (0.0, 0.0, 0.0))  # the stage has since moved on
    projection = BeamStageProjection.from_image(image)
    base = image.metadata.microscope_state.stage_position
    m.move_stage_absolute(projection.from_plane(DX, DY, base))
    return _moves(f)


def _saved_positions(m, f):
    """A position saved at each pose, near the base, as the driver read it back."""
    saved = []
    for pose in POSES:
        position = _go(m, f, pose, _offset())
        position.name = pose
        saved.append(position)
    return saved


def _reprojection(m, f, pose, beam):
    """Where positions saved at each pose are drawn on an image taken at ``pose``."""
    saved = _saved_positions(m, f)
    _go(m, f, pose)
    image = image_at(m, f, beam)
    points = reproject_stage_positions_onto_image2(image, saved)
    return {pt.name: _point(pt) for pt in points}


def _image_to_plane(m, f, pose, beam):
    """The same, through the projection the overview draws with (plane offsets)."""
    saved = _saved_positions(m, f)
    _go(m, f, pose)
    image = image_at(m, f, beam)
    projection = BeamStageProjection.from_image(image)
    base = image.metadata.microscope_state.stage_position
    return {p.name: [float(v) for v in projection.to_plane(p, base)] for p in saved}


BEAM_CASES: Dict[str, Callable] = {
    "stable_move": _stable_move,
    "project_stable_move": _project_stable_move_then_move,
    "vertical_move": _vertical_move,
    "live_click": _live_click,
    "live_to_plane": _live_to_plane,
    "image_click": _image_click,
    "reprojection": _reprojection,
    "image_to_plane": _image_to_plane,
}


def _relative_move(m, f, pose):
    _go(m, f, pose)
    m.move_stage_relative(FibsemStagePosition(x=10e-6, y=-20e-6, z=5e-6, r=0.0, t=0.0))
    return _moves(f)


def _move_to_pose(m, f, pose):
    """Going to a pose from elsewhere, and what the arrival is classified as."""
    _go(m, f, "SEM", (0.0, 0.0, 0.0))
    orientation = m.get_orientation(pose)
    m.move_stage_absolute(FibsemStagePosition(r=orientation.r, t=orientation.t))
    return {
        "moves": _moves(f),
        "classified": m.get_stage_orientation(m.get_stage_position()),
    }


def _absolute_return(m, f, pose):
    """Going back to a position read at ``pose`` after the stage has moved away."""
    saved = _go(m, f, pose, _offset())
    _go(m, f, "SEM", (0.0, 0.0, 0.0))
    m.move_stage_absolute(saved)
    return _moves(f)


POSE_CASES: Dict[str, Callable] = {
    "relative_move": _relative_move,
    "move_to_pose": _move_to_pose,
    "absolute_return": _absolute_return,
}


def _case_ids():
    for pretilt, pose in itertools.product(PRETILTS, POSES):
        for name in POSE_CASES:
            yield f"pretilt{pretilt:g}-{pose}-{name}"
        for beam, scan, name in itertools.product(BEAMS, SCAN_ROTATIONS, BEAM_CASES):
            yield f"pretilt{pretilt:g}-{pose}-{beam.name}-scan{scan:g}-{name}"


def run_case(monkeypatch, case_id: str) -> Any:
    parts = case_id.split("-")
    pretilt = float(parts[0][len("pretilt") :])
    pose = parts[1]
    microscope, fake = connect(monkeypatch, _system(pretilt))
    microscope._update_orientations()
    try:
        if len(parts) == 3:
            return POSE_CASES[parts[2]](microscope, fake, pose)
        beam = BeamType[parts[2]]
        scan = float(parts[3][len("scan") :])
        fake.column(beam).Optics.image_rotation = scan
        fake.log.clear()
        return BEAM_CASES[parts[4]](microscope, fake, pose, beam)
    except Exception as e:  # a refusal is behaviour too
        return {"raises": type(e).__name__}


def _load_pins() -> Dict[str, Any]:
    with open(PINS_PATH) as f:
        return json.load(f)


def _assert_same(actual, expected, where=""):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), where
        assert sorted(actual) == sorted(expected), where
        for key in expected:
            _assert_same(actual[key], expected[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), where
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_same(a, e, f"{where}[{i}]")
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9), where
    else:
        assert actual == expected, where


CASE_IDS = list(_case_ids())


def test_every_case_is_pinned():
    assert sorted(_load_pins()) == sorted(CASE_IDS)


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_tescan_behaviour_is_unchanged(monkeypatch, case_id):
    expected = _load_pins()[case_id]
    actual = json.loads(json.dumps(run_case(monkeypatch, case_id)))
    _assert_same(actual, expected, case_id)


def _write_pins():
    pins = {}
    for case_id in CASE_IDS:
        with pytest.MonkeyPatch.context() as monkeypatch:
            pins[case_id] = run_case(monkeypatch, case_id)
    with open(PINS_PATH, "w") as f:
        json.dump(pins, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"wrote {len(pins)} cases to {PINS_PATH}")


if __name__ == "__main__":
    if "--write" in sys.argv:
        _write_pins()
