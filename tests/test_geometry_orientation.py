"""A saved image names the view it was taken in, with no microscope to ask (FIB-811).

`FibsemMicroscope.get_stage_orientation` and `get_current_milling_angle` read
everything they need from the instrument's configuration, and every image records
that configuration's geometry. So the rule is in `fibsem.geometry.orientation`, the
microscope delegates to it, and a saved image is classified by the same rule against
its own geometry. These pin that the two answers agree, and what an image answers.

The extraction itself was checked separately: the old methods and the delegating ones
were run across 388,944 combinations of stage type, pre-tilt, column tilt, reference
rotation, target milling angle, tilt and rotation, with byte-identical output.
"""

import itertools
import json
import os

import numpy as np
import pytest

from fibsem import utils
from fibsem.geometry.orientation import (
    image_milling_angle,
    image_orientation,
    stage_milling_angle,
    stage_orientation,
)
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    FibsemStagePosition,
)

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "metadata")


@pytest.fixture(params=[False, True], ids=["offset-stage", "compustage"])
def microscope(request):
    scope, _ = utils.setup_session(manufacturer="Demo")
    scope.stage_is_compustage = request.param
    scope._update_orientations()
    return scope


def _configure(microscope, pretilt, rotation_reference, milling_angle):
    stage = microscope.system.stage
    stage.shuttle_pre_tilt = pretilt
    stage.rotation_reference = rotation_reference
    stage.milling_angle = milling_angle
    microscope._update_orientations()


@pytest.mark.parametrize(
    "pretilt,rotation_reference,milling_angle",
    list(itertools.product([0, 35, 45], [0, 180], [12, 20])),
)
def test_the_image_and_the_live_microscope_agree(
    microscope, pretilt, rotation_reference, milling_angle
):
    """Same pose, same geometry: the same orientation and the same milling angle.

    The live side reads the microscope's configuration; the offline side only the
    geometry an image would record, which has no target milling angle in it.
    """
    _configure(microscope, pretilt, rotation_reference, milling_angle)
    geometry = microscope.hardware_geometry()

    seen = set()
    for tilt, rotation in itertools.product(
        range(-200, 100, 2), [-90, 0, 3, 90, 180, 183, 270, 365]
    ):
        position = FibsemStagePosition(r=np.radians(rotation), t=np.radians(tilt))
        live = microscope.get_stage_orientation(position)
        assert stage_orientation(position, geometry) == live, (tilt, rotation)
        assert stage_milling_angle(position, geometry) == pytest.approx(
            microscope.get_current_milling_angle(position)
        ), (tilt, rotation)
        seen.add(live)

    # The sweep reaches every orientation the stage has, so none agrees vacuously.
    expected = {"SEM", "MILLING", "FIB", "NONE"}
    if microscope.stage_is_compustage:
        expected.add("FM")
    assert seen == expected


def test_a_lamella_milled_at_another_angle_is_still_at_milling(microscope):
    """An image does not record the site's target milling angle, and needs not to:
    any tilt off flat at the SEM rotation is the milling orientation."""
    _configure(microscope, pretilt=0, rotation_reference=0, milling_angle=12)
    geometry = microscope.hardware_geometry()
    stage_tilt = np.radians(20 - 90 + 52)  # milled at 20, with 52 column tilt

    position = FibsemStagePosition(r=0.0, t=stage_tilt)

    assert stage_orientation(position, geometry) == "MILLING"
    assert stage_milling_angle(position, geometry) == pytest.approx(20)


def test_a_saved_image_names_the_view_it_was_taken_in(microscope, tmp_path):
    for orientation in ("SEM", "FIB", "MILLING"):
        microscope.move_to_orientation(orientation)
        reported = microscope.get_stage_orientation()
        angle = microscope.get_current_milling_angle()
        image = microscope.acquire_image(beam_type=BeamType.ION)
        path = image.save(tmp_path / orientation)

        metadata = FibsemImage.load(path).metadata

        assert reported == orientation
        assert image_orientation(metadata) == orientation
        assert image_milling_angle(metadata) == pytest.approx(angle)


@pytest.mark.parametrize(
    "name,orientation,angle",
    [
        # stage tilt -23 under a 52 degree ion column, no pre-tilt
        ("thermo_arctis_compustage_v3.json", "MILLING", 15.0),
        # stage tilt 6 on a 35 degree shuttle at its reference rotation
        ("thermo_aquilos_v3.json", "MILLING", 9.0),
    ],
)
def test_a_real_instrument_image_is_named_offline(name, orientation, angle):
    """Written before images recorded a geometry record (v3): the geometry recovered
    from their embedded configuration is enough."""
    with open(os.path.join(FIXTURE_DIR, name)) as f:
        metadata = FibsemImageMetadata.from_dict(json.load(f))

    assert image_orientation(metadata) == orientation
    assert image_milling_angle(metadata) == pytest.approx(angle, abs=0.05)


def test_an_image_without_geometry_or_pose_is_unknown_not_none_of_them():
    """None, not "NONE": "NONE" says the stage was somewhere unnamed; None says
    nothing records where it was."""
    with open(os.path.join(FIXTURE_DIR, "thermo_arctis_compustage_v3.json")) as f:
        settings = json.load(f)

    no_geometry = FibsemImageMetadata.from_dict(settings)
    no_geometry.hardware_geometry = None
    assert image_orientation(no_geometry) is None
    assert image_milling_angle(no_geometry) is None

    no_pose = FibsemImageMetadata.from_dict(settings)
    no_pose.microscope_state.stage_position.t = None
    assert image_orientation(no_pose) is None
    assert image_milling_angle(no_pose) is None
