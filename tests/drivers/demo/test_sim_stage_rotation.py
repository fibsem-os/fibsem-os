"""The simulated sample turns with a rotating stage.

On a stage that reaches the ion beam by a half turn, fibsemOS assumes the
sample turns with it: `target_position` reflects x, y through the rotation
centre, and the projection flips positions recorded half a turn away. The
scene must agree, or a site milled in one orientation is not in view after
the move to the other. The default simulator is a compustage, where the half
turn never happens, so this runs on a rotating configuration.

The mark is located from the scene's milled geometry, not from images: frames
carry random noise and the periodic mesh fools correlation.
"""

import numpy as np
import pytest

from fibsem.drivers.demo.sim_scene import SampleScene, fm_channel_weights
from fibsem.geometry.projection import BeamStageProjection, FMStageProjection
from fibsem.structures.settings import BeamType, FibsemRectangleSettings
from tests.fixtures.microscopes import demo_microscope

microscope = demo_microscope("sim-iflm-configuration.yaml", scope="function")

# a site well away from the world origin, so a mirrored mark misses by twice
# its distance from it
SITE = (200e-6, 150e-6)


@pytest.fixture
def scene(microscope):
    scene = SampleScene(cell_type="none")
    scene.anchor(microscope._connection.scene_stage_position())
    microscope._connection.sample_scene = scene
    return scene


def _move_to(microscope, orientation, dx=0.0, dy=0.0):
    position = microscope.get_target_position(
        microscope.get_stage_position(), orientation
    )
    position.x += dx
    position.y += dy
    microscope.move_stage_absolute(position)
    return microscope._connection.scene_stage_position()


def _mill_at_crosshair(microscope, scene, beam_type, size=2e-6):
    projection = BeamStageProjection.from_microscope(microscope, beam_type=beam_type)
    pattern = FibsemRectangleSettings(
        width=size, height=size, depth=1e-6, centre_x=0.0, centre_y=0.0
    )
    position = microscope._connection.scene_stage_position()
    (mark,) = scene.mill([pattern], beam_type, position, projection)
    return mark.points.mean(axis=0)


def test_mark_milled_in_fib_orientation_is_under_the_sem_crosshair(microscope, scene):
    _move_to(microscope, "FIB", *SITE)
    mark = _mill_at_crosshair(microscope, scene, BeamType.ION)

    position = _move_to(microscope, "SEM")
    projection = BeamStageProjection.from_microscope(
        microscope, beam_type=BeamType.ELECTRON
    )
    centre = scene.view_to_world(BeamType.ELECTRON, position, projection, 0.0, 0.0)

    assert np.hypot(*(mark - centre)) < 3e-6


def test_mark_milled_in_sem_orientation_is_centred_in_the_fm(microscope, scene):
    """The offset FM images in the FIB pose: half a turn from the SEM."""
    _move_to(microscope, "SEM", *SITE)
    _mill_at_crosshair(microscope, scene, BeamType.ELECTRON, size=5e-6)

    movement = microscope.stage_movement
    microscope.move_stage_absolute(
        movement.target_position(
            microscope.get_stage_position(), "FIB", target_device="FM"
        )
    )

    # where the FM render puts the trench, from the mask it draws
    masks = []
    milled_mask = scene.milled_mask
    scene.milled_mask = lambda xs, ys: masks.append(milled_mask(xs, ys)) or masks[-1]
    projection = FMStageProjection.from_microscope(microscope)
    height, width = projection.shape
    scene.render_fm(
        microscope._connection.scene_stage_position_at_fm(),
        (width, height),
        projection,
        weights=fm_channel_weights(None),
    )

    rows, cols = np.nonzero(masks[0])
    assert rows.size, "the mark is out of the FM's view"
    offset = np.hypot(cols.mean() - width / 2, rows.mean() - height / 2)
    assert offset * projection.pixel_size < 3e-6
