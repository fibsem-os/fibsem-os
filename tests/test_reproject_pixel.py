"""A pixel measured on one frame, drawn on another (FIB-1411): the agent's
values are kept with the frame they were measured on and shown on whatever
frame a view holds now."""

from __future__ import annotations

import os

import pytest

from fibsem import utils
from fibsem.geometry.projection import image_point_to_stage, reproject_pixel
from fibsem.structures import BeamType, FibsemImage, FibsemStagePosition, ImageSettings

SHAPE = (1024, 1536)


@pytest.fixture(scope="module")
def scope():
    import fibsem.config as fibsem_config

    path = os.path.join(
        os.path.dirname(fibsem_config.__file__), "config", "sim-arctis-configuration.yaml"
    )
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    yield microscope
    microscope.disconnect()


def _frame(scope, hfw, beam_type=BeamType.ION, dx=0.0, geometry=True):
    h, w = SHAPE
    image = FibsemImage.generate_blank_image(resolution=(w, h), hfw=hfw)
    image.metadata.image_settings = ImageSettings(hfw=hfw, beam_type=beam_type)
    if geometry:
        milling = scope.get_orientation("MILLING")
        state = scope.get_microscope_state(beam_type=beam_type)
        state.stage_position = FibsemStagePosition(
            x=dx, y=0.0, z=0.0, r=milling.r, t=milling.t
        )
        image.metadata.microscope_state = state
        image.metadata.system_info = scope.system.info
        image.metadata.hardware_geometry = scope.hardware_geometry()
    return image


def test_a_frame_onto_itself_is_the_same_pixel(scope):
    frame = _frame(scope, 150e-6)
    assert reproject_pixel(frame, frame, 100.0, 200.0) == (100.0, 200.0)


@pytest.mark.parametrize("geometry", [True, False])
def test_another_field_of_view_scales_about_the_centre(scope, geometry):
    wide = _frame(scope, 150e-6, geometry=geometry)
    close = _frame(scope, 75e-6, geometry=geometry)
    x, y = reproject_pixel(wide, close, 768 + 100, 512 - 50)
    assert (x, y) == pytest.approx((768 + 200, 512 - 100), abs=1e-6)


def test_after_a_stage_move_the_pixel_is_the_same_place_on_the_sample(scope):
    before = _frame(scope, 150e-6)
    after = _frame(scope, 100e-6, dx=10e-6)
    x, y = reproject_pixel(before, after, 900.0, 300.0)
    there = image_point_to_stage(before, 900.0, 300.0)
    seen = image_point_to_stage(after, x, y)
    assert (seen.x, seen.y, seen.z) == pytest.approx(
        (there.x, there.y, there.z), abs=1e-9
    )
    # Moved 10 µm in x, the place sits 10 µm (about 15 px at 100 µm) to the left.
    assert x == pytest.approx(768 + (900 - 768) * 1.5 - 10e-6 / (100e-6 / 1536), abs=0.5)


def test_a_beam_point_onto_the_other_beam(scope):
    """A point measured in the SEM, drawn on the FIB: through the stage."""
    sem = _frame(scope, 150e-6, beam_type=BeamType.ELECTRON)
    fib = _frame(scope, 150e-6, beam_type=BeamType.ION)
    x, y = reproject_pixel(sem, fib, 900.0, 300.0)
    there = image_point_to_stage(sem, 900.0, 300.0)
    seen = image_point_to_stage(fib, x, y)
    assert (seen.x, seen.y, seen.z) == pytest.approx(
        (there.x, there.y, there.z), abs=1e-9
    )
