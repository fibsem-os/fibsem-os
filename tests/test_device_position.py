"""A device's place as a whole position: the origin's axes, the sample's for the rest.

The FM origin is partial on purpose -- a METEOR's is odemis's CALIB dx, dy -- and a
reader that filled the gaps with 0 put the FM overview's sample plane 32 mm below the
sample (odemis simulator, 2026-10-26). `device_position` fills them from the sample,
as odemis takes z from each grid's SAMPLE_CENTERS; `sample_height` is that z.
"""

import os
from datetime import datetime, timezone

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.geometry.stage_places import sample_height
from fibsem.structures.settings import FibsemStagePosition, SlotCalibration

SAMPLE_Z = 32.0e-3


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")
    )
    return microscope


def test_the_origin_axes_are_kept_and_the_rest_are_the_samples(microscope):
    origin = microscope.stage_movement.device_origin("FM")
    sample = FibsemStagePosition(x=1e-3, y=2e-3, z=SAMPLE_Z, r=0.0, t=0.6)

    place = microscope.stage_movement.device_position("FM", sample)

    for axis in ("x", "y", "z"):
        expected = getattr(origin, axis)
        expected = getattr(sample, axis) if expected is None else expected
        assert getattr(place, axis) == pytest.approx(expected), axis
    assert place.z == pytest.approx(SAMPLE_Z), "z invented rather than the sample's"


def test_the_pose_is_the_devices_imaging_pose(microscope):
    fib = microscope.get_orientation("FIB")
    place = microscope.stage_movement.device_position(
        "FM", FibsemStagePosition(x=0.0, y=0.0, z=SAMPLE_Z, r=0.0, t=0.6)
    )
    assert (place.r, place.t) == pytest.approx((fib.r, fib.t))


def test_without_a_position_it_is_the_stages(microscope):
    sem = microscope.get_orientation("SEM")
    microscope.move_stage_absolute(
        FibsemStagePosition(x=0.0, y=0.0, z=SAMPLE_Z, r=sem.r, t=sem.t)
    )
    assert microscope.stage_movement.device_position("FM").z == pytest.approx(SAMPLE_Z)


def test_the_sample_height_is_a_calibrated_slots(microscope):
    slots = list(microscope.sample_handling.holder.slots.values())
    assert sample_height(microscope) is None, "an uncalibrated holder has no height"

    slot = slots[0]
    slot.position = FibsemStagePosition(
        name=slot.name, x=0.0, y=0.0, z=SAMPLE_Z, r=0.0, t=np.radians(35.0)
    )
    slot.calibration = SlotCalibration(
        orientation="SEM",
        pre_tilt=35.0,
        rotation_reference=0.0,
        captured_at=datetime.now(timezone.utc).isoformat(),
    )
    assert sample_height(microscope) == pytest.approx(SAMPLE_Z)
