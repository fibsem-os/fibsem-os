"""Tests for deriving a lamella's milling angle from its milling-pose stage tilt."""

import numpy as np
import pytest

import fibsem.utils as utils
from fibsem.applications.autolamella.structures import Lamella
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.structures import FibsemStagePosition, MicroscopeState


@pytest.fixture(scope="module")
def microscope() -> DemoMicroscope:
    return DemoMicroscope(utils.load_microscope_configuration().system)


def _make_lamella(tmp_path, tilt_rad) -> Lamella:
    lamella = Lamella(petname="test-lamella", path=str(tmp_path / "test-lamella"), number=1)
    lamella.milling_pose = MicroscopeState(
        stage_position=FibsemStagePosition(
            x=0.0, y=0.0, z=0.0, r=0.0, t=tilt_rad, coordinate_system="RAW"
        )
    )
    return lamella


def test_update_milling_angle_from_pose(microscope, tmp_path):
    """The milling angle is computed from the stored milling-pose stage tilt."""
    lamella = _make_lamella(tmp_path, tilt_rad=np.radians(20))

    lamella.update_milling_angle(microscope)

    # milling angle = stage tilt + pretilt (see get_current_milling_angle)
    expected = microscope.get_current_milling_angle(
        stage_position=lamella.stage_position
    )
    assert lamella.milling_angle == pytest.approx(expected)
    assert lamella.milling_angle == pytest.approx(23.0, abs=0.1)


def test_update_milling_angle_missing_tilt_is_noop(microscope, tmp_path):
    """If the stage tilt is unavailable, the existing milling angle is left untouched."""
    lamella = _make_lamella(tmp_path, tilt_rad=np.radians(15))
    lamella.milling_angle = 15.0
    lamella.milling_pose.stage_position.t = None

    lamella.update_milling_angle(microscope)  # must not raise

    assert lamella.milling_angle == 15.0


def test_update_milling_angle_no_milling_pose_is_noop(microscope, tmp_path):
    """A lamella without a milling pose is handled gracefully."""
    lamella = Lamella(petname="no-pose", path=str(tmp_path / "no-pose"), number=1)
    lamella.update_milling_angle(microscope)  # must not raise
    assert lamella.milling_angle is None
