"""Tests for deriving a lamella's milling angle from its milling-pose stage tilt."""

import numpy as np
import pytest

import fibsem.utils as utils
from fibsem.applications.autolamella.structures import Lamella
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.structures import FibsemStagePosition, MicroscopeState


@pytest.fixture(scope="module")
def microscope() -> DemoMicroscope:
    microscope = DemoMicroscope(utils.load_microscope_configuration().system)
    # Pin the milling geometry so the expected angle is independent of the
    # machine's default configuration: the bundled configs disagree on the
    # shuttle pre-tilt (0 in sim-arctis, 35 in microscope-configuration), which
    # changes the computed milling angle.
    microscope.system.stage.shuttle_pre_tilt = 35.0
    microscope.system.ion.column_tilt = 52.0
    return microscope


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

    # milling_angle = 90 - column_tilt + stage_tilt - pretilt = 90 - 52 + 20 - 35 = 23
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
