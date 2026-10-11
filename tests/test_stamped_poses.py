"""Images stamp the stage's declared poses (FIB-1101).

An image records the pose the stage declared for each orientation name, so readers
working from a saved image can use the whole pose rather than `rotation_180` alone.
An image from before this has none, and `declared_poses` rebuilds them the way its
readers interpreted it: the rebuild must agree with what the stage declares today on
every shipped stage type, or an old image and a new one would read differently.
"""

import dataclasses
import math
import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import FibsemHardwareGeometry

STAGE_TYPES = [
    "microscope-configuration.yaml",
    "tescan-configuration.yaml",
    "sim-arctis-configuration.yaml",
]


def _microscope(filename):
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, filename), manufacturer="Demo"
    )
    return microscope


def _deg(poses):
    return {
        name: (round(math.degrees(p.r) % 360, 9), round(math.degrees(p.t), 9))
        for name, p in poses.items()
    }


@pytest.mark.parametrize("filename", STAGE_TYPES)
def test_an_image_stamps_what_the_stage_declares(filename):
    microscope = _microscope(filename)
    geometry = microscope.hardware_geometry()

    assert _deg(geometry.declared_poses()) == _deg(microscope._stage_poses())
    assert "MILLING" not in geometry.poses


@pytest.mark.parametrize("filename", STAGE_TYPES)
def test_an_old_image_rebuilds_the_same_poses(filename):
    """Without stamped poses, the rebuild from `rotation_180` and `is_compustage`
    matches what the stage declares."""
    geometry = _microscope(filename).hardware_geometry()
    old = dataclasses.replace(geometry, poses={})

    assert _deg(old.declared_poses()) == _deg(geometry.declared_poses())


def test_the_poses_survive_a_round_trip():
    geometry = _microscope("sim-arctis-configuration.yaml").hardware_geometry()
    loaded = FibsemHardwareGeometry.from_dict(geometry.to_dict())

    assert loaded == geometry
    assert set(loaded.poses) == {"SEM", "FIB", "FM"}


def test_a_file_without_poses_loads_with_none():
    ddict = FibsemHardwareGeometry().to_dict()
    del ddict["poses"]

    assert FibsemHardwareGeometry.from_dict(ddict).poses == {}


def test_an_fm_position_without_r_or_t_takes_the_stamped_pose():
    """A stage that keeps its rotation and tilts to the ion beam (JEOL-style)
    completes to the FIB pose it stamped, not to a half turn from the reference."""
    from fibsem.correlation.geometry import _complete_fm_pose
    from fibsem.structures import FibsemStagePosition

    geometry = FibsemHardwareGeometry(
        rotation_reference=0.0,
        rotation_180=180.0,
        fib_column_tilt=52.0,
        shuttle_pre_tilt=0.0,
        poses={"SEM": (0.0, 0.0), "FIB": (0.0, 52.0)},
    )
    completed = _complete_fm_pose(FibsemStagePosition(x=0, y=0, z=0), geometry)

    assert math.degrees(completed.r) == pytest.approx(0.0)
    assert math.degrees(completed.t) == pytest.approx(52.0)
