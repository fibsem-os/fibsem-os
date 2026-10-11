"""What each reader of `stage_is_compustage` answers today, per stage type.

The flag is being retired: each reader moves to the stage device, or to the question it
actually means (`_fm_is_a_pose`, `Stage.turned_over`, "has an r axis"). These pins hold
the answers still while that happens, so each PR can show it changed nothing.

The microscopes are built from the shipped simulator configurations, never by flipping
the flag after connect. A flipped flag is what the readers are moving away from, so a
pin built that way would stop meaning anything halfway through.
"""

import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.poses import _to_milling
from fibsem.microscopes._stage import COMPUSTAGE_HOLDER_NAME
from fibsem.structures import FibsemStagePosition

ARCTIS = "sim-arctis-configuration.yaml"
IFLM = "sim-iflm-configuration.yaml"


@pytest.fixture(scope="module", params=[ARCTIS, IFLM])
def scope(request):
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, request.param), manufacturer="Demo"
    )
    return request.param, microscope


def _compustage(scope) -> bool:
    return scope[0] == ARCTIS


def _position(t_deg: float = 0.0) -> FibsemStagePosition:
    return FibsemStagePosition(x=1e-4, y=2e-4, z=0.0, r=0.0, t=np.radians(t_deg))


# --- microscope.py -------------------------------------------------------------------


def test_an_unconfigured_fm_defaults_on_only_on_a_compustage(scope):
    compustage = _compustage(scope)
    assert scope[1]._fluorescence_default() is compustage


def test_compucentric_rotation_leaves_a_compustage_position_alone(scope):
    compustage = _compustage(scope)
    rotated = scope[1]._get_compucentric_rotation_position(_position())
    if compustage:
        assert (rotated.x, rotated.y, rotated.r) == (1e-4, 2e-4, 0.0)
    else:
        assert np.isclose(rotated.x, -1e-4) and np.isclose(rotated.y, -2e-4)
        assert np.isclose(rotated.r, np.pi)


def test_the_stage_poses(scope):
    compustage = _compustage(scope)
    poses = {
        name: (round(np.degrees(p.r), 6), round(np.degrees(p.t), 6))
        for name, p in scope[1]._stage_poses().items()
    }
    if compustage:
        assert poses == {"SEM": (0.0, 0.0), "FIB": (0.0, -128.0), "FM": (0.0, -180.0)}
    else:
        assert poses == {"SEM": (0.0, 35.0), "FIB": (180.0, 17.0)}


# Tilts each stage can reach: a compustage tilts to -195, an offset stage stops at -10.
@pytest.mark.parametrize("which", ["low", "high"])
def test_the_milling_angle_unwinds_a_turned_over_compustage(scope, which):
    compustage = _compustage(scope)
    tilt, expected = {
        (True, "low"): (-170.0, 48.0),
        (True, "high"): (30.0, 68.0),
        (False, "low"): (-5.0, -2.0),
        (False, "high"): (30.0, 33.0),
    }[(compustage, which)]
    angle = scope[1].get_current_milling_angle(_position(tilt))
    assert np.isclose(angle, expected)


def test_images_stamp_the_stage_type(scope):
    compustage = _compustage(scope)
    assert scope[1].hardware_geometry().is_compustage is compustage


# --- fm/microscope.py -------------------------------------------------------------


def test_the_fm_objective_faces_up_from_under_a_compustage(scope):
    compustage = _compustage(scope)
    microscope = scope[1]
    expected = 180.0 if compustage else microscope.system.ion.column_tilt
    assert microscope.fm.camera_tilt == expected


# --- autolamella/poses.py -----------------------------------------------------------


def test_a_lamella_is_marked_from_fluorescence_only_on_a_compustage(scope):
    compustage = _compustage(scope)
    fm_position = FibsemStagePosition(x=0, y=0, z=0, r=0.0, t=np.radians(-180))
    if compustage:
        milling = _to_milling(scope[1], fm_position)
        assert np.isclose(np.degrees(milling.t), np.degrees(-0.4014257279586957))
    else:
        with pytest.raises(ValueError, match="offset mount"):
            _to_milling(scope[1], fm_position)


# --- microscopes/_stage.py ----------------------------------------------------------


def test_a_compustage_holds_one_built_in_shuttle(scope):
    compustage = _compustage(scope)
    stage = scope[1]._stage
    if compustage:
        assert (stage.holder.name, stage.holder.capacity) == (COMPUSTAGE_HOLDER_NAME, 1)
        assert stage.loader is not None
    else:
        assert stage.holder.name != COMPUSTAGE_HOLDER_NAME
        assert stage.loader is None


# --- the Demo's stage and devices ---------------------------------------------------


def test_the_demo_stage_links_and_rotates_only_off_a_compustage(scope):
    compustage = _compustage(scope)
    stage = scope[1].stage_device
    assert stage.available_linked() is (not compustage)
    assert ("r" in stage.axes) is (not compustage)
    assert scope[1].system.stage.rotation is (not compustage)


def test_the_demo_builds_a_sample_loader_only_for_a_compustage(scope):
    compustage = _compustage(scope)
    assert ("sample_loader" in scope[1].devices) is compustage
