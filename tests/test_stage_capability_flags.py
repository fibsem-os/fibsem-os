"""`stage.tilt` is gone, and `stage.rotation` is not.

Two booleans sat side by side in the stage configuration and looked like a pair. They
were not. `rotation` distinguishes the two mountings this project drives -- it is what
`rotation_180` is derived from (FIB-834) -- while `tilt` had one reachable value, was
read by nothing, and existed only as somewhere for a typo to sit.

The tests below pin both halves of that: that nothing lost a capability it was using,
and that the asymmetry with the *manipulator*, which keeps its own `tilt`, is deliberate
rather than an edit that missed a line.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscopes.simulator import (
    STAGE_LIMITS_COMPUSTAGE,
    STAGE_LIMITS_DEFAULT,
)
from fibsem.structures import StageSystemSettings

# Named rather than globbed. `fibsem/config/*.yaml` is gitignored with an allowlist
# for the shipped files, so a configuration the developer saved from the wizard sits
# in this same folder -- and a glob would sweep it in, failing on one machine and not
# another for a reason that looks nothing like the cause.
CONFIGS = (
    "microscope-configuration.yaml",
    "odemis-configuration.yaml",
    "sim-arctis-configuration.yaml",
    "sim-iflm-configuration.yaml",
    "tescan-configuration.yaml",
    "tfs-aquilos2-configuration.yaml",
    "tfs-arctis-configuration.yaml",
    "tfs-hydra-configuration.yaml",
)


def _stage_block(filename: str) -> dict:
    return utils.load_yaml(os.path.join(cfg.CONFIG_PATH, filename))["stage"]


@pytest.mark.parametrize("filename", CONFIGS)
def test_no_shipped_file_states_a_stage_tilt(filename: str):
    assert "tilt" not in _stage_block(filename)


@pytest.mark.parametrize("filename", CONFIGS)
def test_the_manipulator_keeps_its_own_tilt(filename: str):
    """The asymmetry is the point, so it is asserted rather than left to be noticed.

    A manipulator that cannot tilt is an ordinary manipulator; a stage that cannot tilt
    is not a stage this project drives. The two keys were spelled the same and meant
    different things, which is exactly how a scoped edit goes wrong -- a `tilt:` line
    removed from the wrong block would silently give every manipulator a tilt axis.
    """
    config = utils.load_yaml(os.path.join(cfg.CONFIG_PATH, filename))
    assert "tilt" in config["manipulator"]


@pytest.mark.parametrize(
    ("label", "limits"),
    [("default", STAGE_LIMITS_DEFAULT), ("compustage", STAGE_LIMITS_COMPUSTAGE)],
)
def test_every_stage_has_a_tilt_axis(label: str, limits: dict):
    """Why the flag had one reachable value.

    A compustage is the stage that differs, and it differs by dropping `r` -- it reaches
    the other side of the grid by tilting, so it needs `t` more than an ordinary stage
    does, not less. Nothing here can answer "no" to tilt, which is what made a
    configurable `tilt` a way to describe a machine that does not exist.
    """
    assert "t" in limits, label


def test_only_the_compustage_drops_the_rotation_axis():
    """The counterpart, and the reason `rotation` stays.

    This is a real difference between two real mountings, and it is the input to the
    `rotation_180` derivation, so a stage misdescribed here reaches the orientation
    table and the compucentric correction.
    """
    assert "r" in STAGE_LIMITS_DEFAULT
    assert "r" not in STAGE_LIMITS_COMPUSTAGE


def test_a_stored_stage_tilt_is_ignored_rather_than_rejected():
    """A configuration written before the removal still loads.

    Dropped rather than honoured, for the same reason as `rotation_180`: there is no
    field left to put it in, and refusing the file instead would make a dead flag able
    to stop an instrument from starting.
    """
    stage = StageSystemSettings.from_dict(
        {
            "rotation_reference": 0.0,
            "shuttle_pre_tilt": 35.0,
            "manipulator_height_limit": 0.0037,
            "tilt": False,
        }
    )
    assert not hasattr(stage, "tilt")
    assert "tilt" not in stage.to_dict()
    assert stage.rotation_180 == 180.0


def test_the_live_microscope_still_answers_for_rotation():
    """`is_available` lost `stage_tilt` and kept `stage_rotation`."""
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
        manufacturer="Demo",
    )
    assert microscope.is_available("stage_rotation") is False
    assert microscope.is_available("stage") is True
    # Unknown subsystems fall through to False rather than raising, so a caller left
    # asking the old question gets a quiet "no" -- worth knowing, since that is what a
    # stale plugin would see.
    assert microscope.is_available("stage_tilt") is False
