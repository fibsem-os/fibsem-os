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

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscopes.simulator import (
    STAGE_LIMITS_COMPUSTAGE,
    STAGE_LIMITS_DEFAULT,
)
from fibsem.structures import StageSystemSettings, SystemSettings

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
@pytest.mark.parametrize("key", ["tilt", "rotation"])
def test_no_shipped_file_states_a_stage_capability(filename: str, key: str):
    assert key not in _stage_block(filename)


@pytest.mark.parametrize("filename", CONFIGS)
@pytest.mark.parametrize("key", ["tilt", "rotation"])
def test_the_manipulator_keeps_its_own(filename: str, key: str):
    """The asymmetry is the point, so it is asserted rather than left to be noticed.

    A manipulator that cannot tilt is an ordinary manipulator; a stage that cannot tilt
    is not a stage this project drives. The two blocks spell these keys the same and
    mean different things, which is exactly how a scoped edit goes wrong -- a line
    removed from the wrong block would silently give every manipulator an axis it does
    not have, and no other test in the suite would notice.
    """
    config = utils.load_yaml(os.path.join(cfg.CONFIG_PATH, filename))
    assert key in config["manipulator"]


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


# ---------------------------------------------------------------------------
# The capability now comes from the instrument
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("filename", "compustage", "rotates", "opposite"),
    [
        ("sim-arctis-configuration.yaml", True, False, 0.0),
        ("microscope-configuration.yaml", False, True, 180.0),
        ("tescan-configuration.yaml", False, True, 0.0),
    ],
)
def test_connecting_fills_the_capability_from_the_stage(
    filename: str, compustage: bool, rotates: bool, opposite: float
):
    """No file states `rotation`, so this is the only thing that answers it.

    Both mountings are here rather than only the compustage: a test that just asserted
    "False for the Arctis" would pass equally well if the capability were never filled
    and something else happened to be false.
    """
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, filename), manufacturer="Demo"
    )

    assert microscope.stage_is_compustage is compustage
    assert microscope.system.stage.rotation is rotates
    assert microscope.system.stage.rotation_180 == opposite


def test_applying_a_configuration_does_not_overwrite_the_capability():
    """The regression this change is most likely to grow.

    `apply_configuration` replaces `system.stage` wholesale from a settings object the
    user chose, and it is reachable from a button in the system setup widget. Since no
    file states `rotation` any more, the replacement carries the field default of
    `True` -- so on a compustage, pressing Apply would move the FIB orientation half a
    turn away and hand the compucentric correction a rotation the stage cannot make.

    The settings applied here are a real load of the same configuration, which is what
    that button does.
    """
    path = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    microscope, _ = utils.setup_session(config_path=path, manufacturer="Demo")
    assert microscope.system.stage.rotation is False

    applied = SystemSettings.from_dict(utils.load_yaml(path))
    assert applied.stage.rotation is True, "the file cannot say, so it says the default"

    microscope.apply_configuration(applied)

    assert microscope.system.stage.rotation is False
    assert microscope.system.stage.rotation_180 == 0.0
    assert np.isclose(microscope.get_orientation("FIB").r, 0.0)


def test_the_thermo_backend_reports_a_compustage_without_a_rotation_axis():
    """The real Arctis path, which no test on CI can connect to.

    `ThermoMicroscope._get_axis_limits` is where an AutoScript compustage becomes "no
    `r` axis", and it is what `_read_stage_capabilities` asks. CI has no AutoScript, and
    the simulator cannot stand in for this branch -- it has its own `_get_axis_limits`.
    So the branch is asserted from the source, the way FIB-500 pinned its
    `r=0.0` literal: crude, but it fails if someone deletes the short-circuit, which is
    the failure that would otherwise reach an instrument.
    """
    import ast
    import inspect
    import textwrap

    from fibsem.microscope import ThermoMicroscope

    tree = ast.parse(
        textwrap.dedent(inspect.getsource(ThermoMicroscope._get_axis_limits))
    )
    returns_under_a_compustage_test = [
        node.body[0].value.id
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and node.test.attr == "stage_is_compustage"
        and isinstance(node.body[0], ast.Return)
        and isinstance(node.body[0].value, ast.Name)
    ]
    assert returns_under_a_compustage_test == ["STAGE_LIMITS_COMPUSTAGE"]
    assert "r" not in STAGE_LIMITS_COMPUSTAGE
