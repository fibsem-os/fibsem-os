"""`stage.rotation_180` is derived, not configured (FIB-834).

It was a field in every shipped configuration, and in every one of them it held
`(rotation_reference + 180) % 360` -- except on the two compustages, which set it
*equal* to the reference to mean "this stage does not turn round". A value doing a
boolean's job, next to the boolean, which already said so.

The risk in deriving it is not the arithmetic. It is that three movement paths pick
`PRETILT_SIGN` by comparing the live stage rotation against `rotation_reference` and
`rotation_180` in turn, so on a compustage -- where the two were both 0 -- *both*
comparisons matched and the second always won. Derive it to a half turn and that
coincidence breaks and the sign flips. The derivation below keeps it at the reference
for a stage that does not rotate, which is why it does not.
`test_the_shipped_rotating_stages_derive_what_they_used_to_state` holds the
arithmetic config by config, `test_a_connected_compustage_derives_no_opposite_rotation`
holds the case a file can no longer describe, and
`tests/test_view_corrected_movement.py` holds the movement itself.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import StageSystemSettings


def _stage(**overrides) -> StageSystemSettings:
    fields = dict(
        rotation_reference=0.0,
        shuttle_pre_tilt=35.0,
        manipulator_height_limit=0.0037,
    )
    fields.update(overrides)
    return StageSystemSettings(**fields)


# The values the eight shipped files stated before FIB-834 removed the key,
# transcribed from the files as they were. Every row has to survive the derivation,
# because every row is a stage somebody runs.
#
# All eight are *rotating* stages as far as a file can tell, and since the capability
# moved to the instrument that is all a file can tell. The two compustage rows are not
# here: their answer is no longer in the file, and asserting the field default against
# them would only be re-testing the default. They are covered by
# `test_a_connected_compustage_derives_no_opposite_rotation` below, through a
# microscope, which is the only thing that now knows.
SHIPPED = {
    "microscope-configuration.yaml": 180.0,
    "odemis-configuration.yaml": 180.0,
    "sim-iflm-configuration.yaml": 180.0,
    "tescan-configuration.yaml": 0.0,
    "tfs-aquilos2-configuration.yaml": 180.0,
    "tfs-hydra-configuration.yaml": 180.0,
}

COMPUSTAGE_CONFIGS = (
    "sim-arctis-configuration.yaml",
    "tfs-arctis-configuration.yaml",
)
ALL_CONFIGS = tuple(SHIPPED) + COMPUSTAGE_CONFIGS


@pytest.mark.parametrize("filename", sorted(SHIPPED))
def test_the_shipped_rotating_stages_derive_what_they_used_to_state(filename: str):
    """The regression guard for the arithmetic, on the stages a file can describe.

    Tescan is the row worth having: reference 180, so its opposite is 0 and not 360.
    """
    config = utils.load_yaml(os.path.join(cfg.CONFIG_PATH, filename))
    stage = StageSystemSettings.from_dict(config["stage"])

    assert stage.rotation_180 == SHIPPED[filename]


def test_a_connected_compustage_derives_no_opposite_rotation():
    """The compustage half, which only a microscope can answer.

    This is what the two removed rows became. `sim-arctis` is the configuration that
    stands in for an Arctis, and it declares `sim.is_compustage`, so the simulator
    reports a stage with no `r` axis and the capability is read from that rather than
    from anything in the file.

    `tfs-arctis` is deliberately not exercised here. It is a *ThermoFisher* file, and
    its compustage is reported by AutoScript at connect; run against the simulator it
    would describe an ordinary stage, which is correct -- the simulator it was pointed
    at does not have a compustage. That is the change working, not a gap in it.
    """
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
        manufacturer="Demo",
    )

    assert microscope.stage_is_compustage
    assert microscope.system.stage.rotation is False
    assert microscope.system.stage.rotation_180 == 0.0


def test_a_connected_rotating_stage_sits_half_a_turn_away():
    """The counterpart, so the test above is not passing for a trivial reason."""
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml"),
        manufacturer="Demo",
    )

    assert not microscope.stage_is_compustage
    assert microscope.system.stage.rotation is True
    assert microscope.system.stage.rotation_180 == 180.0


@pytest.mark.parametrize("filename", sorted(ALL_CONFIGS))
def test_no_shipped_file_still_states_it(filename: str):
    config = utils.load_yaml(os.path.join(cfg.CONFIG_PATH, filename))
    assert "rotation_180" not in config["stage"]


def test_a_rotating_stage_sits_half_a_turn_from_its_reference():
    assert _stage(rotation_reference=0.0).rotation_180 == 180.0


def test_the_half_turn_wraps():
    """Tescan's reference is 180, so its opposite is 0 and not 360.

    The modulo is the difference between a file that reads `0` and one that reads `360`.
    Both compare equal through `rotation_angle_is_smaller`, so nothing would have broken
    -- it would just have been written in a spelling no other configuration uses, for a
    reader to wonder about. `fibsem/configuration.py` omitted the modulo for years.
    """
    assert _stage(rotation_reference=180.0).rotation_180 == 0.0


def test_a_stage_that_does_not_rotate_has_no_opposite():
    """The compustage. It reaches the other side of the grid by tilting.

    Returning the reference rather than raising or returning None keeps every consumer
    branch-free: the movement paths compare a live rotation against both values, and
    two equal targets is exactly the "there is only one side" they already handled.
    """
    assert _stage(rotation_reference=0.0, rotation=False).rotation_180 == 0.0


def test_a_stored_value_is_ignored_rather_than_honoured():
    """A configuration written before FIB-834 still loads, and its number is dropped.

    Honouring it would preserve the one thing removing the field was meant to stop --
    a stated opposite that disagrees with the reference beside it. This file says the
    stage sits at 0 and turns round to 99; it turns round to 180.
    """
    stage = StageSystemSettings.from_dict(
        {
            "rotation_reference": 0.0,
            "rotation_180": 99.0,
            "shuttle_pre_tilt": 35.0,
            "manipulator_height_limit": 0.0037,
        }
    )
    assert stage.rotation_180 == 180.0


def test_it_is_not_written_back_out():
    """`to_dict` feeds both the saved file and the RPC client's `SystemSettings`.

    The client rebuilds from this dict, so what matters is that the two inputs to the
    derivation cross the wire -- not the derived value itself, which would only give the
    two ends a way to disagree.
    """
    stage = _stage(rotation_reference=180.0)
    payload = stage.to_dict()

    assert "rotation_180" not in payload
    assert payload["rotation_reference"] == 180.0
    assert payload["rotation"] is True
    assert StageSystemSettings.from_dict(payload).rotation_180 == 0.0
