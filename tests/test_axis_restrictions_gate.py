"""Which stage types drop z and rotation from an absolute move.

`ThermoMicroscope.move_stage_absolute` blanks two axes when it believes the
microscope will refuse them. That guard has only ever run on a compustage, because
`microscope.fm` is None on every other system -- and opening the connection gate is
what makes it reachable elsewhere.

It is not safe to let that happen. The axes are not equivalent across stage types:
`stage_position_to_autoscript` returns a `CompustagePosition(x, y, z, a)` with no `r`
field at all, so dropping `r` there has never done anything, while on an offset mount
it would drop a real rotation axis. Every absolute move would silently half-succeed --
land at x and y, no z, no rotation.

So the objective half is compustage-gated **temporarily**, and FIB-640 owns removing
it. That issue is explicit that the branch should *not* stay gated once it is correct
-- an iFLM has an objective too -- and it is also where the axis pair gets settled: it
measured z and t, not z and r.

The predicate is tested rather than the move: AutoScript is not installed in CI, so
`ThermoMicroscope.move_stage_absolute` cannot be executed at all.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils

IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


def _microscope(config_path: str):
    microscope, _ = utils.setup_session(config_path=config_path)
    return microscope


def test_a_compustage_with_the_objective_in_still_restricts():
    """Unchanged. This is the only place the guard has ever run."""
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.fm.objective.insert()

    assert microscope._axis_restrictions_apply() is True


def test_a_compustage_with_the_objective_out_does_not():
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.fm.objective.retract()
    microscope.move_to_orientation("SEM")

    assert microscope._axis_restrictions_apply() is False


def test_an_offset_mount_with_the_objective_in_does_not_restrict():
    """The gate. Without it, every absolute move on an offset system with an FM would
    silently lose its z and its rotation -- behaviour that has executed on no
    instrument. Remove this with FIB-640, once the axis pair is settled on hardware."""
    microscope = _microscope(IFLM_CONFIG)
    microscope.move_to_orientation("FIB")
    microscope.fm.objective.insert()

    assert microscope.stage_is_compustage is False
    assert microscope.fm.objective.state == "Inserted"
    assert microscope._axis_restrictions_apply() is False


def test_a_system_with_no_fluorescence_microscope_does_not_restrict():
    """Where every offset system sits today, gate closed."""
    microscope = _microscope(cfg.MICROSCOPE_CONFIGURATION_PATH)

    assert microscope.fm is None
    assert microscope._axis_restrictions_apply() is False


def test_the_orientation_half_needs_no_gate_of_its_own():
    """It is confined to the compustage by the orientations themselves.

    `get_stage_orientation` can never return "FM" on an offset mount -- the FM is a
    device there, and `orientations["FM"]` is a byte-identical copy of the FIB entry,
    which is matched first. So that half is dead off a compustage without anything
    saying so, and gating it would be describing the same fact twice.
    """
    microscope = _microscope(IFLM_CONFIG)
    microscope.move_to_orientation("FIB")
    microscope.move_to_microscope("FM")

    assert microscope.get_current_device() == "FM"
    assert microscope.get_stage_orientation() != "FM"


@pytest.mark.parametrize("config_path", [ARCTIS_CONFIG, IFLM_CONFIG])
def test_the_predicate_reads_the_microscope_rather_than_being_told(config_path: str):
    """It answers from live objective state, so inserting flips it where it applies."""
    microscope = _microscope(config_path)
    microscope.move_to_orientation("SEM")
    microscope.fm.objective.retract()
    before = microscope._axis_restrictions_apply()

    microscope.fm.objective.insert()
    after = microscope._axis_restrictions_apply()

    assert before is False
    assert after is microscope.stage_is_compustage
