"""Pins `get_target_position` before it is refactored (FIB-829).

`get_target_position` converts a stage position from one named orientation to another.
It is six hand-written branches over ``(from, to)``, and the plan is to replace that
table with composable per-leg transforms so a *device* translation can be added as one
more leg rather than edited into every branch.

Nothing about the behaviour is supposed to change. This file is what says so.

Why it needs writing first
--------------------------
The four existing tests in ``test_microscope.py`` cover MILLING <-> FM on a compustage,
the non-compustage refusal, and the same-orientation no-op. **Every pair that does
positional work is untested** -- SEM/MILLING <-> FIB apply a compucentric rotation to
x/y, and nothing currently notices if that stops happening. A refactor of stage-motion
arithmetic with four tests behind it is not a refactor, it is a rewrite.

How it pins
-----------
Two halves, because the behaviour has two independent parts:

* **Which legs move x/y at all** -- the classification table below. Stated as
  ``PRESERVE`` / ``COMPUCENTRIC`` / ``RAISES`` per (stage, from, to), which is far more
  legible than twenty-one hardcoded coordinate pairs and says the same thing.
* **What the compucentric legs actually compute** -- pinned separately, against a
  *non-zero* rotation offset. The base ``_get_compucentric_rotation_offset`` returns
  ``(0, 0)``, so on a simulator the transform degenerates to plain negation and a
  refactor could drop the offset term entirely without any of the table noticing.

Do not fold FIB-655 in here
---------------------------
``_get_compucentric_rotation_offset`` is itself under review -- FIB-655 found the two
compucentric implementations in the tree disagree by 55 um. This file pins *current*
behaviour on purpose. Changing the offset and re-baselining these expectations in the
same PR would leave nothing checking the refactor.
"""

from copy import deepcopy

import numpy as np
import pytest

from fibsem import utils
from fibsem.structures import FibsemStagePosition

ORIENTATIONS = ["SEM", "FIB", "MILLING", "FM"]

# A source position with a distinctive, asymmetric x/y so a sign flip or an axis swap
# cannot hide behind coincidence, and a non-zero z that no branch should touch.
SOURCE_X = 1.0e-3
SOURCE_Y = 2.0e-3
SOURCE_Z = 3.0e-3

# A rotation-centre offset that is non-zero and asymmetric. Only used by the arithmetic
# test below; the classification table runs against the (0, 0) default so it stays a
# statement about *which* legs move x/y rather than by how much.
OFFSET_X = 0.4e-3
OFFSET_Y = -0.7e-3

PRESERVE = "preserve"  # x/y come through untouched; only r/t are rewritten
COMPUCENTRIC = "compucentric"  # x/y go through the compucentric rotation
RAISES = "raises"  # the conversion is refused
UNBUILDABLE = "unbuildable"  # the source orientation does not exist on this mounting

# What each (from, to) does today. Read off the implementation and confirmed by running
# it; see the module docstring for why this is a classification rather than a table of
# coordinates.
#
# The compustage column is entirely PRESERVE because
# `_get_compucentric_rotation_position` returns its argument unchanged on a compustage
# ("compustage does not support compucentric rotation"), so even the branches that call
# it are positionally inert there.
COMPUSTAGE_BEHAVIOUR = {
    ("SEM", "FIB"): PRESERVE,
    ("SEM", "MILLING"): PRESERVE,
    ("SEM", "FM"): PRESERVE,
    ("FIB", "SEM"): PRESERVE,
    ("FIB", "MILLING"): PRESERVE,
    ("FIB", "FM"): PRESERVE,
    ("MILLING", "SEM"): PRESERVE,
    ("MILLING", "FIB"): PRESERVE,
    ("MILLING", "FM"): PRESERVE,
    ("FM", "SEM"): PRESERVE,
    ("FM", "FIB"): PRESERVE,
    ("FM", "MILLING"): PRESERVE,
}

# The offset column is where the branches actually differ.
#
# The FM rows used to pin consequences of `orientations["FM"]` being a verbatim copy
# of the FIB one off a compustage -- FM -> FIB handed back the caller's own position
# because the source *classified as FIB*, and the other FM sources took the FIB
# branches the same way. The change this file said "FIB-830 is where they change"
# about has arrived: the copy is deleted, FM is not an orientation on this mounting
# at all, and a source position "at the FM orientation" cannot even be built --
# `get_orientation("FM")` refuses by name. The FM *target* rows still refuse, in
# `get_target_position` itself.
OFFSET_BEHAVIOUR = {
    ("SEM", "FIB"): COMPUCENTRIC,
    ("SEM", "MILLING"): PRESERVE,
    ("SEM", "FM"): RAISES,
    ("FIB", "SEM"): COMPUCENTRIC,
    ("FIB", "MILLING"): COMPUCENTRIC,
    ("FIB", "FM"): RAISES,
    ("MILLING", "SEM"): PRESERVE,
    ("MILLING", "FIB"): COMPUCENTRIC,
    ("MILLING", "FM"): RAISES,
    ("FM", "SEM"): UNBUILDABLE,
    ("FM", "FIB"): UNBUILDABLE,
    ("FM", "MILLING"): UNBUILDABLE,
}


def _microscope(compustage: bool):
    """A Demo microscope with realistic stage geometry for the given mounting.

    The stage settings are set explicitly rather than loaded from a configuration file:
    `setup_session` resolves a *user* configuration path, so a test reading one would
    depend on whichever system the developer last connected to. The values below match
    the shipped configurations.

    `stage.rotation` is what separates the two mountings. It is the capability the FIB
    rotation is derived from (FIB-834), so setting it is setting the geometry -- there
    is no second number to keep in step, and no way to write the pair that the stage
    could not physically be.
    """
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = compustage

    stage = microscope.system.stage
    stage.rotation_reference = 0
    if compustage:
        # Arctis: no wedge and no rotation; every orientation is reached by tilting.
        stage.shuttle_pre_tilt = 0
        stage.rotation = False
    else:
        # Pre-tilted shuttle: 35 degree wedge, and the ion beam is reached by turning
        # half way round.
        stage.shuttle_pre_tilt = 35
        stage.rotation = True

    microscope._update_orientations()
    return microscope


def _position_at(microscope, orientation: str) -> FibsemStagePosition:
    """A stage position sitting at the named orientation."""
    pose = microscope.get_orientation(orientation)
    return FibsemStagePosition(x=SOURCE_X, y=SOURCE_Y, z=SOURCE_Z, r=pose.r, t=pose.t)


@pytest.mark.parametrize(
    ("compustage", "behaviour"),
    [(True, COMPUSTAGE_BEHAVIOUR), (False, OFFSET_BEHAVIOUR)],
    ids=["compustage", "offset"],
)
@pytest.mark.parametrize(
    ("source", "target"),
    [(f, t) for f in ORIENTATIONS for t in ORIENTATIONS if f != t],
    ids=[f"{f}_to_{t}" for f in ORIENTATIONS for t in ORIENTATIONS if f != t],
)
def test_every_pair_keeps_its_current_behaviour(
    compustage: bool, behaviour: dict, source: str, target: str
):
    """All twelve ordered pairs, on both mountings, do exactly what they do today.

    This is the refactor's contract: every cell here has to come out the same
    afterwards, and any that does not is a behaviour change that has to be argued for
    rather than discovered.
    """
    microscope = _microscope(compustage)
    expected = behaviour[(source, target)]

    if expected is UNBUILDABLE:
        with pytest.raises(ValueError, match="not supported"):
            _position_at(microscope, source)
        return

    position = _position_at(microscope, source)

    if expected is RAISES:
        with pytest.raises(ValueError):
            microscope.get_target_position(
                deepcopy(position), target_orientation=target
            )
        return

    result = microscope.get_target_position(
        deepcopy(position), target_orientation=target
    )

    # Whatever happens to x/y, the pose is always rewritten to the target's r and t.
    target_pose = microscope.get_orientation(target)
    assert np.isclose(result.r, target_pose.r, atol=1e-9)
    assert np.isclose(result.t, target_pose.t, atol=1e-9)

    # z is never a term in any branch.
    assert np.isclose(result.z, SOURCE_Z, atol=1e-12)

    if expected is PRESERVE:
        assert np.isclose(result.x, SOURCE_X, atol=1e-12)
        assert np.isclose(result.y, SOURCE_Y, atol=1e-12)
    else:
        # With the default (0, 0) rotation offset the compucentric transform is a
        # negation. The offset term is pinned separately, below.
        assert np.isclose(result.x, -SOURCE_X, atol=1e-12)
        assert np.isclose(result.y, -SOURCE_Y, atol=1e-12)


@pytest.mark.parametrize(
    ("source", "target"),
    [pair for pair, kind in OFFSET_BEHAVIOUR.items() if kind is COMPUCENTRIC],
    ids=[
        f"{f}_to_{t}"
        for (f, t), kind in OFFSET_BEHAVIOUR.items()
        if kind is COMPUCENTRIC
    ],
)
def test_compucentric_legs_carry_the_offset_term(source: str, target: str):
    """A compucentric leg is ``p -> -p - 2 * offset``, not merely ``p -> -p``.

    The simulator's rotation offset is ``(0, 0)``, so every compucentric leg degenerates
    to a plain negation and the table above would stay green even if the offset term
    were dropped entirely. Giving the microscope a real offset is what makes this a test
    of the arithmetic rather than of the sign.
    """
    microscope = _microscope(compustage=False)
    offset = FibsemStagePosition(x=OFFSET_X, y=OFFSET_Y)
    microscope._get_compucentric_rotation_offset = lambda: offset

    position = _position_at(microscope, source)
    result = microscope.get_target_position(
        deepcopy(position), target_orientation=target
    )

    assert np.isclose(result.x, -SOURCE_X - 2 * OFFSET_X, atol=1e-12)
    assert np.isclose(result.y, -SOURCE_Y - 2 * OFFSET_Y, atol=1e-12)


def test_a_compucentric_round_trip_returns_to_the_start():
    """SEM -> FIB -> SEM lands where it started, offset and all.

    A property rather than a pinned value, and the one the composable form has to keep:
    ``p -> -p - 2 * offset`` is its own inverse for any offset, so a leg and its reverse
    cancel exactly. Worth stating now because the current pair table cannot be checked
    this way on a compustage, where both directions are positionally inert.
    """
    microscope = _microscope(compustage=False)
    offset = FibsemStagePosition(x=OFFSET_X, y=OFFSET_Y)
    microscope._get_compucentric_rotation_offset = lambda: offset

    start = _position_at(microscope, "SEM")
    at_fib = microscope.get_target_position(deepcopy(start), target_orientation="FIB")
    back = microscope.get_target_position(deepcopy(at_fib), target_orientation="SEM")

    assert np.isclose(back.x, start.x, atol=1e-12)
    assert np.isclose(back.y, start.y, atol=1e-12)
    assert np.isclose(back.r, start.r, atol=1e-9)
    assert np.isclose(back.t, start.t, atol=1e-9)


def test_going_via_a_third_orientation_agrees_with_the_direct_pair():
    """SEM -> FIB -> MILLING equals SEM -> MILLING.

    The property the refactor is *for*: if every pair is a composition of legs, routing
    through an intermediate has to give the same answer as the direct conversion. It
    already holds here, which is worth knowing before rather than after -- it means the
    composable form has a consistent table to reproduce, not a contradictory one to
    pick a winner from.
    """
    microscope = _microscope(compustage=False)
    offset = FibsemStagePosition(x=OFFSET_X, y=OFFSET_Y)
    microscope._get_compucentric_rotation_offset = lambda: offset

    start = _position_at(microscope, "SEM")

    direct = microscope.get_target_position(
        deepcopy(start), target_orientation="MILLING"
    )
    at_fib = microscope.get_target_position(deepcopy(start), target_orientation="FIB")
    via_fib = microscope.get_target_position(
        deepcopy(at_fib), target_orientation="MILLING"
    )

    assert np.isclose(via_fib.x, direct.x, atol=1e-12)
    assert np.isclose(via_fib.y, direct.y, atol=1e-12)
    assert np.isclose(via_fib.r, direct.r, atol=1e-9)
    assert np.isclose(via_fib.t, direct.t, atol=1e-9)


@pytest.mark.parametrize("compustage", [True, False], ids=["compustage", "offset"])
def test_converting_to_the_orientation_it_is_already_at_returns_the_same_object(
    compustage: bool,
):
    """The early return hands back the caller's own position, not a copy.

    ``if current == target: return stage_position`` sits above the ``deepcopy``, so a
    caller that mutates the result mutates its own input. Pinned because it is a real
    aliasing hazard the refactor should fix *deliberately* -- if this test starts
    failing because a copy is returned, that is an improvement, and the failure is how
    it gets noticed rather than shipped by accident.
    """
    microscope = _microscope(compustage)
    position = _position_at(microscope, "SEM")

    result = microscope.get_target_position(position, target_orientation="SEM")

    assert result is position


# ── how much rotation, not whether any ───────────────────────────────

# (source rotation, target rotation, does the compucentric correction apply)
#
# The correction is `p -> -p - 2 * offset` -- a half turn or nothing -- so it applies
# when the rotation *is* a half turn, however that half turn happens to be written.
ROTATION_CASES = [
    (0, 180, True, "half_turn"),
    (0, -180, True, "half_turn_written_negative"),
    (-90, 90, True, "half_turn_across_zero"),
    (270, 90, True, "half_turn_from_a_wound_on_rotation"),
    (-90, 270, False, "same_rotation_written_two_ways"),
    (0, 360, False, "same_rotation_a_full_turn_apart"),
    (0, 0, False, "no_rotation"),
    (0, -0.0001, False, "a_hair_either_side_of_zero"),
    (359.9999, 0, False, "a_hair_either_side_of_zero_the_other_way"),
    (0, 90, False, "a_quarter_turn_is_not_a_half_turn"),
]


@pytest.mark.parametrize(
    ("reference_deg", "fib_deg", "expect_compucentric", "label"),
    ROTATION_CASES,
    ids=[case[3] for case in ROTATION_CASES],
)
def test_the_correction_applies_to_half_turns_however_they_are_written(
    reference_deg: float, fib_deg: float, expect_compucentric: bool, label: str
):
    """A stage rotates continuously, so one rotation has many spellings.

    Three groups here, and the middle and last are the ones worth having.

    **Equivalent rotations** -- 270 and -90, 0 and 360 -- are one rotation and must not
    be read as two. Plain modulo gets those right but breaks either side of zero:
    -0.0001 degrees becomes 359.9999, which compares nowhere near 0.

    **A quarter turn gets nothing.** The correction cannot represent it, so applying it
    would be wrong by the whole grid rather than by the offset. No shipped configuration
    has an orientation at 90 degrees; this pins what happens if one is ever added.

    None of these are reachable from a shipped configuration, where every value is a
    whole number of degrees and every pair is 0 or 180 apart. Pinned because
    "unreachable given the data we happen to ship" is not the same as correct, and this
    is a stage-motion path.
    """
    microscope = _microscope(compustage=False)
    stage = microscope.system.stage
    stage.rotation_reference = reference_deg
    microscope._update_orientations()
    # The FIB rotation is written straight onto the orientation table rather than
    # configured, because since FIB-834 there is no configuration that produces these
    # pairs -- the derivation only ever yields a half turn. That is the point: what is
    # under test is `get_target_position` reading an orientation table, and the table is
    # the seam a future orientation at some other angle would arrive through.
    microscope.orientations["FIB"].r = np.radians(fib_deg)

    position = _position_at(microscope, "SEM")
    result = microscope.get_target_position(
        deepcopy(position), target_orientation="FIB"
    )

    if expect_compucentric:
        assert np.isclose(result.x, -SOURCE_X, atol=1e-12)
        assert np.isclose(result.y, -SOURCE_Y, atol=1e-12)
    else:
        assert np.isclose(result.x, SOURCE_X, atol=1e-12)
        assert np.isclose(result.y, SOURCE_Y, atol=1e-12)
