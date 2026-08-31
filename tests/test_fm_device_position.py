"""Where the stage travels for an instrument to see the sample.

A compustage's fluorescence objective sits under the grid: the stage turns over and
the FM is genuinely an *orientation*. An offset FM -- METEOR, iFLM -- is off to the
side, and the stage travels ~48.8 mm to reach it while rotation and tilt do not change
at all. There the FM is a **place**, and a place is not something
`get_stage_orientation` can report: measured on the offset simulator, the FM
orientation is byte-identical to the FIB one, so it never comes back.

So the device is its own axis, alongside orientation. This file pins the first half of
it: the devices are configuration rather than three constants inlined in
`move_to_microscope`, and `is_at_device` can ask the question that had no asker.

See FIB-830.
"""

import os
from copy import deepcopy

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import (
    DEFAULT_DEVICE_RANGE,
    DEFAULT_STAGE_DEVICES,
    FibsemStagePosition,
    StageDeviceSettings,
)

IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
# The other mounting. Its objective is under the grid, and it declares no `devices:`
# block at all -- so it is also the test of what saying nothing gets you.
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")

# Between the two windows: past the beams' 20 mm, short of the FM's 28.8 mm.
# Mid-traverse, and at no device.
IN_THE_GAP_MM = 24.0


def _microscope(config_path: str = IFLM_CONFIG):
    microscope, _ = utils.setup_session(config_path=config_path)
    return microscope


def _at_fib(microscope):
    """`move_to_microscope("FM")` refuses from any other orientation -- see FIB-832."""
    microscope.move_to_orientation("FIB")
    return microscope


# ── the devices are configuration ────────────────────────────────────


def test_the_devices_come_from_the_configuration_file():
    """The three numbers that used to be inline in `move_to_microscope`.

    `TRANSLATION_DX` carried its own note -- *"THIS needs to be configurable for
    different microscopes"* -- and the two windows it had to agree with were separate
    literals that nothing kept in step with it.
    """
    microscope = _microscope()

    assert microscope.get_device_origin("FIBSEM").x == pytest.approx(0.0)
    assert microscope.get_device_origin("FM").x == pytest.approx(48.8e-3)
    assert microscope.system.stage.device_range.x == pytest.approx(20.0e-3)


def test_a_configuration_that_says_nothing_gets_the_objective_under_the_grid():
    """Saying nothing describes the *common* chamber, not the rare one.

    The defaults used to be the TFS SDB layout -- FM 48.8 mm along x -- because they
    were lifted from constants inlined in `move_to_microscope`, which only ever ran on
    an offset mount. Since nothing but the offset simulator declares a `devices:`
    block, every other configuration inherited a fluorescence microscope somewhere its
    stage never goes. Now the default is the objective under the grid, and a site
    whose objective is offset says so.
    """
    default, _ = utils.setup_session(config_path=cfg.MICROSCOPE_CONFIGURATION_PATH)

    assert "devices" not in utils.load_yaml(cfg.MICROSCOPE_CONFIGURATION_PATH)["stage"]
    assert default.system.stage.devices == DEFAULT_STAGE_DEVICES
    assert default.system.stage.device_range == DEFAULT_DEVICE_RANGE


def test_a_device_is_a_place_and_leaves_the_pose_alone():
    """Only x. Rotation and tilt are the *orientation* axis, not the device one.

    A device that also fixed r or t would be the same conflation this is undoing, so
    the configuration refuses to express one.
    """
    position = _microscope().get_device_origin("FM")

    assert position.x is not None
    assert (position.y, position.z, position.r, position.t) == (None, None, None, None)

    with pytest.raises(ValueError, match="Unsupported device"):
        StageDeviceSettings.from_dict({"origin": {"x": 48.8e-3, "r": 3.14}})


def test_the_configuration_survives_a_round_trip():
    """`system.to_dict()` is served over the API and written into image metadata."""
    stage = _microscope().system.stage
    restored = type(stage).from_dict(stage.to_dict())

    assert restored.devices == stage.devices
    assert restored.device_range == stage.device_range


# ── the other half of a device: which poses it can image from ────────


def _can_see_the_sample(microscope, device: str) -> bool:
    """The conjunction, spelled out.

    It is written inline here rather than called because it does not exist yet: this
    file pins the *data* that makes one expression work on both mountings, and the
    method that asks it lands next (FIB-839). Written out, the two terms and the two
    remedies stay visible.

    An empty list is vacuously TRUE -- the device does not constrain the pose -- so
    the place term alone decides. The beams are that case.
    """
    orientations = microscope.system.stage.devices[device].acquisition_orientations
    return microscope.is_at_device(device) and (
        not orientations or microscope.get_stage_orientation() in orientations
    )


def test_a_device_says_which_poses_it_can_image_from():
    """A place is only half of a device. The other half is what it can see from there.

    The offset simulator's objective images the sample held in the FIB pose -- not
    because the ion beam is looking at it, it is 48.8 mm away, but because that is the
    pose the holder grips it in and the pose is a property of the holder.
    """
    microscope = _microscope()

    assert microscope.system.stage.devices["FM"].acquisition_orientations == ["FIB"]


def test_the_beams_say_nothing_about_the_pose():
    """Empty means unconstrained: the orientation term is vacuously true.

    SEM, FIB and MILLING are all views of the sample from the beams, and choosing
    between them is not the device axis's business -- so the place term alone decides,
    and the beams can see the sample in every pose. The other reading, "can never
    image", is deliberately unrepresentable: a device that can never image should not
    be declared, and an empty list must not silently mean a dead instrument.
    """
    microscope = _microscope()

    assert microscope.system.stage.devices["FIBSEM"].acquisition_orientations == []
    for orientation in ("SEM", "FIB", "MILLING"):
        microscope.move_to_orientation(orientation)
        assert _can_see_the_sample(microscope, "FIBSEM") is True


def test_a_misspelt_orientation_is_refused_at_load():
    """A typo here would otherwise be perfectly quiet.

    The conjunction that reads this list would simply never be true, and the
    instrument would be dead with no error -- undetectable in normal operation,
    because the load-bearing field on each mounting is the other mounting's inert one.
    """
    with pytest.raises(ValueError, match=r"Unknown acquisition orientation.*FIBB"):
        StageDeviceSettings.from_dict(
            {"origin": {"x": 48.8e-3}, "acquisition_orientations": ["FIBB"]}
        )


def test_the_fm_object_reads_the_device_declaration():
    """One source of truth, not two fields with the same name.

    `FluorescenceMicroscope.acquisition_orientations` used to hardcode
    `[default_orientation]` -- `["FM"]` on every mounting, which on offset names a
    pose the classifier never returns there. It now reads the device declaration, so
    the widget gates that consume it can be true at the actual FM.
    """
    offset = _microscope()
    compustage = _microscope(ARCTIS_CONFIG)

    assert offset.fm.acquisition_orientations == ["FIB"]
    assert compustage.fm.acquisition_orientations == ["FM"]


def test_neither_axis_answers_on_both_mountings():
    """The finding this whole field exists for. Measured, and exact mirror images.

    At the fluorescence microscope on each mounting:

    | | compustage | offset |
    | -- | -- | -- |
    | `get_stage_orientation() == "FM"` | True | False -- it reads FIB |
    | `is_at_device("FM")` | False* | True |

    (*before this change; the compustage FM now shares the beams' origin, which is
    what makes the row below true rather than accidental.)

    So "replace the orientation check with a device check" breaks the compustage as
    thoroughly as the orientation check breaks the offset mount.
    """
    compustage = _microscope(ARCTIS_CONFIG)
    compustage.move_to_microscope("FM")

    offset = _at_fib(_microscope())
    offset.move_to_microscope("FM")

    assert compustage.get_stage_orientation() == "FM"
    assert offset.get_stage_orientation() == "FIB"


def test_the_same_question_answers_at_the_fm_on_both_mountings():
    """One expression, no `stage_is_compustage`. The mounting lives in configuration.

    Each mounting makes a *different* term trivially true -- the compustage's FM is
    where the beams are, so the place carries nothing and the pose carries it all; the
    offset FM images from the pose it was already in, so the pose carries nothing and
    the place carries it all. Which is why the conjunction discriminates on both.
    """
    compustage = _microscope(ARCTIS_CONFIG)
    compustage.move_to_microscope("FM")

    offset = _at_fib(_microscope())
    offset.move_to_microscope("FM")

    assert _can_see_the_sample(compustage, "FM") is True
    assert _can_see_the_sample(offset, "FM") is True


def test_the_term_that_fails_names_the_remedy():
    """A wrong place means travel; a wrong pose means re-pose. Different remedies.

    Today they collapse into one `False`, so a refusal cannot say which. Both rows
    below are live states an operator reaches by doing something ordinary: the
    compustage one by looking at the sample with a beam, the offset one by not having
    travelled yet.
    """
    compustage = _microscope(ARCTIS_CONFIG)
    compustage.move_to_orientation("SEM")

    offset = _at_fib(_microscope())

    # Right place, wrong pose: flip it over.
    assert compustage.is_at_device("FM") is True
    assert compustage.get_stage_orientation() == "SEM"

    # Right pose, wrong place: drive it out.
    assert offset.is_at_device("FM") is False
    assert offset.get_stage_orientation() == "FIB"

    assert _can_see_the_sample(compustage, "FM") is False
    assert _can_see_the_sample(offset, "FM") is False


def test_the_acquisition_orientations_survive_a_round_trip():
    """`system.to_dict()` is served over the API and written into image metadata."""
    devices = _microscope().system.stage.devices

    restored = StageDeviceSettings.from_dict(devices["FM"].to_dict())

    assert restored == devices["FM"]
    assert restored.acquisition_orientations == ["FIB"]


def test_a_configuration_that_says_nothing_can_still_see_the_sample():
    """The default has to be a working compustage, not an empty list.

    Neither shipped compustage configuration declares a `devices:` block, so if the
    default said nothing about the pose the conjunction would be false at the one
    place an Arctis takes fluorescence images.
    """
    microscope = _microscope(ARCTIS_CONFIG)

    assert "devices" not in utils.load_yaml(ARCTIS_CONFIG)["stage"]
    assert microscope.system.stage.devices["FM"].acquisition_orientations == ["FM"]


# ── the question nothing could ask ───────────────────────────────────


def test_is_at_device_answers_where_get_stage_orientation_cannot():
    """The stage starts at the beams, which is not where the FM is.

    `get_stage_orientation` cannot make this distinction on an offset mount at all:
    the FM orientation there is a copy of the FIB one, so it reports "FIB" at both
    ends of a 48.8 mm traverse.
    """
    microscope = _microscope()

    assert microscope.is_at_device("FIBSEM") is True
    assert microscope.is_at_device("FM") is False


def test_the_window_is_wider_than_the_origin():
    """A stage never lands on the nominal value, so "at" is a window, not a point."""
    microscope = _microscope()
    position = FibsemStagePosition(x=10.0e-3, y=0.0, z=0.0, r=0.0, t=0.0)

    assert microscope.is_at_device("FIBSEM", position) is True
    assert microscope.is_at_device("FM", position) is False


def test_between_the_two_devices_is_neither():
    """The windows do not tile the axis, and that is deliberate.

    The beam window ends at 20 mm and the fluorescence one begins at 28.8 mm, so 8.8
    mm of travel belongs to neither. A stage there is mid-traverse, which is a real
    state and not one to guess a device for.
    """
    microscope = _microscope()
    position = FibsemStagePosition(x=IN_THE_GAP_MM * 1e-3, y=0.0, z=0.0, r=0.0, t=0.0)

    assert microscope.is_at_device("FIBSEM", position) is False
    assert microscope.is_at_device("FM", position) is False


def test_the_axes_a_device_does_not_constrain_do_not_decide():
    """y, z, r and t are free at a device: it is an x location, nothing more."""
    microscope = _microscope()
    position = FibsemStagePosition(x=0.0, y=5.0e-3, z=2.0e-3, r=3.14, t=0.3)

    assert microscope.is_at_device("FIBSEM", position) is True


def test_a_device_the_range_cannot_decide_is_never_arrived_at():
    """The asymmetry is the reason: a wrong "no" costs a move, a wrong "yes" skips one.

    Two ways to be unanswerable, and both say no. A device whose origin constrains an
    axis the range says nothing about cannot be decided; nor can a device with no
    origin at all, which is what a device that comes to the sample rather than being
    travelled to looks like -- a needle or a knife inserts, and position is simply not
    the question for it. See FIB-839.
    """
    device = StageDeviceSettings(origin=FibsemStagePosition(x=48.8e-3))
    here = FibsemStagePosition(x=48.8e-3, y=0.0, z=0.0)

    assert device.contains(here, FibsemStagePosition(y=1.0e-3)) is False
    assert (
        StageDeviceSettings(origin=FibsemStagePosition()).contains(
            here, DEFAULT_DEVICE_RANGE
        )
        is False
    )


def test_an_unconfigured_device_is_refused_by_name():
    microscope = _microscope()

    with pytest.raises(ValueError, match=r"Configured devices: \['FIBSEM', 'FM'\]"):
        microscope.is_at_device("TEM")


# ── the traverse ─────────────────────────────────────────────────────


def test_the_traverse_lands_at_the_fm():
    microscope = _at_fib(_microscope())

    microscope.move_to_microscope("FM")

    assert microscope.get_stage_position().x == pytest.approx(48.8e-3)
    assert microscope.is_at_device("FM") is True
    assert microscope.is_at_device("FIBSEM") is False


def test_the_traverse_round_trips():
    microscope = _at_fib(_microscope())
    start = deepcopy(microscope.get_stage_position())

    microscope.move_to_microscope("FM")
    microscope.move_to_microscope("FIBSEM")

    assert microscope.get_stage_position().is_close(start, tol=1e-9)


def test_the_traverse_leaves_the_orientation_alone():
    """The point of the model: travelling to a device is not re-posing the sample."""
    microscope = _at_fib(_microscope())
    before = deepcopy(microscope.get_stage_position())

    microscope.move_to_microscope("FM")
    after = microscope.get_stage_position()

    assert (after.r, after.t) == (before.r, before.t)


def test_arriving_where_it_already_is_does_not_move():
    microscope = _at_fib(_microscope())
    microscope.move_to_microscope("FM")

    microscope.move_to_microscope("FM")

    assert microscope.get_stage_position().x == pytest.approx(48.8e-3)


def test_the_traverse_is_the_gap_between_the_devices_not_a_constant():
    """Move a device in configuration and the traverse follows it.

    This is what the separate `TRANSLATION_DX` could not do: it was a third number
    that had to be kept consistent with two windows by hand. Now there is one origin
    to move, and the traverse and the window both follow it.
    """
    microscope = _at_fib(_microscope())
    microscope.system.stage.devices["FM"] = StageDeviceSettings(
        origin=FibsemStagePosition(x=30.0e-3)
    )

    microscope.move_to_microscope("FM")

    assert microscope.get_stage_position().x == pytest.approx(30.0e-3)
    assert microscope.is_at_device("FM") is True


# ── the window has to survive the traverse ───────────────────────────

# Every corner of the beam window, because the traverse carries the grid position
# across rather than discarding it: at beam x the stage arrives at x + 48.8.
BEAM_WINDOW_MM = [-19.0, -10.0, 0.0, 5.0, 11.0, 15.0, 19.0]


def _at_beam_x(microscope, beam_x_mm: float):
    """Put the stage at the FIB orientation, `beam_x_mm` along the grid."""
    microscope.move_to_orientation("FIB")
    microscope.move_stage_relative(
        FibsemStagePosition(x=beam_x_mm * 1e-3, y=0.0, z=0.0, r=0.0, t=0.0)
    )
    assert microscope.is_at_device("FIBSEM") is True
    return microscope


@pytest.mark.parametrize("beam_x_mm", BEAM_WINDOW_MM)
def test_anywhere_in_the_beam_window_traverses_to_somewhere_in_the_fm_window(
    beam_x_mm: float,
):
    """The property, not the arithmetic: the two ranges must agree with the traverse.

    They did not. The FM window was `(40, 60)` -- about 10 mm around its origin --
    against the beams' 20 mm, and the traverse carries the offset across unchanged.
    Above beam x = 11.2 mm the stage arrived at the FM and reported that it had not.

    Now that both ranges come from one value this holds by arithmetic rather than by
    agreement: `|arrival - target| = |start - source|`. That is why
    `move_to_microscope` checks where the stage *starts* and never checks where it
    will land -- a destination check could not fire. This test is what makes that
    claim checkable instead of asserted.
    """
    microscope = _at_beam_x(_microscope(), beam_x_mm)

    microscope.move_to_microscope("FM")

    assert microscope.get_stage_position().x == pytest.approx((beam_x_mm + 48.8) * 1e-3)
    assert microscope.is_at_device("FM") is True


@pytest.mark.parametrize("beam_x_mm", BEAM_WINDOW_MM)
def test_asking_twice_does_not_traverse_twice(beam_x_mm: float):
    """What the mismatched window cost: a second request moved the stage again.

    From beam x = 15 mm the old pair landed at 63.8 mm, called it "not at the FM", and
    a second `move_to_microscope("FM")` translated another 48.8 mm to 112.6 mm.
    """
    microscope = _at_beam_x(_microscope(), beam_x_mm)
    microscope.move_to_microscope("FM")
    arrived = microscope.get_stage_position().x

    microscope.move_to_microscope("FM")

    assert microscope.get_stage_position().x == pytest.approx(arrived)


# ── travelling from where the stage is ───────────────────────────────


def test_the_source_is_where_the_stage_is_not_the_other_device():
    """`get_current_device` reports the device, and `None` between them."""
    microscope = _microscope()

    assert microscope.get_current_device() == "FIBSEM"
    assert (
        microscope.get_current_device(
            FibsemStagePosition(x=48.8e-3, y=0, z=0, r=0, t=0)
        )
        == "FM"
    )
    assert (
        microscope.get_current_device(
            FibsemStagePosition(x=24.0e-3, y=0, z=0, r=0, t=0)
        )
        is None
    )


def test_travelling_from_neither_device_is_refused_rather_than_guessed():
    """Mid-traverse the old code translated anyway, and landed nowhere in particular.

    It assumed the source was "the other device". A visible refusal the operator can
    report beats a move that silently ends up 24 mm past the objective -- the same
    preference FIB-640 argues for.
    """
    microscope = _microscope()
    microscope.move_to_orientation("FIB")
    microscope.move_stage_relative(
        FibsemStagePosition(x=IN_THE_GAP_MM * 1e-3, y=0, z=0, r=0, t=0)
    )
    assert microscope.get_current_device() is None

    with pytest.raises(ValueError, match="not at any configured device"):
        microscope.move_to_microscope("FM")


# ── the objective is out before the stage moves ──────────────────────


def test_the_objective_is_retracted_before_the_stage_travels():
    microscope = _at_beam_x(_microscope(), 0.0)
    microscope.fm.objective.insert()

    moved_at = []
    original = microscope.move_stage_relative

    def record(position):
        moved_at.append(microscope.fm.objective.state)
        return original(position)

    microscope.move_stage_relative = record
    microscope.move_to_microscope("FM")

    assert moved_at == ["Retracted"]


def test_no_move_means_no_retraction():
    """The objective comes out for the move, and there is no move here.

    Retracting anyway would pull the objective off the sample to accomplish nothing,
    which is a real cost on a call that is otherwise free -- and asking for the device
    the stage is already at is an ordinary thing for a caller to do.
    """
    microscope = _at_beam_x(_microscope(), 0.0)
    microscope.fm.objective.insert()

    microscope.move_to_microscope("FIBSEM")  # already there

    assert microscope.fm.objective.state == "Inserted"


def test_arriving_at_the_fm_leaves_the_objective_inserted_either_way():
    """The postcondition is the device *and* the objective state, so it holds on the
    no-move path too -- otherwise a redundant call would leave the FM blind."""
    microscope = _at_beam_x(_microscope(), 0.0)
    microscope.move_to_microscope("FM")

    microscope.move_to_microscope("FM")

    assert microscope.fm.objective.state == "Inserted"


def test_a_refused_traverse_leaves_the_objective_alone():
    """A call that refuses changes nothing, the objective included.

    The source is resolved before the objective is touched, so a refusal is inert
    rather than half-done. Retracting first would mean a rejected request still moved
    hardware -- and the operator would be left with the objective out and no
    explanation for it.
    """
    microscope = _at_beam_x(_microscope(), 0.0)
    microscope.move_to_microscope("FM")
    assert microscope.fm.objective.state == "Inserted"

    microscope.move_stage_relative(
        FibsemStagePosition(x=(IN_THE_GAP_MM - 48.8) * 1e-3, y=0, z=0, r=0, t=0)
    )  # into the gap

    with pytest.raises(ValueError, match="not at any configured device"):
        microscope.move_to_microscope("FIBSEM")

    assert microscope.fm.objective.state == "Inserted"


# ── one range, not a window per device ───────────────────────────────


def test_the_window_is_the_range_placed_at_each_origin():
    """The number that used to be written out per device, and drifted.

    Both windows come from one range, so no device can be given a region that
    disagrees with the traverse that gets to it.
    """
    microscope = _microscope()
    device_range = microscope.system.stage.device_range.x

    for device, origin in (("FIBSEM", 0.0), ("FM", 48.8e-3)):
        inside = FibsemStagePosition(x=origin + device_range * 0.99, y=0.0, z=0.0)
        outside = FibsemStagePosition(x=origin + device_range * 1.01, y=0.0, z=0.0)

        assert microscope.is_at_device(device, inside) is True
        assert microscope.is_at_device(device, outside) is False


def test_widening_the_range_widens_every_device_at_once():
    """One number, so the two windows cannot be changed out of step with each other."""
    microscope = _microscope()
    just_past_the_beams = FibsemStagePosition(x=24.0e-3, y=0.0, z=0.0)

    assert microscope.get_current_device(just_past_the_beams) is None

    microscope.system.stage.device_range = FibsemStagePosition(x=25.0e-3)

    assert microscope.get_current_device(just_past_the_beams) == "FIBSEM"


def test_devices_are_allowed_to_overlap():
    """A compustage is exactly that case, and the model has to be able to say it.

    There the objective is under the grid: the beams and the FM are the *same* place,
    reached by flipping rather than travelling, so the two devices share an origin.
    Forbidding overlap would make an Arctis unrepresentable. Position stops deciding
    which device it is -- correctly, because there it is the orientation that does.
    """
    microscope = _microscope()
    microscope.system.stage.devices["FM"] = StageDeviceSettings(
        origin=FibsemStagePosition(x=0.0)
    )

    assert microscope.is_at_device("FIBSEM") is True
    assert microscope.is_at_device("FM") is True


# ── what the traverse does not do ────────────────────────────────────


def test_the_traverse_commands_only_the_axes_the_devices_differ_along():
    """A device change is a translation, not a re-pose.

    Worth pinning because the obvious refactor breaks it: `get_target_position` also
    answers "where should the stage go", but it snaps r and t to the target
    orientation's canonical pose. Reaching for it here would quietly discard a milling
    angle an operator had dialled in. It converts a pose; the traverse relocates one.
    """
    microscope = _at_beam_x(_microscope(), 7.0)
    # Three degrees off the nominal FIB tilt -- still inside the 5 degree band that
    # `get_stage_orientation` calls FIB, so the traverse accepts it. Off-nominal is
    # the whole point: at the canonical pose a snap is invisible, and this test would
    # pass against the very refactor it exists to refuse.
    microscope.move_stage_relative(
        FibsemStagePosition(x=0.0, y=3.0e-3, z=1.0e-3, r=0.0, t=np.radians(3))
    )
    before = deepcopy(microscope.get_stage_position())

    microscope.move_to_microscope("FM")
    after = microscope.get_stage_position()

    assert after.x == pytest.approx(before.x + 48.8e-3)
    assert after.y == pytest.approx(before.y)
    assert after.z == pytest.approx(before.z)
    assert after.r == pytest.approx(before.r)
    assert after.t == pytest.approx(before.t)
