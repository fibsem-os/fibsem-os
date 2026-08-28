"""Converting across both axes at once: the device and the orientation.

On an offset mount these are independent. The device is a *place* the stage travels
to; the orientation is the pose the sample is held in once it is there. So a target
position is a pair, and `get_target_position` takes both.

The thing worth reading this file for is the **order**. The two legs do not commute,
and only one arrangement round-trips -- see
`test_the_orientation_leg_is_bracketed_by_the_device_legs`, which is where the wrong
one was caught.

See FIB-830.
"""

import itertools
import os

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import FibsemStagePosition

IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")

ORIENTATIONS = ("SEM", "FIB", "MILLING")
TRAVERSE = 48.8e-3

# Across the beam device's range, which is what the traverse has to carry.
GRID_X_MM = (-19.0, -12.0, 0.0, 5.0, 15.0, 19.0)


def _microscope(config_path: str = IFLM_CONFIG):
    microscope, _ = utils.setup_session(config_path=config_path)
    return microscope


def _at(microscope, orientation: str, x_mm: float, y_mm: float = 2.0):
    """A position at `orientation`, `x_mm` along the grid. Constructed, not moved to.

    Relative moves accumulate, and a few of them in a row walk the stage into the gap
    between the two devices, where the conversion is refused -- correctly, and
    confusingly if it happens inside a loop.
    """
    pose = microscope.get_orientation(orientation)
    return FibsemStagePosition(
        x=x_mm * 1e-3,
        y=y_mm * 1e-3,
        z=0.0,
        r=pose.r,
        t=pose.t,
        coordinate_system="RAW",
    )


# ── the device leg ───────────────────────────────────────────────────


def test_asking_only_for_a_device_is_the_traverse():
    """No orientation change, so the whole answer is the gap between the origins."""
    microscope = _microscope()
    position = _at(microscope, "FIB", 5.0)

    target = microscope.get_target_position(position, "FIB", target_device="FM")

    assert target.x == pytest.approx(position.x + TRAVERSE)
    assert target.y == pytest.approx(position.y)
    assert microscope.is_at_device("FM", target) is True


def test_fm_milling_is_a_device_and_an_orientation_not_a_fifth_orientation():
    """The request that settles the design (FIB-833): at the FM, posed for milling.

    Nothing was added to `orientations` for this. It is a pair the model could already
    express the moment the device became its own axis.
    """
    microscope = _microscope()

    target = microscope.get_target_position(
        _at(microscope, "SEM", 0.0), "MILLING", target_device="FM"
    )

    assert microscope.get_current_device(target) == "FM"
    assert microscope.get_stage_orientation(target) == "MILLING"
    assert target.t == pytest.approx(microscope.get_orientation("MILLING").t)


# ── the order of the two legs ────────────────────────────────────────


def test_the_orientation_leg_is_bracketed_by_the_device_legs():
    """The arrangement, stated as the number it produces.

    `_get_compucentric_rotation_position` is a half turn about a chamber-fixed centre,
    and the beams are the only place it has ever been applied. So a position is
    carried into that frame, re-posed, and carried back out to the device asked for.

    The alternative -- re-pose wherever the stage is, then translate -- gives the same
    answer going out and a different one coming back, because the return journey
    rotates about a centre 48.8 mm away. From the beams at x = 0 it returns
    **-97.6 mm** rather than 0. That is how this was caught; it is not a stylistic
    preference between two correct orders.
    """
    microscope = _microscope()
    start = _at(microscope, "SEM", 0.0)

    at_the_fm = microscope.get_target_position(start, "FIB", target_device="FM")
    returned = microscope.get_target_position(at_the_fm, "SEM", target_device="FIBSEM")

    assert at_the_fm.x == pytest.approx(TRAVERSE)
    assert returned.x == pytest.approx(0.0, abs=1e-12)
    assert returned.x != pytest.approx(-2 * TRAVERSE)  # the arrangement that lost


@pytest.mark.parametrize("x_mm", GRID_X_MM)
@pytest.mark.parametrize(
    ("there", "back"), list(itertools.product(ORIENTATIONS, repeat=2))
)
def test_every_pair_round_trips_from_anywhere_on_the_grid(
    x_mm: float, there: str, back: str
):
    """`f(p) = rotate(p - source) + target`, and `rotate` is an involution.

    So swapping source and target inverts it, which is exactly what the return call
    does. Pinned across the beam device's whole range because the traverse carries the
    grid offset -- an error in the bracketing shows up as a function of x, and at
    x = 0 with the simulator's zero compucentric offset it would hide entirely.
    """
    microscope = _microscope()
    start = _at(microscope, back, x_mm)

    away = microscope.get_target_position(start, there, target_device="FM")
    returned = microscope.get_target_position(away, back, target_device="FIBSEM")

    assert returned.x == pytest.approx(start.x, abs=1e-12)
    assert returned.y == pytest.approx(start.y, abs=1e-12)
    assert np.isclose(returned.r, start.r)
    assert np.isclose(returned.t, start.t)


@pytest.mark.parametrize("x_mm", GRID_X_MM)
def test_the_traverse_lands_inside_the_fm_range_whatever_the_pose(x_mm: float):
    """A converted position has to be somewhere the device axis recognises.

    Otherwise the caller has a target it cannot then be told it has reached.
    """
    microscope = _microscope()

    for orientation in ORIENTATIONS:
        target = microscope.get_target_position(
            _at(microscope, "SEM", x_mm), orientation, target_device="FM"
        )
        assert microscope.get_current_device(target) == "FM"


# ── what did not change ──────────────────────────────────────────────


def test_a_compustage_brackets_with_nothing():
    """Its objective is under the grid: the FM is reached by flipping, not travelling.

    So the device legs are a zero translation there and the compustage takes the same
    path to the same answer it gave before -- no stage-type branch in the transform.
    """
    microscope = _microscope(ARCTIS_CONFIG)
    start = _at(microscope, "SEM", 0.0)

    with_device = microscope.get_target_position(start, "FM", target_device="FM")
    orientation_only = microscope.get_target_position(start, "FM")

    assert with_device.x == pytest.approx(orientation_only.x)
    assert with_device.y == pytest.approx(orientation_only.y)


def test_asking_for_no_device_is_the_transform_that_was_already_there():
    """Every existing caller passes no device, and gets what it always got."""
    microscope = _microscope()
    start = _at(microscope, "SEM", 5.0)

    target = microscope.get_target_position(start, "FIB")

    assert target.x == pytest.approx(-start.x)  # the compucentric half turn, alone
    assert np.isclose(target.r, microscope.get_orientation("FIB").r)


def test_converting_to_where_it_already_is_still_returns_the_same_object():
    """The aliasing early return, which `tests/test_orientation_transform_parity.py`
    pins. A device argument is what takes the conversion off that path."""
    microscope = _microscope()
    start = _at(microscope, "SEM", 0.0)

    assert microscope.get_target_position(start, "SEM") is start
    assert microscope.get_target_position(start, "SEM", target_device="FM") is not start


def test_the_fm_is_still_not_an_orientation_on_an_offset_mount():
    """`orientations["FM"]` is a copy of the FIB entry there, carrying no positional
    term. The device argument is how to ask for the FM; the orientation is not."""
    microscope = _microscope()

    with pytest.raises(ValueError, match="Cannot move to FM position"):
        microscope.get_target_position(_at(microscope, "SEM", 0.0), "FM")


def test_converting_from_between_the_devices_is_refused():
    """Mid-traverse there is no source frame to convert out of, so it says so."""
    microscope = _microscope()
    stranded = _at(microscope, "SEM", 24.0)  # the gap between the two ranges

    assert microscope.get_current_device(stranded) is None

    with pytest.raises(ValueError, match="not at any configured device"):
        microscope.get_target_position(stranded, "FIB", target_device="FM")
