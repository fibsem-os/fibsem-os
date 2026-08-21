import numpy as np
import pytest

from fibsem import movement


def test_angle_difference():
    assert np.isclose(movement.angle_difference(np.deg2rad(0), np.deg2rad(0)), 0)
    assert np.isclose(movement.angle_difference(np.deg2rad(0), np.deg2rad(360)), 0)
    assert np.isclose(movement.angle_difference(np.deg2rad(0), np.deg2rad(180)), np.pi)
    assert np.isclose(
        movement.angle_difference(np.deg2rad(2), np.deg2rad(358)), np.deg2rad(4)
    )
    assert np.isclose(
        movement.angle_difference(np.deg2rad(-45), np.deg2rad(45)), np.pi / 2
    )
    assert np.isclose(movement.angle_difference(np.deg2rad(-360), np.deg2rad(360)), 0)
    assert np.isclose(movement.angle_difference(-4 * np.pi, 4 * np.pi), 0)


def test_rotation_angle_is_larger():
    assert movement.rotation_angle_is_larger(np.deg2rad(0), np.deg2rad(0)) == False
    assert movement.rotation_angle_is_larger(np.deg2rad(0), np.deg2rad(360)) == False
    assert movement.rotation_angle_is_larger(np.deg2rad(-45), np.deg2rad(45)) == False
    assert movement.rotation_angle_is_larger(np.deg2rad(0), np.deg2rad(180)) == True
    assert movement.rotation_angle_is_larger(np.deg2rad(-90), np.deg2rad(90)) == True
    assert movement.rotation_angle_is_larger(np.deg2rad(0), np.deg2rad(720)) == False


def test_rotation_angle_is_smaller():
    assert movement.rotation_angle_is_smaller(np.deg2rad(0), np.deg2rad(0), 5) == True
    assert movement.rotation_angle_is_smaller(np.deg2rad(0), np.deg2rad(360), 5) == True
    assert movement.rotation_angle_is_smaller(np.deg2rad(0), np.deg2rad(180)) == False
    assert (
        movement.rotation_angle_is_smaller(np.deg2rad(-90), np.deg2rad(90), 180)
        == False
    )
    assert (
        movement.rotation_angle_is_smaller(np.deg2rad(0), np.deg2rad(-360), 1) == True
    )


# ── the simulator pretends a stage move takes time ───────────────────────────


def _demo_microscope():
    import os

    import fibsem.config as fibsem_config
    from fibsem import utils

    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    return scope


def test_a_simulated_stage_move_takes_time(monkeypatch):
    """The simulator models how long an acquisition takes and modelled a stage move as
    instantaneous, which put anything the UI says *during* a move out of reach: the
    message went up and came down inside one event loop iteration (FIB-765).

    Asserted through `sim_sleep` rather than a wall clock, so the test neither sleeps
    nor becomes flaky on a loaded machine.
    """
    from fibsem.microscopes import simulator
    from fibsem.structures import FibsemStagePosition

    slept = []
    monkeypatch.setattr(simulator, "sim_sleep", lambda seconds: slept.append(seconds))

    scope = _demo_microscope()
    base = scope.get_stage_position()
    # Cleared here: reading the stage position has its own 0.1 s in the simulator, which
    # predates this and would otherwise be counted as part of the move.
    slept.clear()

    scope.move_stage_absolute(
        FibsemStagePosition(x=base.x + 50e-6, y=base.y, z=base.z, r=base.r, t=base.t)
    )

    # First, before anything else the move does. The delay is taken *before* the
    # position is assigned, so a reader during the move sees where the stage set off
    # from rather than where it is going.
    assert slept[0] == simulator.STAGE_MOVEMENT_SLEEP_TIME
    assert slept.count(simulator.STAGE_MOVEMENT_SLEEP_TIME) == 1


def test_a_relative_move_takes_time_too(monkeypatch):
    """`stable_move` and `vertical_move` both land here, so a delay only on the absolute
    path would leave the two gestures that drive coincidence instant."""
    from fibsem.microscopes import simulator
    from fibsem.structures import FibsemStagePosition

    slept = []
    monkeypatch.setattr(simulator, "sim_sleep", lambda seconds: slept.append(seconds))

    scope = _demo_microscope()
    slept.clear()

    scope.move_stage_relative(FibsemStagePosition(x=10e-6, y=0.0, z=0.0))

    assert slept[0] == simulator.STAGE_MOVEMENT_SLEEP_TIME
    assert slept.count(simulator.STAGE_MOVEMENT_SLEEP_TIME) == 1


def test_the_delay_is_off_under_the_test_flag():
    """`FIBSEM_SIM_NO_DELAY` is what keeps the suite fast, and it is set for the whole
    run in `tests/conftest.py`. If that ever stops being true, every stage move in the
    suite starts costing a second and this says so rather than the suite just getting
    slow."""
    import os
    import time

    from fibsem.structures import FibsemStagePosition

    assert os.environ.get("FIBSEM_SIM_NO_DELAY") == "1", (
        "the suite is running with simulator delays enabled"
    )

    scope = _demo_microscope()
    base = scope.get_stage_position()
    started = time.perf_counter()
    scope.move_stage_absolute(
        FibsemStagePosition(x=base.x + 10e-6, y=base.y, z=base.z, r=base.r, t=base.t)
    )

    assert time.perf_counter() - started < 0.5
