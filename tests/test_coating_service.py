"""The coating service on the Arctis simulator: a deposition's sequence and its
clean-up, a sputter coat, and the service built only where a device is."""

import math
import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.services.coating import (
    ARCTIS_DEPOSITION_POSITION,
    CoatingStatus,
    position_text,
)
from fibsem.structures import InsertableDeviceState


@pytest.fixture
def arctis():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    return microscope


def _moves(microscope, monkeypatch):
    moves = []
    movement = microscope.stage_movement
    move = movement.safe_absolute_stage_movement
    monkeypatch.setattr(
        movement,
        "safe_absolute_stage_movement",
        lambda position: (moves.append(position), move(position))[1],
    )
    return moves


def _calls(device, monkeypatch, names):
    calls = []
    for name in names:
        original = getattr(device, name)

        def record(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(device, name, record)
    return calls


GIS_STEPS = ("insert", "heater_on", "open", "close", "retract")


def test_an_arctis_deposits_from_its_position_and_moves_back(arctis, monkeypatch):
    coating = arctis.coating
    gis = arctis.devices["gis"]
    start = arctis.get_stage_position()
    moves = _moves(arctis, monkeypatch)
    calls = _calls(gis, monkeypatch, GIS_STEPS)

    assert coating.deposition_position.get_value() == ARCTIS_DEPOSITION_POSITION
    assert coating.deposit(2) is CoatingStatus.FINISHED

    assert calls == list(GIS_STEPS)
    assert moves[0] == ARCTIS_DEPOSITION_POSITION
    assert moves[-1].z == pytest.approx(start.z)
    assert gis.state.get_value() is InsertableDeviceState.RETRACTED
    assert not gis.opened.get_value()
    assert coating.progress.get_value().status is CoatingStatus.FINISHED


def test_the_arctis_position_reads_as_its_numbers():
    assert position_text(ARCTIS_DEPOSITION_POSITION) == (
        "x 0 µm, y 0 µm, z −990 µm, tilt 10°"
    )
    assert position_text(None) == "where the stage is"
    assert math.degrees(ARCTIS_DEPOSITION_POSITION.t) == pytest.approx(10)


def test_a_failed_deposition_closes_retracts_and_moves_back(arctis, monkeypatch):
    coating = arctis.coating
    gis = arctis.devices["gis"]
    moves = _moves(arctis, monkeypatch)

    def fail(seconds):
        raise RuntimeError("the valve stuck")

    monkeypatch.setattr(coating, "_wait", fail)
    with pytest.raises(RuntimeError, match="the valve stuck"):
        coating.deposit(2)

    assert not gis.opened.get_value()
    assert gis.state.get_value() is InsertableDeviceState.RETRACTED
    assert len(moves) == 2  # there, and back
    report = coating.progress.get_value()
    assert report.status is CoatingStatus.FAILED
    assert "the valve stuck" in report.error


def test_a_stop_closes_the_valve_early(arctis, monkeypatch):
    coating = arctis.coating
    gis = arctis.devices["gis"]
    waits = []

    def wait(seconds):
        waits.append(seconds)
        coating.stop()

    monkeypatch.setattr(coating, "_wait", wait)
    assert coating.deposit(10) is CoatingStatus.CANCELLED
    assert waits == [1.0]
    assert not gis.opened.get_value()
    assert gis.state.get_value() is InsertableDeviceState.RETRACTED


def test_a_deposition_without_a_position_stays_put(arctis, monkeypatch):
    coating = arctis.coating
    coating._position = None
    moves = _moves(arctis, monkeypatch)
    assert "deposition_position" not in coating.parameters or (
        coating.deposition_position.get_value() is None
    )
    assert coating.deposit(1) is CoatingStatus.FINISHED
    assert moves == []


def test_a_sputter_coat_runs_at_the_current_asked(arctis):
    coater = arctis.devices["sputter_coater"]
    assert arctis.coating.sputter(3, current=2e-7) is CoatingStatus.FINISHED
    assert coater.sim_runs == [(3.0, None, 2e-7)]


def test_a_deposition_refuses_no_time(arctis):
    with pytest.raises(ValueError):
        arctis.coating.deposit(0)


def test_no_gis_and_no_coater_means_no_coating():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    if "gis" in microscope.devices or "sputter_coater" in microscope.devices:
        pytest.skip("the default simulator lists a GIS or a coater")
    assert microscope.coating is None
    assert "coating" not in microscope.services


def test_the_agent_may_coat_but_never_drive_the_devices_one_by_one(arctis):
    from fibsem.ai.agent.blocklist import blocked

    for command in ("deposit", "sputter", "stop"):
        assert blocked(arctis.coating, "command", command) is None
    assert blocked(arctis.devices["gis"], "command", "open")
    assert blocked(arctis.devices["sputter_coater"], "command", "run")
