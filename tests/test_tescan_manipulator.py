"""Tescan reports no manipulator.

fibsem no longer drives the Tescan Nanomanipulator: the backend reports none, so
the app hides the manipulator widget, and the manipulator methods are the base
class's, which raise without touching the instrument.

No hardware or Tescan SDK required: the microscope is connected over the fake SDK
(``tests/fixtures/tescan_sdk.py``).
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import FibsemManipulatorPosition
from tests.fixtures.tescan_sdk import connect


@pytest.fixture
def connected(monkeypatch):
    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    return connect(monkeypatch, system)


def test_it_reports_no_manipulator(connected):
    microscope, _ = connected
    assert microscope.is_available("manipulator") is False
    assert microscope.system.manipulator.enabled is False
    assert microscope.capability_sources["manipulator"] == "backend"
    assert microscope.manipulator_device is None
    assert microscope.manipulator_named_positions() == []


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.insert_manipulator("Standby"),
        lambda m: m.retract_manipulator(),
        lambda m: m.move_manipulator_relative(FibsemManipulatorPosition(x=1e-6)),
        lambda m: m.move_manipulator_absolute(FibsemManipulatorPosition()),
        lambda m: m.move_manipulator_to_named_position("Parking"),
    ],
)
def test_its_manipulator_moves_raise_without_touching_the_instrument(connected, call):
    microscope, fake = connected
    with pytest.raises(NotImplementedError):
        call(microscope)
    assert not [path for path, _, _ in fake.log if "Nanomanipulator" in path]
