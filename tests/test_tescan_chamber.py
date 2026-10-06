"""Tescan has no chamber.

fibsem no longer drives the Tescan chamber: no chamber device is built, and the
chamber keys are unknown to ``_get``/``_set``, so reading them, pumping or venting
never reaches the instrument.

No hardware or Tescan SDK required: the microscope is connected over the fake SDK
(``tests/fixtures/tescan_sdk.py``).
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from tests.fixtures.tescan_sdk import connect


@pytest.fixture
def connected(monkeypatch):
    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    return connect(monkeypatch, system)


def test_it_has_no_chamber_device(connected):
    microscope, _ = connected
    assert microscope.chamber_device is None


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m._get_impl("chamber_state"),
        lambda m: m._get_impl("chamber_pressure"),
        lambda m: m.pump(),
        lambda m: m.vent(),
    ],
)
def test_the_chamber_is_never_touched(connected, call):
    microscope, fake = connected
    assert call(microscope) is None
    assert not [path for path, _, _ in fake.log if "Chamber" in path]
