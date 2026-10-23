"""Tescan has no chamber.

fibsem no longer drives the Tescan chamber: no chamber device is built, the
chamber keys are unsupported (they read None), and pumping or venting raises, so
none of them reaches the instrument.

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
        os.path.join(cfg.SHIPPED_CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    return connect(monkeypatch, system)


def test_it_has_no_chamber_device(connected):
    microscope, _ = connected
    assert microscope.chamber_device is None


@pytest.mark.parametrize("method", ["pump", "vent"])
def test_pumping_or_venting_raises_unsupported(connected, method):
    microscope, fake = connected
    with pytest.raises(NotImplementedError, match=f"does not support {method}"):
        getattr(microscope, method)()
    assert not [path for path, _, _ in fake.log if "Chamber" in path]
