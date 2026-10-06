"""What the Tescan API has no control for is answered by the wrappers, quietly.

The FIB has no working distance in the Tescan API and neither column has a detector
mode. State restores call ``set_working_distance`` and ``set_detector_mode`` every time,
so ``TescanMicroscope`` answers them itself: None, nothing sent to the instrument, and
nothing logged above debug. The old ``_get``/``_set`` branches for them are deleted, so
the string keys are unknown.

No hardware or Tescan SDK required: the microscope is connected over the fake SDK
(``tests/fixtures/tescan_sdk.py``).
"""

import logging
import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import BeamType
from tests.fixtures.tescan_sdk import connect

E, I = BeamType.ELECTRON, BeamType.ION


@pytest.fixture
def connected(monkeypatch):
    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    return connect(monkeypatch, system)


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.get_working_distance(I),
        lambda m: m.set_working_distance(5e-3, I),
        lambda m: m.get_detector_mode(E),
        lambda m: m.set_detector_mode("SE", E),
        lambda m: m.get_detector_mode(I),
        lambda m: m.set_detector_mode("SE", I),
    ],
)
def test_it_answers_none_without_the_instrument_or_a_warning(connected, call, caplog):
    microscope, fake = connected
    with caplog.at_level(logging.INFO):
        assert call(microscope) is None
    assert fake.log == []
    assert caplog.records == []


def test_the_electron_working_distance_still_reaches_the_column(connected):
    microscope, fake = connected
    microscope.set_working_distance(5e-3, E)
    assert ["SEM.Optics.SetWD", [5.0], {}] in fake.log


@pytest.mark.parametrize(
    "key, beam_type",
    [("detector_mode", E), ("detector_mode", I), ("working_distance", I)],
)
def test_the_keys_are_unknown(connected, key, beam_type, caplog):
    microscope, _ = connected
    with caplog.at_level(logging.WARNING):
        microscope._set_impl(key, None, beam_type)
    assert [r.getMessage() for r in caplog.records] == [
        f"Unknown key: {key}, value: None ({beam_type})"
    ]
