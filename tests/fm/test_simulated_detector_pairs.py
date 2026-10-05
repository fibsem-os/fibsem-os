"""Detector reads and writes take the shared imaging channel in two steps (FIB-544).

``ThermoMicroscope._get`` and ``._set`` do this for the four detector properties::

    self.set_channel(beam_type)                  # 2 RPCs
    return self.connection.detector.type.value   # 1 RPC, against the *active device*

``connection.detector`` resolves against whatever device is active when the read lands,
so anything taking the channel in between makes the read answer from the other column,
silently, and makes a write land on the other column's detector and stay there.
``get_detector_settings`` is four of these back to back, and ``get_microscope_state``
calls it once per beam, polled from the GUI thread.

These run on ``LegacyDemoMicroscope``, whose simulated column has one detector on a
shared channel, as the hardware has. The device Demo gives each beam device its own
detector, so it has no channel to lose here. ``ThermoMicroscope`` cannot be built here
(no AutoScript SDK), so its side is pinned structurally in
``tests/test_imaging_channel_lock.py``.
"""

import logging

import pytest

from fibsem.microscopes.simulator import DETECTOR_KEYS, FM_ACTIVE_VIEW
from fibsem.structures import BeamType
from tests._legacy_demo import setup_legacy_session

STOLEN = "Imaging channel changed"


@pytest.fixture
def microscope():
    scope, _ = setup_legacy_session(ip_address="localhost")
    return scope


def steal_after_set_channel(microscope, monkeypatch) -> None:
    """Have the FM take the channel the instant a beam operation claims it.

    A detector read has no duration to hook, so the window is opened where it is:
    between the ``set_channel`` and the access that follows it.
    """
    original = type(microscope).set_channel

    def set_then_lose(self, beam_type):
        original(self, beam_type)
        self.imaging_system.active_view = FM_ACTIVE_VIEW

    monkeypatch.setattr(type(microscope), "set_channel", set_then_lose)


@pytest.mark.parametrize("key", DETECTOR_KEYS)
def test_a_read_notices_the_channel_moving_under_it(
    microscope, monkeypatch, caplog, key
):
    """Asserted on the warning: hardware returns the wrong value with no way to tell,
    so the simulator makes it visible rather than being wrong a second way."""
    steal_after_set_channel(microscope, monkeypatch)

    with caplog.at_level(logging.WARNING):
        microscope.get(key, BeamType.ELECTRON)

    assert STOLEN in caplog.text


def test_the_channel_is_the_beams_when_nothing_steals_it(microscope, caplog):
    """The other direction, so the test above cannot pass by warning always."""
    with caplog.at_level(logging.WARNING):
        microscope.get("detector_type", BeamType.ION)

    assert STOLEN not in caplog.text
    assert microscope.imaging_system.active_view == BeamType.ION.value


@pytest.mark.parametrize(
    "key, value", [("detector_brightness", 0.5), ("detector_contrast", 0.5)]
)
def test_a_write_notices_the_channel_moving_under_it(
    microscope, monkeypatch, caplog, key, value
):
    """The dangerous half: a write reconfigures the other column's detector."""
    steal_after_set_channel(microscope, monkeypatch)

    with caplog.at_level(logging.WARNING):
        microscope.set(key, value, BeamType.ELECTRON)

    assert STOLEN in caplog.text


def test_get_detector_settings_claims_the_channel_for_each_property(
    microscope, monkeypatch
):
    """Four properties, four claims: why ``ThermoMicroscope`` holds the lock across
    the group rather than only around each one."""
    claims = []
    original = type(microscope).set_channel
    monkeypatch.setattr(
        type(microscope),
        "set_channel",
        lambda self, beam_type: (claims.append(beam_type), original(self, beam_type)),
    )

    microscope.get_detector_settings(BeamType.ELECTRON)

    assert claims == [BeamType.ELECTRON] * 4
