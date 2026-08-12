"""Detector reads and writes take the shared imaging channel in two steps (FIB-544).

`ThermoMicroscope._get` and `._set` both do this for the four detector properties:

    self.set_channel(beam_type)          # 2 RPCs
    return self.connection.detector.type.value   # 1 RPC, resolves against the *active
                                                 # device*

Two RPCs, no lock. `connection.detector` resolves against whatever device is active when
the second one lands, so anything taking the channel in between makes the read answer
from the other column -- silently, because the value is a bare float or string with
nothing in it to say which device replied. The write half is worse: brightness or
contrast lands on a detector nobody asked about, and stays there.

This is FIB-542's defect on the property path, and it is more exposed than the
acquisition pair was. `get_detector_settings` is four of these back to back, and
`get_microscope_state()` calls it once per beam -- so one state read with no `beam_type`
is eight unguarded pairs, and it is polled from the GUI thread by the minimap's stage
refresh and the coincidence viewer.

FIB-601 raised the stakes: holding the FM channel for a whole autofocus sweep means the
FM now sits on the channel for an uninterrupted stretch rather than handing it back
between steps, so a poll landing mid-sweep is likelier to have the view moved under it.

These run against the simulator, which models the shared channel on both sides since
FIB-518 and the detector pair since this change. The hardware class cannot be built here
(no AutoScript SDK), so its side is pinned structurally in
`tests/test_imaging_channel_lock.py`.
"""

import logging

import pytest

from fibsem.microscopes.simulator import FM_ACTIVE_VIEW
from fibsem.structures import BeamType

STOLEN = "Imaging channel changed"

DETECTOR_PROPERTIES = [
    "detector_type",
    "detector_mode",
    "detector_brightness",
    "detector_contrast",
]


@pytest.fixture
def microscope():
    from fibsem import utils

    scope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    assert scope.fm is not None, "the simulated compustage system should have an FM"
    return scope


def steal_after_set_channel(microscope, monkeypatch) -> None:
    """Have the FM take the channel the instant a beam operation claims it.

    The other pairs in this suite hook `sim_sleep`, which stands in for the operation's
    duration. A detector read has no duration to hook -- it is two RPCs back to back --
    so the window is opened where it actually is: between the `set_channel` and the read
    that follows it. Patching `set_channel` to hand the channel straight to the FM models
    another thread winning in exactly that gap.
    """
    original = type(microscope).set_channel

    def set_then_lose(self, beam_type):
        original(self, beam_type)
        self.imaging_system.active_view = FM_ACTIVE_VIEW

    monkeypatch.setattr(type(microscope), "set_channel", set_then_lose)


class TestTheReadPath:
    @pytest.mark.parametrize("key", DETECTOR_PROPERTIES)
    def test_a_read_notices_the_channel_moving_under_it(
        self, microscope, monkeypatch, caplog, key
    ):
        """The defect. Without the lock the read lands on the FM's device.

        Asserted on the warning rather than on the returned value, matching
        `_warn_if_channel_moved`'s deliberate choice not to answer wrongly: what hardware
        does here is return the wrong number with no way to tell, so the simulator makes
        it visible instead of adding a second way to be wrong.
        """
        steal_after_set_channel(microscope, monkeypatch)

        with caplog.at_level(logging.WARNING):
            microscope.get(key, BeamType.ELECTRON)

        assert STOLEN in caplog.text, (
            f"reading {key} did not notice the channel moving between its set and its "
            f"read -- on hardware that returns the other column's detector, silently"
        )

    def test_the_channel_is_the_beams_when_nothing_steals_it(self, microscope, caplog):
        """The other direction, so the test above cannot pass by warning always."""
        with caplog.at_level(logging.WARNING):
            microscope.get("detector_type", BeamType.ION)

        assert STOLEN not in caplog.text
        assert microscope.imaging_system.active_view == BeamType.ION.value


class TestTheWritePath:
    @pytest.mark.parametrize("key, value", [
        ("detector_brightness", 0.5),
        ("detector_contrast", 0.5),
    ])
    def test_a_write_notices_the_channel_moving_under_it(
        self, microscope, monkeypatch, caplog, key, value
    ):
        """The dangerous half: this one reconfigures hardware rather than misreporting."""
        steal_after_set_channel(microscope, monkeypatch)

        with caplog.at_level(logging.WARNING):
            microscope.set(key, value, BeamType.ELECTRON)

        assert STOLEN in caplog.text, (
            f"writing {key} did not notice the channel moving -- on hardware that "
            f"applies the value to whichever detector is active, and leaves it there"
        )


class TestWhatTheStateReadCosts:
    def test_get_detector_settings_is_four_pairs(self, microscope, monkeypatch):
        """Why this is more exposed than the acquisition pair was.

        Four properties, four independent set-then-read pairs. Locking each one makes
        every value correct on its own and still lets something interleave *between*
        them, so the four can describe two different states -- which is why the fix
        holds the lock across the group rather than only per pair.
        """
        claims = []
        original = type(microscope).set_channel
        monkeypatch.setattr(
            type(microscope),
            "set_channel",
            lambda self, beam_type: (claims.append(beam_type), original(self, beam_type))[1],
        )

        microscope.get_detector_settings(BeamType.ELECTRON)

        assert len(claims) == 4, (
            f"expected one channel claim per detector property, got {len(claims)} -- "
            f"if this changed, the interleaving window changed with it"
        )
        assert all(bt is BeamType.ELECTRON for bt in claims)
