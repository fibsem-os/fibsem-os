"""The simulator models the imaging channel the FM and the beams share (FIB-518).

On a TFS system the FM and the beams are one connection with one active view and one
active device -- whoever writes last owns the microscope. Every bug in that class
(FIB-517, FIB-542, FIB-544, FIB-545) had to be found on hardware, because the simulator
modelled the beam half only: `imaging_system.active_view` existed, but the simulated FM
never contended for it, so no test could show a beam operation losing the channel. One
of those bugs stopped a running workflow task.

The behavioural counterpart to the two tests that had to settle for less:
`tests/fm/test_active_channel.py` pins the driver's contract against a hand copy, and
`tests/test_imaging_channel_lock.py` pins the beam side by reading its source -- both
because `fibsem/microscope.py`'s TFS half needs an SDK that is not installed off the
microscope. Here the contract can be run instead of inspected.
"""

import logging
import threading

import pytest

from fibsem.microscopes.simulator import FM_ACTIVE_DEVICE, FM_ACTIVE_VIEW
from fibsem.structures import BeamType, ImageSettings

STOLEN = "Imaging channel changed"


@pytest.fixture
def microscope():
    from fibsem import utils

    scope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    assert scope.fm is not None, "the simulated compustage system should have an FM"
    return scope


@pytest.fixture
def settings():
    return ImageSettings(
        hfw=80e-6,
        resolution=(256, 256),
        beam_type=BeamType.ION,
        dwell_time=1e-6,
    )


def frame_seconds(settings: ImageSettings) -> float:
    """How long the simulator spends on the frame itself.

    The simulator sleeps twice per acquisition -- once reading the microscope state for
    the metadata, once for the frame -- and only the second is inside the channel lock.
    Tests that want to act *during the frame* discriminate on this.
    """
    return settings.dwell_time * settings.resolution[0] * settings.resolution[1]


class TestTheFMTakesTheChannel:
    """The half the simulator was missing entirely."""

    def test_bringing_the_fm_up_leaves_the_channel_on_it(self, microscope):
        """As `ThermoMicroscope.__init__` does. Taking it back is the next beam
        operation's job, and the fact that this is the starting state is half of why
        that matters."""
        assert microscope.imaging_system.active_view == FM_ACTIVE_VIEW

    def test_a_bare_set_leaves_it_on_the_fm(self, microscope):
        """The shape of the bug: this is what a property getter does. `objective.state`
        called it on every stage poll, and the poll runs constantly."""
        microscope.set_channel(BeamType.ELECTRON)

        microscope.fm.set_active_channel()

        assert microscope.imaging_system.active_view == FM_ACTIVE_VIEW
        assert microscope.imaging_system.active_device == FM_ACTIVE_DEVICE

    def test_the_scope_hands_it_back(self, microscope):
        microscope.set_channel(BeamType.ELECTRON)

        with microscope.fm.active_channel():
            assert microscope.imaging_system.active_view == FM_ACTIVE_VIEW

        assert microscope.imaging_system.active_view == BeamType.ELECTRON.value

    def test_it_restores_whatever_was_there(self, microscope):
        """Not a hardcoded beam view: it puts back what it found."""
        for beam_type in (BeamType.ELECTRON, BeamType.ION):
            microscope.set_channel(beam_type)

            with microscope.fm.active_channel():
                pass

            assert microscope.imaging_system.active_view == beam_type.value

    def test_the_device_comes_back_with_the_view(self, microscope):
        """A view left correct with the device still on the FM would be a half-restore
        that reads as clean."""
        microscope.set_channel(BeamType.ION)

        with microscope.fm.active_channel():
            pass

        assert microscope.imaging_system.active_device == BeamType.ION.value

    def test_it_hands_it_back_when_the_body_fails(self, microscope):
        """An FM that is off, or answering slowly, must not strand the beam side."""
        microscope.set_channel(BeamType.ION)

        with pytest.raises(RuntimeError):
            with microscope.fm.active_channel():
                raise RuntimeError("detector did not answer")

        assert microscope.imaging_system.active_view == BeamType.ION.value

    def test_only_the_outermost_scope_restores(self, microscope):
        """A tileset holds the channel for the whole run and each tile opens a scope
        inside it. An inner scope that restored would put the beam view back between
        every pair of tiles."""
        microscope.set_channel(BeamType.ION)

        with microscope.fm.active_channel():
            with microscope.fm.active_channel():
                assert microscope.imaging_system.active_view == FM_ACTIVE_VIEW
            assert microscope.imaging_system.active_view == FM_ACTIVE_VIEW, (
                "the inner scope restored"
            )

        assert microscope.imaging_system.active_view == BeamType.ION.value

    def test_acquiring_an_fm_image_hands_the_channel_back(self, microscope):
        """The path everything above the driver actually takes:
        `FluorescenceMicroscope.acquire_image` is the chokepoint every FM image passes
        through, and it is scoped."""
        microscope.set_channel(BeamType.ELECTRON)

        microscope.fm.acquire_image()

        assert microscope.imaging_system.active_view == BeamType.ELECTRON.value


class TestTheBeamNoticesATheft:
    """The other half: a beam operation can now tell that it lost the channel."""

    def test_a_beam_acquisition_takes_the_channel(self, microscope, settings):
        microscope.fm.set_active_channel()

        microscope.acquire_image(image_settings=settings)

        assert microscope.imaging_system.active_view == settings.beam_type.value

    def test_a_clean_acquisition_says_nothing(self, microscope, settings, caplog):
        with caplog.at_level(logging.WARNING):
            microscope.acquire_image(image_settings=settings)

        assert STOLEN not in caplog.text

    def test_an_unguarded_fm_read_during_the_frame_is_reported(
        self, microscope, settings, monkeypatch, caplog
    ):
        """The failure FIB-517 was, reproduced without hardware.

        The FM read is unscoped -- `set_active_channel` and walk away -- which is what
        every property getter did before FIB-517, and what the getters FIB-544 covers
        still do. It lands mid-frame, where on hardware `grab_frame` would then read the
        FM's buffer and hand it back labelled as an ion image.
        """
        during_the_frame = frame_seconds(settings)

        def steal(seconds: float) -> None:
            if seconds == during_the_frame:
                microscope.fm.set_active_channel()

        monkeypatch.setattr("fibsem.microscopes.simulator.sim_sleep", steal)

        with caplog.at_level(logging.WARNING):
            microscope.acquire_image(image_settings=settings)

        assert STOLEN in caplog.text
        assert "found view 3" in caplog.text

    def test_the_scoped_form_cannot_steal_during_the_frame(
        self, microscope, settings, monkeypatch, caplog
    ):
        """Why the scoped form is the fix and not just good manners.

        `active_channel()` takes the same lock the acquisition holds over its channel
        and its frame, so an FM operation on another thread -- the live stream, an
        overview run -- queues behind the frame instead of landing inside it. The thief
        here is a real second thread, and the test asserts it was genuinely blocked
        during the frame *and* that it eventually got in, so a thief that never ran
        cannot pass this by doing nothing.
        """
        during_the_frame = frame_seconds(settings)
        at_the_door = threading.Event()
        got_in = threading.Event()
        blocked_during_the_frame = []

        def steal() -> None:
            at_the_door.wait(timeout=5)
            with microscope.fm.active_channel():
                got_in.set()

        thief = threading.Thread(target=steal, daemon=True)
        thief.start()

        def frame(seconds: float) -> None:
            if seconds != during_the_frame:
                return
            at_the_door.set()
            # Long enough that a thief who could get in, would have.
            blocked_during_the_frame.append(not got_in.wait(timeout=0.5))

        monkeypatch.setattr("fibsem.microscopes.simulator.sim_sleep", frame)

        with caplog.at_level(logging.WARNING):
            microscope.acquire_image(image_settings=settings)

        thief.join(timeout=5)
        assert got_in.is_set(), "the thief never ran -- the test proved nothing"
        assert blocked_during_the_frame == [True], "the scoped form got in mid-frame"
        assert STOLEN not in caplog.text
