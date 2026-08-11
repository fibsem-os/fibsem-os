"""The other operations that take the shared imaging channel (FIB-569, FIB-545).

FIB-542 locked the beam's set-and-grab. Sweeping `ThermoMicroscope` for the same shape
found three more places where the action is documented as operating on the *active view*
and the pair is unguarded, plus one that takes the channel and never gives it back:

| operation | the vendor doc's words | why it matters |
| -- | -- | -- |
| `auto_focus` | "Runs the automatic focus routine in the active view" | a **write** -- focuses the wrong column and leaves it there |
| `autocontrast` | "optimizes contrast and brightness of the active detector in the active view" | a **write**, same |
| `last_image` | "Retrieves a microscope image currently present in the active view" | FIB-542's twin on the retrieval path |
| `acquire_chamber_image` | takes view 4 / device 3 | never restores, so it strands whoever had the channel |

The autofunctions are the dangerous pair: a stolen grab returns one wrong image, a stolen
autofocus misconfigures the beam for everything after it. `imaging/tiled.py:359` runs
`auto_focus` once per tile of an unattended tileset.

These run against the simulator, which models the shared channel since FIB-518. The
hardware class cannot be built here (no SDK), so its side is pinned structurally in
`tests/test_imaging_channel_lock.py`.
"""

import logging
import threading

import pytest

from fibsem.microscopes.simulator import CHAMBER_ACTIVE_VIEW, FM_ACTIVE_VIEW
from fibsem.structures import BeamType

STOLEN = "Imaging channel changed"


@pytest.fixture
def microscope():
    from fibsem import utils

    scope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    assert scope.fm is not None, "the simulated compustage system should have an FM"
    return scope


def steal_during(monkeypatch, thief) -> None:
    """Run `thief` inside the operation's simulated duration.

    The autofunctions sleep to emulate the routine, and that sleep is the window an
    unguarded FM read would land in -- the same hook `test_simulated_shared_channel.py`
    uses for the acquisition. Patched on the module, so it fires for whichever
    `sim_sleep` the operation under test reaches.
    """
    monkeypatch.setattr("fibsem.microscopes.simulator.sim_sleep", lambda seconds: thief())


class TestTheAutofunctionsHoldTheChannel:
    """The two writes. Each asserts the channel is theirs when the routine runs."""

    @pytest.mark.parametrize("operation", ["autocontrast", "auto_focus"])
    def test_it_claims_the_channel(self, microscope, operation):
        """Neither took the channel at all before FIB-569 -- the simulated versions did
        not touch it, and neither does the hardware one until it is fixed."""
        microscope.fm.set_active_channel()

        getattr(microscope, operation)(BeamType.ION)

        assert microscope.imaging_system.active_view == BeamType.ION.value

    @pytest.mark.parametrize("operation", ["autocontrast", "auto_focus"])
    def test_a_clean_run_says_nothing(self, microscope, operation, caplog):
        with caplog.at_level(logging.WARNING):
            getattr(microscope, operation)(BeamType.ION)

        assert STOLEN not in caplog.text

    @pytest.mark.parametrize("operation", ["autocontrast", "auto_focus"])
    def test_an_unguarded_fm_read_during_the_routine_is_reported(
        self, microscope, operation, monkeypatch, caplog
    ):
        """The FIB-517 shape: a getter takes the channel and walks away, mid-routine.

        On hardware the routine then tunes the FM's view instead of the beam's, and
        nothing downstream can tell -- which is why this needs to be caught here.
        """
        steal_during(monkeypatch, microscope.fm.set_active_channel)

        with caplog.at_level(logging.WARNING):
            getattr(microscope, operation)(BeamType.ION)

        assert STOLEN in caplog.text
        assert f"found view {FM_ACTIVE_VIEW}" in caplog.text

    def test_the_scoped_form_cannot_steal_during_the_routine(
        self, microscope, monkeypatch, caplog
    ):
        """An FM operation that uses `active_channel()` queues behind the routine
        instead of landing inside it, because both take the same lock.

        The thief is a real second thread, and the test asserts it was blocked *and*
        that it eventually ran, so a thief that never started cannot pass by doing
        nothing.
        """
        at_the_door = threading.Event()
        got_in = threading.Event()
        blocked = []

        def steal():
            at_the_door.wait(timeout=5)
            with microscope.fm.active_channel():
                got_in.set()

        thief = threading.Thread(target=steal, daemon=True)
        thief.start()

        def during_the_routine():
            at_the_door.set()
            blocked.append(not got_in.wait(timeout=0.5))

        steal_during(monkeypatch, during_the_routine)

        with caplog.at_level(logging.WARNING):
            microscope.auto_focus(BeamType.ION)

        thief.join(timeout=5)
        assert got_in.is_set(), "the thief never ran -- the test proved nothing"
        assert blocked == [True], "the scoped form got the channel mid-routine"
        assert STOLEN not in caplog.text


class TestLastImageClaimsTheChannel:
    def test_it_takes_the_channel_before_reading(self, microscope):
        """`get_image` reads the active view's buffer, so a `last_image` that does not
        set the channel first returns whoever owns it."""
        settings_beam = BeamType.ION
        microscope.acquire_image(beam_type=settings_beam)
        microscope.fm.set_active_channel()

        microscope.last_image(settings_beam)

        assert microscope.imaging_system.active_view == settings_beam.value


class TestTheChamberCameraGivesTheChannelBack:
    """FIB-545. Different failure: not a race, a channel taken and abandoned."""

    def test_it_takes_the_channel(self, microscope, monkeypatch):
        """The premise -- if it did not take the channel there would be nothing to give
        back, and the test below would pass for the wrong reason.

        Observed at the grab, not after the call: by the time it returns the channel is
        back where it started, which is the whole point.
        """
        seen = {}
        real_randint = __import__("numpy").random.randint

        def watching_randint(*args, **kwargs):
            seen["view"] = microscope.imaging_system.active_view
            return real_randint(*args, **kwargs)

        monkeypatch.setattr("numpy.random.randint", watching_randint)
        microscope.set_channel(BeamType.ELECTRON)

        microscope.acquire_chamber_image()

        assert seen["view"] == CHAMBER_ACTIVE_VIEW

    @pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
    def test_it_puts_the_channel_back(self, microscope, beam_type):
        """A glance at the chamber must not strand the beam that had the microscope."""
        microscope.set_channel(beam_type)

        microscope.acquire_chamber_image()

        assert microscope.imaging_system.active_view == beam_type.value
        assert microscope.imaging_system.active_device == beam_type.value

    def test_it_puts_it_back_even_when_the_grab_fails(self, microscope, monkeypatch):
        microscope.set_channel(BeamType.ION)

        def boom(*args, **kwargs):
            raise RuntimeError("chamber camera did not answer")

        monkeypatch.setattr("numpy.random.randint", boom)

        with pytest.raises(RuntimeError):
            microscope.acquire_chamber_image()

        assert microscope.imaging_system.active_view == BeamType.ION.value
