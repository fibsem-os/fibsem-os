"""A multi-step FM routine holds the channel for its duration.

The tileset runner already did this — `active_channel()`'s depth count exists for it:
"A tileset holds the channel for the whole run; each tile's acquisition opens a scope
inside it and must not restore the beam view between tiles." The autofocus sweep and the
z-stack never got the same treatment, so every step took the channel and handed it back.

Measured on the simulator before this change, starting from a beam view:

    autofocus sweep                 43 scopes, 43 hand-backs to the beam
    z-stack, 1 channel x 11 slices  23 scopes, 23 hand-backs
    z-stack, 2 channels x 11 slices 45 scopes, 45 hand-backs

Each hand-back is a restore on hardware and one visible flick of the active view in the
microscope's own UI — reported from the instrument as the view flicking while nothing
should have been touching it.

Note what this does *not* change: `active_channel()` calls `set_active_channel()` on
every entry regardless of depth, so the number of *sets* is unchanged. What goes away is
the restore between steps, and with it the window where another client can take the
channel mid-routine.
"""

import threading
from contextlib import contextmanager

import pytest

from fibsem.microscopes.simulator import FM_ACTIVE_VIEW

BEAM_VIEW = 1


@pytest.fixture
def microscope():
    from fibsem import utils

    scope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    assert scope.fm is not None
    scope.fm.objective.insert()
    return scope


@pytest.fixture
def channel(microscope):
    from fibsem.fm.structures import ChannelSettings

    return ChannelSettings(name="Channel-01")


@pytest.fixture
def zparams():
    from fibsem.fm.structures import ZParameters

    return ZParameters(zmin=-5e-6, zmax=5e-6, zstep=1e-6)


@contextmanager
def watching(microscope):
    """Record every moment the view returns to the beam while a routine runs."""
    imaging = microscope.imaging_system
    imaging.active_view = BEAM_VIEW
    fm_cls = type(microscope.fm)
    original = fm_cls.set_active_channel
    hand_backs = []
    started = []

    def counting(self):
        # On a beam view when a set is about to happen, after the routine has already
        # claimed it once, means it was given back in between.
        if started and imaging.active_view == BEAM_VIEW:
            hand_backs.append(1)
        started.append(1)
        original(self)

    fm_cls.set_active_channel = counting
    try:
        yield hand_backs
    finally:
        fm_cls.set_active_channel = original


class TestTheRoutineHoldsIt:
    def test_the_autofocus_sweep_does_not_hand_the_channel_back(
        self, microscope, channel
    ):
        from fibsem.fm.calibration import run_autofocus

        with watching(microscope) as hand_backs:
            run_autofocus(microscope.fm, channel)

        assert hand_backs == [], (
            f"the sweep gave the channel back {len(hand_backs)} times mid-routine — each "
            f"is a restore on hardware and a visible flick of the microscope's own view"
        )

    def test_the_z_stack_does_not_hand_the_channel_back(
        self, microscope, channel, zparams
    ):
        from fibsem.fm.acquisition import acquire_z_stack

        with watching(microscope) as hand_backs:
            acquire_z_stack(microscope.fm, [channel], zparams)

        assert hand_backs == []

    def test_a_multi_channel_z_stack_does_not_hand_the_channel_back(
        self, microscope, channel, zparams
    ):
        """Two channels doubles the steps, so it doubled the hand-backs."""
        from fibsem.fm.acquisition import acquire_z_stack
        from fibsem.fm.structures import ChannelSettings

        channels = [channel, ChannelSettings(name="Channel-02")]
        with watching(microscope) as hand_backs:
            acquire_z_stack(microscope.fm, channels, zparams)

        assert hand_backs == []


class TestItStillGivesItBackAtTheEnd:
    """Holding it for the routine must not mean keeping it afterwards — that is FIB-517."""

    def test_the_autofocus_sweep_restores_the_beam_view(self, microscope, channel):
        from fibsem.fm.calibration import run_autofocus

        microscope.imaging_system.active_view = BEAM_VIEW

        run_autofocus(microscope.fm, channel)

        assert microscope.imaging_system.active_view == BEAM_VIEW, (
            "the sweep left the connection on the FM — the next beam operation would "
            "read the FM's buffer (FIB-517)"
        )

    def test_the_z_stack_restores_the_beam_view(self, microscope, channel, zparams):
        from fibsem.fm.acquisition import acquire_z_stack

        microscope.imaging_system.active_view = BEAM_VIEW

        acquire_z_stack(microscope.fm, [channel], zparams)

        assert microscope.imaging_system.active_view == BEAM_VIEW

    def test_a_cancelled_sweep_still_restores(self, microscope, channel):
        """The `with` covers the early returns too, which a manual claim would miss."""
        from fibsem.fm.calibration import run_autofocus

        microscope.imaging_system.active_view = BEAM_VIEW
        stop = threading.Event()
        stop.set()

        run_autofocus(microscope.fm, channel, stop_event=stop)

        assert microscope.imaging_system.active_view == BEAM_VIEW


class TestTheClaimIsStructural:
    """The wrap is one line and easy to lose in a later edit of a long function."""

    @pytest.mark.parametrize(
        "module,function",
        [
            ("fm/calibration/__init__.py", "run_autofocus"),
            ("fm/acquisition.py", "acquire_z_stack"),
        ],
    )
    def test_the_routine_opens_a_channel_scope(self, module, function):
        import ast
        from pathlib import Path

        import fibsem

        tree = ast.parse(
            (Path(fibsem.__file__).parent / module).read_text(encoding="utf-8")
        )
        fn = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == function
        )
        scoped = any(
            isinstance(item.context_expr, ast.Call)
            and isinstance(item.context_expr.func, ast.Attribute)
            and item.context_expr.func.attr == "active_channel"
            for node in ast.walk(fn)
            if isinstance(node, ast.With)
            for item in node.items
        )
        assert scoped, (
            f"{function} no longer opens an `active_channel()` scope, so every step "
            f"inside it takes and returns the channel on its own again"
        )
