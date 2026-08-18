"""A channel scope that changes nothing takes no lock.

Found at the microscope: moving the objective while the FM streamed was unusable, and
focusing was close to impossible. The live stream holds the shared lock across every
frame in a loop with nothing between iterations, so it releases and immediately
re-acquires — and Python locks are not fair. Anything else wanting that lock is
*starved*, not queued.

Two observations rule out the queueing explanation, and both matter because a queueing
problem has a completely different fix:

* the exposure was 2 ms, so waiting on a frame cannot account for seconds of lag;
* halving the number of acquisitions per move (by disabling the objective's readback)
  changed nothing, where a queueing problem would have roughly halved.

The fix is to stop entering the race. When the connection is already on the FM there is
no view to set and therefore none to put back, so the scope does no work and takes no
lock. Safe against FIB-517 by construction: the fast path performs no writes, so it
cannot leave the channel anywhere the next beam operation does not expect.
"""

import threading
import time

import pytest

from fibsem.microscopes.simulator import FM_ACTIVE_VIEW


@pytest.fixture
def microscope():
    from fibsem import utils

    scope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    assert scope.fm is not None, "the simulated compustage system should have an FM"
    return scope


class TestTheFastPath:
    def test_a_scope_that_changes_nothing_does_not_wait_for_the_lock(self, microscope):
        """The property the whole change exists for.

        Another thread holds the shared lock for the duration. If the scope needed it,
        this blocks for the full hold; the fast path does not ask for it at all.
        """
        fm = microscope.fm
        microscope.imaging_system.active_view = FM_ACTIVE_VIEW  # the stream owns it

        holding = threading.Event()
        release = threading.Event()

        def hog():
            with microscope._threading_lock:
                holding.set()
                release.wait(timeout=5)

        t = threading.Thread(target=hog, daemon=True)
        t.start()
        assert holding.wait(timeout=5), "the hogging thread never took the lock"

        started = time.perf_counter()
        with fm.active_channel():
            pass
        elapsed = time.perf_counter() - started

        release.set()
        t.join(timeout=5)

        assert elapsed < 0.2, (
            f"entering the channel scope took {elapsed:.2f}s while another thread held "
            f"the lock — it is still contending, which is what starves the objective "
            f"against a live stream"
        )

    def test_an_objective_move_completes_against_a_stream_that_never_lets_go(
        self, microscope
    ):
        """The reported scenario, as close as the simulator gets to it.

        A loop that takes the lock, does a little work and immediately re-takes it —
        the shape of `_start_fast_acquisition`. Before the fast path an objective move
        could wait behind an unbounded number of iterations.
        """
        fm = microscope.fm
        microscope.imaging_system.active_view = FM_ACTIVE_VIEW
        stop = threading.Event()

        def stream():
            while not stop.is_set():
                with microscope._threading_lock:
                    microscope.imaging_system.active_view = FM_ACTIVE_VIEW
                    time.sleep(0.002)  # the 2 ms exposure, under the lock

        t = threading.Thread(target=stream, daemon=True)
        t.start()
        time.sleep(0.05)  # let it get into its stride

        started = time.perf_counter()
        for _ in range(20):
            fm.objective.move_relative(1e-6)
        elapsed = time.perf_counter() - started

        stop.set()
        t.join(timeout=5)

        assert elapsed < 1.0, (
            f"20 objective steps took {elapsed:.2f}s against a live stream — focusing "
            f"is a burst of steps, so this is what made it unusable"
        )

    def test_it_writes_nothing_when_the_channel_is_already_ours(self, microscope):
        """No writes is what makes skipping the lock safe: FIB-517 was a stray *write*."""
        fm = microscope.fm
        imaging = microscope.imaging_system
        imaging.active_view = FM_ACTIVE_VIEW
        imaging.active_device = fm._active_device

        before = (imaging.active_view, imaging.active_device)
        with fm.active_channel():
            during = (imaging.active_view, imaging.active_device)
        after = (imaging.active_view, imaging.active_device)

        assert before == during == after


class TestTheSlowPathIsUnchanged:
    def test_a_beam_view_is_still_taken_and_put_back(self, microscope):
        """The FIB-517 guarantee, which the fast path must not weaken."""
        fm = microscope.fm
        imaging = microscope.imaging_system
        imaging.active_view = 1  # a beam view — not ours

        with fm.active_channel():
            assert imaging.active_view == FM_ACTIVE_VIEW, "the scope did not take the channel"

        assert imaging.active_view == 1, "the scope did not put the beam view back"

    def test_nested_scopes_restore_once_at_the_outermost(self, microscope):
        """The tileset case: an inner acquisition must not restore between tiles."""
        fm = microscope.fm
        imaging = microscope.imaging_system
        imaging.active_view = 1

        with fm.active_channel():
            with fm.active_channel():
                assert imaging.active_view == FM_ACTIVE_VIEW
            assert imaging.active_view == FM_ACTIVE_VIEW, (
                "the inner scope restored the beam view mid-run"
            )

        assert imaging.active_view == 1

    def test_a_scope_entered_from_a_beam_view_still_serialises(self, microscope):
        """It still takes the lock when it has real work to do — only the no-op skips."""
        fm = microscope.fm
        microscope.imaging_system.active_view = 1  # not ours, so the slow path

        holding = threading.Event()
        release = threading.Event()
        entered = threading.Event()

        def hog():
            with microscope._threading_lock:
                holding.set()
                release.wait(timeout=5)

        def enter():
            with fm.active_channel():
                entered.set()

        t = threading.Thread(target=hog, daemon=True)
        t.start()
        assert holding.wait(timeout=5)

        e = threading.Thread(target=enter, daemon=True)
        e.start()
        blocked = not entered.wait(timeout=0.3)
        release.set()
        t.join(timeout=5)
        e.join(timeout=5)

        assert blocked, (
            "a scope that has to change the channel entered while another thread held "
            "the lock — the set-then-restore bookkeeping is no longer serialised"
        )


class TestBothDriversHaveIt:
    """Structural: `ThermoFisherFluorescenceMicroscope` needs an SDK absent off the
    microscope, and the hardware driver is the one that actually hurt."""

    def test_the_driver_and_the_simulator_both_short_circuit(self):
        import ast
        from pathlib import Path

        import fibsem

        root = Path(fibsem.__file__).parent
        for module, cls_name in (
            ("fm/autoscript.py", "ThermoFisherFluorescenceMicroscope"),
            ("microscopes/simulator.py", "SimulatedFluorescenceMicroscope"),
        ):
            tree = ast.parse((root / module).read_text(encoding="utf-8"))
            cls = next(
                n for n in ast.walk(tree)
                if isinstance(n, ast.ClassDef) and n.name == cls_name
            )
            names = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
            assert "_channel_is_ours" in names, f"{cls_name} lost its fast-path check"

            scope = next(
                n for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "active_channel"
            )
            calls = {
                c.func.attr for c in ast.walk(scope)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
            }
            assert "_channel_is_ours" in calls, (
                f"{cls_name}.active_channel no longer consults `_channel_is_ours`, so "
                f"every scope contends for the lock again and the objective starves "
                f"against a live stream"
            )
