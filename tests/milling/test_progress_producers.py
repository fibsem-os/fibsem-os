"""What the backends and strategies put on ``milling_progress_signal`` (FIB-797).

The three backend poll loops and the two built-in strategies now emit a
``MillingProgress`` rather than a nested dict. The task layer still emits dicts; the
consumers decode either, which is what allows the producers to be flipped in two steps.

The guard at the bottom is the part worth keeping after the migration. On FIB-402 the
equivalent test was keyed on the signal name alone, and two emit sites hiding behind a
relay method slipped through it -- so this one is keyed on the payload's *shape* at every
emit site, and it names the files that have not been flipped yet rather than ignoring
them.
"""

import ast
import time
from pathlib import Path

import pytest

import fibsem
from fibsem import utils
from fibsem.milling import FibsemMillingStage
from fibsem.milling.progress import MillingProgress, MillingStatus
from fibsem.milling.strategy.coincidence import (
    CoincidenceMillingStrategy,
    CoincidenceMillingStrategyConfig,
)
from fibsem.milling.strategy.standard import StandardMillingStrategy
from fibsem.structures import MillingState


@pytest.fixture
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo")
    return scope


def _collect(scope):
    """Subscribe to the signal and return the list the reports land in."""
    emitted = []
    scope.milling_progress_signal.connect(emitted.append)
    return emitted


class TestADelegatingStrategy:
    """`StandardMillingStrategy` emits its label once and hands the loop to the
    backend's `run_milling`, which knows nothing about the strategy."""

    def test_every_report_is_typed(self, microscope):
        emitted = _collect(microscope)
        StandardMillingStrategy().run(microscope, FibsemMillingStage(name="Rough Mill"))

        assert emitted, "the strategy emitted nothing at all"
        assert all(isinstance(report, MillingProgress) for report in emitted), [
            type(report).__name__ for report in emitted
        ]

    def test_the_strategy_says_what_it_is_doing(self, microscope):
        """The feature this signal keeps a `message` for. Before this change the
        strategy's report carried no `state`, matched no consumer branch, and its words
        reached no screen."""
        emitted = _collect(microscope)
        StandardMillingStrategy().run(microscope, FibsemMillingStage(name="Rough Mill"))

        first = emitted[0]
        assert first.message == "Running Rough Mill..."
        assert first.stage_name == "Rough Mill"
        # A status a consumer actually renders, which is what makes the message visible.
        assert first.status is MillingStatus.STAGE_UPDATE

    def test_the_backends_ticks_carry_the_countdown_and_the_instrument_state(
        self, microscope
    ):
        emitted = _collect(microscope)
        StandardMillingStrategy().run(microscope, FibsemMillingStage(name="Rough Mill"))

        ticks = [r for r in emitted if r.remaining_time is not None]
        assert ticks, "the backend poll loop reported no countdown"
        assert all(t.status is MillingStatus.STAGE_UPDATE for t in ticks)
        # Carried on the report so a consumer does not have to poll for it -- and on
        # ThermoFisher that poll sets the active view as a side effect.
        assert all(isinstance(t.milling_state, MillingState) for t in ticks)

    def test_the_backend_does_not_re_say_what_the_strategy_said(self, microscope):
        """A backend has no idea what the strategy calls itself, so it says nothing and
        the consumer keeps the last words standing. The alternative -- threading the
        message through `run_milling` -- changes three backend signatures and still
        leaves the backend not owning the words."""
        emitted = _collect(microscope)
        StandardMillingStrategy().run(microscope, FibsemMillingStage(name="Rough Mill"))

        ticks = [r for r in emitted if r.remaining_time is not None]
        assert all(t.message is None for t in ticks)


class TestTheCoincidenceStrategy:
    """Self-driving: it runs its own monitor loop rather than delegating to
    `run_milling`, which is the whole point of the strategy abstraction."""

    def _monitor_once(self, microscope):
        strategy = CoincidenceMillingStrategy()
        strategy.microscope = microscope
        strategy.stage = FibsemMillingStage(name="Coincidence Mill")
        strategy.parent_ui = None
        strategy.config = CoincidenceMillingStrategyConfig()
        strategy._drop_detected = False
        emitted = _collect(microscope)
        # An estimate of zero puts the timeout in the past, so the loop emits exactly
        # once and breaks. It still sleeps its one real second first.
        strategy._monitor_milling_progress(estimated_time=0.0)
        return emitted

    def test_it_declines_to_read_the_milling_state(self, microscope):
        """Not sloppiness, and not something to "fix" by calling the getter like
        everyone else. On ThermoFisher `get_milling_state()` sets the active view, and
        this strategy is running a fluorescence acquisition that holds the view for its
        whole duration -- asking would yank it away mid-acquisition."""
        calls = []
        original = microscope.get_milling_state
        microscope.get_milling_state = lambda *a, **k: (
            calls.append(1) or original(*a, **k)
        )
        try:
            emitted = self._monitor_once(microscope)
        finally:
            microscope.get_milling_state = original

        assert emitted
        assert calls == [], "the monitor loop polled the milling state"
        assert all(r.milling_state is MillingState.UNKNOWN for r in emitted)

    def test_the_gap_is_an_enum_member_not_a_string(self, microscope):
        """Every other producer sends a `MillingState`. This one used to send the string
        `"UNKNOWN"`, so a consumer had to pattern-match a magic value to tell the two
        apart."""
        emitted = self._monitor_once(microscope)
        assert emitted[0].milling_state is MillingState.UNKNOWN
        assert not isinstance(emitted[0].milling_state, str)

    def test_a_self_driving_strategy_stamps_every_report(self, microscope):
        """Unlike a delegating one, nothing else emits on its behalf, so it can carry
        its own words on every tick and needs no stickiness."""
        emitted = self._monitor_once(microscope)
        assert all(
            r.message == "Coincidence milling: Coincidence Mill" for r in emitted
        )
        assert all(r.status is MillingStatus.STAGE_UPDATE for r in emitted)


# --------------------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------------------

# Every module that emits on `milling_progress_signal`, and whether it has been flipped.
# `tasks.py` is False on purpose: its two emit sites go in the next PR, along with the
# relay they hide behind. Naming it rather than omitting it is what makes forgetting it
# a failing test rather than a silently narrower scan.
EMITTERS = {
    "microscope.py": True,
    "microscopes/simulator.py": True,
    "microscopes/tescan.py": True,
    "milling/strategy/standard.py": True,
    "milling/strategy/coincidence.py": True,
    "milling/tasks.py": False,
}


# Names that put a payload on the signal. `_handle_progress` is `MillingTask`'s relay --
# a pass-through that emits whatever it is handed -- and it is why this scan cannot be
# keyed on the signal name alone.
#
# This is not hypothetical. Written the obvious way, keyed on
# `milling_progress_signal.emit`, this scan reported `milling/tasks.py` as already
# flipped while both of its emit sites were still building dicts, because neither
# mentions the signal. That is precisely the blind spot that let two `_stitch` emits
# through the equivalent guard on FIB-402.
_PAYLOAD_SINKS = ("milling_progress_signal", "_handle_progress")


def _dict_emits(path: Path):
    """Every line in *path* that hands a dict literal to the milling progress signal,
    directly or through a relay."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (node.args and isinstance(node.args[0], ast.Dict)):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        # `<...>.milling_progress_signal.emit({...})`
        direct = (
            func.attr == "emit"
            and isinstance(func.value, ast.Attribute)
            and (func.value.attr in _PAYLOAD_SINKS)
        )
        # `self._handle_progress({...})`
        via_relay = func.attr in _PAYLOAD_SINKS
        if direct or via_relay:
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("relative,flipped", sorted(EMITTERS.items()))
def test_the_flipped_producers_no_longer_emit_dicts(relative, flipped):
    path = Path(fibsem.__file__).parent / relative
    assert path.exists(), f"{relative} moved; update EMITTERS"
    hits = _dict_emits(path)
    if flipped:
        assert hits == [], (
            f"{relative} still emits a dict literal on milling_progress_signal at "
            f"line(s) {hits}"
        )
    else:
        assert hits, (
            f"{relative} no longer emits any dict, so it has been flipped -- set its "
            "entry in EMITTERS to True."
        )


def test_the_scan_sees_through_the_relay(tmp_path):
    """Shown rather than asserted, because the obvious version of this scan does not.

    Keyed on `milling_progress_signal.emit` alone it reads line 3 and misses line 5 --
    which is exactly what happened when this file was first written, and what let two
    `_stitch` emits through the equivalent guard on FIB-402.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "class X:\n"
        "    def direct(self):\n"
        "        self.microscope.milling_progress_signal.emit({'progress': {}})\n"
        "    def relayed(self):\n"
        "        self._handle_progress({'progress': {}})\n"
        "    def typed(self):\n"
        "        self._handle_progress(MillingProgress(MillingStatus.TASK_FINISHED))\n",
        encoding="utf-8",
    )
    assert _dict_emits(probe) == [3, 5]


def test_a_stage_run_end_to_end_emits_no_dict(microscope):
    """The static scan cannot see a dict built somewhere else and passed in, so this
    watches a real run."""
    emitted = _collect(microscope)
    start = time.time()
    StandardMillingStrategy().run(microscope, FibsemMillingStage(name="Rough Mill"))
    assert time.time() - start < 60, "the simulated mill did not skip its delays"

    dicts = [r for r in emitted if isinstance(r, dict)]
    assert dicts == [], f"{len(dicts)} dict payload(s) still reached the signal"
