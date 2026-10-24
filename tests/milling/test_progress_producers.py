"""What the backends and strategies report on ``microscope.milling.progress`` (FIB-797).

Every producer reports a ``MillingProgress``: the backends' run loop, both built-in
strategies, and the task layer. All of them report on the milling service's
``progress`` parameter, and the signal sends on what it reports.

The guard at the bottom keeps it that way: nothing emits on a progress signal itself.
"""

import threading
import time

from fibsem.milling import FibsemMillingStage
from fibsem.milling.progress import MillingProgress, MillingProgressStatus
from fibsem.milling.strategy.coincidence import (
    CoincidenceMillingStrategy,
    CoincidenceMillingStrategyConfig,
)
from fibsem.milling.strategy.standard import StandardMillingStrategy
from fibsem.structures.settings import MillingState
from tests.fixtures.microscopes import demo_microscope

microscope = demo_microscope(scope="function")


def _collect(scope):
    """Listen on milling progress and return the list the reports land in."""
    emitted = []
    scope.milling.progress.changed.connect(emitted.append)
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
        assert first.status is MillingProgressStatus.STAGE_UPDATE

    def test_the_backends_ticks_carry_the_countdown_and_the_instrument_state(
        self, microscope
    ):
        emitted = _collect(microscope)
        StandardMillingStrategy().run(microscope, FibsemMillingStage(name="Rough Mill"))

        ticks = [r for r in emitted if r.remaining_time is not None]
        assert ticks, "the backend poll loop reported no countdown"
        assert all(t.status is MillingProgressStatus.STAGE_UPDATE for t in ticks)
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
        state = microscope.milling.state
        original = state.get_value
        state.get_value = lambda *a, **k: calls.append(1) or original(*a, **k)
        try:
            emitted = self._monitor_once(microscope)
        finally:
            del state.get_value

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
        assert all(r.status is MillingProgressStatus.STAGE_UPDATE for r in emitted)


# --------------------------------------------------------------------------------------
# The guard: one source
# --------------------------------------------------------------------------------------

# Milling and spot burn progress has one source: the services' `progress`
# parameters, which the device API and the agent server read too. The microscope
# has no progress signal of its own to emit on beside them.


def test_the_microscope_has_no_progress_signals_of_its_own(microscope):
    assert not hasattr(microscope, "milling_progress_signal")
    assert not hasattr(microscope, "spot_burn_progress_signal")


def test_a_strategy_report_is_the_service_progress(microscope):
    """What a strategy reports is what `microscope.milling.progress` reads."""
    emitted = _collect(microscope)
    StandardMillingStrategy().run(microscope, FibsemMillingStage(name="Rough Mill"))
    assert emitted[0].message == "Running Rough Mill..."
    assert microscope.milling.progress.get_value() == emitted[-1]


def test_without_milling_a_report_goes_nowhere(microscope):
    from fibsem.services.milling import report_progress

    emitted = _collect(microscope)
    microscope.milling = None
    report_progress(microscope, MillingProgress(MillingProgressStatus.TASK_FINISHED))
    assert emitted == []


def test_a_stage_run_end_to_end_emits_no_dict(microscope):
    """The static scan cannot see a dict built somewhere else and passed in, so this
    watches a real run."""
    emitted = _collect(microscope)
    start = time.time()
    StandardMillingStrategy().run(microscope, FibsemMillingStage(name="Rough Mill"))
    assert time.time() - start < 60, "the simulated mill did not skip its delays"

    dicts = [r for r in emitted if isinstance(r, dict)]
    assert dicts == [], f"{len(dicts)} dict payload(s) still reached the signal"


# --------------------------------------------------------------------------------------
# The task layer, and the defect it carried
# --------------------------------------------------------------------------------------


def _task(microscope, stages, name="Trench"):
    from fibsem.milling.tasks import FibsemMillingTask, FibsemMillingTaskConfig

    config = FibsemMillingTaskConfig(name=name, stages=stages)
    return FibsemMillingTask(microscope=microscope, config=config)


def _stage(name):
    stage = FibsemMillingStage(name=name)
    stage.strategy = StandardMillingStrategy()
    return stage


class TestHowATaskEnds:
    """Before FIB-797 both `except` blocks logged and fell through to a `finally` that
    emitted `finished` regardless, so a mill the user cancelled and a mill that crashed
    both told the UI they had finished. The status bar rendered "Done" either way and the
    exception text reached only the logfile."""

    def test_a_completed_task_says_so(self, microscope):
        task = _task(microscope, [_stage("Rough Mill")])
        emitted = _collect(microscope)
        task.run()

        assert emitted[-1].status is MillingProgressStatus.TASK_FINISHED
        assert emitted[-1].error is None

    def test_a_cancelled_task_does_not_claim_to_have_finished(self, microscope):
        task = _task(microscope, [_stage("Rough Mill")])
        task._stop_event = threading.Event()
        task._stop_event.set()
        emitted = _collect(microscope)
        task.run()

        assert emitted[-1].status is MillingProgressStatus.TASK_CANCELLED

    def test_a_cancelled_task_is_not_painted_as_a_failure(self, microscope):
        """A cancel is someone getting what they asked for, so it is a distinct status
        rather than an error -- nothing here should render red."""
        task = _task(microscope, [_stage("Rough Mill")])
        task._stop_event = threading.Event()
        task._stop_event.set()
        emitted = _collect(microscope)
        task.run()

        assert emitted[-1].status is not MillingProgressStatus.TASK_FAILED
        assert emitted[-1].error is None

    def test_a_failed_task_carries_why(self, microscope):
        task = _task(microscope, [_stage("Rough Mill")])

        def boom(*a, **k):
            raise RuntimeError("the column tripped")

        task._configure_path = boom
        emitted = _collect(microscope)
        task.run()

        assert emitted[-1].status is MillingProgressStatus.TASK_FAILED
        assert emitted[-1].error == "the column tripped"

    def test_the_terminal_is_the_last_thing_the_task_says(self, microscope):
        """Whatever the outcome, exactly one terminal report and nothing after it -- or a
        consumer that hides its bar on `is_terminal` shows it again for the rest of the
        session."""
        task = _task(microscope, [_stage("Rough Mill"), _stage("Polish")])
        emitted = _collect(microscope)
        task.run()

        terminals = [i for i, r in enumerate(emitted) if r.status.is_terminal]
        assert len(terminals) == 1
        assert terminals[0] == len(emitted) - 1


class TestTheTwoScales:
    def test_a_stage_starting_is_not_the_task_starting(self, microscope):
        """The old vocabulary emitted `start` per stage, N times, and `finished` once per
        task. They read like a matched pair and were not one."""
        task = _task(microscope, [_stage("Rough Mill"), _stage("Polish")])
        emitted = _collect(microscope)
        task.run()

        assert [r.status for r in emitted].count(
            MillingProgressStatus.TASK_STARTED
        ) == 1
        assert [r.status for r in emitted].count(
            MillingProgressStatus.STAGE_STARTED
        ) == 2

    def test_each_stage_reports_its_own_index_zero_based(self, microscope):
        task = _task(microscope, [_stage("Rough Mill"), _stage("Polish")])
        emitted = _collect(microscope)
        task.run()

        starts = [r for r in emitted if r.status is MillingProgressStatus.STAGE_STARTED]
        assert [r.current_stage for r in starts] == [0, 1]
        assert [r.display_stage for r in starts] == [1, 2]
        assert [r.stage_name for r in starts] == ["Rough Mill", "Polish"]
        assert all(r.total_stages == 2 for r in starts)

    def test_a_stage_finishing_is_not_terminal(self, microscope):
        task = _task(microscope, [_stage("Rough Mill"), _stage("Polish")])
        emitted = _collect(microscope)
        task.run()

        finishes = [
            r for r in emitted if r.status is MillingProgressStatus.STAGE_FINISHED
        ]
        assert len(finishes) == 2
        assert not any(r.status.is_terminal for r in finishes)

    def test_every_report_carries_the_task_identity(self, microscope):
        """What the `_emit` helper buys over the pass-through relay it replaces: the
        stamp lives in one place instead of being repeated at each call site."""
        task = _task(microscope, [_stage("Rough Mill")], name="Trench")
        emitted = _collect(microscope)
        task.run()

        from_task = [r for r in emitted if r.milling_run_id is not None]
        assert from_task, "no report carried a milling run id"
        assert all(r.milling_task_name == "Trench" for r in from_task)
        assert len({r.milling_run_id for r in from_task}) == 1

    def test_a_stage_update_says_which_run_and_stage(self, microscope):
        """The backend's ticks name neither; the milling service gives them the run
        and stage the last stage start named (FIB-1356)."""
        task = _task(microscope, [_stage("Rough Mill")], name="Trench")
        emitted = _collect(microscope)
        task.run()

        updates = [r for r in emitted if r.status is MillingProgressStatus.STAGE_UPDATE]
        assert updates, "no stage update was reported"
        for update in updates:
            assert update.milling_run_id == task.task_id
            assert update.milling_task_name == "Trench"
            assert update.stage_name == "Rough Mill"
            assert update.current_stage == 0
            assert update.total_stages == 1


def test_the_relay_no_longer_exists():
    """`_handle_progress` was a pass-through that emitted whatever it was handed,
    handled nothing, and described the traffic backwards. Deleting it is only safe
    because the guard above now follows `_emit` instead."""
    from fibsem.milling.tasks import FibsemMillingTask

    assert not hasattr(FibsemMillingTask, "_handle_progress")
