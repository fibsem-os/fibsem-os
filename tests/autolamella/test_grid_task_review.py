"""Grid tasks propose their result and are decided, the way lamella tasks are.

Real overview tasks on the Arctis simulator, run through a real GridTaskManager:
every run records a task_result proposal on the grid pointing at the stitched
overview; automated, the task confirms its own; under review (the protocol says
so and the preference is on), the task ends AwaitingDecision and a decision
finishes it.
"""

import os
import threading
import time
from pathlib import Path

import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.proposals import (
    TASK_RESULT,
    AuthorKind,
    Decision,
    DecisionOutcome,
)
from fibsem.applications.autolamella.structures import (
    Attention,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
    GridTaskProtocol,
    Lamella,
)
from fibsem.applications.autolamella.task_outputs import latest_grid_output
from fibsem.applications.autolamella.workflows.tasks.grid import (
    BeamOverviewGridTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME,
    GridTaskManager,
)
from fibsem.structures import BeamType, ImageSettings, OverviewAcquisitionSettings

GRID = "Grid-01"
OTHER_GRID = "Grid-02"
OVERVIEW = "overview_sem"
LATER = "overview_fib"


class _Signal:
    def emit(self, payload) -> None:
        pass


class _UI:
    """What the manager emits status on, in place of AutoLamellaUI."""

    def __init__(self):
        self.workflow_status_signal = _Signal()
        self._task_manager = None


def _small_settings() -> OverviewAcquisitionSettings:
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(
            resolution=(128, 128), hfw=200e-6, beam_type=BeamType.ELECTRON
        ),
        nrows=1,
        ncols=1,
    )


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    return microscope


@pytest.fixture
def experiment(tmp_path, microscope):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.grid_protocol.add(
        BeamOverviewGridTaskConfig(task_name=OVERVIEW, settings=_small_settings())
    )
    exp.sync_grids_from_inventory(microscope._stage)
    return exp


def _run(microscope, experiment, review_enabled: bool) -> GridTaskManager:
    ui = _UI()
    manager = GridTaskManager(microscope, experiment, parent_ui=ui)
    ui._task_manager = manager
    manager.review_enabled = review_enabled  # the preference, as read for a run
    manager.run([OVERVIEW], [GRID])
    return manager


# ---------------------------------------------------------------------------
# The config
# ---------------------------------------------------------------------------


class TestAttentionOnTheConfig:
    def test_defaults_to_automated_and_is_not_a_form_parameter(self):
        config = BeamOverviewGridTaskConfig(task_name=OVERVIEW)
        assert config.attention is Attention.automated
        assert "attention" not in config.parameters

    def test_round_trips_through_the_protocol(self):
        protocol = GridTaskProtocol()
        protocol.add(
            BeamOverviewGridTaskConfig(task_name=OVERVIEW, attention=Attention.review)
        )
        data = yaml.safe_load(yaml.safe_dump(protocol.to_dict()))
        assert data["tasks"][OVERVIEW]["attention"] == "review"
        again = GridTaskProtocol.from_dict(data)
        assert again.task_config[OVERVIEW].attention is Attention.review

    def test_an_unknown_value_reads_as_automated_and_keeps_the_task(self):
        data = BeamOverviewGridTaskConfig(task_name=OVERVIEW).to_dict()
        data["attention"] = "sometimes"
        again = GridTaskProtocol.from_dict({"tasks": {OVERVIEW: data}})
        assert again.task_config[OVERVIEW].attention is Attention.automated

    def test_requires_round_trips_and_is_not_a_form_parameter(self):
        protocol = GridTaskProtocol()
        protocol.add(BeamOverviewGridTaskConfig(task_name=OVERVIEW))
        protocol.add(BeamOverviewGridTaskConfig(task_name=LATER, requires=[OVERVIEW]))
        assert "requires" not in protocol.task_config[LATER].parameters
        again = GridTaskProtocol.from_dict(
            yaml.safe_load(yaml.safe_dump(protocol.to_dict()))
        )
        assert again.requirements(LATER) == [OVERVIEW]
        assert again.requirements(OVERVIEW) == []
        assert again.requirements("not a task") == []

    def test_a_malformed_requires_reads_as_none_and_keeps_the_task(self):
        data = BeamOverviewGridTaskConfig(task_name=LATER).to_dict()
        data["requires"] = OVERVIEW  # a string, not a list
        again = GridTaskProtocol.from_dict({"tasks": {LATER: data}})
        assert again.requirements(LATER) == []

    def test_a_protocol_written_before_attention_loads_as_automated(self):
        data = BeamOverviewGridTaskConfig(task_name=OVERVIEW).to_dict()
        del data["attention"]
        again = GridTaskProtocol.from_dict({"tasks": {OVERVIEW: data}})
        assert again.task_config[OVERVIEW].attention is Attention.automated


# ---------------------------------------------------------------------------
# A run proposes
# ---------------------------------------------------------------------------


class TestAGridTaskProposes:
    def test_automated_confirms_its_own_result(self, microscope, experiment):
        _run(microscope, experiment, review_enabled=True)
        grid = experiment.get_grid_by_name(GRID)

        assert grid.task_history[-1].status is AutoLamellaTaskStatus.Completed
        proposal = grid.proposals[OVERVIEW]
        assert proposal.kind == TASK_RESULT
        assert not proposal.pending
        assert proposal.decisions[-1].author.kind is AuthorKind.automated

    def test_the_proposal_points_at_the_stitched_overview_not_the_thumbnail(
        self, microscope, experiment
    ):
        _run(microscope, experiment, review_enabled=False)
        grid = experiment.get_grid_by_name(GRID)

        reference = grid.proposals[OVERVIEW].provenance["reference_image"]
        overview = latest_grid_output(experiment, grid, OVERVIEW)
        assert reference.endswith(".tif") and "thumbnail" not in reference
        path = Path(experiment.item_path(grid)) / reference
        # the record keeps a forward slash on every platform; compare the files
        assert path.exists() and os.path.normpath(path) == os.path.normpath(overview)

    def test_under_review_the_task_waits_and_a_decision_finishes_it(
        self, microscope, experiment
    ):
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.review
        _run(microscope, experiment, review_enabled=True)
        grid = experiment.get_grid_by_name(GRID)

        assert grid.is_awaiting_decision(OVERVIEW)
        assert grid.task_state.status is AutoLamellaTaskStatus.AwaitingDecision
        assert grid.proposals[OVERVIEW].pending
        assert not grid.has_completed_task(OVERVIEW)
        assert grid.proposals[OVERVIEW].task_id == grid.task_history[-1].task_id

        result = experiment.decide(
            grid.id,
            OVERVIEW,
            Decision(
                outcome=DecisionOutcome.Confirmed,
                author="human:op",
                task_id=grid.proposals[OVERVIEW].task_id,
            ),
        )
        assert result.applied is True
        assert grid.has_completed_task(OVERVIEW)

    def test_review_in_the_protocol_runs_automated_while_the_preference_is_off(
        self, microscope, experiment
    ):
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.review
        _run(microscope, experiment, review_enabled=False)
        grid = experiment.get_grid_by_name(GRID)

        assert grid.task_history[-1].status is AutoLamellaTaskStatus.Completed
        assert not grid.proposals[OVERVIEW].pending

    def test_a_failed_task_proposes_its_failure_and_stays_failed(
        self, microscope, experiment
    ):
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.review
        # an orientation the stage does not have: a real failure, inside the
        # task and before any beam
        experiment.grid_protocol.task_config[OVERVIEW].orientation = "NOWHERE"
        grid = experiment.get_grid_by_name(GRID)

        _run(microscope, experiment, review_enabled=True)

        assert grid.task_history[-1].status is AutoLamellaTaskStatus.Failed
        proposal = grid.proposals[OVERVIEW]
        assert proposal.provenance["failure"]
        assert proposal.provenance["reference_image"] == ""
        assert not grid.is_awaiting_decision(OVERVIEW), "a failure does not wait"


def test_item_path_is_a_lamellas_own_and_a_grids_derived(tmp_path, experiment):
    grid = experiment.get_grid_by_name(GRID)
    assert experiment.item_path(grid) == experiment.grid_path(grid)
    lamella = Lamella(path=tmp_path / "lam-01", number=1, petname="lam-01")
    assert experiment.item_path(lamella) == Path(lamella.path)


# ---------------------------------------------------------------------------
# The run waits on a decision (FIB-1002 PR 4)
# ---------------------------------------------------------------------------


def _with_later_task(experiment, review_wait, requires=(OVERVIEW,)):
    """The SEM overview under review, then an automated FIB overview that
    requires it (a stand-in for a task that uses the overview)."""
    experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.review
    experiment.grid_protocol.add(
        BeamOverviewGridTaskConfig(
            task_name=LATER,
            requires=list(requires),
            orientation="FIB",
            settings=OverviewAcquisitionSettings(
                image_settings=ImageSettings(
                    resolution=(128, 128), hfw=200e-6, beam_type=BeamType.ION
                ),
                nrows=1,
                ncols=1,
            ),
        )
    )
    experiment.task_protocol.options.review_wait = review_wait


def _manager(microscope, experiment, hook_manager=None) -> GridTaskManager:
    manager = GridTaskManager(microscope, experiment, hook_manager=hook_manager)
    manager.review_enabled = True
    return manager


def _decide_when(experiment, grid_name, ready, outcome, reason=""):
    """Decide the SEM overview on a thread once ``ready()`` holds. Through
    _decide, the unmarshalled inner: the wake-up is the subject here, the main
    thread hop is covered in tests/ui/test_decide_main_thread.py."""

    def run():
        deadline = time.monotonic() + 30
        while not ready() and time.monotonic() < deadline:
            time.sleep(0.05)
        grid = experiment.get_grid_by_name(grid_name)
        experiment._decide(
            grid.id,
            OVERVIEW,
            Decision(
                outcome=outcome,
                author="human:op",
                reason=reason,
                task_id=grid.proposals[OVERVIEW].task_id,
            ),
        )

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def _ran(grid, task_name):
    return [t for t in grid.task_history if t.name == task_name]


class TestTheGridRunWaitsOnADecision:
    def test_a_later_task_waits_and_the_run_stalls_when_nobody_decides(
        self, microscope, experiment
    ):
        from fibsem.hooks import FunctionHook, HookEvent, HookManager

        _with_later_task(experiment, review_wait=0)
        fired = []
        hooks = HookManager()
        hooks.register(
            FunctionHook(name="rec", events=list(HookEvent), callback=fired.append)
        )
        manager = _manager(microscope, experiment, hook_manager=hooks)

        manager.run([OVERVIEW, LATER], [GRID])

        grid = experiment.get_grid_by_name(GRID)
        assert grid.is_awaiting_decision(OVERVIEW)
        assert _ran(grid, LATER) == [], "the later task did not run"
        assert manager.queue.has_pending_pair(GRID, LATER), "deferred, not retired"
        assert manager.stalled is True
        assert "Decide in the Review tab, then Run again." in manager.closing_note()
        assert fired[-1].event == "workflow_stalled"
        assert fired[-1].decisions_pending == 1

    def test_a_confirm_wakes_the_run_and_the_later_task_runs(
        self, microscope, experiment
    ):
        _with_later_task(experiment, review_wait=30.0)
        manager = _manager(microscope, experiment)
        grid = experiment.get_grid_by_name(GRID)
        thread = _decide_when(
            experiment,
            GRID,
            lambda: grid.is_awaiting_decision(OVERVIEW) and manager.deferred_items(),
            DecisionOutcome.Confirmed,
        )
        started = time.monotonic()

        manager.run([OVERVIEW, LATER], [GRID])
        thread.join(5)

        assert time.monotonic() - started < 25, "woke on the decision, not the timeout"
        assert manager.stalled is False
        assert grid.has_completed_task(OVERVIEW)
        assert [t.status for t in _ran(grid, LATER)] == [
            AutoLamellaTaskStatus.Completed
        ]
        assert manager.closing_note() == ""

    def test_a_reject_fails_the_task_and_skips_what_requires_it(
        self, microscope, experiment
    ):
        from fibsem.hooks import FunctionHook, HookEvent, HookManager

        _with_later_task(experiment, review_wait=30.0)
        fired = []
        hooks = HookManager()
        hooks.register(
            FunctionHook(name="rec", events=list(HookEvent), callback=fired.append)
        )
        manager = _manager(microscope, experiment, hook_manager=hooks)
        grid = experiment.get_grid_by_name(GRID)
        thread = _decide_when(
            experiment,
            GRID,
            lambda: grid.is_awaiting_decision(OVERVIEW) and manager.deferred_items(),
            DecisionOutcome.Rejected,
            reason="all ice",
        )

        manager.run([OVERVIEW, LATER], [GRID])
        thread.join(5)

        assert _ran(grid, OVERVIEW)[-1].status is AutoLamellaTaskStatus.Failed
        assert "all ice" in _ran(grid, OVERVIEW)[-1].status_message
        assert _ran(grid, LATER) == [], "skipped, never run"
        (later,) = [i for i in manager.queue.items if i.task_name == LATER]
        assert later.status is AutoLamellaTaskStatus.Skipped
        skipped = [c for c in fired if c.event == "task_skipped"]
        assert [(c.task_name, c.skip_reason) for c in skipped] == [
            (LATER, "missing_prereqs")
        ]
        assert not manager.stalled

    def test_the_run_moves_on_to_the_next_grid_while_one_waits(
        self, microscope, experiment
    ):
        """Waiting on Grid-01 does not hold the beam: Grid-02 is loaded and its
        overview taken. Each decision then releases its own grid's later task,
        Grid-01's with an exchange back to it."""
        _with_later_task(experiment, review_wait=60.0)
        manager = _manager(microscope, experiment)
        first = experiment.get_grid_by_name(GRID)
        second = experiment.get_grid_by_name(OTHER_GRID)
        both_waiting = lambda: (  # noqa: E731
            first.is_awaiting_decision(OVERVIEW)
            and second.is_awaiting_decision(OVERVIEW)
        )
        decide_first = _decide_when(
            experiment, GRID, both_waiting, DecisionOutcome.Confirmed
        )
        decide_second = _decide_when(
            experiment,
            OTHER_GRID,
            lambda: bool(_ran(first, LATER)),
            DecisionOutcome.Confirmed,
        )

        manager.run([OVERVIEW, LATER], [GRID, OTHER_GRID])
        decide_first.join(5)
        decide_second.join(5)

        assert not manager.stalled
        assert first.has_completed_task(LATER) and second.has_completed_task(LATER)
        assert _ran(second, OVERVIEW)[-1].start_timestamp < (
            _ran(first, LATER)[-1].start_timestamp
        ), "Grid-02 was run while Grid-01 waited"
        assert len(_ran(first, LOAD_ENTRY_NAME)) == 2, "and Grid-01 was loaded again"

    def test_run_leaves_out_a_task_already_awaiting_a_decision(
        self, microscope, experiment
    ):
        _with_later_task(experiment, review_wait=0)
        _manager(microscope, experiment).run([OVERVIEW], [GRID])
        grid = experiment.get_grid_by_name(GRID)
        assert grid.is_awaiting_decision(OVERVIEW)
        runs = len(_ran(grid, OVERVIEW))

        again = _manager(microscope, experiment)
        again.run([OVERVIEW], [GRID])

        assert len(_ran(grid, OVERVIEW)) == runs, "not re-run over the pending look"
        assert not again.queue.has_pending_pair(GRID, OVERVIEW)
        assert grid.proposals[OVERVIEW].pending

    def test_review_on_a_task_nothing_requires_holds_nothing(
        self, microscope, experiment
    ):
        """The overview waits in the Review tab to be looked at; the run goes on."""
        _with_later_task(experiment, review_wait=0, requires=())
        manager = _manager(microscope, experiment)

        manager.run([OVERVIEW, LATER], [GRID])

        grid = experiment.get_grid_by_name(GRID)
        assert grid.is_awaiting_decision(OVERVIEW)
        assert grid.has_completed_task(LATER)
        assert not manager.stalled and manager.deferred_items() == []

    def test_a_requirement_queued_later_runs_first(self, microscope, experiment):
        """Selected in the other order, the task waits for its requirement's
        turn (prereq_pending) instead of skipping over it."""
        _with_later_task(experiment, review_wait=0)
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.automated
        manager = _manager(microscope, experiment)

        manager.run([LATER, OVERVIEW], [GRID])

        grid = experiment.get_grid_by_name(GRID)
        assert grid.has_completed_task(OVERVIEW) and grid.has_completed_task(LATER)
        assert _ran(grid, OVERVIEW)[-1].end_timestamp <= (
            _ran(grid, LATER)[-1].start_timestamp
        )

    def test_a_failed_requirement_skips_the_task_without_waiting(
        self, microscope, experiment
    ):
        _with_later_task(experiment, review_wait=0)
        # an orientation the stage does not have: the overview fails in the task
        experiment.grid_protocol.task_config[OVERVIEW].orientation = "NOWHERE"
        manager = _manager(microscope, experiment)

        manager.run([OVERVIEW, LATER], [GRID])

        grid = experiment.get_grid_by_name(GRID)
        assert _ran(grid, OVERVIEW)[-1].status is AutoLamellaTaskStatus.Failed
        assert _ran(grid, LATER) == []
        assert not manager.stalled

    def test_with_review_off_nothing_waits(self, microscope, experiment):
        _with_later_task(experiment, review_wait=0)
        manager = _manager(microscope, experiment)
        manager.review_enabled = False

        manager.run([OVERVIEW, LATER], [GRID])

        grid = experiment.get_grid_by_name(GRID)
        assert grid.has_completed_task(OVERVIEW) and grid.has_completed_task(LATER)
        assert not manager.stalled


# ---------------------------------------------------------------------------
# The latest run of a requirement counts (FIB-1006)
# ---------------------------------------------------------------------------


class TestTheLatestRunOfARequirementCounts:
    def _succeed_once(self, microscope, experiment):
        _with_later_task(experiment, review_wait=30.0)
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.automated
        _manager(microscope, experiment).run([OVERVIEW], [GRID])
        grid = experiment.get_grid_by_name(GRID)
        assert grid.has_completed_task(OVERVIEW)
        return grid

    def test_a_failed_rerun_skips_the_task_that_requires_it(
        self, microscope, experiment
    ):
        grid = self._succeed_once(microscope, experiment)
        experiment.grid_protocol.task_config[OVERVIEW].orientation = "NOWHERE"

        _manager(microscope, experiment).run([OVERVIEW, LATER], [GRID])

        assert _ran(grid, OVERVIEW)[-1].status is AutoLamellaTaskStatus.Failed
        assert _ran(grid, LATER) == [], "the old success did not license it"

    def test_a_rejected_rerun_skips_the_task_that_requires_it(
        self, microscope, experiment
    ):
        grid = self._succeed_once(microscope, experiment)
        experiment.grid_protocol.task_config[OVERVIEW].attention = Attention.review
        manager = _manager(microscope, experiment)
        thread = _decide_when(
            experiment,
            GRID,
            lambda: grid.is_awaiting_decision(OVERVIEW) and manager.deferred_items(),
            DecisionOutcome.Rejected,
            reason="all ice",
        )

        manager.run([OVERVIEW, LATER], [GRID])
        thread.join(5)

        assert _ran(grid, OVERVIEW)[-1].status is AutoLamellaTaskStatus.Failed
        assert _ran(grid, LATER) == []

    def test_a_rerun_queued_after_its_consumer_is_waited_for(
        self, microscope, experiment
    ):
        grid = self._succeed_once(microscope, experiment)
        experiment.grid_protocol.task_config[OVERVIEW].orientation = "NOWHERE"
        manager = _manager(microscope, experiment)

        manager.run([LATER, OVERVIEW], [GRID])

        assert _ran(grid, OVERVIEW)[-1].status is AutoLamellaTaskStatus.Failed
        assert _ran(grid, LATER) == [], "waited for the rerun, then skipped"
        (later,) = [i for i in manager.queue.items if i.task_name == LATER]
        assert later.status is AutoLamellaTaskStatus.Skipped


# ---------------------------------------------------------------------------
# A grid is loaded for work that can run (FIB-1005)
# ---------------------------------------------------------------------------


def _loaded(microscope, name):
    return microscope._stage.holder.find_slot_by_grid_name(name) is not None


class TestAGridIsLoadedForWorkThatCanRun:
    def test_no_exchange_when_every_selected_task_will_be_skipped(
        self, microscope, experiment
    ):
        """Grid-01 is in; on Grid-02 the FIB overview requires an SEM overview
        that never ran. Nothing there can run, so Grid-01 stays in."""
        _with_later_task(experiment, review_wait=0)
        _manager(microscope, experiment).run([OVERVIEW], [GRID])
        assert _loaded(microscope, GRID)
        second = experiment.get_grid_by_name(OTHER_GRID)

        manager = _manager(microscope, experiment)
        manager.run([LATER], [OTHER_GRID])

        assert _loaded(microscope, GRID) and not _loaded(microscope, OTHER_GRID)
        assert _ran(second, LOAD_ENTRY_NAME) == [], "no exchange was attempted"
        statuses = {i.task_name: i.status for i in manager.queue.items}
        assert statuses == {
            LOAD_ENTRY_NAME: AutoLamellaTaskStatus.Skipped,
            LATER: AutoLamellaTaskStatus.Skipped,
        }
        assert not manager.stalled

    def test_a_grid_is_loaded_once_its_waiting_work_can_run(
        self, microscope, experiment
    ):
        """Grid-02's FIB overview waits on a decision about its SEM overview from
        an earlier run: Grid-02 is not loaded for it until the decision lands."""
        _with_later_task(experiment, review_wait=0)
        _manager(microscope, experiment).run([OVERVIEW], [OTHER_GRID])
        second = experiment.get_grid_by_name(OTHER_GRID)
        assert second.is_awaiting_decision(OVERVIEW)
        _manager(microscope, experiment).run([OVERVIEW], [GRID])  # Grid-01 back in
        assert _loaded(microscope, GRID)
        loads_before = len(_ran(second, LOAD_ENTRY_NAME))
        experiment.task_protocol.options.review_wait = 30.0

        manager = _manager(microscope, experiment)
        seen = {}

        def parked():
            if manager.deferred_items():
                seen.setdefault("grid_01_in", _loaded(microscope, GRID))
                seen.setdefault("loads", len(_ran(second, LOAD_ENTRY_NAME)))
                return True
            return False

        thread = _decide_when(experiment, OTHER_GRID, parked, DecisionOutcome.Confirmed)
        manager.run([LATER], [OTHER_GRID])
        thread.join(5)

        assert seen == {"grid_01_in": True, "loads": loads_before}, (
            "while it waited, nothing was exchanged"
        )
        assert _ran(second, LATER)[-1].status is AutoLamellaTaskStatus.Completed
        assert len(_ran(second, LOAD_ENTRY_NAME)) == loads_before + 1
        waiting = [r for i, r in manager.deferred_items()]
        assert waiting == []

    def test_a_load_with_nothing_behind_it_still_loads(self, microscope, experiment):
        _with_later_task(experiment, review_wait=0)
        manager = _manager(microscope, experiment)
        manager.queue.build_from_pairs([(OTHER_GRID, LOAD_ENTRY_NAME)])
        manager._run_queue()
        assert _loaded(microscope, OTHER_GRID)
