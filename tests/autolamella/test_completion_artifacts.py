"""An artifact is written by whatever finishes the work, not by a person clicking.

Per lamella, because a lamella is what gets delivered to a TEM and because an
experiment with one abandoned lamella never completes at all. A placeholder for the
PDF and overview PNG while the trigger is proven end to end. See FIB-461.

The hook is commented out in AutoLamellaUI.setup_hooks rather than shipped, so nothing
writes these files during a real run yet. These tests are what keeps the writer honest
in the meantime: they register the same hook themselves, so the wiring is exercised
even while it is dormant.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    AutoLamellaWorkflowConfig,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.tools.artifacts import (
    COMPLETION_SUMMARY_FILENAME,
    write_completion_summary,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.hooks import FunctionHook, HookContext, HookEvent, HookManager

REQUIRED_TASKS = ["MillTrench", "MillUndercut"]


@pytest.fixture
def experiment(tmp_path: Path) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(name=name, supervise=False, required=True)
                for name in REQUIRED_TASKS
            ]
        )
    )
    # Constructing an Experiment deliberately does not create its directory (FIB-420).
    os.makedirs(exp.path, exist_ok=True)
    return exp


def _new_lamella(experiment: Experiment, name: str) -> Lamella:
    """As Experiment._create_lamella builds one: its path is its own directory under
    the experiment, which is what the artifact is written into."""
    return Lamella(
        path=Path(os.path.join(experiment.path, name)),
        number=len(experiment.positions) + 1,
        petname=name,
    )


def _complete_lamella(experiment: Experiment, name: str) -> Lamella:
    lamella = _new_lamella(experiment, name)
    for task_name in REQUIRED_TASKS:
        lamella.task_history.append(
            AutoLamellaTaskState(name=task_name, status=AutoLamellaTaskStatus.Completed)
        )
    experiment.positions.append(lamella)
    return lamella


def _summary(lamella: Lamella) -> dict:
    with open(os.path.join(lamella.path, COMPLETION_SUMMARY_FILENAME)) as f:
        return json.load(f)


def _context(lamella: Lamella, experiment: Experiment, **overrides) -> HookContext:
    fields = dict(
        event=HookEvent.ITEM_COMPLETED,
        item_name=lamella.name,
        item_id=lamella.id,
        experiment_id=experiment.id,
        experiment_name=experiment.name,
    )
    fields.update(overrides)
    return HookContext(**fields)


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------

def test_the_summary_says_which_lamella_which_experiment_and_when(experiment):
    lamella = _complete_lamella(experiment, "lam-1")
    fired_at = 1_754_000_000.0

    write_completion_summary(
        experiment, _context(lamella, experiment, timestamp=fired_at)
    )

    summary = _summary(lamella)
    assert summary["item_id"] == lamella.id
    assert summary["item_name"] == lamella.name
    assert summary["experiment_id"] == experiment.id
    assert summary["experiment_name"] == "test-exp"
    assert summary["event"] == "item_completed"
    # ISO 8601 to the second, and it round-trips to the event's timestamp. Asserted as
    # a property rather than a literal, which would encode the runner's timezone.
    assert datetime.fromisoformat(summary["completed_at"]).timestamp() == fired_at


def test_it_records_the_work_that_was_delivered(experiment):
    lamella = _complete_lamella(experiment, "lam-1")

    write_completion_summary(experiment, _context(lamella, experiment))

    tasks = _summary(lamella)["tasks_completed"]
    assert [t["name"] for t in tasks] == REQUIRED_TASKS
    assert [t["task_id"] for t in tasks] == [t.task_id for t in lamella.task_history]


def test_it_records_the_files_each_task_left_behind(experiment):
    """The point of the artifact: the recipient gets the folder and needs to know
    which files in it are the result."""
    lamella = _complete_lamella(experiment, "lam-1")
    lamella.task_history[0].outputs = {
        "final_sem": ["MillTrench-final-sem.tif"],
        "final_fib": ["MillTrench-final-fib.tif"],
    }

    write_completion_summary(experiment, _context(lamella, experiment))

    tasks = _summary(lamella)["tasks_completed"]
    assert tasks[0]["outputs"] == {
        "final_sem": ["MillTrench-final-sem.tif"],
        "final_fib": ["MillTrench-final-fib.tif"],
    }
    # relative to the lamella directory, which is where the artifact itself lives, so
    # they resolve as-is for whoever receives the folder
    assert os.path.isabs(tasks[0]["outputs"]["final_sem"][0]) is False
    assert tasks[1]["outputs"] == {}


def test_it_records_how_long_each_task_took(experiment):
    lamella = _complete_lamella(experiment, "lam-1")
    task = lamella.task_history[0]
    task.end_timestamp = task.start_timestamp + 412.34

    write_completion_summary(experiment, _context(lamella, experiment))

    record = _summary(lamella)["tasks_completed"][0]
    assert record["duration_s"] == 412.3
    assert (
        datetime.fromisoformat(record["completed_at"]).timestamp()
        == pytest.approx(task.end_timestamp, abs=1)
    )


def test_a_task_still_running_has_no_completion_time(experiment):
    """end_timestamp is None until post_task lands; null rather than a crash."""
    lamella = _complete_lamella(experiment, "lam-1")
    lamella.task_history[0].end_timestamp = None

    write_completion_summary(experiment, _context(lamella, experiment))

    record = _summary(lamella)["tasks_completed"][0]
    assert record["completed_at"] is None
    assert record["duration_s"] == 0


def test_a_task_that_failed_is_not_reported_as_delivered(experiment):
    """task_history holds every terminal outcome (FIB-490), so this filters on status
    -- a failed task's outputs are not results anyone should be handed."""
    lamella = _complete_lamella(experiment, "lam-1")
    lamella.task_history.append(
        AutoLamellaTaskState(
            name="MillPolish",
            status=AutoLamellaTaskStatus.Failed,
            outputs={"final_sem": ["MillPolish-final-sem.tif"]},
        )
    )

    write_completion_summary(experiment, _context(lamella, experiment))

    tasks = _summary(lamella)["tasks_completed"]
    assert "MillPolish" not in [t["name"] for t in tasks]


def test_a_retried_task_appears_as_both_attempts(experiment):
    """Two completions of the same task are two entries, distinguished by task_id --
    the history is reported as it happened rather than collapsed by name."""
    lamella = _complete_lamella(experiment, "lam-1")
    retry = AutoLamellaTaskState(
        name="MillTrench", status=AutoLamellaTaskStatus.Completed
    )
    lamella.task_history.append(retry)

    write_completion_summary(experiment, _context(lamella, experiment))

    tasks = _summary(lamella)["tasks_completed"]
    trench = [t for t in tasks if t["name"] == "MillTrench"]
    assert len(trench) == 2
    assert trench[0]["task_id"] != trench[1]["task_id"]


def test_it_lands_beside_the_lamella_not_in_the_experiment_root(experiment):
    """Two finished lamellae must not overwrite each other's artifact."""
    lam1 = _complete_lamella(experiment, "lam-1")
    lam2 = _complete_lamella(experiment, "lam-2")

    write_completion_summary(experiment, _context(lam1, experiment))
    write_completion_summary(experiment, _context(lam2, experiment))

    assert _summary(lam1)["item_name"] == lam1.name
    assert _summary(lam2)["item_name"] == lam2.name
    assert not os.path.exists(
        os.path.join(experiment.path, COMPLETION_SUMMARY_FILENAME)
    )


def test_it_records_when_the_work_finished_not_when_the_file_was_written(experiment):
    """The event's timestamp, not now(): the two differ if a hook defers work."""
    import time

    lamella = _complete_lamella(experiment, "lam-1")
    fired_at = time.time() - 3600

    write_completion_summary(
        experiment, _context(lamella, experiment, timestamp=fired_at)
    )

    expected = datetime.fromtimestamp(fired_at).isoformat(timespec="seconds")
    assert _summary(lamella)["completed_at"] == expected


def test_it_falls_back_to_the_experiment_when_the_context_is_bare(experiment):
    """A context fired without the run fields still produces an identified artifact."""
    lamella = _complete_lamella(experiment, "lam-1")

    write_completion_summary(
        experiment,
        HookContext(event=HookEvent.ITEM_COMPLETED, item_name=lamella.name),
    )

    summary = _summary(lamella)
    assert summary["experiment_id"] == experiment.id
    assert summary["experiment_name"] == "test-exp"


def test_the_lamella_is_found_by_id_when_it_has_been_renamed(experiment):
    """The id is the stable key; the petname is editable. A context in flight while
    someone renames the lamella still resolves."""
    lamella = _complete_lamella(experiment, "lam-1")
    context = _context(lamella, experiment)
    lamella.name = "renamed-by-a-user"

    write_completion_summary(experiment, context)

    assert _summary(lamella)["item_name"] == "renamed-by-a-user"


def test_an_unresolvable_item_writes_nothing(experiment, caplog):
    import logging

    _complete_lamella(experiment, "lam-1")

    with caplog.at_level(logging.WARNING):
        result = write_completion_summary(
            experiment, HookContext(event=HookEvent.ITEM_COMPLETED, item_name="ghost")
        )

    assert result is None
    assert any("ghost" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# through the hook, as the UI wires it
# ---------------------------------------------------------------------------

def _manager_with_summary_hook(experiment: Experiment) -> TaskManager:
    """Mirrors the registration commented out in AutoLamellaUI.setup_hooks, so the two
    have to be changed together when it is turned back on."""
    class _NoMicroscope:
        fm = None

    hooks = HookManager()
    hooks.register(
        FunctionHook(
            name="completion_summary",
            events=[HookEvent.ITEM_COMPLETED],
            callback=lambda ctx: write_completion_summary(experiment, ctx),
        )
    )
    return TaskManager(
        microscope=_NoMicroscope(),
        experiment=experiment,
        parent_ui=None,
        hook_manager=hooks,
    )


def test_finishing_a_lamella_writes_its_artifact(experiment):
    manager = _manager_with_summary_hook(experiment)
    lamella = _complete_lamella(experiment, "lam-1")
    manager._completed_lamella = set()  # pretend it finished during this run

    manager._maybe_fire_lamella_completed(lamella, "MillUndercut")

    summary = _summary(lamella)
    assert summary["item_name"] == lamella.name
    assert summary["event"] == "item_completed"


def test_an_abandoned_neighbour_does_not_stop_the_artifact(experiment):
    """The reason this hangs off the item: one defective lamella must not suppress the
    artifact for the ones that finished."""
    from fibsem.applications.autolamella.structures import DefectType

    manager = _manager_with_summary_hook(experiment)
    finished = _complete_lamella(experiment, "lam-1")
    abandoned = _new_lamella(experiment, "lam-2")
    abandoned.defect.state = DefectType.FAILURE
    experiment.positions.append(abandoned)
    manager._completed_lamella = set()

    manager._maybe_fire_lamella_completed(finished, "MillUndercut")

    assert _summary(finished)["item_name"] == finished.name


def test_an_unfinished_lamella_writes_nothing(experiment):
    manager = _manager_with_summary_hook(experiment)
    lamella = _new_lamella(experiment, "lam-1")
    lamella.task_history.append(
        AutoLamellaTaskState(name="MillTrench", status=AutoLamellaTaskStatus.Completed)
    )
    experiment.positions.append(lamella)
    manager._snapshot_completion()

    manager._maybe_fire_lamella_completed(lamella, "MillTrench")

    assert not os.path.exists(
        os.path.join(lamella.path, COMPLETION_SUMMARY_FILENAME)
    )


def test_a_failing_artifact_does_not_break_the_run(experiment, caplog):
    """FIB-461: a report that cannot be generated is a warning, never an exception
    propagating into workflow teardown. HookManager.fire already contains it."""
    import logging

    class _NoMicroscope:
        fm = None

    hooks = HookManager()
    hooks.register(
        FunctionHook(
            name="completion_summary",
            events=[HookEvent.ITEM_COMPLETED],
            callback=lambda ctx: (_ for _ in ()).throw(OSError("disk full")),
        )
    )
    manager = TaskManager(
        microscope=_NoMicroscope(), experiment=experiment,
        parent_ui=None, hook_manager=hooks,
    )
    lamella = _complete_lamella(experiment, "lam-1")
    manager._completed_lamella = set()

    with caplog.at_level(logging.ERROR):
        manager._maybe_fire_lamella_completed(lamella, "MillUndercut")  # must not raise

    assert any("completion_summary" in r.message for r in caplog.records)


def test_the_event_the_dormant_wiring_names_still_exists():
    """The registration in setup_hooks is commented out, so nothing type-checks the
    event it names. This does, and fails loudly if ITEM_COMPLETED is ever renamed while
    the wiring is asleep."""
    assert HookEvent.ITEM_COMPLETED in list(HookEvent)
