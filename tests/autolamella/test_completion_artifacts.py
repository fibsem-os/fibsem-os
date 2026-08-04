"""An artifact is written by whatever finishes the run, not by a person clicking.

A placeholder for the PDF and overview PNG while the trigger is proven end to end.
See FIB-461.
"""

import json
import os
from pathlib import Path
from typing import List

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


def _summary(experiment: Experiment) -> dict:
    with open(os.path.join(experiment.path, COMPLETION_SUMMARY_FILENAME)) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------

def test_the_summary_says_which_experiment_and_when(experiment):
    from datetime import datetime

    fired_at = 1_754_000_000.0
    context = HookContext(
        event=HookEvent.EXPERIMENT_COMPLETED,
        experiment_id=experiment.id,
        experiment_name=experiment.name,
        timestamp=fired_at,
    )

    write_completion_summary(experiment, context)

    summary = _summary(experiment)
    assert summary["experiment_id"] == experiment.id
    assert summary["experiment_name"] == "test-exp"
    assert summary["event"] == "experiment_completed"
    # ISO 8601 to the second, and it round-trips to the event's timestamp. Asserted as
    # a property rather than a literal, which would encode the runner's timezone.
    assert datetime.fromisoformat(summary["completed_at"]).timestamp() == fired_at


def test_it_records_when_the_run_finished_not_when_the_file_was_written(experiment):
    """The event's timestamp, not now(): the two differ if a hook defers work."""
    import time

    fired_at = time.time() - 3600
    write_completion_summary(
        experiment,
        HookContext(event=HookEvent.EXPERIMENT_COMPLETED, timestamp=fired_at),
    )

    from datetime import datetime

    expected = datetime.fromtimestamp(fired_at).isoformat(timespec="seconds")
    assert _summary(experiment)["completed_at"] == expected


def test_it_falls_back_to_the_experiment_when_the_context_is_bare(experiment):
    """A context fired without the run fields still produces an identified artifact."""
    write_completion_summary(
        experiment, HookContext(event=HookEvent.EXPERIMENT_COMPLETED)
    )

    summary = _summary(experiment)
    assert summary["experiment_id"] == experiment.id
    assert summary["experiment_name"] == "test-exp"


# ---------------------------------------------------------------------------
# through the hook, as the UI wires it
# ---------------------------------------------------------------------------

def _manager_with_summary_hook(experiment: Experiment) -> TaskManager:
    """Mirrors what AutoLamellaUI.setup_hooks registers."""
    class _NoMicroscope:
        fm = None

    hooks = HookManager()
    hooks.register(
        FunctionHook(
            name="completion_summary",
            events=[HookEvent.EXPERIMENT_COMPLETED],
            callback=lambda ctx: write_completion_summary(experiment, ctx),
        )
    )
    return TaskManager(
        microscope=_NoMicroscope(),
        experiment=experiment,
        parent_ui=None,
        hook_manager=hooks,
    )


def _complete_lamella(experiment: Experiment, name: str) -> Lamella:
    lamella = Lamella(path=experiment.path, number=1, petname=name)
    for task_name in REQUIRED_TASKS:
        lamella.task_history.append(
            AutoLamellaTaskState(name=task_name, status=AutoLamellaTaskStatus.Completed)
        )
    experiment.positions.append(lamella)
    return lamella


def test_finishing_the_experiment_writes_the_artifact(experiment):
    manager = _manager_with_summary_hook(experiment)
    _complete_lamella(experiment, "lam-1")
    manager._completed_lamella = set()  # pretend it finished during this run

    manager._maybe_fire_experiment_completed()

    summary = _summary(experiment)
    assert summary["experiment_name"] == "test-exp"
    assert summary["event"] == "experiment_completed"


def test_an_unfinished_experiment_writes_nothing(experiment):
    manager = _manager_with_summary_hook(experiment)
    lamella = Lamella(path=experiment.path, number=1, petname="lam-1")
    lamella.task_history.append(
        AutoLamellaTaskState(name="MillTrench", status=AutoLamellaTaskStatus.Completed)
    )
    experiment.positions.append(lamella)
    manager._snapshot_completion()

    manager._maybe_fire_experiment_completed()

    assert not os.path.exists(
        os.path.join(experiment.path, COMPLETION_SUMMARY_FILENAME)
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
            events=[HookEvent.EXPERIMENT_COMPLETED],
            callback=lambda ctx: (_ for _ in ()).throw(OSError("disk full")),
        )
    )
    manager = TaskManager(
        microscope=_NoMicroscope(), experiment=experiment,
        parent_ui=None, hook_manager=hooks,
    )
    _complete_lamella(experiment, "lam-1")
    manager._completed_lamella = set()

    with caplog.at_level(logging.ERROR):
        manager._maybe_fire_experiment_completed()  # must not raise

    assert any("completion_summary" in r.message for r in caplog.records)


def test_the_event_the_ui_subscribes_to_exists():
    """Guards the wiring in setup_hooks, which tests/ui cannot construct headlessly."""
    assert HookEvent.EXPERIMENT_COMPLETED in list(HookEvent)
