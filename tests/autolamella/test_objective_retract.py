"""Tests for retracting the FM objective when a fluorescence task finishes (FIB-376).

The objective is inserted by the fluorescence tasks and, before this, was only
retracted once at the very end of a whole workflow run — so it stayed inserted
across every task in between.
"""
from unittest.mock import MagicMock

import pytest

from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (
    AcquireFluorescenceImageConfig,
)
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask


class _StubTask(AutoLamellaTask):
    """Concrete subclass — AutoLamellaTask is abstract on ``_run``."""

    def _run(self) -> None:  # pragma: no cover - never called
        raise NotImplementedError


def _task_with_objective(state: str) -> AutoLamellaTask:
    """A bare task instance wired to a mock microscope, for the retract helper."""
    task = _StubTask.__new__(_StubTask)
    task.microscope = MagicMock()
    task.microscope.fm.objective.state = state
    task.config = MagicMock(task_name="Acquire Fluorescence Image")
    task.lamella = MagicMock()
    task.parent_ui = None
    task.log_status_message = MagicMock()
    return task


# --- task-level helper ------------------------------------------------------


def test_retract_objective_when_inserted():
    task = _task_with_objective("Inserted")
    task._retract_objective()
    assert task.microscope.fm.objective.retract.called


def test_no_retract_when_already_retracted():
    task = _task_with_objective("Retracted")
    task._retract_objective()
    assert not task.microscope.fm.objective.retract.called


def test_no_retract_without_an_fm():
    task = _task_with_objective("Inserted")
    task.microscope.fm = None
    task._retract_objective()  # must not raise


def test_retract_failure_does_not_fail_the_task():
    """A task that otherwise completed shouldn't be reported failed by the retract."""
    task = _task_with_objective("Inserted")
    task.microscope.fm.objective.retract.side_effect = RuntimeError("stuck")
    task._retract_objective()  # swallowed and logged


# --- config round-trip ------------------------------------------------------


@pytest.mark.parametrize("value", [True, False])
def test_retract_objective_survives_a_config_round_trip(value):
    """to_dict/from_dict are hand-written here — a new field is easy to drop."""
    config = AcquireFluorescenceImageConfig(task_name="FM", retract_objective=value)
    assert AcquireFluorescenceImageConfig.from_dict(config.to_dict()).retract_objective is value


def test_retract_objective_defaults_true_for_older_protocols():
    assert AcquireFluorescenceImageConfig.from_dict({"task_name": "FM"}).retract_objective is True
