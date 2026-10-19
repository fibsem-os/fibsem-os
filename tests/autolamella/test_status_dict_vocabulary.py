"""The status dict and HookContext name the same thing the same way.

Both streams describe one task lifecycle. They stay separate on purpose — see the
_emit_report docstring — but a subscriber reading one and then the other should not
have to learn two words for the item. See FIB-464.
"""

from pathlib import Path
from typing import List

import pytest

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
from fibsem.hooks import HookContext
from fibsem.workflows.status import TaskStatus


class _FakeSignal:
    def __init__(self) -> None:
        self.payloads: List[dict] = []

    def emit(self, payload: dict) -> None:
        self.payloads.append(payload)


class _FakeUI:
    """Hears the run's ``status`` as the window does, which is all
    _emit_report says on."""

    def __init__(self, manager: TaskManager) -> None:
        self.workflow_status_signal = _FakeSignal()
        manager.status.connect(self.workflow_status_signal.emit)


def _listened(manager: TaskManager) -> TaskManager:
    manager.ui = _FakeUI(manager)
    return manager


@pytest.fixture
def manager(tmp_path: Path) -> TaskManager:
    class _NoMicroscope:
        fm = None

    experiment = Experiment(path=tmp_path, name="test-exp")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    return _listened(
        TaskManager(
            microscope=_NoMicroscope(),
            experiment=experiment,
        )
    )


def _emit(manager: TaskManager, name: str = "lam-1") -> dict:
    lamella = Lamella(path=manager.experiment.path, number=1, petname=name)
    items = manager.queue.build_from_matrix(["MillTrench"], [lamella.name])
    manager._emit_report(
        item=items[0],
        item_name=lamella.name,
        status=TaskStatus.InProgress,
    )
    return manager.ui.workflow_status_signal.payloads[-1].report


def test_the_status_dict_names_the_item_the_way_the_hook_does(manager: TaskManager):
    status = _emit(manager)

    assert status.item_name == "lam-1"
    # the hook payload names it identically
    assert "item_name" in HookContext(event="task_started").__dict__


def test_the_positioning_fields_keep_their_names(manager: TaskManager):
    """The position fields are timeline positioning, not lifecycle facts — they have
    no hook counterpart to agree with, so the vocabulary alignment leaves them alone.

    They are no longer the task x lamella matrix indices this originally asserted:
    those were replaced by a position in the live queue, because the queue can be
    added to and reordered mid-run and a matrix coordinate then has no answer. That
    is FIB-472, not a vocabulary change — the point this test makes is unaffected.
    """
    import dataclasses

    from fibsem.workflows.status import (
        WorkflowStatusUpdate,
    )

    _emit(manager)

    # Field names on the type rather than keys in a dict, now that the payload is a
    # record (FIB-827). A stronger check than membership: a field cannot be present
    # on one emission and absent on another.
    fields = {f.name for f in dataclasses.fields(WorkflowStatusUpdate)}
    assert {"queue_position", "queue_total"} <= fields
