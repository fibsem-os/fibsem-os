"""An image should say which item and task produced it, not rely on its folder.

The association existed only as filesystem position -- every task hand-set
`image_settings.path = self.lamella.path` -- which is lost the moment a file is
copied, emailed or pulled into an export. See FIB-466.

It is recorded on `FibsemExperimentRef` alongside the experiment, because that is one
answer to one question. Two things have to hold for that to be safe: the workflow half
is *cleared* on every path out of a task, and every image gets its own copy. A stale
record stamps the previous lamella onto an image, and a shared one rewrites images
already acquired -- both silent, both worse than recording nothing.
"""

import pytest

from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskConfig,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.structures import BeamType, FibsemExperimentRef


@pytest.fixture
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo")
    scope.experiment = FibsemExperimentRef(id="exp-1", name="an-experiment")
    return scope


@pytest.fixture
def lamella(tmp_path) -> Lamella:
    return Lamella(path=str(tmp_path), petname="grand-dodo", number=1)


class _Config(AutoLamellaTaskConfig):
    task_type = "test-task"
    display_name = "Test Task"


class _Task(AutoLamellaTask):
    """A task whose body is supplied per test, so failure paths are reachable."""

    def __init__(self, microscope, lamella, body=None, **kwargs):
        super().__init__(
            microscope=microscope,
            lamella=lamella,
            config=_Config(task_name="Test Task"),
            parent_ui=None,
            **kwargs,
        )
        self._body = body or (lambda: None)

    def _run(self) -> None:
        self._body()


def _stamped(microscope) -> FibsemExperimentRef:
    """What an image acquired right now would carry."""
    return microscope.acquire_image(beam_type=BeamType.ELECTRON).metadata.experiment


def test_outside_a_workflow_there_is_no_item(microscope) -> None:
    """The experiment is still recorded; the item simply is not, which is correct."""
    stamped = _stamped(microscope)

    assert stamped.id == "exp-1"
    assert stamped.item_id is None
    assert stamped.task_id is None


def test_an_image_acquired_during_a_task_carries_both(microscope, lamella) -> None:
    captured = {}

    task = _Task(
        microscope, lamella, body=lambda: captured.update(seen=_stamped(microscope))
    )
    task.run()

    stamped = captured["seen"]
    assert stamped.id == "exp-1"
    assert stamped.item_id == lamella.id
    assert stamped.item_name == lamella.name
    assert stamped.task_id == task.task_id
    assert stamped.task_name == "Test Task"


def test_the_workflow_half_is_cleared_after_the_task_completes(
    microscope, lamella
) -> None:
    _Task(microscope, lamella).run()

    assert _stamped(microscope).item_id is None


def test_the_workflow_half_is_cleared_when_the_task_fails(microscope, lamella) -> None:
    """The case the `finally` exists for.

    `post_task` never runs when the body raises, so clearing there alone would leave
    the record set. The next thing acquired -- an operator looking at what went wrong
    -- would claim to belong to the task that just failed.
    """

    def explode():
        raise RuntimeError("the task went wrong")

    with pytest.raises(RuntimeError, match="the task went wrong"):
        _Task(microscope, lamella, body=explode).run()

    assert _stamped(microscope).item_id is None


def test_clearing_the_workflow_keeps_the_experiment(microscope, lamella) -> None:
    """The reason clearing is a method rather than four assignments at the call site."""
    _Task(microscope, lamella).run()

    assert microscope.experiment.id == "exp-1"
    assert microscope.experiment.name == "an-experiment"


def test_a_failing_task_still_stamped_while_it_ran(microscope, lamella) -> None:
    """Clearing afterwards must not mean images from a failed task lose it.

    Those are the ones worth having.
    """
    captured = {}

    def acquire_then_fail():
        captured["seen"] = _stamped(microscope)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _Task(microscope, lamella, body=acquire_then_fail).run()

    assert captured["seen"].item_name == lamella.name


def test_a_second_task_replaces_the_first(microscope, lamella) -> None:
    first = _Task(microscope, lamella)
    first.run()

    captured = {}
    second = _Task(
        microscope, lamella, body=lambda: captured.update(seen=_stamped(microscope))
    )
    second.run()

    assert captured["seen"].task_id == second.task_id
    assert captured["seen"].task_id != first.task_id


def test_each_image_keeps_its_own_copy(microscope, lamella) -> None:
    """The record is mutated per task, so sharing one object would rewrite history.

    This is what makes it safe to keep the item on the same record as the experiment:
    without the deepcopy in `_set_additional_metadata`, moving to the next lamella
    would silently relabel every image already acquired.
    """
    captured = {}

    _Task(
        microscope,
        lamella,
        body=lambda: captured.update(
            image=microscope.acquire_image(beam_type=BeamType.ELECTRON)
        ),
    ).run()

    image = captured["image"]
    assert image.metadata.experiment.item_name == lamella.name

    microscope.experiment.set_workflow_metadata(item_name="a-different-lamella")

    assert image.metadata.experiment.item_name == lamella.name


def test_the_record_round_trips(microscope, lamella) -> None:
    captured = {}

    _Task(
        microscope, lamella, body=lambda: captured.update(seen=_stamped(microscope))
    ).run()

    stamped = captured["seen"]
    assert FibsemExperimentRef.from_dict(stamped.to_dict()) == stamped


def test_a_pre_v8_file_has_no_item() -> None:
    """Images written before this existed load with the workflow half absent."""
    legacy = {"id": "exp-1", "name": "an-experiment", "date": 1700000000.0}

    ref = FibsemExperimentRef.from_dict(legacy)

    assert ref.id == "exp-1"
    assert ref.item_id is None
    assert ref.item_name is None
    assert ref.task_id is None
    assert ref.task_name is None
