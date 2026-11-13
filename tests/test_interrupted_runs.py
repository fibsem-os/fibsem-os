"""A run that ended without unwinding (engineering review R1, R4): a task left
InProgress on disk is not running when the experiment is read again, and a beam
that will not go off does not leave the other on."""

from types import SimpleNamespace

import pytest

from fibsem.applications.autolamella.config_edits import _is_running
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.structures.settings import BeamType, MicroscopeState
from fibsem.workflows.manager import BaseTaskManager
from fibsem.workflows.status import INTERRUPTED, TaskStatus


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="interrupted")
    (tmp_path / "interrupted").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    for name in ("running", "waiting"):
        exp.add_new_lamella(
            microscope_state=MicroscopeState(),
            task_config=exp.task_protocol.task_config,
            name=name,
        )
    return exp


def _saved_mid_task(experiment) -> str:
    """Save as a decision or an edit does while a task runs; the file's path."""
    state = experiment.positions[0].task_state
    state.name = "Rough Milling"
    state.status = TaskStatus.InProgress
    state.end_timestamp = None
    experiment.save()
    return experiment.path + "/experiment.yaml"


def test_a_task_left_in_progress_is_read_as_failed(experiment):
    loaded = Experiment.load(_saved_mid_task(experiment))
    running, waiting = loaded.positions
    assert running.task_state.status is TaskStatus.Failed
    assert running.task_state.status_message == INTERRUPTED
    assert running.task_state.end_timestamp is not None
    assert not _is_running(running, "Rough Milling")
    assert waiting.task_state.status is TaskStatus.NotStarted


def test_the_interrupted_run_is_in_the_history_once(experiment):
    path = _saved_mid_task(experiment)
    loaded = Experiment.load(path)
    task_id = loaded.positions[0].task_state.task_id
    assert [s.task_id for s in loaded.positions[0].task_history] == [task_id]
    assert loaded.positions[0].task_history[0].status is TaskStatus.Failed
    loaded.save()
    again = Experiment.load(path)
    assert [s.task_id for s in again.positions[0].task_history] == [task_id]


def test_a_finished_task_is_left_as_it_ended(experiment):
    state = experiment.positions[0].task_state
    state.status = TaskStatus.Completed
    experiment.save()
    loaded = Experiment.load(experiment.path + "/experiment.yaml")
    assert loaded.positions[0].task_state.status is TaskStatus.Completed
    assert loaded.end_what_was_running(INTERRUPTED) == 0


class _Beams:
    def __init__(self, stuck: BeamType):
        self.stuck = stuck
        self.asked = []

    def turn_off(self, beam: BeamType) -> None:
        self.asked.append(beam)
        if beam is self.stuck:
            raise RuntimeError(f"{beam.name} will not turn off")


def test_a_beam_that_will_not_go_off_does_not_leave_the_other_on():
    microscope = _Beams(stuck=BeamType.ELECTRON)
    with pytest.raises(RuntimeError, match="ELECTRON will not turn off"):
        BaseTaskManager._turn_beams_off(SimpleNamespace(microscope=microscope))
    assert microscope.asked == [BeamType.ELECTRON, BeamType.ION]
