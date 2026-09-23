"""A workflow run without the GUI records its events, the same way a GUI run
does (FIB-1044).

The task manager, the one path headless and GUI runs share, finds the
microscope's recorder -- the app keeps one from connecting -- or makes one for
the run, and registers its lifecycle hook for the run. Real tasks through
``run_tasks`` on Demo, read back from ``events.jsonl``.
"""

import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.event_recording import (
    EVENTS_FILENAME,
    EventRecorder,
    read_events,
    recorder_for,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks import manager as manager_module
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    run_grid_tasks,
)
from fibsem.applications.autolamella.workflows.tasks.manager import run_tasks
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTaskConfig,
)
from fibsem.hooks import HookManager

SETUP = "Setup Lamella Position"
CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


@pytest.fixture
def experiment(microscope, tmp_path):
    exp = Experiment(path=tmp_path, name="headless")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[AutoLamellaTaskDescription(name=SETUP, required=True)]
        )
    )
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        EventedDict(
            {
                SETUP: SelectMillingPositionTaskConfig(
                    task_name=SETUP, use_autofocus=False
                )
            }
        ),
    )
    exp.positions[0].path.mkdir(parents=True, exist_ok=True)
    exp.positions[0].milling_pose = microscope.get_microscope_state()
    yield exp
    assert recorder_for(microscope) is None, "a run left its recorder open"


def _kinds(records):
    return [r["kind"] for r in records]


def test_a_run_without_the_gui_records_its_events(microscope, experiment):
    run_tasks(microscope, experiment, [SETUP])

    records = list(read_events(Path(experiment.path) / EVENTS_FILENAME))
    kinds = _kinds(records)
    assert kinds.count("workflow_started") == 1
    assert kinds.count("task_started") == 1
    assert kinds.count("task_completed") == 1
    assert kinds.count("workflow_completed") == 1
    assert "task_step" in kinds and "stage_moved" in kinds
    lamella = experiment.positions[0]
    images = [r for r in records if r["kind"] == "image_acquired"]
    assert images, "the task's reference images"
    assert all(r["item"]["name"] == lamella.name for r in images)
    # nobody marked a script's own calls, and a run's are the task's
    assert {r["actor"] for r in records if r["kind"] == "task_step"} == {"task"}


def test_a_run_records_through_the_recorder_the_app_keeps(microscope, experiment):
    """The GUI's recorder is found, not doubled: each lifecycle event once, in
    its stream, and it is still open after the run. The run's hook is taken
    off the caller's hook manager afterwards."""
    kept = EventRecorder(
        microscope, experiment_path=experiment.path, experiment=experiment
    )
    hooks = HookManager()
    try:
        run_tasks(microscope, experiment, [SETUP], hook_manager=hooks)

        kinds = _kinds(kept.buffer.events_since(0)["events"])
        assert kinds.count("task_started") == 1
        assert kinds.count("workflow_completed") == 1
        assert recorder_for(microscope) is kept and kept.writer.alive
        assert kept.lifecycle_hook not in hooks._hooks
    finally:
        kept.close()


def test_a_run_that_raises_still_closes_the_recorder_it_made(
    microscope, experiment, monkeypatch
):
    def fail(self):
        raise RuntimeError("the run broke")

    monkeypatch.setattr(manager_module.TaskManager, "_run_items", fail)

    with pytest.raises(RuntimeError):
        run_tasks(microscope, experiment, [SETUP])

    # closed (the fixture checks), and what it recorded is on disk
    kinds = _kinds(read_events(Path(experiment.path) / EVENTS_FILENAME))
    assert kinds == ["workflow_started"]


def test_a_run_that_raises_leaves_the_recorder_it_found_open(
    microscope, experiment, monkeypatch
):
    def fail(self):
        raise RuntimeError("the run broke")

    monkeypatch.setattr(manager_module.TaskManager, "_run_items", fail)
    kept = EventRecorder(microscope, experiment_path=experiment.path)
    try:
        with pytest.raises(RuntimeError):
            run_tasks(microscope, experiment, [SETUP])

        assert recorder_for(microscope) is kept and kept.writer.alive
    finally:
        kept.close()


def test_a_grid_run_without_the_gui_records_its_events(microscope, experiment):
    run_grid_tasks(microscope, experiment, task_names=[], grid_names=[])

    kinds = _kinds(read_events(Path(experiment.path) / EVENTS_FILENAME))
    assert kinds == ["workflow_started", "workflow_completed"]


def _writers():
    import threading

    return sum(t.name == "fibsem-event-writer" for t in threading.enumerate())


def test_a_recorder_that_cannot_be_made_leaves_no_writer_running():
    """Nothing holds a recorder whose construction failed, so nothing would
    ever close the writer thread it had started."""

    class NotAMicroscope:
        pass

    before = _writers()
    with pytest.raises(AttributeError):
        EventRecorder(NotAMicroscope())
    assert _writers() == before


def test_a_run_on_a_microscope_that_cannot_be_recorded_leaves_nothing_running(
    experiment,
):
    """A stand-in microscope, as tests and scripts use: the run goes ahead
    unrecorded, and leaves no writer thread behind."""

    class StandIn:
        pass

    manager = manager_module.TaskManager(microscope=StandIn(), experiment=experiment)
    before = _writers()
    with manager._recording():
        pass
    assert _writers() == before
