"""A task run writes nothing outside the experiment (FIB-1252).

Every built-in task that runs headless on Demo, through ``run_tasks``, with the
working directory and the two global data folders (``DATA_CC_PATH``,
``DATA_ML_PATH``) pointed at empty temporary folders. Those are where core code
falls back to when a caller does not say where to save, so a run that reaches
them has scattered files into the installed package or wherever the app was
started from -- silently, on the microscope. Milling alignment and acquisition
are switched on, so the saves nested inside milling (the drift alignment, the
per-stage images) are exercised too.

Inside the experiment, everything a task writes is in its lamella's folder; the
experiment folder itself holds only the experiment's own records.

Guards the explicit output folder (FIB-1253): a boundary it misses fails here.
"""

import os
from pathlib import Path
from typing import List

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks import get_tasks
from fibsem.applications.autolamella.workflows.tasks.manager import run_tasks

CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")

# Built-in tasks that cannot run here, and why. Any other built-in task runs in
# this test, so a new one is guarded from the day it is registered; one that
# cannot run headless on Demo fails below, asking to be listed here instead.
CANNOT_RUN_HERE = {
    "MILL_UNDERCUT": "needs a segmentation model",
    "SELECT_FLUORESCENCE_POSITION": "needs a fluorescence microscope",
    "ACQUIRE_FLUORESCENCE_IMAGE": "needs a fluorescence microscope",
    "SETUP_COINCIDENCE_MILLING": "needs an operator at the microscope",
    "MILL_COINCIDENT": "needs a fluorescence microscope and Setup Coincidence Milling",
}

# What the experiment writes for itself, beside the lamella folders.
EXPERIMENT_RECORDS = {"experiment.yaml", "events.jsonl", "logfile.log"}

BUILT_IN_TASKS = sorted(
    task_type
    for task_type, task_cls in get_tasks().items()
    if task_cls.__module__.startswith("fibsem.")
)


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=CONFIG, setup_logging=False
    )
    yield microscope
    microscope.disconnect()


@pytest.fixture
def elsewhere(tmp_path, monkeypatch) -> List[Path]:
    """The places a run must not write to, each empty to start with."""
    folders = [tmp_path / "cwd", tmp_path / "DATA_CC_PATH", tmp_path / "DATA_ML_PATH"]
    for folder in folders:
        folder.mkdir()
    monkeypatch.chdir(folders[0])
    monkeypatch.setattr(cfg, "DATA_CC_PATH", str(folders[1]))
    monkeypatch.setattr(cfg, "DATA_ML_PATH", str(folders[2]))
    return folders


def _experiment(microscope, path: Path, task_type: str) -> Experiment:
    config_cls = get_tasks()[task_type].config_cls
    name = config_cls.display_name
    config = config_cls(task_name=name)
    for milling in config.milling.values():
        milling.alignment.enabled = True
        milling.acquisition.acquire_sem = True
        milling.acquisition.acquire_fib = True

    exp = Experiment(path=path, name=task_type)
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[AutoLamellaTaskDescription(name=name, required=True)]
        )
    )
    exp.add_new_lamella(microscope.get_microscope_state(), EventedDict({name: config}))
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    return exp


def _files(folder: Path) -> List[str]:
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file())


@pytest.mark.parametrize("task_type", BUILT_IN_TASKS)
def test_a_task_run_writes_only_into_its_lamella_folder(
    microscope, elsewhere, tmp_path, task_type
):
    if task_type in CANNOT_RUN_HERE:
        pytest.skip(CANNOT_RUN_HERE[task_type])
    exp = _experiment(microscope, tmp_path / "experiment", task_type)
    lamella = exp.positions[0]

    run_tasks(microscope, exp, [exp.task_protocol.workflow_config.tasks[0].name])

    # a task that stopped early wrote nothing, and would pass for that reason
    assert lamella.task_state.status is AutoLamellaTaskStatus.Completed, (
        f"{task_type} did not run headless on Demo "
        f"({lamella.task_state.status_message!r}); "
        "if it cannot, list it in CANNOT_RUN_HERE with the reason"
    )
    assert _files(lamella.path), f"{task_type} wrote nothing, so this proves nothing"
    for folder in elsewhere:
        assert _files(folder) == [], f"{task_type} wrote into {folder.name}"
    outside_lamella = [
        f
        for f in _files(Path(exp.path))
        if not (Path(exp.path) / f).is_relative_to(lamella.path)
    ]
    assert set(outside_lamella) <= EXPERIMENT_RECORDS, (
        f"{task_type} wrote into the experiment folder, outside its lamella"
    )
