"""A task's operations write where its output folder says (FIB-1253).

``AutoLamellaTask.output_dir`` is the one place a task decides where its
alignment runs, autofocus runs and milling are saved. Pointed somewhere other
than the lamella's folder, they all follow it: real Setup and Rough Milling
runs on Demo, through ``run_tasks``. Reference images are still placed by hand
in each task and stay in the lamella's folder until that moves too.

The alignment and autofocus wrappers also return what they measured, instead of
dropping it.
"""

import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.alignment import AlignmentResult
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.base import (
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
    AutoLamellaTask,
)
from fibsem.applications.autolamella.workflows.tasks.manager import run_tasks
from fibsem.applications.autolamella.workflows.tasks.rough import (
    MillRoughTask,
    MillRoughTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTaskConfig,
)
from fibsem.autofunctions.autofocus import AutoFocusResult
from fibsem.structures import BeamType

CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")
SETUP = "Setup Lamella Position"
ROUGH = "Rough Milling"


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=CONFIG, setup_logging=False
    )
    yield microscope
    microscope.disconnect()


@pytest.fixture
def experiment(microscope, tmp_path):
    rough = MillRoughTaskConfig(task_name=ROUGH)
    for milling in rough.milling.values():
        milling.alignment.enabled = True
    exp = Experiment(path=tmp_path / "experiment", name="output-dir")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(name=SETUP, required=True),
                AutoLamellaTaskDescription(name=ROUGH, required=True),
            ]
        )
    )
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        EventedDict(
            {
                SETUP: SelectMillingPositionTaskConfig(
                    task_name=SETUP, use_autofocus=True
                ),
                ROUGH: rough,
            }
        ),
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    return exp


def _top_level(folder: Path):
    return {p.name for p in folder.iterdir()} if folder.exists() else set()


def test_alignment_autofocus_and_milling_follow_the_output_folder(
    microscope, experiment, tmp_path, monkeypatch
):
    elsewhere = tmp_path / "elsewhere"
    monkeypatch.setattr(
        AutoLamellaTask, "output_dir", property(lambda _: str(elsewhere))
    )
    lamella = experiment.positions[0]

    run_tasks(microscope, experiment, [SETUP, ROUGH])

    assert lamella.task_state.status is AutoLamellaTaskStatus.Completed
    assert {"Alignment", "autofunctions", "Milling"} <= _top_level(elsewhere)
    in_lamella = _top_level(lamella.path)
    assert not {"Alignment", "autofunctions", "Milling"} & in_lamella, in_lamella
    # the reference both tasks share is read from the lamella's folder
    assert ALIGNMENT_REFERENCE_IMAGE_FILENAME in in_lamella


def test_the_output_folder_is_the_lamella_folder_for_now(microscope, experiment):
    lamella = experiment.positions[0]
    task = MillRoughTask(microscope, lamella.task_config[ROUGH], lamella)

    assert task.output_dir == str(lamella.path)


def test_the_wrappers_return_what_they_measured(microscope, experiment):
    lamella = experiment.positions[0]
    run_tasks(microscope, experiment, [SETUP])  # leaves the alignment reference
    task = MillRoughTask(microscope, lamella.task_config[ROUGH], lamella)

    assert isinstance(
        task._align_reference_image(ALIGNMENT_REFERENCE_IMAGE_FILENAME), AlignmentResult
    )
    assert task._align_reference_image("no-such-reference") is None
    assert isinstance(task._run_autofocus(BeamType.ELECTRON), AutoFocusResult)
