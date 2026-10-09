"""A mill the microscope rejects fails the AutoLamella task (FIB-1112).

On an Aquilos every pattern call was rejected, each milling stage logged the
error and moved on, and a headless run recorded Rough Milling Completed with
nothing milled -- which also satisfies Polishing's prerequisite. Real task
through ``run_tasks`` on Demo; only the microscope's mill is replaced.
"""

import os

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
from fibsem.applications.autolamella.workflows.tasks.manager import run_tasks
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig

ROUGH = "Rough Milling"
CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


@pytest.fixture
def experiment(microscope, tmp_path):
    exp = Experiment(path=tmp_path, name="failed-mill")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[AutoLamellaTaskDescription(name=ROUGH, required=True)]
        )
    )
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        EventedDict({ROUGH: MillRoughTaskConfig(task_name=ROUGH)}),
    )
    exp.positions[0].path.mkdir(parents=True, exist_ok=True)
    exp.positions[0].milling_pose = microscope.get_microscope_state()
    return exp


def test_a_rejected_mill_fails_the_task(microscope, experiment):
    milled = []

    def run_milling(*args, **kwargs):
        milled.append(args)
        raise RuntimeError("could not load type")

    microscope.run_milling = run_milling

    run_tasks(microscope, experiment, [ROUGH])

    lamella = experiment.positions[0]
    assert len(milled) == 1, "the stages after the rejected one do not mill"
    assert lamella.task_state.status is AutoLamellaTaskStatus.Failed
    assert ROUGH not in lamella.completed_tasks
