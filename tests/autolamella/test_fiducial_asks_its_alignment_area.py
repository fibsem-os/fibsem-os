"""Mill Fiducial asks its alignment area on the image the mill left behind
(FIB-1053): declared as a question with a switch, and sat on the milling
session's ``finished`` acquisition rather than the pre-mill reference."""

import os
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.proposals import ALIGNMENT_AREA
from fibsem.applications.autolamella.protocol.constants import FIDUCIAL_KEY
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.workflows.tasks.fiducial import (
    MillFiducialTask,
    MillFiducialTaskConfig,
)

FIDUCIAL = "Mill Fiducial"
CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


def test_the_area_is_a_question_with_a_switch():
    assert MillFiducialTask.questions == (ALIGNMENT_AREA,)
    assert MillFiducialTask.sessions == ("milling",)
    on = MillFiducialTaskConfig(task_name=FIDUCIAL)
    off = MillFiducialTaskConfig(task_name=FIDUCIAL, confirm_alignment_area=False)
    assert MillFiducialTask.questions_for(on) == (ALIGNMENT_AREA,)
    assert MillFiducialTask.questions_for(off) == ()


def test_the_question_sits_on_the_image_the_mill_left_behind(microscope, tmp_path):
    exp = Experiment(path=tmp_path, name="fid-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)
    config = MillFiducialTaskConfig(task_name=FIDUCIAL)
    exp.add_new_lamella(
        microscope.get_microscope_state(), EventedDict({FIDUCIAL: config})
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    task = MillFiducialTask(
        microscope=microscope,
        config=config,
        lamella=lamella,
        parent_ui=None,
        task_manager=None,
    )
    milling = config.milling[FIDUCIAL_KEY]

    assert task._milling_result_image_file(milling) == "", "nothing saved yet"

    milling.acquisition.imaging.filename = "Fiducial-Milling_finished_12-00-00"
    Path(lamella.path, "Fiducial-Milling_finished_12-00-00_ib.tif").touch()
    Path(lamella.path, "Fiducial-Milling_finished_12-00-00_eb.tif").touch()

    assert (
        task._milling_result_image_file(milling)
        == "Fiducial-Milling_finished_12-00-00_ib.tif"
    )
