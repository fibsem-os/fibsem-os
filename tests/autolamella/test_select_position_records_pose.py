"""SelectMillingPositionTask records the milling pose and the alignment
reference whether or not coincidence alignment is on.

Regression: the task's tail -- the position confirmation, the point of
interest, the alignment area, the reference images and the milling pose --
had ended up inside the coincidence helper, which only runs with
``auto_milling_alignment`` on and returns early when the stage is already at
the milling angle. With the default configuration the task moved, imaged,
and recorded nothing.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks.base import (
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
)
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
)

CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


@pytest.mark.parametrize("auto_milling_alignment", [False, True])
def test_the_task_records_pose_and_reference_either_way(
    microscope, tmp_path, auto_milling_alignment
):
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    assert lamella.milling_angle is None
    config = SelectMillingPositionTaskConfig(
        auto_milling_alignment=auto_milling_alignment,
        use_autofocus=False,
        select_poi=True,
    )
    task = SelectMillingPositionTask(
        microscope=microscope, config=config, lamella=lamella
    )

    task._run()

    assert lamella.milling_angle is not None, "milling pose and angle were stored"
    assert os.path.exists(
        os.path.join(lamella.path, ALIGNMENT_REFERENCE_IMAGE_FILENAME)
    ), "the alignment reference was acquired"
