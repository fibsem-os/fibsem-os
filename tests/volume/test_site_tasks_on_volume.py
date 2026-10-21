"""The shared site tasks run on a volume position: added from volume's own
registry, run once on a position before its slices."""

import os

from fibsem.applications.volume.manager import VolumeTaskManager
from fibsem.applications.volume.tasks import get_volume_tasks
from fibsem.workflows.attention import Attention
from fibsem.workflows.site_tasks import BUILTIN_SITE_TASKS
from fibsem.workflows.status import TaskStatus

from .conftest import set_up_position

REFERENCE = "Reference Image"
MILL = "Volume FIB Milling"
IMAGE = "Volume SEM Acquisition"


def test_volume_offers_every_site_task():
    assert set(BUILTIN_SITE_TASKS) <= set(get_volume_tasks())
    assert {"ACQUIRE_REFERENCE_IMAGE", "MILL", "SPOT_BURN_FIDUCIAL"} <= set(
        BUILTIN_SITE_TASKS
    )


def test_a_site_task_runs_once_before_the_slices(microscope, experiment):
    position = set_up_position(experiment, microscope, "aspen", iterations=2)
    config = experiment.add_task("ACQUIRE_REFERENCE_IMAGE", REFERENCE)
    config.attention = Attention.automated

    VolumeTaskManager(microscope=microscope, experiment=experiment).execute(
        [REFERENCE, MILL, IMAGE]
    )

    done = TaskStatus.Completed
    assert [(s.name, s.iteration, s.status) for s in position.task_history] == [
        (REFERENCE, None, done),
        (MILL, 0, done),
        (IMAGE, 0, done),
        (MILL, 1, done),
        (IMAGE, 1, done),
    ]
    outputs = position.task_history[0].outputs["final_fib"]
    folder = experiment.position_path(position)
    assert outputs and all(os.path.exists(folder / p) for p in outputs)
