"""Setup starts the fiducial at the centre of the alignment area it chose
(FIB-1296), so a lamella whose point of interest is off-centre does not get a
fiducial on its trenches. A fiducial that does not fit in the area keeps its
configured position, and the switch turns the sync off."""

import os

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.protocol.constants import (
    FIDUCIAL_KEY,
    MILL_ROUGH_KEY,
)
from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.workflows.tasks.fiducial import (
    MillFiducialTaskConfig,
    fiducial_point_in_alignment_area,
)
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
)
from fibsem.milling.patterning.utils import get_patterns_bounding_box
from fibsem.structures import FibsemImage, FibsemRectangle, Point

CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")
SETUP, FIDUCIAL, ROUGH = "Setup", "Mill Fiducial", "Rough Milling"

# 01-alive-dory's area on the FIB-1035 test run, away from an off-centre POI
AREA = FibsemRectangle(left=0.66, top=0.22, width=0.28, height=0.25)
POI = Point(x=27.5e-6, y=-9.6e-6)


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


def test_the_point_is_the_area_centre_in_the_image_frame():
    """Fractions of the image the area was chosen on, at that image's field
    of view: the worked example from the issue, 100 um at 1536 x 1024."""
    image = FibsemImage.generate_blank_image(resolution=(1536, 1024), hfw=100e-6)
    milling = MillFiducialTaskConfig().milling[FIDUCIAL_KEY]

    point = fiducial_point_in_alignment_area(milling, AREA, image)

    assert point is not None
    assert point.x == pytest.approx(30.0e-6, abs=0.05e-6)
    assert point.y == pytest.approx(10.3e-6, abs=0.05e-6)


def test_a_fiducial_that_does_not_fit_has_no_point():
    image = FibsemImage.generate_blank_image(resolution=(1536, 1024), hfw=100e-6)
    milling = MillFiducialTaskConfig().milling[FIDUCIAL_KEY]
    small = FibsemRectangle(left=0.7, top=0.3, width=0.03, height=0.03)

    assert fiducial_point_in_alignment_area(milling, small, image) is None


def _lamella(tmp_path, microscope, area: FibsemRectangle) -> Lamella:
    lamella = Lamella(path=tmp_path / "lam", number=0, petname="test")
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    lamella.task_config = EventedDict(
        {
            FIDUCIAL: MillFiducialTaskConfig(task_name=FIDUCIAL),
            ROUGH: MillRoughTaskConfig(task_name=ROUGH),
        }
    )
    lamella.poi = POI
    lamella.sync_tasks_to_poi()
    lamella.alignment_area = area
    return lamella


def _run_setup(microscope, lamella, **config) -> SelectMillingPositionTask:
    task = SelectMillingPositionTask(
        microscope=microscope,
        config=SelectMillingPositionTaskConfig(
            task_name=SETUP, use_autofocus=False, select_poi=False, **config
        ),
        lamella=lamella,
    )
    task._run()
    return task


def test_setup_moves_the_fiducial_off_the_trenches(microscope, tmp_path):
    lamella = _lamella(tmp_path, microscope, AREA)
    fiducial = lamella.task_config[FIDUCIAL].milling[FIDUCIAL_KEY]
    rough = lamella.task_config[ROUGH].milling[MILL_ROUGH_KEY]

    task = _run_setup(microscope, lamella)

    image = task._last_fib_image
    height, width = image.data.shape[:2]
    pixelsize = image.metadata.pixel_size.x
    point = fiducial.stages[0].pattern.point
    assert point.x == pytest.approx((0.80 - 0.5) * width * pixelsize)
    assert point.y == pytest.approx((0.5 - 0.345) * height * pixelsize)

    fx0, fy0, fx1, fy1 = get_patterns_bounding_box(
        [s.pattern for s in fiducial.stages], image
    )
    rx0, ry0, rx1, ry1 = get_patterns_bounding_box(
        [s.pattern for s in rough.stages], image
    )
    assert fx1 < rx0 or rx1 < fx0 or fy1 < ry0 or ry1 < fy0, "no overlap"


def test_the_switch_and_a_small_area_keep_the_configured_position(microscope, tmp_path):
    configured = MillFiducialTaskConfig().milling[FIDUCIAL_KEY].stages[0].pattern.point

    lamella = _lamella(tmp_path / "off", microscope, AREA)
    _run_setup(microscope, lamella, sync_fiducial_position=False)
    assert (
        lamella.task_config[FIDUCIAL].milling[FIDUCIAL_KEY].stages[0].pattern.point
        == configured
    )

    small = FibsemRectangle(left=0.7, top=0.3, width=0.03, height=0.03)
    lamella = _lamella(tmp_path / "small", microscope, small)
    _run_setup(microscope, lamella)
    assert (
        lamella.task_config[FIDUCIAL].milling[FIDUCIAL_KEY].stages[0].pattern.point
        == configured
    )
    assert SelectMillingPositionTaskConfig().sync_fiducial_position is True
