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
from fibsem.applications.autolamella.workflows.tasks.base import (
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
)
from fibsem.applications.autolamella.workflows.tasks.fiducial import (
    MillFiducialTaskConfig,
    fiducial_point_in_alignment_area,
    follow_alignment_area,
)
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTask,
    SelectMillingPositionTaskConfig,
)
from fibsem.milling.patterning.utils import get_patterns_bounding_box
from fibsem.structures import FibsemImage, FibsemRectangle, Point
from tests.autolamella._on_the_record import manager_for
from tests.protocols import planned, task_protocol

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
        task_manager=manager_for(microscope, lamella),
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


@pytest.mark.parametrize("numpy_area", [False, True])
def test_setup_again_after_the_fiducial_saves_and_reloads(
    microscope, tmp_path, monkeypatch, numpy_area
):
    """The FIB-1326 sequence, through the task manager on the Demo
    microscope: Setup, Mill Fiducial (which re-records the alignment area
    from the milled pattern's bounding box, numpy min/max), then Setup again
    with the sync on and the area accepted unchanged; the experiment then
    saves and reloads. On main the second Setup's fiducial point came out as
    np.float64 and the save failed. ``numpy_area`` also starts from an area
    assigned numpy values after it is built, which the dataclass cannot
    convert, so the YAML writer has to cope."""
    import numpy as np

    from fibsem.applications.autolamella.structures import (
        Experiment,
    )
    from fibsem.applications.autolamella.workflows.tasks.fiducial import (
        MillFiducialTask,
    )
    from tests.run_state import run_lamellae

    exp = Experiment(path=tmp_path, name="fid-save")
    exp.task_protocol = task_protocol(
        planned(name=SETUP, required=True), planned(name=FIDUCIAL, required=True)
    )
    os.makedirs(exp.path, exist_ok=True)
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        EventedDict(
            {
                SETUP: SelectMillingPositionTaskConfig(
                    task_name=SETUP, use_autofocus=False, select_poi=False
                ),
                FIDUCIAL: MillFiducialTaskConfig(task_name=FIDUCIAL),
            }
        ),
    )
    lamella = exp.positions[0]
    lamella.path.mkdir(parents=True, exist_ok=True)
    lamella.milling_pose = microscope.get_microscope_state()
    if numpy_area:
        area = FibsemRectangle()
        area.left, area.top, area.width, area.height = (
            np.float64(v) for v in (AREA.left, AREA.top, AREA.width, AREA.height)
        )
        lamella.alignment_area = area
    # the mill itself is not the point: the session hands back its config
    monkeypatch.setattr(
        MillFiducialTask, "update_milling_config_ui", lambda self, cfg, msg: cfg
    )

    run_lamellae(microscope, exp, [SETUP, FIDUCIAL])
    run_lamellae(microscope, exp, [SETUP])

    assert lamella.has_completed_task(SETUP)
    assert lamella.has_completed_task(FIDUCIAL)
    point = lamella.task_config[FIDUCIAL].milling[FIDUCIAL_KEY].stages[0].pattern.point
    exp.save()
    loaded = Experiment.load(os.path.join(exp.path, "experiment.yaml"))
    reloaded_lamella = loaded.positions[0]
    reloaded = (
        reloaded_lamella.task_config[FIDUCIAL].milling[FIDUCIAL_KEY].stages[0]
    ).pattern.point
    assert reloaded.x == pytest.approx(float(point.x))
    assert reloaded.y == pytest.approx(float(point.y))
    assert reloaded_lamella.alignment_area.left == pytest.approx(
        float(lamella.alignment_area.left)
    )


def _save_alignment_reference(microscope, lamella, hfw=100e-6):
    """The alignment reference Setup leaves in the lamella's folder, cropped to
    the area, as Setup takes it."""
    from fibsem import acquire
    from fibsem.structures import BeamType, ImageSettings

    acquire.acquire_image(
        microscope,
        ImageSettings(
            hfw=hfw,
            resolution=(1536, 1024),
            beam_type=BeamType.ION,
            reduced_area=lamella.alignment_area,
            save=True,
            path=str(lamella.path),
            filename="ref_alignment",
        ),
    )
    assert os.path.isfile(
        os.path.join(lamella.path, ALIGNMENT_REFERENCE_IMAGE_FILENAME)
    )


def test_an_area_edit_after_setup_moves_the_fiducial(microscope, tmp_path):
    """FIB-1430: the fiducial follows the area to its new centre, in the full
    frame the area is a fraction of."""
    lamella = _lamella(tmp_path, microscope, AREA)
    _save_alignment_reference(microscope, lamella)
    lamella.alignment_area = FibsemRectangle(
        left=0.66, top=0.12, width=0.28, height=0.35
    )

    assert follow_alignment_area(lamella) == [FIDUCIAL]

    point = lamella.task_config[FIDUCIAL].milling[FIDUCIAL_KEY].stages[0].pattern.point
    assert point.x == pytest.approx((0.80 - 0.5) * 100e-6)
    assert point.y == pytest.approx((0.5 - 0.295) * 100e-6 * 1024 / 1536)


def test_before_setup_or_once_milled_the_fiducial_stays(microscope, tmp_path):
    from fibsem.workflows.status import TaskState, TaskStatus

    lamella = _lamella(tmp_path, microscope, AREA)
    stage = lamella.task_config[FIDUCIAL].milling[FIDUCIAL_KEY].stages[0]
    before = stage.pattern.point
    assert follow_alignment_area(lamella) == []  # no reference: Setup syncs

    _save_alignment_reference(microscope, lamella)
    lamella.task_history.append(TaskState(name=FIDUCIAL, status=TaskStatus.Completed))
    assert follow_alignment_area(lamella) == []  # on the sample already
    assert stage.pattern.point == before
