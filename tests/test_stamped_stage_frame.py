"""Images stamp the frame their stage position is in (FIB-1114).

Every stage device reports fibsem's frame, Tescan's included since TescanStage converts
(FIB-1114). An image records it, so a Tescan image from before the conversion, which
has no stamp, can be told apart from one after.
"""

import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import (
    STAGE_FRAME_FIBSEM,
    STAGE_FRAME_TESCAN,
    BeamType,
    FibsemHardwareGeometry,
    FibsemImageMetadata,
)
from tests.fixtures.tescan_sdk import connect, image_at


def _tescan(monkeypatch, stage=True):
    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    system.stage.enabled = stage
    return connect(monkeypatch, system)


@pytest.mark.parametrize(
    "filename",
    [
        "microscope-configuration.yaml",
        "tescan-configuration.yaml",
        "sim-arctis-configuration.yaml",
    ],
)
def test_a_demo_stage_stamps_fibsem(filename):
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.CONFIG_PATH, filename), manufacturer="Demo"
    )
    assert microscope.hardware_geometry().stage_frame == STAGE_FRAME_FIBSEM


def test_a_tescan_stage_stamps_fibsem(monkeypatch):
    microscope, _ = _tescan(monkeypatch)
    assert microscope.hardware_geometry().stage_frame == STAGE_FRAME_FIBSEM


def test_without_a_stage_device_tescan_stamps_its_own_frame(monkeypatch):
    microscope, _ = _tescan(monkeypatch, stage=False)
    assert microscope.hardware_geometry().stage_frame == STAGE_FRAME_TESCAN


def test_a_tescan_image_records_its_stage_in_fibsem_frame(monkeypatch):
    microscope, fake = _tescan(monkeypatch)
    fake.Stage.position = [1.2, -0.8, 29.0, 180.0, 30.0]
    image = image_at(microscope, fake, BeamType.ELECTRON)
    loaded = FibsemImageMetadata.from_dict(image.metadata.to_dict())

    assert loaded.hardware_geometry.stage_frame == STAGE_FRAME_FIBSEM
    stamped = loaded.microscope_state.stage_position
    live = microscope.get_stage_position()
    for axis in "xyzrt":
        assert getattr(stamped, axis) == pytest.approx(getattr(live, axis))


def test_a_file_without_the_stamp_loads_with_none():
    ddict = FibsemHardwareGeometry(stage_frame=STAGE_FRAME_FIBSEM).to_dict()
    del ddict["stage_frame"]

    assert FibsemHardwareGeometry.from_dict(ddict).stage_frame is None
