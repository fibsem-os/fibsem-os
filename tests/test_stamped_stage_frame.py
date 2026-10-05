"""Images stamp the frame their stage position is in (FIB-1114).

Tescan's stage still reports Tescan's own frame; every other stage reports fibsem's. An
image records which, so that when TescanStage converts to fibsem's frame, an image
taken before and one taken after can be told apart. Nothing reads the stamp yet.
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


def _tescan(monkeypatch):
    system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
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


def test_a_tescan_stage_stamps_its_own_frame(monkeypatch):
    microscope, _ = _tescan(monkeypatch)
    assert microscope.stage.frame == STAGE_FRAME_TESCAN
    assert microscope.hardware_geometry().stage_frame == STAGE_FRAME_TESCAN


def test_a_tescan_image_round_trips_its_frame(monkeypatch):
    microscope, fake = _tescan(monkeypatch)
    image = image_at(microscope, fake, BeamType.ELECTRON)
    loaded = FibsemImageMetadata.from_dict(image.metadata.to_dict())

    assert loaded.hardware_geometry.stage_frame == STAGE_FRAME_TESCAN


def test_a_file_without_the_stamp_loads_with_none():
    ddict = FibsemHardwareGeometry(stage_frame=STAGE_FRAME_FIBSEM).to_dict()
    del ddict["stage_frame"]

    assert FibsemHardwareGeometry.from_dict(ddict).stage_frame is None
