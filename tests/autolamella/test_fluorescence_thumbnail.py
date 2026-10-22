"""A lamella with only a fluorescence stack shows its projection on its card."""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (
    AcquireFluorescenceImageTask,
)
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)


@pytest.fixture
def lamella(tmp_path):
    from fibsem.applications.autolamella.structures import Lamella

    folder = tmp_path / "01-lamella"
    folder.mkdir()
    return Lamella(path=folder, number=1, petname="01-lamella")


def _stack() -> FluorescenceImage:
    data = np.zeros((1, 3, 32, 32), dtype=np.uint16)
    data[0, 1, 8:24, 8:24] = 4000
    channel = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
        color="green",
    )
    metadata = FluorescenceImageMetadata(
        acquisition_datetime="2026-01-01T00:00:00",
        pixel_size_x=1e-7,
        pixel_size_y=1e-7,
        resolution=(32, 32),
        channels=[channel],
    )
    return FluorescenceImage(data=data, metadata=metadata)


def _task(lamella) -> AcquireFluorescenceImageTask:
    task = AcquireFluorescenceImageTask.__new__(AcquireFluorescenceImageTask)
    task.lamella = lamella
    return task


def test_the_first_stack_becomes_the_thumbnail(lamella):
    from PIL import Image

    _task(lamella)._thumbnail_if_none(_stack())

    thumb = np.asarray(Image.open(os.path.join(lamella.path, "thumbnail.png")))
    assert thumb[16, 16, 1] > 200 and thumb[16, 16, 0] < 50  # green, as drawn


def test_a_beam_thumbnail_is_kept(lamella):
    path = os.path.join(lamella.path, "thumbnail.png")
    with open(path, "wb") as f:
        f.write(b"beam")

    _task(lamella)._thumbnail_if_none(_stack())

    with open(path, "rb") as f:
        assert f.read() == b"beam"


def test_a_thumbnail_that_cannot_be_written_does_not_fail_the_task(lamella):
    lamella_mock = MagicMock(path=str(lamella.path))
    lamella_mock.save_thumbnail.side_effect = OSError("disk full")

    _task(lamella_mock)._thumbnail_if_none(_stack())  # logged, not raised
