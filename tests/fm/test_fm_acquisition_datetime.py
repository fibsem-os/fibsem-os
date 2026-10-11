"""Every fluorescence image records when it was acquired, with its UTC offset.

`FM.acquire_frame` stamps the frame as the acquisition starts, as `Beam.acquire` does
for a beam image, and `acquisition_datetime` carries it into the image; the legacy
`acquisition_date` keeps the same value for OME and odemis. Older images, whose
`acquisition_date` is the acquiring machine's clock time alone, still load and read
through `acquisition_datetime_of` without being given a zone they never had.
"""

import os
from datetime import datetime, timezone

import pytest

from fibsem.devices.wire import Frame
from fibsem.drivers.demo.microscope import DemoFluorescenceMicroscope
from fibsem.fm.structures import (
    ChannelSettings,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.util.timestamps import acquisition_datetime_of, format_time, zone_known

LEGACY = os.path.join(os.path.dirname(__file__), "..", "fixtures", "fm_legacy")


@pytest.fixture()
def fm():
    return DemoFluorescenceMicroscope()


def _channel() -> ChannelSettings:
    return ChannelSettings(
        name="GFP", excitation_wavelength=488, emission_wavelength=None, power=0.1
    )


def test_an_acquired_image_records_when_with_its_offset(fm):
    before = datetime.now(timezone.utc)
    image = fm.acquire_image(_channel())
    after = datetime.now(timezone.utc)

    md = image.metadata
    assert zone_known(md.acquisition_datetime)
    assert before <= md.acquisition_datetime <= after
    # The legacy field, still written, with the same value.
    assert md.acquisition_date == md.acquisition_datetime.isoformat()


def test_a_frame_without_an_offset_gets_the_time_the_acquisition_started(fm):
    """A frame from a driver or server that writes the old naive string: its zone is
    unknown, so the time acquire_frame took is recorded instead."""
    device = fm.devices["fm"]
    original = device._acquire_frame

    def naive(channel):
        frame = original(channel)
        frame.metadata["acquisition_date"] = "2026-07-22T10:00:00"
        return frame

    device._acquire_frame = naive
    frame = device.acquire_frame(_channel().to_dict())
    assert isinstance(frame, Frame)
    assert zone_known(frame.metadata["acquisition_date"])


def test_a_drivers_own_time_with_an_offset_is_kept(fm):
    device = fm.devices["fm"]
    original = device._acquire_frame
    vendor = "2026-07-22T10:00:00-06:00"

    def stamped(channel):
        frame = original(channel)
        frame.metadata["acquisition_date"] = vendor
        return frame

    device._acquire_frame = stamped
    image = fm.acquire_image(_channel())
    assert image.metadata.acquisition_datetime == datetime.fromisoformat(vendor)
    assert image.metadata.acquisition_datetime.utcoffset().total_seconds() == -6 * 3600


def test_it_survives_a_save_and_load(fm, tmp_path):
    image = fm.acquire_image(_channel())
    path = str(tmp_path / "image.ome.tiff")
    image.save(path)
    loaded = FluorescenceImage.load(path)
    assert loaded.metadata.acquisition_datetime == image.metadata.acquisition_datetime


def test_a_projection_keeps_the_time_of_what_it_projects():
    image = FluorescenceImage.generate_blank_image(resolution=(8, 8), zlevels=3)
    image.metadata.acquisition_datetime = datetime.fromisoformat(
        "2026-09-13T21:19:40.974286-06:00"
    )
    projected = image.max_intensity_projection()
    assert (
        projected.metadata.acquisition_datetime == image.metadata.acquisition_datetime
    )


def test_ome_records_the_instant(fm, tmp_path):
    """Read back through the file's own OME, as software other than fibsemOS would."""
    image = fm.acquire_image(_channel())
    path = str(tmp_path / "image.ome.tiff")
    image.save(path)
    from fibsem.fm.structures import safe_ome_from_tiff

    from_ome = FluorescenceImageMetadata.from_ome(safe_ome_from_tiff(path))
    assert from_ome.acquisition_datetime == image.metadata.acquisition_datetime


@pytest.mark.parametrize("name", ["c2z2.ome.tiff", "tile.ome.tiff"])
def test_an_older_image_keeps_its_clock_time_and_no_zone(name):
    image = FluorescenceImage.load(os.path.join(LEGACY, name))
    md = image.metadata
    assert md.acquisition_datetime is None
    when = acquisition_datetime_of(md)
    assert when is not None and when.tzinfo is None
    assert when == datetime.fromisoformat(md.acquisition_date)
    # Displayed as it was before: the acquiring machine's clock, unmoved.
    expected = datetime.fromisoformat(md.acquisition_date).strftime("%Y-%m-%d %H:%M")
    assert format_time(when) == expected
    # and survives a save without gaining a zone it never had
    again = FluorescenceImageMetadata.from_dict(md.to_dict())
    assert again.acquisition_datetime is None
    assert again.acquisition_date == md.acquisition_date


def test_an_image_built_rather_than_acquired_has_no_acquisition_time():
    image = FluorescenceImage.generate_blank_image(resolution=(8, 8))
    assert image.metadata.acquisition_datetime is None
    assert zone_known(image.metadata.acquisition_date)
