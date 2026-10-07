"""Every beam image records when it was acquired, as ISO 8601 with a UTC offset.

`Beam.acquire` stamps `acquisition_datetime` just before the driver runs, so every
backend gets it; older files fall back through `acquisition_datetime_of`. The legacy
values come from the real metadata fixtures in tests/fixtures/metadata.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from fibsem import config as cfg
from fibsem import utils
from fibsem.fm.structures import FluorescenceChannelMetadata, FluorescenceImageMetadata
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    ImageSettings,
    MicroscopeState,
    Point,
)
from fibsem.util.timestamps import acquisition_datetime_of, format_time, zone_known

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "metadata")


def _fixture(name: str) -> FibsemImageMetadata:
    with open(os.path.join(FIXTURE_DIR, name)) as f:
        return FibsemImageMetadata.from_dict(json.load(f))


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        config_path=cfg.DEFAULT_CONFIGURATION_PATH,
        manufacturer="Demo",
        setup_logging=False,
    )
    yield microscope
    microscope.stop_acquisition()


def _settings(**kwargs) -> ImageSettings:
    return ImageSettings(resolution=(64, 64), dwell_time=1e-9, hfw=100e-6, **kwargs)


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_a_demo_image_records_when_it_was_acquired(microscope, beam_type):
    before = datetime.now(timezone.utc)
    image = microscope.acquire_image(_settings(beam_type=beam_type))
    after = datetime.now(timezone.utc)

    recorded = image.metadata.acquisition_datetime
    assert zone_known(recorded)
    assert before <= recorded <= after


def test_acquiring_with_the_current_settings_records_it_too(microscope):
    image = microscope.acquire_image(beam_type=BeamType.ELECTRON)
    assert zone_known(image.metadata.acquisition_datetime)


def test_a_drivers_own_time_is_kept(microscope, monkeypatch):
    """A driver that reads the time off the instrument sets it; the default doesn't
    overwrite it."""
    vendor = datetime.fromisoformat("2026-07-16T11:07:27-06:00")
    original = microscope._demo_acquire

    def acquire_with_a_vendor_time(*args, **kwargs):
        image = original(*args, **kwargs)
        image.metadata.acquisition_datetime = vendor
        return image

    monkeypatch.setattr(microscope, "_demo_acquire", acquire_with_a_vendor_time)
    image = microscope.acquire_image(_settings(beam_type=BeamType.ELECTRON))
    assert image.metadata.acquisition_datetime == vendor


def test_a_saved_image_keeps_it(microscope, tmp_path):
    image = microscope.acquire_image(_settings(beam_type=BeamType.ELECTRON))
    image.save(str(tmp_path / "image.tif"))
    loaded = FibsemImage.load(str(tmp_path / "image.tif"))

    assert loaded.metadata.version == cfg.METADATA_VERSION == "v11"
    assert loaded.metadata.acquisition_datetime == image.metadata.acquisition_datetime
    assert loaded.metadata.acquisition_date == acquisition_datetime_of(image.metadata)


def test_an_image_built_rather_than_acquired_records_none():
    image = FibsemImage.generate_blank_image(resolution=(8, 8), hfw=1e-6)
    assert image.metadata.acquisition_datetime is None
    assert "acquisition_datetime" not in image.metadata.to_dict()
    assert image.metadata.acquisition_date is None


# --- older files: read through acquisition_datetime_of, never rewritten -------------


@pytest.mark.parametrize(
    "name", ["thermo_aquilos_v3.json", "thermo_arctis_compustage_v3.json"]
)
def test_a_thermofisher_image_reads_as_the_instruments_clock_time(name):
    """AutoScript's string in microscope_state.timestamp: the instrument PC's clock,
    zone unknown. `acquisition_date` raised on these before (FIB-487)."""
    metadata = _fixture(name)
    assert metadata.acquisition_datetime is None
    when = metadata.acquisition_date
    assert when == datetime.strptime(
        metadata.microscope_state.timestamp, "%m/%d/%Y %H:%M:%S"
    )
    assert when.tzinfo is None


@pytest.mark.parametrize("name", ["demo_simulator_v4.json", "demo_simulator_v5.json"])
def test_an_older_demo_image_reads_its_posix_timestamp(name):
    metadata = _fixture(name)
    when = metadata.acquisition_date
    assert when.tzinfo is not None
    assert when.timestamp() == pytest.approx(metadata.microscope_state.timestamp)


def test_the_recorded_time_wins_over_the_states():
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(),
        pixel_size=Point(1e-9, 1e-9),
        microscope_state=MicroscopeState(timestamp=0.0),
        acquisition_datetime=datetime.fromisoformat("2026-09-13T21:19:40.974286-06:00"),
    )
    expected = datetime(2026, 9, 14, 3, 19, 40, 974286, tzinfo=timezone.utc)
    assert acquisition_datetime_of(metadata) == expected


def test_a_beam_image_without_a_state_has_no_time():
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(), pixel_size=Point(), microscope_state=None
    )
    assert acquisition_datetime_of(metadata) is None


def test_a_fluorescence_image_falls_back_to_acquisition_date():
    metadata = FluorescenceImageMetadata(
        acquisition_date="2026-09-13T21:19:40.974286",
        pixel_size_x=1e-7,
        pixel_size_y=1e-7,
        channels=[
            FluorescenceChannelMetadata(
                name="c",
                excitation_wavelength=488,
                power=1,
                exposure_time=0.1,
                gain=1,
                offset=0,
            )
        ],
    )
    assert acquisition_datetime_of(metadata) == datetime(
        2026, 9, 13, 21, 19, 40, 974286
    )


def test_two_images_taken_together_read_the_same_whatever_their_zone():
    """An offset pins the instant: the same moment written in two zones compares
    equal, which a naive time cannot."""
    a = FibsemImageMetadata(
        image_settings=ImageSettings(),
        pixel_size=Point(),
        microscope_state=None,
        acquisition_datetime=datetime.fromisoformat("2026-09-13T21:19:40-06:00"),
    )
    b = FibsemImageMetadata(
        image_settings=ImageSettings(),
        pixel_size=Point(),
        microscope_state=None,
        acquisition_datetime=datetime.fromisoformat("2026-09-14T13:19:40+10:00"),
    )
    assert acquisition_datetime_of(a) == acquisition_datetime_of(b)
    assert acquisition_datetime_of(a) - acquisition_datetime_of(b) == timedelta(0)


# --- older files display exactly as they did before v11 -------------------------------


def _as_displayed_before(value) -> str:
    """What the export bar showed before FIB-1190: a POSIX time in the viewer's zone,
    a string as it was recorded."""
    if isinstance(value, float):
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M")
    try:
        return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return datetime.strptime(value, "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%d %H:%M")


@pytest.mark.parametrize(
    "name",
    [
        "demo_simulator_v4.json",
        "demo_simulator_v5.json",
        "thermo_aquilos_v3.json",
        "thermo_arctis_compustage_v3.json",
    ],
)
def test_an_older_beam_image_displays_as_it_did(name):
    metadata = _fixture(name)
    expected = _as_displayed_before(metadata.microscope_state.timestamp)
    assert format_time(acquisition_datetime_of(metadata)) == expected
    # and survives a save: nothing is added that the file did not record
    again = FibsemImageMetadata.from_dict(metadata.to_dict())
    assert again.acquisition_datetime is None
    assert again.microscope_state.timestamp == metadata.microscope_state.timestamp
