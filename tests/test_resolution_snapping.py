"""An acquire at a resolution the beam can't scan takes the nearest one it can.

Resolutions are fixed in a few places (defaults, tiling, strategies), and some
instruments (Thermo, JEOL) scan only the ones they list, so the beam snaps rather than
fail. A driver says what it can scan with ``_acquire_resolution``; the default scans
whatever is asked.
"""

import logging
import types

import pytest

from fibsem import config as cfg
from fibsem import utils
from fibsem.devices.beam import nearest_resolution
from fibsem.drivers.autoscript.devices import AutoscriptBeam
from fibsem.structures import BeamType, ImageSettings

# what a JEOL-like scan lists: 4:3 frames only
FOUR_THREE = [(640, 480), (1280, 960), (2560, 1920)]


@pytest.mark.parametrize(
    "requested, expected",
    [
        ((1280, 960), (1280, 960)),  # listed: as asked
        ((1536, 1024), (1280, 960)),
        ((3072, 2048), (2560, 1920)),
        ((384, 256), (640, 480)),
        ((1024, 1024), (1280, 960)),
    ],
)
def test_the_nearest_listed_resolution(requested, expected):
    assert nearest_resolution(requested, FOUR_THREE) == expected


def test_nothing_listed_keeps_the_request():
    assert nearest_resolution((1000, 700), None) == (1000, 700)
    assert nearest_resolution((1000, 700), []) == (1000, 700)


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        config_path=cfg.DEFAULT_CONFIGURATION_PATH,
        manufacturer="Demo",
        setup_logging=False,
    )
    yield microscope
    microscope.stop_acquisition()


def _settings(resolution):
    return ImageSettings(
        resolution=resolution,
        dwell_time=1e-6,
        hfw=150e-6,
        beam_type=BeamType.ELECTRON,
    )


def test_a_beam_that_lists_its_resolutions_snaps_and_warns(microscope, caplog):
    beam = microscope.beams[BeamType.ELECTRON]
    beam._acquire_resolution = lambda requested: nearest_resolution(
        requested, FOUR_THREE
    )
    settings = _settings((1536, 1024))
    with caplog.at_level(logging.WARNING):
        image = beam.acquire(settings)
    assert image.data.shape[:2] == (960, 1280)
    assert tuple(image.metadata.image_settings.resolution) == (1280, 960)
    assert image.metadata.pixel_size.x == pytest.approx(150e-6 / 1280)
    # the caller's settings are left as they were
    assert tuple(settings.resolution) == (1536, 1024)
    assert "1536x1024" in caplog.text and "1280x960" in caplog.text


def test_by_default_a_beam_scans_what_is_asked(microscope, caplog):
    beam = microscope.beams[BeamType.ELECTRON]
    with caplog.at_level(logging.WARNING):
        image = beam.acquire(_settings((1000, 700)))
    assert image.data.shape[:2] == (700, 1000)
    assert "nearest it can" not in caplog.text


def test_a_thermo_beam_scans_only_what_its_scan_lists():
    beam = types.SimpleNamespace(resolution=types.SimpleNamespace(choices=FOUR_THREE))
    assert AutoscriptBeam._acquire_resolution(beam, (1536, 1024)) == (1280, 960)
    assert AutoscriptBeam._acquire_resolution(beam, (640, 480)) == (640, 480)
