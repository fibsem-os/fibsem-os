"""Apply sets what the configuration decides, not the column's alignment.

The working distance, stigmation and beam shift are the column's current alignment,
and the detector's brightness and contrast are what the last autocontrast left. A
configuration that did not state them loaded them as 0 -- the working distance as the
eucentric height -- and Apply pushed those: refocusing both columns, discarding the
beam shift and blacking out the detectors. A saved configuration stated them too.
"""

import os

import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import (
    NOT_BEAM_DEFAULTS,
    NOT_IMAGING_DEFAULTS,
    BeamType,
    MicroscopeSettings,
    Point,
)

SHIPPED = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        config_path=SHIPPED, manufacturer="Demo", setup_logging=False
    )
    yield microscope
    microscope.disconnect()


@pytest.mark.parametrize("beam_type", [BeamType.ELECTRON, BeamType.ION])
def test_apply_leaves_the_alignment_and_the_detector_levels(microscope, beam_type):
    microscope.set_working_distance(0.0123, beam_type)
    microscope.set_stigmation(Point(1e-3, 2e-3), beam_type)
    microscope.set_beam_shift(Point(1e-6, -2e-6), beam_type)
    microscope.set("detector_brightness", 0.4, beam_type)
    microscope.set("detector_contrast", 0.6, beam_type)

    microscope.apply_configuration()

    assert microscope.get_working_distance(beam_type) == pytest.approx(0.0123)
    assert microscope.get_stigmation(beam_type) == Point(1e-3, 2e-3)
    assert microscope.get_beam_shift(beam_type) == Point(1e-6, -2e-6)
    detector = microscope.get_detector_settings(beam_type)
    assert detector.brightness == pytest.approx(0.4)
    assert detector.contrast == pytest.approx(0.6)


def test_apply_still_sets_the_defaults(microscope):
    microscope.set_beam_voltage(5000, BeamType.ELECTRON)
    microscope.set_detector_mode("BackscatterElectrons", BeamType.ELECTRON)

    microscope.apply_configuration()

    configured = microscope.system.electron
    assert microscope.get_beam_voltage(BeamType.ELECTRON) == configured.beam.voltage
    assert (
        microscope.get_detector_settings(BeamType.ELECTRON).mode
        == configured.detector.mode
    )


def test_a_saved_configuration_does_not_record_alignment_or_session_state():
    written = utils.load_microscope_configuration(SHIPPED).to_dict()

    for beam in ("electron", "ion"):
        assert not set(NOT_BEAM_DEFAULTS) & set(written["defaults"][beam])
        assert not set(NOT_BEAM_DEFAULTS) & set(
            utils.configuration_device(written, beam)
        )
    assert not set(NOT_IMAGING_DEFAULTS) & set(written["defaults"]["imaging"])


def test_a_file_that_states_them_still_loads_and_they_are_reported(tmp_path):
    """Files saved before this carry the keys. They load; the keys are ignored."""
    config = utils.load_yaml(SHIPPED)
    config["defaults"]["electron"]["shift"] = {"x": 1e-6, "y": 0.0}
    config["defaults"]["electron"]["detector_brightness"] = 0.0
    config["defaults"]["imaging"]["path"] = "/somewhere/else"
    path = tmp_path / "saved-before.yaml"
    path.write_text(yaml.safe_dump(config))

    settings = utils.load_microscope_configuration(str(path))

    assert isinstance(settings, MicroscopeSettings)
    unread = utils.unrecognised_configuration_keys(config)
    assert "defaults.electron.shift" in unread
    assert "defaults.imaging.path" in unread
