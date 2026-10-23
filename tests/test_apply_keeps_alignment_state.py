"""A saved configuration records what the configuration decides, not the column's
alignment or the detector levels the last autocontrast left."""

import os

import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import (
    NOT_BEAM_DEFAULTS,
    NOT_IMAGING_DEFAULTS,
    MicroscopeSettings,
)

SHIPPED = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        config_path=SHIPPED, manufacturer="Demo", setup_logging=False
    )
    yield microscope
    microscope.disconnect()


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
