"""The objective's calibration lives in the microscope configuration.

`focus_position` and `limit_position` -- where this objective is in focus, and how far
it may safely be inserted -- lived only in `fm-configuration.yaml`, which a one-second
debounced autosave rewrites after any channel edit. A safety limit carried by a file
that turns over while someone adjusts colours.

Now they are `calibration.objective` in the microscope configuration, written by one
action ("Save Focus and Limit as Calibration") and read at every connect. A site that
has not pressed it sees no change at all: the configuration states nothing, and the
working-state file answers exactly as before.
"""

import copy
import os

import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import MicroscopeSettings, SystemSettings

IFLM = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")


def _iflm_with_objective(tmp_path, focus, limit) -> str:
    config = utils.load_yaml(IFLM)
    config["calibration"]["objective"] = {
        "focus_position": focus,
        "limit_position": limit,
    }
    path = tmp_path / "iflm.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


# ---------------------------------------------------------------------------
# The record and the file
# ---------------------------------------------------------------------------


def test_an_unstated_calibration_is_none_not_a_number():
    """A default would silently become the answer on every site that has not
    pressed Save -- and for `limit_position` that moves where the objective may stop."""
    fm = SystemSettings.from_dict({}).fm
    assert fm.focus_position is None
    assert fm.limit_position is None


def test_it_is_written_under_calibration_and_round_trips():
    system = SystemSettings.from_dict(
        {
            "calibration": {
                "objective": {"focus_position": 7.6e-3, "limit_position": 8.6e-3}
            }
        }
    )
    assert system.fm.focus_position == 7.6e-3
    assert system.fm.limit_position == 8.6e-3

    written = system.to_dict()
    assert written["calibration"]["objective"] == {
        "focus_position": 7.6e-3,
        "limit_position": 8.6e-3,
    }
    assert "focus_position" not in written["hardware"]["fm"]
    assert SystemSettings.from_dict(copy.deepcopy(written)).fm == system.fm


def test_the_keys_are_known_to_the_schema():
    assert (
        utils.unrecognised_configuration_keys(
            {
                "calibration": {
                    "objective": {"focus_position": 1.0, "limit_position": 2.0}
                }
            }
        )
        == []
    )


# ---------------------------------------------------------------------------
# Connect applies it; silence leaves the objective alone
# ---------------------------------------------------------------------------


def test_connect_applies_the_configured_calibration(tmp_path):
    path = _iflm_with_objective(tmp_path, focus=7.1e-3, limit=8.2e-3)
    microscope, _ = utils.setup_session(config_path=path, manufacturer="Demo")
    try:
        assert microscope.fm is not None
        assert microscope.fm.objective.focus_position == pytest.approx(7.1e-3)
        assert microscope.fm.objective.limit_position == pytest.approx(8.2e-3)
        assert microscope.configuration_path == path
    finally:
        microscope.disconnect()


def test_an_unstated_calibration_leaves_the_objective_alone():
    """Set a baseline *after* connect, then apply a silent configuration.

    The baseline has to be set by the test: connect itself applies the calibration,
    so reading the objective right after connecting reads a value that may already
    have been touched.
    """
    microscope, _ = utils.setup_session(config_path=IFLM, manufacturer="Demo")
    try:
        assert microscope.system.fm.focus_position is None
        microscope.fm.objective.focus_position = 5.5e-3
        microscope.fm.objective.limit_position = 9.9e-3

        microscope._apply_fluorescence_calibration()

        assert microscope.fm.objective.focus_position == 5.5e-3
        assert microscope.fm.objective.limit_position == 9.9e-3
    finally:
        microscope.disconnect()


# ---------------------------------------------------------------------------
# The calibration action writes one block and preserves the rest
# ---------------------------------------------------------------------------


def test_writing_goes_to_the_file_named_and_keeps_its_shape(tmp_path):
    """Three ways the generic YAML writer would have gone wrong.

    It forces a `.yaml` suffix, so `site.yml` gets a sibling written and stays
    untouched; it sorts keys, so the sections come back alphabetical; and its
    dumper writes a numpy scalar as a tag `safe_load` refuses -- one instrument
    value of the wrong type and the configuration no longer loads.
    """
    import numpy as np

    path = tmp_path / "site.yml"
    original = utils.load_yaml(IFLM)
    path.write_text(yaml.safe_dump(original, sort_keys=False))
    order = list(original)

    utils.write_objective_calibration(path, np.float64(7.3e-3), np.float32(8.4e-3))

    assert sorted(p.name for p in tmp_path.iterdir()) == ["site.yml"]
    written = utils.load_yaml(str(path))  # safe_load: raises on a numpy tag
    assert list(written) == order
    assert written["calibration"]["objective"]["focus_position"] == pytest.approx(
        7.3e-3
    )
    assert "numpy" not in path.read_text()


def test_writing_the_calibration_touches_only_that_block(tmp_path):
    path = tmp_path / "site.yaml"
    original = utils.load_yaml(IFLM)
    original["info"]["name"] = "Bay 2"  # something a person wrote
    path.write_text(yaml.safe_dump(original))

    utils.write_objective_calibration(path, 7.3e-3, 8.4e-3)

    written = utils.load_yaml(str(path))
    assert written["calibration"]["objective"] == {
        "focus_position": 7.3e-3,
        "limit_position": 8.4e-3,
    }
    expected = copy.deepcopy(original)
    expected["calibration"]["objective"] = written["calibration"]["objective"]
    assert written == expected
    # and it loads
    assert MicroscopeSettings.from_dict(written).system.fm.focus_position == 7.3e-3
