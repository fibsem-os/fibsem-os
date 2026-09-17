"""The one writer of the configuration file from the application.

`write_configuration` sets nested keys and preserves everything else at the parsed
level, so a calibration action and the defaults panel can each write their own
section without disturbing the other's, or anything a person wrote by hand.
"""

import copy
import os

import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import MicroscopeSettings

SHIPPED = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


def _site_file(tmp_path):
    original = utils.load_yaml(SHIPPED)
    original["info"]["name"] = "Bay 2"  # something a person wrote
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(original))
    return path, original


def test_a_nested_update_touches_only_the_keys_it_names(tmp_path):
    path, original = _site_file(tmp_path)

    utils.write_configuration(path, {"defaults": {"electron": {"voltage": 5000}}})

    written = utils.load_yaml(str(path))
    expected = copy.deepcopy(original)
    expected["defaults"]["electron"]["voltage"] = 5000
    assert written == expected


def test_a_dict_merges_and_a_value_replaces(tmp_path):
    path, original = _site_file(tmp_path)

    utils.write_configuration(
        path, {"calibration": {"objective": {"focus_position": 1.0}}, "version": 1}
    )

    written = utils.load_yaml(str(path))
    assert written["calibration"]["objective"] == {"focus_position": 1.0}
    assert (
        written["calibration"]["shuttle_pre_tilt"]
        == original["calibration"]["shuttle_pre_tilt"]
    )
    assert written["hardware"] == original["hardware"]


def test_writing_into_an_old_flat_file_retires_the_old_copy(tmp_path):
    """A file from before the sections keeps its flat blocks, and the reader
    accepts them. Writing `defaults.electron.voltage` must not leave
    `electron.voltage` beside it -- two homes for one value, the new one silently
    winning and the legacy alias hiding the warning."""
    path = tmp_path / "old.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "electron": {"voltage": 2000, "column_tilt": 0, "hfw": 150e-6},
                "imaging": {"hfw": 150e-6, "beam_type": "ELECTRON"},
            }
        )
    )

    utils.write_configuration(
        path,
        {"defaults": {"electron": {"voltage": 5000}, "imaging": {"hfw": 80e-6}}},
    )

    written = utils.load_yaml(str(path))
    assert written["defaults"]["electron"]["voltage"] == 5000
    assert "voltage" not in written["electron"], "two homes for one value"
    assert written["electron"] == {
        "column_tilt": 0,
        "hfw": 150e-6,
    }  # untouched keys stay
    assert written["imaging"] == {"beam_type": "ELECTRON"}
    assert MicroscopeSettings.from_dict(written).system.electron.beam.voltage == 5000


def test_an_emptied_legacy_block_is_removed(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text(yaml.safe_dump({"imaging": {"hfw": 150e-6}}))

    utils.write_configuration(path, {"defaults": {"imaging": {"hfw": 80e-6}}})

    assert "imaging" not in utils.load_yaml(str(path))


def test_two_sections_written_separately_do_not_disturb_each_other(tmp_path):
    path, _ = _site_file(tmp_path)

    utils.write_objective_calibration(path, 7.0e-3, 8.0e-3)
    utils.write_configuration(path, {"defaults": {"ion": {"voltage": 8000}}})

    written = utils.load_yaml(str(path))
    assert written["calibration"]["objective"]["focus_position"] == 7.0e-3
    assert written["defaults"]["ion"]["voltage"] == 8000
    assert written["defaults"]["electron"]["voltage"] == 2000
    # and it still loads with nothing unrecognised
    assert utils.unrecognised_configuration_keys(written) == []
    assert MicroscopeSettings.from_dict(written).system.ion.beam.voltage == 8000
