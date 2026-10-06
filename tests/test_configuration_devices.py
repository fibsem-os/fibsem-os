"""Configuration version 2: `hardware:` is a list of devices.

The list is an overlay on what a backend builds. An entry changes a device (switches it
off, names its driver, gives it keys) or adds one the backend cannot find for itself; a
device the file does not name is built as before. Version 1 files, with a block per
device, load unchanged, and are converted the first time anything saves them.
"""

import copy
import os

import pytest
import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import (
    CONFIGURATION_VERSION,
    DeviceEntry,
    MicroscopeSettings,
    SystemSettings,
)

# The shipped files as they were in configuration version 1.
VERSION_1_FILES = os.path.join(
    os.path.dirname(__file__), "fixtures", "configuration_v1"
)
VERSION_1 = os.path.join(VERSION_1_FILES, "tfs-arctis-configuration.yaml")


def _version_1() -> dict:
    config = utils.load_yaml(VERSION_1)
    assert config["version"] == 1, "fixture no longer exercises a version 1 file"
    return config


def _names(config: dict) -> list:
    return [DeviceEntry.from_dict(d).name for d in config["hardware"]["devices"]]


# ---------------------------------------------------------------------------
# Version 1 and version 2 say the same thing
# ---------------------------------------------------------------------------


def test_the_writer_lists_the_devices():
    written = MicroscopeSettings.from_dict(_version_1()).to_dict()

    assert written["version"] == CONFIGURATION_VERSION == 2
    assert list(written["hardware"]) == ["devices"]
    assert _names(written) == ["stage", "electron", "ion", "fm"]
    ion = utils.configuration_device(written, "ion")
    assert ion["type"] == "beam"
    assert ion["plasma_gas"] == "Xenon"
    assert "beam_type" not in ion, "the entry's name says which column"


def test_a_version_1_file_and_its_upgrade_load_the_same():
    config = _version_1()
    upgraded = utils.upgrade_configuration(copy.deepcopy(config))

    assert "stage" not in upgraded["hardware"]
    assert upgraded["version"] == CONFIGURATION_VERSION
    assert (
        MicroscopeSettings.from_dict(upgraded).system.to_dict()
        == MicroscopeSettings.from_dict(config).system.to_dict()
    )


def test_the_written_list_reads_back_as_it_was_written():
    written = MicroscopeSettings.from_dict(_version_1()).system.to_dict()

    assert SystemSettings.from_dict(written).to_dict() == written


def test_a_list_entry_wins_over_a_version_1_block_key_by_key():
    config = _version_1()
    config["hardware"]["devices"] = [{"name": "ion", "eucentric_height": 0.02}]

    ion = SystemSettings.from_dict(config).ion
    assert ion.eucentric_height == 0.02
    assert ion.plasma_gas == "Xenon"  # from the block, which the entry left alone


# ---------------------------------------------------------------------------
# What an entry may leave out
# ---------------------------------------------------------------------------


def test_the_name_defaults_to_the_type():
    assert DeviceEntry.from_dict({"type": "gis"}).name == "gis"


@pytest.mark.parametrize(
    "name, expected", [("ion", "beam"), ("fm", "fm"), ("manipulator", "manipulator")]
)
def test_the_type_follows_from_a_name_the_backend_builds(name, expected):
    assert DeviceEntry.from_dict({"name": name}).type == expected


def test_an_entry_naming_neither_is_an_error():
    with pytest.raises(ValueError, match="neither"):
        DeviceEntry.from_dict({"enabled": False})


def test_a_new_device_must_say_what_it_is():
    with pytest.raises(ValueError, match="type"):
        DeviceEntry.from_dict({"name": "knife"})


def test_a_beam_is_named_for_its_column():
    with pytest.raises(ValueError, match="electron"):
        DeviceEntry.from_dict({"type": "beam"})


def test_two_entries_with_one_name_are_an_error():
    """Two GIS are two names; one name twice cannot be told apart."""
    config = {"hardware": {"devices": [{"type": "gis"}, {"name": "gis"}]}}
    with pytest.raises(ValueError, match="twice"):
        SystemSettings.from_dict(config)


def test_two_devices_of_one_type_are_two_names():
    config = {
        "hardware": {
            "devices": [
                {"type": "gis"},
                {"name": "gis-w", "type": "gis", "driver": "remote", "port": 9},
            ]
        }
    }
    system = SystemSettings.from_dict(config)
    assert [e.name for e in system.other_devices] == ["gis", "gis-w"]
    assert system.other_devices[1].options == {"port": 9}


# ---------------------------------------------------------------------------
# The entries fill the records every reader uses
# ---------------------------------------------------------------------------


def test_a_switched_off_column_is_not_enabled():
    config = {"hardware": {"devices": [{"name": "ion", "enabled": False}]}}
    system = SystemSettings.from_dict(config)
    assert system.ion.enabled is False
    assert system.electron.enabled is True


def test_the_fm_entry_names_its_driver_and_its_keys_sit_beside_it():
    config = {
        "hardware": {
            "devices": [
                {"name": "fm", "driver": "remote", "address": "10.0.0.2", "port": 8001}
            ]
        }
    }
    fm = SystemSettings.from_dict(config).fm
    assert (fm.driver, fm.address, fm.port) == ("remote", "10.0.0.2", 8001)
    assert fm.enabled is None, "absent is the backend's default, not false"


def test_a_device_without_a_record_is_kept_and_written_back():
    entry = {"name": "gis", "type": "gis", "enabled": False}
    config = {"hardware": {"devices": [entry]}}

    written = SystemSettings.from_dict(config).to_dict()

    assert written["hardware"]["devices"][-1] == entry


def test_a_version_1_block_for_another_device_is_not_read():
    """Version 1 read four blocks under `hardware:`. A fifth was read by nothing, and
    must not start building a device now."""
    config = _version_1()
    config["hardware"]["manipulator"] = {"enabled": True}

    assert SystemSettings.from_dict(config).other_devices == []
    assert "hardware.manipulator" in utils.unrecognised_configuration_keys(config)


# ---------------------------------------------------------------------------
# Unrecognised keys, in either shape
# ---------------------------------------------------------------------------


def test_a_typo_in_a_listed_device_is_reported_where_it_is():
    config = MicroscopeSettings.from_dict(_version_1()).to_dict()
    utils.configuration_device(config, "stage")["nonsense"] = 1

    assert utils.unrecognised_configuration_keys(config) == [
        "hardware.devices.stage.nonsense"
    ]


def test_a_written_file_reports_nothing():
    config = MicroscopeSettings.from_dict(_version_1()).to_dict()
    assert utils.unrecognised_configuration_keys(config) == []


def test_a_device_without_a_record_carries_its_drivers_keys_unpoliced():
    config = {"hardware": {"devices": [{"type": "gis", "driver": "x", "anything": 1}]}}
    assert utils.unrecognised_configuration_keys(config) == []


# ---------------------------------------------------------------------------
# Saving converts a version 1 file, once, and keeps the original
# ---------------------------------------------------------------------------


def _site_file(tmp_path):
    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(_version_1(), sort_keys=False))
    return path


def test_saving_a_version_1_file_lists_its_devices(tmp_path):
    path = _site_file(tmp_path)
    before = MicroscopeSettings.from_dict(utils.load_yaml(str(path))).system.to_dict()

    utils.write_configuration(path, {"defaults": {"ion": {"voltage": 16000}}})

    written = utils.load_yaml(str(path))
    assert written["version"] == CONFIGURATION_VERSION
    assert list(written["hardware"]) == ["devices"]
    # What the file said, as entries: it has no `fm:` block, so no FM entry.
    assert _names(written) == ["stage", "electron", "ion"]
    after = MicroscopeSettings.from_dict(written).system.to_dict()
    before["defaults"]["ion"]["voltage"] = 16000
    assert after == before
    assert utils.unrecognised_configuration_keys(written) == []


def test_the_version_1_file_is_kept_once(tmp_path):
    """Version 1 does not read the list; it would load every device's geometry as the
    defaults. The copy is what a site going back restores."""
    path = _site_file(tmp_path)
    original = path.read_text()

    utils.write_configuration(path, {"defaults": {"ion": {"voltage": 16000}}})
    utils.write_configuration(path, {"defaults": {"ion": {"voltage": 8000}}})

    assert utils.configuration_backup_path(path, before=2).read_text() == original
    assert not utils.configuration_backup_path(path).exists()


def test_a_version_2_file_is_not_copied(tmp_path):
    path = _site_file(tmp_path)
    utils.write_configuration(path, {})
    utils.configuration_backup_path(path, before=2).unlink()

    utils.write_configuration(path, {"defaults": {"ion": {"voltage": 8000}}})

    assert not utils.configuration_backup_path(path, before=2).exists()


def test_roles_are_kept_on_every_entry():
    """`roles:` binds one entry to another; nothing reads it yet, but a file stating it
    must not lose it on save, on the four with records as on any other."""
    config = {
        "hardware": {
            "devices": [
                {"name": "electron", "roles": {"scanner": "scan_generator"}},
                {"name": "scan_generator", "type": "scan_generator"},
            ]
        }
    }

    written = SystemSettings.from_dict(config).to_dict()

    assert utils.configuration_device(written, "electron")["roles"] == {
        "scanner": "scan_generator"
    }
    assert "roles" not in utils.configuration_device(written, "ion")
    assert utils.unrecognised_configuration_keys(written) == []


# ---------------------------------------------------------------------------
# Where the stage travels for a device is on that device's entry
# ---------------------------------------------------------------------------

OFFSET_FM = os.path.join(VERSION_1_FILES, "sim-iflm-configuration.yaml")


def _offset_fm_entry(**keys) -> dict:
    return {
        "hardware": {"devices": [dict({"name": "fm", "origin": {"x": 0.05}}, **keys)]}
    }


def test_a_version_1_files_positions_move_onto_the_fm_entry():
    """Its shared `device_range` is copied onto the FM, so the window it had is kept."""
    written = MicroscopeSettings.from_dict(utils.load_yaml(OFFSET_FM)).to_dict()

    fm = utils.configuration_device(written, "fm")
    assert fm["origin"] == {"x": 48.8e-3}
    assert fm["available_orientations"] == ["FIB"]
    assert fm["range"] == {"x": 20.0e-3}
    assert "devices" not in utils.configuration_device(written, "stage")
    assert "device_range" not in utils.configuration_device(written, "stage")
    assert utils.unrecognised_configuration_keys(written) == []


def test_the_positions_read_back_as_they_were_written():
    system = MicroscopeSettings.from_dict(utils.load_yaml(OFFSET_FM)).system
    restored = SystemSettings.from_dict(system.to_dict())

    assert restored.stage.devices == system.stage.devices


def test_a_device_that_states_no_range_has_1_mm_on_each_axis_its_origin_sets():
    fm = SystemSettings.from_dict(_offset_fm_entry()).stage.devices["FM"]

    assert (fm.range.x, fm.range.y, fm.range.z) == (1.0e-3, None, None)


def test_the_beams_have_no_range():
    assert SystemSettings.from_dict({}).stage.devices["FIBSEM"].range is None


def test_a_device_that_states_no_orientations_gets_its_types():
    fm = SystemSettings.from_dict(_offset_fm_entry()).stage.devices["FM"]
    assert fm.available_orientations == ["FM"]


def test_an_empty_orientation_list_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        SystemSettings.from_dict(_offset_fm_entry(available_orientations=[]))


def test_the_version_1_key_left_empty_meant_any_orientation():
    config = {
        "hardware": {
            "stage": {
                "devices": {
                    "FM": {"origin": {"x": 0.05}, "acquisition_orientations": []}
                }
            }
        }
    }
    fm = SystemSettings.from_dict(config).stage.devices["FM"]
    assert fm.available_orientations == ["SEM", "FIB", "MILLING", "FM"]
    assert fm.range.x == 20.0e-3, "a version 1 file keeps its 20 mm window"


# ---------------------------------------------------------------------------
# How the FM camera is mounted
# ---------------------------------------------------------------------------


def test_the_fm_entry_states_how_its_camera_is_mounted():
    from fibsem.structures import CameraImageTransform

    config = {"hardware": {"devices": [{"name": "fm", "mount_transform": "flip-x"}]}}

    system = SystemSettings.from_dict(config)

    assert system.fm.mount_transform is CameraImageTransform.FLIP_X
    fm = utils.configuration_device(system.to_dict(), "fm")
    assert fm["mount_transform"] == "flip-x"
    assert utils.unrecognised_configuration_keys(config) == []


def test_an_fm_entry_that_does_not_say_is_mounted_straight_and_saved_unchanged():
    from fibsem.structures import CameraImageTransform

    system = MicroscopeSettings.from_dict(_version_1()).system

    assert system.fm.mount_transform is CameraImageTransform.NONE
    assert "mount_transform" not in utils.configuration_device(system.to_dict(), "fm")


def test_an_unknown_mount_says_where_it_is_and_what_it_could_be():
    config = {"hardware": {"devices": [{"name": "fm", "mount_transform": "rotate"}]}}

    with pytest.raises(ValueError, match="fm.*none, flip-x, flip-y, flip-xy"):
        SystemSettings.from_dict(config)


def test_the_demo_microscope_hands_the_fm_entry_to_its_fm(tmp_path):
    """The configuration's fm entry reaches the FM's devices as the binder's config,
    so the images come out the way round the mount says."""
    from fibsem.structures import CameraImageTransform

    config = utils.load_yaml(
        os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    )
    config = utils.upgrade_configuration(config)
    utils.configuration_device(config, "fm", create=True)["mount_transform"] = "flip-xy"
    path = tmp_path / "microscope-configuration.yaml"
    path.write_text(yaml.safe_dump(config))

    microscope, _ = utils.setup_session(config_path=str(path))

    assert microscope.fm.devices["camera"].mount_transform.get_value() == "flip-xy"
    assert microscope.fm.mount_transform is CameraImageTransform.FLIP_XY
