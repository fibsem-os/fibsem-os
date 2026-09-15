"""A configuration loads whether or not it says everything.

Two directions, and the readers only handled one of them. A file carrying *extra* keys
has always loaded -- `from_dict` reads what it names and ignores the rest. A file
*missing* a key raised `KeyError`, which is why removing `rotation_180` in FIB-834
needed its reader changed first, and why the six dead `milling:` keys and
`imaging.imaging_current` could not simply be deleted.

Both directions now work, which is what lets the rest of the schema change happen as
one refactor rather than a sequence of them.
"""

import copy
import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import (
    CONFIGURATION_VERSION,
    DEFAULT_FIB_COLUMN_TILT,
    MicroscopeSettings,
    SystemSettings,
)

# Named rather than globbed: `fibsem/config/*.yaml` is gitignored with an allowlist,
# so a configuration saved from the wizard sits in the same folder and a glob would
# sweep it in -- failing on one machine and not another.
SHIPPED = (
    "microscope-configuration.yaml",
    "odemis-configuration.yaml",
    "sim-arctis-configuration.yaml",
    "sim-iflm-configuration.yaml",
    "tescan-configuration.yaml",
    "tfs-aquilos2-configuration.yaml",
    "tfs-arctis-configuration.yaml",
    "tfs-hydra-configuration.yaml",
)


def _load(filename: str) -> dict:
    return utils.load_yaml(os.path.join(cfg.CONFIG_PATH, filename))


# ---------------------------------------------------------------------------
# Nothing that used to load stopped loading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", SHIPPED)
def test_every_shipped_configuration_still_loads(filename: str):
    settings = MicroscopeSettings.from_dict(_load(filename))
    assert settings.system.info.manufacturer
    assert settings.system.electron.beam.voltage is not None


@pytest.mark.parametrize("filename", SHIPPED)
def test_every_shipped_configuration_declares_its_version(filename: str):
    """A format change cannot migrate a file that does not say what format it is.

    Cheap now; impossible to add retrospectively, because a file without the field is
    indistinguishable from one written before the field existed.
    """
    assert _load(filename)["version"] == CONFIGURATION_VERSION


def test_saving_writes_the_version():
    """The writer states the version too, or a file saved from the application
    would drop the one field a future migration needs."""
    written = MicroscopeSettings.from_dict(
        _load("microscope-configuration.yaml")
    ).to_dict()
    assert written["version"] == CONFIGURATION_VERSION


# ---------------------------------------------------------------------------
# A configuration may leave things out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block",
    [
        "version",
        "info",
        "stage",
        "electron",
        "ion",
        "manipulator",
        "gis",
        "imaging",
        "sim",
    ],
)
def test_any_block_may_be_absent(block: str):
    """A configuration that drops a section it does not need is a configuration, not a
    corrupt file. This is the change that unblocks removing keys from the shipped
    files without every existing one raising at load."""
    config = copy.deepcopy(_load("microscope-configuration.yaml"))
    config.pop(block)

    settings = MicroscopeSettings.from_dict(config)
    assert isinstance(settings.system, SystemSettings)


def test_an_empty_configuration_loads():
    """The limit case, and the one that catches a reader missed in a nested class.

    `BeamSettings.from_dict` was the one this found: it is reached through
    `BeamSystemSettings`, so converting the classes named in the audit was not enough
    and a per-field grep did not show it.
    """
    settings = MicroscopeSettings.from_dict({})

    assert settings.system.stage.shuttle_pre_tilt == 0.0
    assert settings.system.electron.beam.voltage is None


def test_a_missing_field_defaults_rather_than_raising():
    config = copy.deepcopy(_load("microscope-configuration.yaml"))
    del config["stage"]["rotation_reference"]
    del config["ion"]["column_tilt"]

    settings = MicroscopeSettings.from_dict(config)
    assert settings.system.stage.rotation_reference == 0.0
    # Not 0: the ion column's default is the angle between the columns, because a
    # dual-beam whose FIB sits at 0 degrees is not a conservative fallback but a
    # different instrument. See `DEFAULT_FIB_COLUMN_TILT`.
    assert settings.system.ion.column_tilt == DEFAULT_FIB_COLUMN_TILT
    assert settings.system.electron.column_tilt == 0.0


def test_reading_does_not_mutate_the_dict_it_was_given():
    """`from_dict` used to stamp `beam_type` into the caller's `electron` block.

    Harmless while one caller loaded one file, and not harmless once the editor holds
    a configuration dict and saves it back -- the stamped key would be written to
    disk.
    """
    config = _load("microscope-configuration.yaml")
    before = copy.deepcopy(config)

    SystemSettings.from_dict(config)

    assert config == before


# ---------------------------------------------------------------------------
# The writer writes what the reader reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", SHIPPED)
def test_a_saved_configuration_reports_nothing_unrecognised(filename: str):
    """Load, save, and every key in the saved file is one this version writes.

    The first attempt at a schema table was written from the shipped YAML rather
    than from `to_dict`, and rejected 23 of the keys the application writes on every
    save -- a user who saved from the setup widget and restarted was told 23 of
    their settings were unsupported. The known set is now derived from the writer,
    and this pins that nothing can put the two back out of step.
    """
    written = MicroscopeSettings.from_dict(_load(filename)).to_dict()
    assert utils.unrecognised_configuration_keys(written) == []


@pytest.mark.parametrize("filename", SHIPPED)
def test_a_round_trip_is_a_fixed_point(filename: str):
    """Load → save → load → save gives the same file as load → save.

    The other half of the guarantee: not only does the writer write what the reader
    reads, the writer writes everything the reader reads. A key the reader takes
    from the file and the writer leaves out comes back as its default on the second
    pass -- which is exactly how a setting vanishes on save.
    """
    loaded = MicroscopeSettings.from_dict(_load(filename))
    reloaded = MicroscopeSettings.from_dict(copy.deepcopy(loaded.to_dict()))
    # Compared as objects, not as dicts: a key the reader reads and the writer
    # forgets would round-trip to the same *dict* (both passes lack it) while the
    # second *object* silently carries the default instead of the file's value.
    assert reloaded.system == loaded.system
    assert reloaded.image == loaded.image
    # Said explicitly as well, because this is the value the round trip most needs
    # to keep and the one a dataclass `__eq__` was blind to for a while.
    assert (
        reloaded.system.stage.shuttle_pre_tilt == loaded.system.stage.shuttle_pre_tilt
    )


# ---------------------------------------------------------------------------
# Extra keys are ignored, and said out loud
# ---------------------------------------------------------------------------


def test_unrecognised_keys_are_reported():
    """Ignoring silently is how `imaging.imaging_current` became a setting a user
    could type, save, reload, and never see again -- `ImageSettings` has no such
    field, so it was dropped on load and nothing said so."""
    config = copy.deepcopy(_load("microscope-configuration.yaml"))
    config["stage"]["nonsense"] = 1
    config["a_block_from_the_future"] = {"x": 1}

    unknown = utils.unrecognised_configuration_keys(config)

    assert "stage.nonsense" in unknown
    assert "a_block_from_the_future" in unknown


def test_no_shipped_configuration_carries_a_key_this_version_ignores():
    """The shipped files say only what this version reads.

    They did not: `imaging.imaging_current` was in all eight and read by nothing, and
    the `milling:` block was six more. Both are gone, and this is what stops another
    one accumulating -- adding a key to a shipped file that `to_dict` does not write
    now fails here rather than being quietly dropped at load.
    """
    offenders = {
        filename: utils.unrecognised_configuration_keys(_load(filename))
        for filename in SHIPPED
    }
    assert {k: v for k, v in offenders.items() if v} == {}


@pytest.mark.parametrize(
    "removed",
    [
        {"milling": {"milling_current": 2.0e-9, "milling_voltage": 30000}},
        {"stage": {"manipulator_height_limit": 0.0037}},
        {"imaging": {"imaging_current": 2.0e-11}},
        {"manipulator": {"rotation": False, "tilt": False}},
        {"electron": {"plasma": False, "plasma_gas": "None"}},
    ],
)
def test_a_configuration_written_before_the_keys_were_removed_still_loads(
    removed: dict,
):
    """The promise made when the keys were deleted: an old file keeps working.

    Every site has a configuration on disk carrying these. They are ignored, not
    honoured and not fatal -- and they are named in the log rather than dropped in
    silence, so a user who set one can find out it does nothing.
    """
    config = copy.deepcopy(_load("microscope-configuration.yaml"))
    for block, keys in removed.items():
        config.setdefault(block, {}).update(keys)

    settings = MicroscopeSettings.from_dict(config)

    assert isinstance(settings.system, SystemSettings)
    reported = utils.unrecognised_configuration_keys(config)
    for block, keys in removed.items():
        for key in keys:
            assert f"{block}.{key}" in reported or block in reported


@pytest.mark.parametrize("block", ["sim", "protocol"])
def test_open_blocks_are_not_policed(block: str):
    """`sim:` and `protocol:` carry backend- and application-specific keys that this
    schema has no business knowing about."""
    unknown = utils.unrecognised_configuration_keys({block: {"anything": 1}})
    assert unknown == []


def test_a_configuration_that_says_nothing_extra_reports_nothing():
    config = {
        "version": 1,
        "info": {"name": "x", "ip_address": "y", "manufacturer": "Demo"},
        "stage": {"rotation_reference": 0.0},
    }
    assert utils.unrecognised_configuration_keys(config) == []


def test_reporting_is_logged_at_load(caplog):
    """One line, at load, naming what was ignored."""
    import logging

    config = copy.deepcopy(_load("microscope-configuration.yaml"))
    config["stage"]["nonsense"] = 1

    with caplog.at_level(logging.INFO):
        utils.report_unrecognised_configuration_keys(config, source="test.yaml")

    assert "stage.nonsense" in caplog.text
    assert "test.yaml" in caplog.text


# ---------------------------------------------------------------------------
# One key for the plasma source
# ---------------------------------------------------------------------------


def test_a_plasma_column_is_one_with_a_gas():
    """`plasma: bool` and `plasma_gas: str` were two keys for one fact and could
    disagree. Now there is the gas, and "is this a plasma column" is derived."""
    ion = MicroscopeSettings.from_dict(
        _load("tfs-arctis-configuration.yaml")
    ).system.ion
    assert ion.plasma_gas == "Xenon"
    assert ion.plasma is True

    ion = MicroscopeSettings.from_dict(_load("tfs-hydra-configuration.yaml")).system.ion
    assert ion.plasma_gas is None
    assert ion.plasma is False


@pytest.mark.parametrize(
    "block, expected",
    [
        ({"plasma": True, "plasma_gas": "Xenon"}, "Xenon"),
        # The flag was what the drivers consulted, so it wins over a stray gas.
        ({"plasma": False, "plasma_gas": "Xenon"}, None),
        # Every shipped file wrote "no gas" as the YAML *string* "None".
        ({"plasma": False, "plasma_gas": "None"}, None),
        ({"plasma_gas": "None"}, None),
        ({"plasma_gas": "none"}, None),
        ({"plasma_gas": ""}, None),
        ({"plasma_gas": None}, None),
        ({}, None),
    ],
)
def test_the_old_two_key_spelling_still_reads(block: dict, expected):
    config = copy.deepcopy(_load("microscope-configuration.yaml"))
    config["ion"].pop("plasma_gas", None)
    config["ion"].update(block)
    assert MicroscopeSettings.from_dict(config).system.ion.plasma_gas == expected


def test_the_old_plasma_flag_is_read_for_migration_and_not_written():
    """A file stating `ion.plasma` is neither warned about nor saved back with it."""
    config = copy.deepcopy(_load("microscope-configuration.yaml"))
    config["ion"]["plasma"] = False
    assert "ion.plasma" not in utils.unrecognised_configuration_keys(config)

    written = MicroscopeSettings.from_dict(config).to_dict()
    assert "plasma" not in written["ion"]
    assert "plasma" not in written["electron"]
    assert "plasma_gas" not in written["electron"]
