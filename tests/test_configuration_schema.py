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
    assert _load(filename)["version"] == 1


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
        "milling",
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


def test_the_dead_key_the_audit_found_is_reported():
    """`imaging.imaging_current` is still in every shipped file and still read by
    nothing. Until it is removed, at least say so."""
    assert "imaging.imaging_current" in utils.unrecognised_configuration_keys(
        _load("microscope-configuration.yaml")
    )


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
