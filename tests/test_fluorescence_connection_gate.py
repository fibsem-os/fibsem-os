"""Which systems get a fluorescence microscope at connection, and which are refused.

`microscope.fm` was built only when the stage was a compustage. That is what made
every offset path unreachable without hardware. Opening it is the last piece of the
offset-FM work, and the way it opens matters more than that it opens:

**Detecting the hardware is the failure mode, not the goal.** There is no
`is_installed` for the FM in AutoScript -- every other subsystem has one -- so the
only capability test available is to select it and see whether the microscope throws.
Running that on every system would mean an Aquilos or Helios with an iFLM fitted finds
half-built offset support in its UI on upgrade. So an explicit flag decides, and the
probe only confirms afterwards.

The flag *widens* the old check rather than replacing it. `stage_is_compustage` is
read from the hardware, not configuration, and no shipped Arctis configuration carries
the flag -- so replacing it would take the FM away from every Arctis site.
"""

import logging
import os

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import FluorescenceSystemSettings

IFLM_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")


def _microscope(config_path: str):
    microscope, _ = utils.setup_session(config_path=config_path)
    return microscope


def _from(settings: dict, tmp_path) -> "object":
    """Connect to a one-off variant of a shipped configuration.

    `setup_session` takes a path, not a blob, so the variant has to go through a file.
    """
    path = tmp_path / "variant-configuration.yaml"
    utils.save_yaml(path, settings)
    return _microscope(str(path))


# ── nothing is detected into existence ───────────────────────────────


def test_a_beam_only_system_gets_no_fluorescence_microscope():
    """The shipped default configuration, and every Thermo site that has not opted in.

    This is the leak the flag exists to prevent: an offset system must not acquire
    in-progress fluorescence support by the software noticing the hardware.
    """
    microscope = _microscope(cfg.MICROSCOPE_CONFIGURATION_PATH)

    assert not microscope.system.fm.enabled
    assert microscope.fm is None


def test_the_flag_defaults_off():
    """Including on systems that do have the hardware. Groundwork ships disabled."""
    assert not FluorescenceSystemSettings().enabled
    assert FluorescenceSystemSettings.from_dict({}).enabled is None
    assert "fm" not in utils.load_yaml(cfg.MICROSCOPE_CONFIGURATION_PATH)


def test_a_configured_offset_system_gets_one():
    """What the whole project was for: an FM on a stage that does not flip."""
    microscope = _microscope(IFLM_CONFIG)

    assert microscope.stage_is_compustage is False
    assert microscope.system.fm.enabled is True
    assert microscope.fm is not None


# ── the Arctis must not lose what it already has ─────────────────────


def test_a_compustage_needs_no_flag():
    """`tfs-arctis-configuration.yaml` has no `fm:` block, and its FM works today.

    `stage_is_compustage` comes from the hardware (`compustage.is_installed`), not
    from configuration, so no Arctis site has ever needed to say anything. Replacing
    the old check with the flag rather than widening it would take the FM away from
    all of them on upgrade -- which is why this is an `or`.
    """
    microscope = _microscope(ARCTIS_CONFIG)
    microscope.system.fm.enabled = None  # as a real Arctis configuration has it

    assert microscope.stage_is_compustage is True
    assert microscope._fluorescence_is_configured() is True


def test_an_explicit_false_turns_a_compustage_fm_off(tmp_path):
    """Absent keeps the compustage default; `false` is a device switched off."""
    settings = utils.load_yaml(ARCTIS_CONFIG)
    settings["hardware"]["fm"]["enabled"] = False

    microscope = _from(settings, tmp_path)

    assert microscope.stage_is_compustage is True
    assert microscope.fm is None


def test_an_absent_flag_survives_a_round_trip():
    """A save must not turn "not stated" into an explicit `false`."""
    fm = FluorescenceSystemSettings.from_dict({})

    assert FluorescenceSystemSettings.from_dict(fm.to_dict()).enabled is None


def test_an_offset_system_does_need_one():
    microscope = _microscope(IFLM_CONFIG)
    microscope.system.fm.enabled = False

    assert microscope.stage_is_compustage is False
    assert microscope._fluorescence_is_configured() is False


# ── configured and detected are different questions ──────────────────


def test_detected_but_not_configured_gets_nothing(tmp_path):
    """The row that matters, and the reason the simulator keeps two separate keys.

    `has_fm` stands in for what the hardware probe would answer; `fm.enabled` is what
    the site said. A system where an FM is present and nothing is configured for it is
    exactly the upgrading site the flag protects, and it has to stay representable --
    inferring one from the other would collapse it into the cases that already work.
    """
    settings = utils.load_yaml(IFLM_CONFIG)
    assert settings["sim"]["has_fm"] is True  # the hardware is "there"
    settings["hardware"]["fm"]["enabled"] = False  # the site has not said so

    assert _from(settings, tmp_path).fm is None


def test_detected_with_no_fm_block_gets_nothing(tmp_path):
    """An offset system with an iFLM fitted and nothing said about it: no FM.

    Absent keeps the old default, which is off on anything but a compustage.
    """
    settings = utils.load_yaml(IFLM_CONFIG)
    assert settings["sim"]["has_fm"] is True
    del settings["hardware"]["fm"]

    microscope = _from(settings, tmp_path)

    assert microscope.stage_is_compustage is False
    assert microscope.system.fm.enabled is None
    assert microscope.fm is None


def test_configured_but_not_detected_gets_nothing_either(tmp_path):
    """The probe still has the last word: a site can be wrong about its own hardware."""
    settings = utils.load_yaml(IFLM_CONFIG)
    settings["sim"]["has_fm"] = False

    microscope = _from(settings, tmp_path)

    assert microscope.system.fm.enabled is True
    assert microscope.fm is None


# ── the configuration itself ─────────────────────────────────────────


def test_the_flag_survives_a_round_trip():
    """`system.to_dict()` is served over the API and written into image metadata."""
    system = _microscope(IFLM_CONFIG).system

    assert type(system).from_dict(system.to_dict()).fm == system.fm


def test_the_config_path_is_still_read_from_the_same_block():
    """Two readers, one `fm:` block, one key each -- this is the hardware fact, the
    other is a path to imaging parameters. Neither should have eaten the other."""
    settings = utils.load_yaml(ARCTIS_CONFIG)

    assert "enabled" in settings["hardware"]["fm"]
    assert _microscope(ARCTIS_CONFIG).system.fm.enabled is True


# ── which driver the FM comes from ───────────────────────────────────


def test_no_driver_key_follows_the_microscope():
    """Every site today: no `driver` key, and the FM each backend always built."""
    fm = FluorescenceSystemSettings.from_dict({"enabled": True})

    assert fm.driver is None
    assert _microscope(IFLM_CONFIG).fm is not None


def test_the_driver_keys_survive_a_round_trip():
    """A remote FM's address must not be dropped when the configuration is saved."""
    block = {"enabled": True, "driver": "remote", "address": "10.0.0.2", "port": 8765}

    assert FluorescenceSystemSettings.from_dict(block).to_dict() == block


def test_the_driver_keys_are_known_configuration_keys():
    """Otherwise loading a remote configuration reports them as ignored."""
    block = {"enabled": True, "driver": "remote", "address": "10.0.0.2", "port": 8765}

    assert utils.unrecognised_configuration_keys({"hardware": {"fm": block}}) == []


def test_a_remote_fm_is_not_looked_for_on_the_beams_connection(tmp_path, caplog):
    """An offset system with a METEOR on its own PC must not get the iFLM driver.

    Nothing serves the FM here, so the site gets none, and the log says why. Getting
    the remote FM when it is served is `tests/server/test_remote_fm_api.py`.
    """
    settings = utils.load_yaml(IFLM_CONFIG)
    settings["hardware"]["fm"].update(driver="remote", address="127.0.0.1", port=1)

    microscope = _from(settings, tmp_path)

    assert microscope.system.fm.driver == "remote"
    assert microscope.fm is None
    assert microscope._fluorescence_uses_own_driver() is False
    # setup_session reconfigures the root logger, which drops caplog's handler
    logging.getLogger().addHandler(caplog.handler)
    caplog.clear()
    assert microscope._connect_remote_fluorescence() is None
    assert "127.0.0.1:1" in caplog.text


def test_a_remote_fm_without_an_address_gets_no_fm(tmp_path, caplog):
    settings = utils.load_yaml(IFLM_CONFIG)
    settings["hardware"]["fm"]["driver"] = "remote"

    microscope = _from(settings, tmp_path)

    assert microscope.fm is None
    logging.getLogger().addHandler(caplog.handler)
    caplog.clear()
    assert microscope._connect_remote_fluorescence() is None
    assert "no `address` and `port`" in caplog.text


def test_an_unknown_driver_gets_no_fm(tmp_path, caplog):
    settings = utils.load_yaml(ARCTIS_CONFIG)
    settings["hardware"]["fm"]["driver"] = "meteor"

    microscope = _from(settings, tmp_path)

    assert microscope.fm is None
    # setup_session reconfigures the root logger, which drops caplog's handler
    logging.getLogger().addHandler(caplog.handler)
    caplog.clear()
    assert microscope._fluorescence_uses_own_driver() is False
    assert "Unknown fluorescence microscope driver 'meteor'" in caplog.text
