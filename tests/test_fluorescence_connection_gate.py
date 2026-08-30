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

    assert microscope.system.fm.enabled is False
    assert microscope.fm is None


def test_the_flag_defaults_off():
    """Including on systems that do have the hardware. Groundwork ships disabled."""
    assert FluorescenceSystemSettings().enabled is False
    assert FluorescenceSystemSettings.from_dict({}).enabled is False
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
    microscope.system.fm.enabled = False  # as a real Arctis configuration has it

    assert microscope.stage_is_compustage is True
    assert microscope._fluorescence_is_configured() is True


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
    settings["fm"]["enabled"] = False  # the site has not said so

    assert _from(settings, tmp_path).fm is None


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

    assert "enabled" in settings["fm"]
    assert _microscope(ARCTIS_CONFIG).system.fm.enabled is True
