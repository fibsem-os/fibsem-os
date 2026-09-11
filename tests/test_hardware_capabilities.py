"""Which subsystems are fitted is not in the configuration file.

`manipulator.enabled`, `gis.enabled`, `gis.multichem` and `gis.sputter_coater` were
configuration keys, which meant a site could describe hardware it does not have, or
omit hardware it does, and nothing would disagree. The same move `rotation` made in
FIB-834, one record over: a backend that can ask the instrument does (AutoScript), and
one that cannot answers for itself with `DEFAULT_FITTED`, which is what its shipped
configuration used to say.

The asymmetry that shapes every test here: a subsystem that wrongly *appears* is a menu
entry that errors when used, while one that wrongly *disappears* is a working
instrument that lost a feature on upgrade, silently. So "cannot say" falls back to the
backend's default, never to "not fitted".
"""

import os
from typing import Optional

import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.structures import BeamType, MicroscopeSettings, SystemSettings
from tests.test_configuration_schema import SHIPPED


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    yield microscope
    microscope.disconnect()


def _load(filename: str) -> dict:
    return utils.load_yaml(os.path.join(cfg.CONFIG_PATH, filename))


# ---------------------------------------------------------------------------
# The file says nothing about it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", SHIPPED)
def test_no_shipped_file_states_what_is_fitted(filename: str):
    config = _load(filename)
    assert "manipulator" not in config["hardware"]
    assert "gis" not in config["hardware"]
    assert "manipulator" not in config
    assert "gis" not in config


def test_the_writer_does_not_state_it_either():
    written = MicroscopeSettings.from_dict(
        _load("microscope-configuration.yaml")
    ).to_dict()
    assert "manipulator" not in written["hardware"]
    assert "gis" not in written["hardware"]


def test_an_old_file_s_capability_keys_are_reported_as_unread():
    """Every configuration in the field carries them. They now do nothing, and the
    load log says so rather than letting a site believe its `enabled: false` held."""
    config = _load("microscope-configuration.yaml")
    config["manipulator"] = {"enabled": False}
    config["gis"] = {"enabled": True, "multichem": True, "sputter_coater": False}

    unknown = utils.unrecognised_configuration_keys(config)

    assert "manipulator" in unknown
    assert "gis" in unknown
    # and it still loads
    assert isinstance(MicroscopeSettings.from_dict(config).system, SystemSettings)


# ---------------------------------------------------------------------------
# "cannot say" falls back to the backend, never to "not fitted"
# ---------------------------------------------------------------------------


def test_a_backend_that_cannot_probe_uses_its_own_default(microscope, monkeypatch):
    """The default probes all return None. This is the case that protects every site
    whose backend has no way to ask -- Tescan and Odemis today. Reading None as False
    would disable the GIS on all of them."""
    for name in (
        "_probe_manipulator_installed",
        "_probe_gis_installed",
        "_probe_multichem_installed",
        "_probe_sputter_coater_installed",
    ):
        monkeypatch.setattr(microscope, name, lambda: None)
    microscope.set_available("gis", False)
    microscope.set_available("manipulator", False)

    microscope._read_hardware_capabilities()

    assert microscope.is_available("gis") is microscope.DEFAULT_FITTED["gis"]
    assert (
        microscope.is_available("manipulator")
        is microscope.DEFAULT_FITTED["manipulator"]
    )


def test_a_probe_that_raises_is_read_as_cannot_say(microscope, monkeypatch):
    """A missing subsystem and a sick connection raise the same way. Reading an
    exception as "not fitted" would let one bad call at connect take the GIS away
    for the whole session."""

    def boom() -> Optional[bool]:
        raise RuntimeError("connection reset")

    monkeypatch.setattr(microscope, "_probe_gis_installed", boom)
    microscope.set_available("gis", False)

    microscope._read_hardware_capabilities()

    assert microscope.is_available("gis") is microscope.DEFAULT_FITTED["gis"]


def test_a_probe_that_answers_wins(microscope, monkeypatch):
    monkeypatch.setattr(microscope, "_probe_gis_installed", lambda: False)
    monkeypatch.setattr(microscope, "_probe_manipulator_installed", lambda: True)
    microscope.set_available("gis", True)
    microscope.set_available("manipulator", False)

    microscope._read_hardware_capabilities()

    assert microscope.is_available("gis") is False
    assert microscope.is_available("manipulator") is True


def test_every_probed_field_is_covered(microscope, monkeypatch):
    for name in (
        "_probe_manipulator_installed",
        "_probe_gis_installed",
        "_probe_multichem_installed",
        "_probe_sputter_coater_installed",
    ):
        monkeypatch.setattr(microscope, name, lambda: True)
    for key in ("manipulator", "gis", "gis_multichem", "gis_sputter_coater"):
        microscope.set_available(key, False)

    microscope._read_hardware_capabilities()

    for key in ("manipulator", "gis", "gis_multichem", "gis_sputter_coater"):
        assert microscope.is_available(key) is True, key


# ---------------------------------------------------------------------------
# The simulator answers from its sim: block, standing in for the probe
# ---------------------------------------------------------------------------


def test_the_demo_default_is_everything_but_a_sputter_coater(microscope):
    assert microscope.is_available("manipulator") is True
    assert microscope.is_available("gis") is True
    assert microscope.is_available("gis_multichem") is True
    assert microscope.is_available("gis_sputter_coater") is False


def test_a_simulated_file_can_say_otherwise():
    """sim-arctis: no manipulator, no multichem -- what its file used to state under
    `manipulator:` and `gis:`, now where the other stand-ins live."""
    path = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    microscope, _ = utils.setup_session(config_path=path, manufacturer="Demo")
    try:
        assert microscope.is_available("manipulator") is False
        assert microscope.is_available("gis") is True
        assert microscope.is_available("gis_multichem") is False
    finally:
        microscope.disconnect()


# ---------------------------------------------------------------------------
# It survives Apply
# ---------------------------------------------------------------------------


def test_applying_a_configuration_keeps_what_the_instrument_said(microscope):
    """`apply_configuration` used to replace `system.gis` and `system.manipulator`
    from the incoming settings. Those now carry only defaults, so replacing would
    restore `False` over what the instrument said at connect."""
    microscope.set_available("gis", True)
    microscope.set_available("gis_multichem", True)
    microscope.set_available("manipulator", True)

    incoming = SystemSettings.from_dict({})
    assert incoming.gis.enabled is False, "fixture no longer exercises the trap"
    microscope.apply_configuration(incoming)

    assert microscope.is_available("gis") is True
    assert microscope.is_available("gis_multichem") is True
    assert microscope.is_available("manipulator") is True


# ---------------------------------------------------------------------------
# The dispatch key that matched nothing
# ---------------------------------------------------------------------------


def test_the_multichem_capability_is_reachable(microscope):
    """`is_available` spells it `gis_multichem`. Two call sites asked for
    `"multichem"`, which matches no branch and falls through to `False`."""
    microscope.set_available("gis_multichem", True)
    assert microscope.is_available("gis_multichem") is True
    assert microscope.is_available("multichem") is False


def test_no_call_site_asks_for_the_key_that_matches_nothing():
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "fibsem"
    offenders = [
        f"{path.relative_to(root.parent)}:{i}"
        for path in root.rglob("*.py")
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r'is_available\(\s*["\']multichem["\']', line)
    ]
    assert offenders == []


def test_a_configuration_that_says_nothing_still_loads_and_images(microscope):
    image = microscope.acquire_image(beam_type=BeamType.ELECTRON)
    assert image is not None
