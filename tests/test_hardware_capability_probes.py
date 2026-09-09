"""Which subsystems are fitted is asked of the instrument, not of a file.

`manipulator.enabled`, `gis.enabled`, `gis.multichem` and `gis.sputter_coater` were
configuration keys, which meant a site could describe hardware it does not have, or
omit hardware it does, and nothing would disagree. The same move `rotation` made in
FIB-834, one record over.

The asymmetry that shapes every test here: a subsystem that wrongly *appears* is a menu
entry that errors when used, while one that wrongly *disappears* is a working
instrument that lost a feature on upgrade, silently. So "cannot say" must never mean
"not fitted".
"""

from typing import Optional

import pytest

from fibsem import utils
from fibsem.structures import BeamType


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    yield microscope
    microscope.disconnect()


# ---------------------------------------------------------------------------
# "cannot say" leaves the configuration alone
# ---------------------------------------------------------------------------


def test_a_backend_that_cannot_probe_keeps_the_configured_answer(microscope):
    """The default probes all return None, and the Demo backend does not override them.

    This is the case that protects every site whose backend has no way to ask -- Tescan
    and Odemis today. Reading None as False would disable the GIS on all of them.
    """
    microscope.system.gis.enabled = True
    microscope.system.gis.multichem = True
    microscope.system.manipulator.enabled = True

    microscope._read_hardware_capabilities()

    assert microscope.system.gis.enabled is True
    assert microscope.system.gis.multichem is True
    assert microscope.system.manipulator.enabled is True


def test_a_probe_that_raises_is_read_as_cannot_say(microscope, monkeypatch):
    """A missing subsystem and a sick connection raise the same way.

    Reading an exception as "not fitted" would let one bad call at connect take the
    GIS away for the whole session.
    """

    def boom() -> Optional[bool]:
        raise RuntimeError("connection reset")

    monkeypatch.setattr(microscope, "_probe_gis_installed", boom)
    microscope.system.gis.enabled = True

    microscope._read_hardware_capabilities()

    assert microscope.system.gis.enabled is True


def test_a_probe_that_answers_overrides_the_configuration(microscope, monkeypatch):
    """The point of the change: the instrument wins over the file."""
    monkeypatch.setattr(microscope, "_probe_gis_installed", lambda: False)
    monkeypatch.setattr(microscope, "_probe_manipulator_installed", lambda: True)
    microscope.system.gis.enabled = True
    microscope.system.manipulator.enabled = False

    microscope._read_hardware_capabilities()

    assert microscope.system.gis.enabled is False
    assert microscope.system.manipulator.enabled is True


def test_every_probed_field_is_covered(microscope, monkeypatch):
    for name in (
        "_probe_manipulator_installed",
        "_probe_gis_installed",
        "_probe_multichem_installed",
        "_probe_sputter_coater_installed",
    ):
        monkeypatch.setattr(microscope, name, lambda: True)
    microscope.system.gis.enabled = False
    microscope.system.gis.multichem = False
    microscope.system.gis.sputter_coater = False
    microscope.system.manipulator.enabled = False

    microscope._read_hardware_capabilities()

    assert microscope.system.gis.enabled
    assert microscope.system.gis.multichem
    assert microscope.system.gis.sputter_coater
    assert microscope.system.manipulator.enabled


# ---------------------------------------------------------------------------
# It survives Apply, which is where the stage capability was lost
# ---------------------------------------------------------------------------


def test_applying_a_configuration_re_reads_the_capabilities(microscope, monkeypatch):
    """`apply_configuration` replaces `system.gis` and `system.manipulator` wholesale.

    The shipped ThermoFisher files no longer state these, so a replacement restores the
    field defaults -- `False` -- over what the instrument said at connect. Exactly the
    trap `rotation` fell into, which is why that call re-reads too.
    """
    monkeypatch.setattr(microscope, "_probe_gis_installed", lambda: True)
    monkeypatch.setattr(microscope, "_probe_manipulator_installed", lambda: True)
    microscope._read_hardware_capabilities()
    assert microscope.system.gis.enabled and microscope.system.manipulator.enabled

    from fibsem.structures import SystemSettings

    # a configuration that says nothing about what is fitted, as the tfs files now do
    incoming = SystemSettings.from_dict({})
    assert incoming.gis.enabled is False, "fixture no longer exercises the trap"
    microscope.apply_configuration(incoming)

    assert microscope.system.gis.enabled is True
    assert microscope.system.manipulator.enabled is True


# ---------------------------------------------------------------------------
# The dispatch key that matched nothing
# ---------------------------------------------------------------------------


def test_the_multichem_capability_is_reachable(microscope):
    """`is_available` spells it `gis_multichem`.

    Two call sites asked for `"multichem"`, which matches no branch and falls through
    to `return False` -- so the multichem path was unreachable regardless of hardware.
    It never showed up because every shipped configuration set `gis.enabled: true`,
    and the `gis` branch is tested first at both sites.
    """
    microscope.system.gis.multichem = True
    assert microscope.is_available("gis_multichem") is True
    assert microscope.is_available("multichem") is False

    microscope.system.gis.multichem = False
    assert microscope.is_available("gis_multichem") is False


def test_no_call_site_asks_for_the_key_that_matches_nothing():
    """A grep, as a test, because the failure is silent and reintroducing it is easy."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "fibsem"
    offenders = [
        f"{path.relative_to(root.parent)}:{i}"
        for path in root.rglob("*.py")
        for i, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r'is_available\(\s*["\']multichem["\']', line)
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# The shipped files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "tfs-aquilos2-configuration.yaml",
        "tfs-arctis-configuration.yaml",
        "tfs-hydra-configuration.yaml",
    ],
)
def test_the_autoscript_configurations_no_longer_state_what_is_fitted(filename):
    import os

    import fibsem.config as cfg

    config = utils.load_yaml(os.path.join(cfg.CONFIG_PATH, filename))

    assert "manipulator" not in config
    assert "gis" not in config


@pytest.mark.parametrize(
    "filename",
    ["tescan-configuration.yaml", "odemis-configuration.yaml"],
)
def test_the_backends_that_cannot_probe_keep_their_keys(filename):
    """Deliberate, not an oversight.

    `list_all_gis_ports` and `specimen.manipulator.is_installed` are AutoScript APIs.
    Nothing here can ask a Tescan or an Odemis what is fitted, so removing their keys
    would disable the GIS on every one of those systems on the strength of a guess.
    """
    import os

    import fibsem.config as cfg

    config = utils.load_yaml(os.path.join(cfg.CONFIG_PATH, filename))

    assert config["gis"]["enabled"] is True
    assert "manipulator" in config


def test_a_configuration_that_says_nothing_still_loads_and_images(microscope):
    """The end state for a ThermoFisher file: no capability keys at all."""
    image = microscope.acquire_image(beam_type=BeamType.ELECTRON)
    assert image is not None
