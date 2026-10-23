"""Which subsystems are fitted is not in the configuration file.

`manipulator.enabled` was a configuration key, which meant a site could describe
hardware it does not have, or omit hardware it does, and nothing would disagree. The
same move `rotation` made in FIB-834, one record over: the device answers
(`Manipulator.is_installed`), and where it cannot say, the backend answers for itself
with `DEFAULT_FITTED`, which is what its shipped configuration used to say. The ion
beam device likewise reports the gas its source runs (`Beam.installed_plasma_gas`).

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
from fibsem.devices.core import Axes
from fibsem.structures import BeamType, MicroscopeSettings, SystemSettings
from tests.fixtures.microscopes import demo_microscope
from tests.test_configuration_schema import SHIPPED

microscope = demo_microscope(scope="function")


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


def test_a_device_that_cannot_say_uses_the_backend_s_default(microscope, monkeypatch):
    """The base device answers None. This is the case that protects every site whose
    device has no way to ask. Reading None as False would disable the manipulator on
    all of them."""
    device = microscope.manipulator_device
    monkeypatch.setattr(device, "is_installed", lambda: None)

    microscope._keep_if_fitted("manipulator")

    assert microscope.DEFAULT_FITTED["manipulator"] is True
    assert microscope.manipulator_device is device
    assert microscope.is_available("manipulator") is True


def test_a_device_that_raises_is_read_as_cannot_say(microscope, monkeypatch):
    """A missing subsystem and a sick connection raise the same way. Reading an
    exception as "not fitted" would let one bad call at connect take the manipulator
    away for the whole session."""

    def boom() -> Optional[bool]:
        raise RuntimeError("connection reset")

    device = microscope.manipulator_device
    monkeypatch.setattr(device, "is_installed", boom)

    microscope._keep_if_fitted("manipulator")

    assert microscope.manipulator_device is device
    assert device.facts()["is_installed"] is None


def test_a_device_that_answers_wins(microscope, monkeypatch):
    monkeypatch.setattr(microscope.manipulator_device, "is_installed", lambda: False)

    microscope._keep_if_fitted("manipulator")

    assert microscope.manipulator_device is None
    assert microscope.is_available("manipulator") is False


# ---------------------------------------------------------------------------
# The simulator answers from its sim: block, standing in for the probe
# ---------------------------------------------------------------------------


def test_the_demo_default_is_a_manipulator(microscope):
    assert microscope.is_available("manipulator") is True


def test_a_simulated_file_can_say_otherwise():
    """sim-arctis: no manipulator -- what its file used to state under
    `manipulator:`, now where the other stand-ins live."""
    path = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    microscope, _ = utils.setup_session(config_path=path, manufacturer="Demo")
    try:
        assert microscope.is_available("manipulator") is False
        assert microscope.capability_sources["manipulator"] == "instrument"
    finally:
        microscope.disconnect()


def test_the_demo_needle_reports_the_sim_block(microscope):
    """The `sim:` block stands in for the instrument; absent is "cannot say"."""
    device = microscope.manipulator_device
    assert device.is_installed() is None
    assert device.facts()["is_installed"] is None

    microscope.system.sim["has_manipulator"] = True
    assert device.facts()["is_installed"] is True


# ---------------------------------------------------------------------------
# It survives Apply
# ---------------------------------------------------------------------------


def test_a_configuration_that_says_nothing_still_loads_and_images(microscope):
    image = microscope.acquire_image(beam_type=BeamType.ELECTRON)
    assert image is not None


# ---------------------------------------------------------------------------
# The plasma gas is the instrument's when it can say
# ---------------------------------------------------------------------------


def test_an_old_plasma_flag_without_a_gas_gets_the_instrument_s_gas(tmp_path):
    """`plasma: true` with `plasma_gas: None` loads as no gas -- a plasma column is
    one with a gas. The instrument names it at connect, so the site keeps its plasma
    controls instead of losing them on upgrade."""
    import yaml

    config = _load("sim-arctis-configuration.yaml")
    config["ion"] = {"plasma": True, "plasma_gas": None}
    config["sim"]["plasma_gas"] = "Xenon"
    assert MicroscopeSettings.from_dict(config).system.ion.plasma is False

    path = tmp_path / "site.yaml"
    path.write_text(yaml.safe_dump(config))
    microscope, _ = utils.setup_session(
        session_path=str(tmp_path), config_path=str(path), setup_logging=False
    )
    try:
        assert microscope.system.ion.plasma_gas == "Xenon"
        assert microscope.is_available("ion_plasma") is True
        # the ion device was built before the gas was known, and gains the control
        ion = microscope.beams[BeamType.ION]
        assert ion.plasma_gas.get_value() == "Xenon"
    finally:
        microscope.disconnect()


def test_a_gas_the_configuration_states_is_left_alone(microscope, monkeypatch):
    """The instrument is not asked, and its answer never replaces the file's."""

    asked = []

    def probe() -> Optional[str]:
        asked.append(True)
        return "Xenon"

    microscope.system.ion.plasma_gas = "Argon"
    monkeypatch.setattr(microscope.beams[BeamType.ION], "installed_plasma_gas", probe)

    microscope._read_plasma_source()

    assert microscope.system.ion.plasma_gas == "Argon"
    assert asked == []


def test_reading_the_gas_does_not_set_one(microscope, monkeypatch):
    """Recording the gas is a read: no call reaches the instrument's setter."""
    calls = []
    original_write = microscope._write_beam
    monkeypatch.setattr(
        microscope,
        "_write_beam",
        lambda name, *a, **k: calls.append(name) or original_write(name, *a, **k),
    )
    microscope.system.ion.plasma_gas = None
    monkeypatch.setattr(
        microscope.beams[BeamType.ION], "installed_plasma_gas", lambda: "Xenon"
    )

    microscope._read_plasma_source()

    assert microscope.system.ion.plasma_gas == "Xenon"
    assert "plasma_gas" not in calls


@pytest.mark.parametrize("answer", [None, "", "raises"])
def test_a_probe_that_cannot_say_leaves_the_file_s_answer(
    microscope, monkeypatch, answer
):
    """On a Ga column AutoScript refuses the question; that must not invent a
    plasma source."""

    def probe() -> Optional[str]:
        if answer == "raises":
            raise RuntimeError("no plasma source")
        return answer

    monkeypatch.setattr(microscope.beams[BeamType.ION], "installed_plasma_gas", probe)
    microscope.system.ion.plasma_gas = None

    microscope._read_plasma_source()

    assert microscope.system.ion.plasma_gas is None


def test_autoscript_asks_the_ion_source():
    """The read the `plasma_gas` parameter makes, asked once at connect."""
    from types import SimpleNamespace

    from fibsem.drivers.autoscript.devices import AutoscriptBeam

    def probe(source, beam_type=BeamType.ION) -> Optional[str]:
        fake = SimpleNamespace(
            beam_type=beam_type, _beam=SimpleNamespace(source=source)
        )
        return AutoscriptBeam.installed_plasma_gas(fake)

    assert (
        probe(SimpleNamespace(plasma_gas=SimpleNamespace(value="Oxygen"))) == "Oxygen"
    )
    assert probe(SimpleNamespace()) is None
    assert probe(SimpleNamespace(plasma_gas=SimpleNamespace(value=""))) is None
    xenon = SimpleNamespace(plasma_gas=SimpleNamespace(value="Xenon"))
    assert probe(xenon, BeamType.ELECTRON) is None


def test_the_ion_beam_reports_its_gas_as_a_fact(microscope):
    """The demo's `sim:` block stands in for the source; a remote copy reads it."""
    ion = microscope.beams[BeamType.ION]
    assert ion.facts() == {"installed_plasma_gas": None}

    microscope.system.sim["plasma_gas"] = "Argon"
    assert ion.facts() == {"installed_plasma_gas": "Argon"}
    assert microscope.beams[BeamType.ELECTRON].facts() == {"installed_plasma_gas": None}


def test_a_plasma_column_is_one_with_a_gas(microscope):
    microscope.system.ion.plasma_gas = None
    assert microscope.is_available("ion_plasma") is False

    microscope.system.ion.plasma_gas = "Xenon"
    assert microscope.is_available("ion_plasma") is True


def test_the_source_of_each_answer_is_recorded(microscope, monkeypatch):
    """The configuration window says whether the instrument answered or the backend's
    default stood in; the two look the same in `is_available`."""
    device = microscope.manipulator_device
    monkeypatch.setattr(device, "is_installed", lambda: True)
    microscope._read_hardware_capabilities()
    assert microscope.capability_sources["manipulator"] == "instrument"

    monkeypatch.setattr(device, "is_installed", lambda: None)
    microscope._read_hardware_capabilities()
    assert microscope.capability_sources["manipulator"] == "backend"

    def boom():
        raise RuntimeError("connection reset")

    monkeypatch.setattr(device, "is_installed", boom)
    microscope._read_hardware_capabilities()
    assert microscope.capability_sources["manipulator"] == "backend"

    # one the device said is not fitted keeps that answer once it is gone
    monkeypatch.setattr(device, "is_installed", lambda: False)
    microscope._keep_if_fitted("manipulator")
    microscope._read_hardware_capabilities()
    assert microscope.capability_sources["manipulator"] == "instrument"


# ---------------------------------------------------------------------------
# What is available is what was built
# ---------------------------------------------------------------------------


def test_availability_is_the_device_map(microscope):
    """No second record: a subsystem is there when its device is."""
    assert microscope.is_available("electron_beam") is (
        "electron" in microscope.devices
    )
    assert microscope.is_available("ion_beam") is ("ion" in microscope.devices)
    assert microscope.is_available("stage") is (microscope.stage is not None)
    assert microscope.is_available("manipulator") is (
        microscope.manipulator_device is not None
    )

    microscope.beams = {BeamType.ELECTRON: microscope.beams[BeamType.ELECTRON]}
    assert microscope.is_available("ion_beam") is False


def test_a_disabled_column_is_not_available(tmp_path):
    """`enabled: false` builds no device, so the column is not there."""
    import yaml

    config = _load("microscope-configuration.yaml")
    config["hardware"]["devices"] = [{"name": "ion", "enabled": False}]
    path = tmp_path / "single-beam.yaml"
    path.write_text(yaml.safe_dump(config))
    microscope, _ = utils.setup_session(
        session_path=str(tmp_path),
        config_path=str(path),
        manufacturer="Demo",
        setup_logging=False,
    )
    try:
        assert "ion" not in microscope.devices
        assert microscope.is_available("ion_beam") is False
        assert microscope.is_available("electron_beam") is True
    finally:
        microscope.disconnect()


def test_the_arm_s_axes_are_its_device_s(microscope, monkeypatch):
    device = microscope.manipulator_device
    assert microscope.is_available("manipulator_rotation") is False

    monkeypatch.setattr(device, "axes", Axes({a: None for a in ("x", "y", "z", "r")}))
    assert microscope.is_available("manipulator_rotation") is True
    assert microscope.is_available("manipulator_tilt") is False

    microscope.manipulator_device = None
    assert microscope.is_available("manipulator_rotation") is False


def test_stage_rotation_is_the_stage_s_r_axis(microscope):
    assert microscope.is_available("stage_rotation") is ("r" in microscope.stage.axes)


def test_an_unknown_subsystem_is_not_available(microscope):
    assert microscope.is_available("gis") is False
