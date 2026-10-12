"""Which subsystems are fitted is not in the configuration file.

`manipulator.enabled` was a configuration key, which meant a site could describe
hardware it does not have, or omit hardware it does, and nothing would disagree. The
same move `rotation` made in FIB-834, one record over: a backend that can ask the
instrument does (AutoScript), and one that cannot answers for itself with
`DEFAULT_FITTED`, which is what its shipped configuration used to say.

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
    would disable the manipulator on all of them."""
    monkeypatch.setattr(microscope, "_probe_manipulator_installed", lambda: None)

    microscope._read_hardware_capabilities()

    assert (
        microscope._is_fitted("manipulator") is microscope.DEFAULT_FITTED["manipulator"]
    )


def test_a_probe_that_raises_is_read_as_cannot_say(microscope, monkeypatch):
    """A missing subsystem and a sick connection raise the same way. Reading an
    exception as "not fitted" would let one bad call at connect take the manipulator
    away for the whole session."""

    def boom() -> Optional[bool]:
        raise RuntimeError("connection reset")

    monkeypatch.setattr(microscope, "_probe_manipulator_installed", boom)

    microscope._read_hardware_capabilities()

    assert (
        microscope._is_fitted("manipulator") is microscope.DEFAULT_FITTED["manipulator"]
    )


def test_a_probe_that_answers_wins(microscope, monkeypatch):
    monkeypatch.setattr(microscope, "_probe_manipulator_installed", lambda: False)

    microscope._read_hardware_capabilities()

    assert microscope._is_fitted("manipulator") is False


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
    finally:
        microscope.disconnect()


# ---------------------------------------------------------------------------
# It survives Apply
# ---------------------------------------------------------------------------


def test_applying_a_configuration_keeps_what_the_instrument_said():
    """`apply_configuration` used to replace `system.manipulator` from the incoming
    settings. That now carries only defaults, so replacing would restore the default
    over what the instrument said at connect. The answer is the device map, which
    Apply does not touch."""
    path = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    microscope, _ = utils.setup_session(config_path=path, manufacturer="Demo")
    try:
        assert microscope.is_available("manipulator") is False

        incoming = SystemSettings.from_dict({})
        assert incoming.manipulator.enabled is True, (
            "fixture no longer exercises the trap"
        )
        microscope.apply_configuration(incoming)

        assert microscope.is_available("manipulator") is False
    finally:
        microscope.disconnect()


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
    finally:
        microscope.disconnect()


def test_a_gas_the_configuration_states_is_left_alone(microscope, monkeypatch):
    """The instrument is not asked, and its answer never replaces the file's."""

    asked = []

    def probe() -> Optional[str]:
        asked.append(True)
        return "Xenon"

    microscope.system.ion.plasma_gas = "Argon"
    monkeypatch.setattr(microscope, "_probe_plasma_gas", probe)

    microscope._read_plasma_source()

    assert microscope.system.ion.plasma_gas == "Argon"
    assert asked == []


def test_reading_the_gas_does_not_set_one(microscope, monkeypatch):
    """Recording the gas is a read: no call reaches the instrument's setter."""
    calls = []
    original_set = microscope.set
    monkeypatch.setattr(
        microscope,
        "set",
        lambda key, *a, **k: calls.append(key) or original_set(key, *a, **k),
    )
    microscope.system.ion.plasma_gas = None
    monkeypatch.setattr(microscope, "_probe_plasma_gas", lambda: "Xenon")

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

    monkeypatch.setattr(microscope, "_probe_plasma_gas", probe)
    microscope.system.ion.plasma_gas = None

    microscope._read_plasma_source()

    assert microscope.system.ion.plasma_gas is None


def test_autoscript_asks_the_ion_source():
    """The call `get("plasma_gas")` already makes, asked once at connect."""
    from types import SimpleNamespace

    from fibsem.drivers.autoscript.microscope import ThermoMicroscope

    def probe(source) -> Optional[str]:
        fake = SimpleNamespace(
            connection=SimpleNamespace(
                beams=SimpleNamespace(ion_beam=SimpleNamespace(source=source))
            )
        )
        return ThermoMicroscope._probe_plasma_gas(fake)

    assert (
        probe(SimpleNamespace(plasma_gas=SimpleNamespace(value="Oxygen"))) == "Oxygen"
    )
    assert probe(SimpleNamespace()) is None
    assert probe(SimpleNamespace(plasma_gas=SimpleNamespace(value=""))) is None


def test_a_plasma_column_is_one_with_a_gas(microscope):
    microscope.system.ion.plasma_gas = None
    assert microscope.is_available("ion_plasma") is False

    microscope.system.ion.plasma_gas = "Xenon"
    assert microscope.is_available("ion_plasma") is True


def test_the_source_of_each_answer_is_recorded(microscope, monkeypatch):
    """The configuration window says whether the instrument answered or the backend's
    default stood in; the two look the same in `is_available`."""
    monkeypatch.setattr(microscope, "_probe_manipulator_installed", lambda: True)
    microscope._read_hardware_capabilities()
    assert microscope.capability_sources["manipulator"] == "instrument"

    monkeypatch.setattr(microscope, "_probe_manipulator_installed", lambda: None)
    microscope._read_hardware_capabilities()
    assert microscope.capability_sources["manipulator"] == "backend"

    def boom():
        raise RuntimeError("connection reset")

    monkeypatch.setattr(microscope, "_probe_manipulator_installed", boom)
    microscope._read_hardware_capabilities()
    assert microscope.capability_sources["manipulator"] == "backend"


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
    assert microscope.get_available_beams() == [BeamType.ELECTRON]


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

    monkeypatch.setattr(device, "axes", lambda: ("x", "y", "z", "r"))
    assert microscope.is_available("manipulator_rotation") is True
    assert microscope.is_available("manipulator_tilt") is False

    microscope.manipulator_device = None
    assert microscope.is_available("manipulator_rotation") is False


def test_stage_rotation_is_the_stage_s_r_axis(microscope):
    assert microscope.is_available("stage_rotation") is ("r" in microscope.stage.axes)


def test_an_unknown_subsystem_is_not_available(microscope):
    assert microscope.is_available("gis") is False
