"""The AutoScript gas injector against a fake of the AutoScript gas API.

The fake follows the AutoScript reference: ``gas.list_all_gis_ports()`` names each
port by its gas, ``gas.get_gis_port(name)`` and ``gas.get_multichem()`` return the
vendor objects, a multichem's heater and temperature take a gas name, and from
AutoScript 4.13.1 a GIS port reads back ``state``, ``is_heated`` and ``is_open``.
Nothing here has run on hardware -- that is the Arctis bench check.
"""

from types import SimpleNamespace
from typing import List

import pytest

from fibsem import utils
from fibsem.drivers.autoscript import devices as autoscript_devices
from fibsem.drivers.autoscript.connection import AutoScriptConnection
from fibsem.drivers.registry import device_builder
from fibsem.structures import DeviceEntry, InsertableDeviceState

# ---------------------------------------------------------------------------
# A fake gas API, shaped like the vendor's
# ---------------------------------------------------------------------------


class FakeGisPort:
    def __init__(self, calls: List[tuple], temperature: float = 310.0) -> None:
        self.calls = calls
        self.state = "Retracted"
        self.is_heated = False
        self.is_open = False
        self.temperature = temperature

    def insert(self, *args) -> None:
        self.calls.append(("insert", *args))
        self.state = "Inserted"

    def retract(self) -> None:
        self.calls.append(("retract",))
        self.state = "Retracted"

    def turn_heater_on(self, *args) -> None:
        self.calls.append(("turn_heater_on", *args))
        self.is_heated = True

    def turn_heater_off(self, *args) -> None:
        self.calls.append(("turn_heater_off", *args))
        self.is_heated = False

    def open(self) -> None:
        self.calls.append(("open",))
        self.is_open = True

    def close(self) -> None:
        self.calls.append(("close",))
        self.is_open = False

    def get_temperature(self, *args) -> float:
        self.calls.append(("get_temperature", *args))
        return self.temperature


class FakeGas:
    def __init__(self, ports=("Pt dep",), multichem: bool = False) -> None:
        self.calls: List[tuple] = []
        self.ports = {name: FakeGisPort(self.calls) for name in ports}
        self.multichem = FakeGisPort(self.calls) if multichem else None

    def list_all_gis_ports(self) -> List[str]:
        return list(self.ports)

    def list_all_multichem_ports(self) -> List[str]:
        return ["Port 1"] if self.multichem else []

    def get_gis_port(self, name: str) -> FakeGisPort:
        return self.ports[name]

    def get_multichem(self) -> FakeGisPort:
        return self.multichem


class FakeSdk:
    def __init__(self, gas: FakeGas, version: str) -> None:
        self.gas = gas
        autoscript = SimpleNamespace(
            client=SimpleNamespace(version=version),
            server=SimpleNamespace(version=version),
        )
        self.service = SimpleNamespace(autoscript=autoscript)


@pytest.fixture(autouse=True)
def no_heater_wait(monkeypatch):
    monkeypatch.setattr(autoscript_devices, "GIS_HEATER_TIMEOUT", 0.0)


def _microscope(
    gas: FakeGas,
    model: str = "Arctis",
    version: str = "4.13.1",
):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.system.info.model = model
    connection = AutoScriptConnection(FakeSdk(gas, version), microscope._threading_lock)
    microscope._connection = connection
    return microscope


def _build(microscope, **options):
    entry = DeviceEntry(name="gis", type="gis", options=options)
    context = SimpleNamespace(microscope=microscope)
    return autoscript_devices.build_autoscript_gis(entry, context)


# ---------------------------------------------------------------------------
# Arctis and Hydra Bio only
# ---------------------------------------------------------------------------


class TestWhereItIsBuilt:
    def test_the_driver_has_a_gis_builder(self):
        assert device_builder("thermofisher", "gis") is not None

    def test_built_by_default_on_an_arctis_with_a_port(self):
        assert autoscript_devices.gis_fitted(_microscope(FakeGas()))

    def test_built_by_default_on_a_hydra_bio_with_a_port(self):
        assert autoscript_devices.gis_fitted(_microscope(FakeGas(), model="Hydra Bio"))

    @pytest.mark.parametrize("model", ["Aquilos 2", "Helios 5", ""])
    def test_not_built_by_default_on_another_instrument(self, model):
        assert not autoscript_devices.gis_fitted(_microscope(FakeGas(), model=model))

    def test_not_built_by_default_on_an_arctis_without_a_gis(self):
        assert not autoscript_devices.gis_fitted(_microscope(FakeGas(ports=())))

    def test_a_configured_gis_is_refused_on_another_instrument(self):
        with pytest.raises(ValueError, match="Arctis or a Hydra Bio only"):
            _build(_microscope(FakeGas(), model="Aquilos 2"))

    def test_an_entry_forces_the_variant_on_a_bench(self):
        gis = _build(_microscope(FakeGas(), model="Aquilos 2"), variant="hydra_bio")
        assert gis.port == "Pt dep"

    def test_an_unknown_variant_is_refused(self):
        with pytest.raises(ValueError, match="variant is one of"):
            _build(_microscope(FakeGas()), variant="helios")

    def test_a_hydra_bio_deposits_from_the_cryo_platinum_port(self):
        gas = FakeGas(ports=("C dep", "Pt dep Cryo"), multichem=True)
        gis = _build(_microscope(gas, model="Hydra Bio"))
        assert isinstance(gis, autoscript_devices.AutoscriptGisPort)
        assert gis.gas.get_value() == "Pt dep Cryo"

    def test_an_arctis_keeps_the_multichem_first(self):
        gas = FakeGas(ports=("C dep", "Pt dep Cryo"), multichem=True)
        gis = _build(_microscope(gas))
        assert isinstance(gis, autoscript_devices.AutoscriptMultichem)

    def test_a_port_the_instrument_does_not_list_is_refused(self):
        with pytest.raises(ValueError, match="no GIS port"):
            _build(_microscope(FakeGas()), port="W dep")


# ---------------------------------------------------------------------------
# A GIS port
# ---------------------------------------------------------------------------


class TestGisPort:
    def test_builds_the_first_port_with_its_gas(self):
        gis = _build(_microscope(FakeGas(ports=("Pt dep", "C dep"))))
        assert isinstance(gis, autoscript_devices.AutoscriptGisPort)
        assert gis.name == "gis"
        assert gis.gas.get_value() == "Pt dep"
        assert gis.facts() == {"is_installed": True}

    def test_reads_the_port_back_from_4_13_1(self):
        gas = FakeGas()
        gis = _build(_microscope(gas))
        assert gis.read_back
        # a change made in xT shows
        gas.ports["Pt dep"].state = "Inserted"
        gas.ports["Pt dep"].is_heated = True
        assert gis.state.get_value() is InsertableDeviceState.INSERTED
        assert gis.heated.get_value() is True

    @pytest.mark.parametrize("version", ["4.13.0", "4.9", "not a version"])
    def test_tracks_its_own_commands_before_4_13_1(self, version):
        gas = FakeGas()
        gis = _build(_microscope(gas, version=version))
        assert not gis.read_back
        gas.ports["Pt dep"].state = "Inserted"  # unseen: no read-back
        assert gis.state.get_value() is InsertableDeviceState.RETRACTED
        assert gis.insert() is InsertableDeviceState.INSERTED
        assert gis.heater_on() is True
        assert gis.open() is True
        assert gis.close() is False

    def test_insert_heat_open_close_retract(self):
        gas = FakeGas()
        gis = _build(_microscope(gas))
        assert gis.insert() is InsertableDeviceState.INSERTED
        assert gis.heater_on() is True
        assert gis.temperature.get_value() == 310.0
        assert gis.open() is True
        assert gis.close() is False
        assert gis.retract() is InsertableDeviceState.RETRACTED
        assert [c[0] for c in gas.calls if c[0] != "get_temperature"] == [
            "insert",
            "turn_heater_on",
            "open",
            "close",
            "retract",
        ]

    def test_heater_on_fails_when_the_gas_stays_cold(self):
        gas = FakeGas()
        gas.ports["Pt dep"].temperature = 250.0
        gis = _build(_microscope(gas))
        with pytest.raises(TimeoutError, match="250.0 K"):
            gis.heater_on()

    def test_a_port_has_no_other_gas(self):
        gis = _build(_microscope(FakeGas()))
        with pytest.raises(ValueError, match="multichem only"):
            gis.heater_on("W(CO)6")

    def test_purge_opens_waits_and_closes(self, monkeypatch):
        gas = FakeGas()
        gis = _build(_microscope(gas))
        monkeypatch.setattr(gis, "_wait", lambda seconds: None)
        assert gis.purge(2.0) is False
        assert [c[0] for c in gas.calls] == ["open", "close"]


# ---------------------------------------------------------------------------
# The multichem
# ---------------------------------------------------------------------------


class TestMultichem:
    def test_preferred_over_a_port(self):
        gis = _build(_microscope(FakeGas(multichem=True)))
        assert isinstance(gis, autoscript_devices.AutoscriptMultichem)
        assert gis.facts() == {"is_installed": True}

    def test_a_named_port_wins(self):
        gis = _build(_microscope(FakeGas(multichem=True)), port="Pt dep")
        assert isinstance(gis, autoscript_devices.AutoscriptGisPort)

    def test_heater_and_temperature_take_the_gas(self):
        gas = FakeGas(multichem=True)
        gis = _build(_microscope(gas))
        assert gis.gas.get_value() == ""
        with pytest.raises(ValueError, match="name the gas"):
            gis.heater_on()
        assert gis.heater_on("Pt") is True
        assert gis.gas.get_value() == "Pt"
        assert gis.heater_off() is False
        assert ("turn_heater_on", "Pt") in gas.calls
        assert ("turn_heater_off", "Pt") in gas.calls
        assert ("get_temperature", "Pt") in gas.calls

    def test_inserts_to_a_named_position_and_reads_its_state(self):
        gas = FakeGas(multichem=True)
        gis = _build(_microscope(gas))
        assert gis.insert("ELECTRON_DEFAULT") is InsertableDeviceState.INSERTED
        assert ("insert", "ELECTRON_DEFAULT") in gas.calls
        gas.multichem.state = "Busy"
        assert gis.state.get_value() is InsertableDeviceState.UNKNOWN
