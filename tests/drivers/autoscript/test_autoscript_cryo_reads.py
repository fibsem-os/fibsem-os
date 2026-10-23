"""The AutoScript cryo reads, against a fake shaped like the vendor reference.

The chamber reads ``state.touch_alarm`` (4.13) and ``state.cryo`` (4.14); the sample
loader reads its two ``CryoCompartment``s (4.10, sensor temperatures from 4.13). All
are read-only, and each is bound only when the AutoScript in use answers it. Nothing
here has run on hardware.
"""

from typing import Dict, List, Optional

import pytest

from fibsem.devices.chamber import CryoCompartment
from fibsem.drivers.autoscript import devices as autoscript_devices
from fibsem.drivers.autoscript.connection import AutoScriptConnection
from tests.fixtures.microscopes import demo_microscope

CRYO_PARAMETERS = {
    "touch_alarm",
    "n2_flow",
    "n2_flow_on",
    "heat_exchanger_on",
    "stage_temperature",
    "shield_temperature",
}


class _Missing:
    """A member this AutoScript does not have: any use raises, as the SDK would."""

    def __getattr__(self, name: str):
        raise AttributeError(name)


class FakeTouchAlarm:
    def __init__(self) -> None:
        self.is_triggered = False
        self.source = "NoTouchAlarm"


class FakeN2Flow:
    def __init__(self) -> None:
        self.is_on = True
        self.value = 2.5e-4


class FakeHeatExchanger:
    def __init__(self) -> None:
        self.is_on = False


class FakeCryoTemperature:
    def __init__(self, readings: Dict[str, float]) -> None:
        self.readings = readings

    def get_available_sensors(self) -> List[str]:
        return list(self.readings)

    def get_temperature(self, sensor: str) -> float:
        return self.readings[sensor]


class FakeCryo:
    def __init__(self) -> None:
        self.n2_flow = FakeN2Flow()
        self.heat_exchanger = FakeHeatExchanger()
        self.temperature = FakeCryoTemperature({"Stage": 80.0, "Shield": 95.5})


class FakeState:
    def __init__(self, touch_alarm: bool = True, cryo: bool = True) -> None:
        self.touch_alarm = FakeTouchAlarm() if touch_alarm else _Missing()
        self.cryo = FakeCryo() if cryo else _Missing()


class FakeVacuum:
    chamber_state = "Pumped"

    class chamber_pressure:
        value = 1e-4


class FakeCompartment:
    """A ``CryoCompartment``; ``renamed`` is 4.14's ``get_temperature``, else 4.13's
    ``get_sensor_temperature``."""

    def __init__(self, readings: Dict[str, float], renamed: bool = True) -> None:
        self.state = "Cryo"
        self.dewar_level = 0.72
        self.readings = readings
        if renamed:
            self.get_temperature = self._read
        else:
            self.get_sensor_temperature = self._read

    def get_available_temperature_sensors(self) -> List[str]:
        return list(self.readings)

    def _read(self, sensor: str) -> float:
        if sensor == "Docker":
            raise RuntimeError("sensor not answering")
        return self.readings[sensor]


class FakeAutoloader:
    is_installed = True

    def __init__(self, compartments: Optional[Dict[str, FakeCompartment]]) -> None:
        self._compartments = compartments

    def get_slots(self, run_inventory: bool) -> list:
        return []

    def get_compartment(self, kind: str) -> FakeCompartment:
        if self._compartments is None:
            raise RuntimeError("get_compartment is not supported")
        return self._compartments[kind]


class FakeSpecimen:
    def __init__(self, autoloader: FakeAutoloader) -> None:
        self.autoloader = autoloader


class FakeConnection:
    def __init__(self, state: FakeState, autoloader: FakeAutoloader) -> None:
        self.vacuum = FakeVacuum()
        self.state = state
        self.specimen = FakeSpecimen(autoloader)


microscope = demo_microscope(scope="function")


Compartments = Optional[Dict[str, FakeCompartment]]


def _connection(microscope, state: FakeState, compartments: Compartments):
    return AutoScriptConnection(
        FakeConnection(state, FakeAutoloader(compartments)),
        microscope._threading_lock,
    )


def _chamber(
    microscope, state: FakeState, compartments: Compartments = None
) -> autoscript_devices.AutoscriptChamber:
    connection = _connection(microscope, state, compartments)
    return autoscript_devices.AutoscriptChamber(microscope, connection).connect()


def _loader(
    microscope, compartments: Compartments
) -> autoscript_devices.AutoscriptSampleLoader:
    connection = _connection(microscope, FakeState(), compartments)
    return autoscript_devices.AutoscriptSampleLoader(microscope, connection).connect()


class TestChamber:
    def test_reads_the_cryo_state(self, microscope):
        state = FakeState()
        chamber = _chamber(microscope, state)
        assert CRYO_PARAMETERS <= set(chamber.parameters)
        assert chamber.touch_alarm.get_value() == ""
        assert chamber.n2_flow.get_value() == pytest.approx(2.5e-4)
        assert chamber.n2_flow_on.get_value() is True
        assert chamber.heat_exchanger_on.get_value() is False
        assert chamber.stage_temperature.get_value() == 80.0
        assert chamber.shield_temperature.get_value() == 95.5

    def test_the_touch_alarm_names_what_touched(self, microscope):
        state = FakeState()
        chamber = _chamber(microscope, state)
        state.touch_alarm.is_triggered = True
        state.touch_alarm.source = "Manipulator"
        assert chamber.touch_alarm.get_value() == "Manipulator"

    def test_cryo_reads_are_read_only(self, microscope):
        chamber = _chamber(microscope, FakeState())
        for name in CRYO_PARAMETERS:
            assert not chamber.parameters[name].metadata.settable, name

    def test_an_older_autoscript_leaves_them_absent(self, microscope):
        """Before 4.13 there is no touch alarm, before 4.14 no ``state.cryo``, and
        before 4.10 no compartments: the chamber keeps its state and pressure."""
        chamber = _chamber(microscope, FakeState(touch_alarm=False, cryo=False))
        assert set(chamber.parameters) == {"state", "pressure"}

    def test_a_missing_sensor_leaves_its_temperature_absent(self, microscope):
        state = FakeState()
        state.cryo.temperature.readings.pop("Shield")
        chamber = _chamber(microscope, state)
        assert "stage_temperature" in chamber.parameters
        assert "shield_temperature" not in chamber.parameters


class TestCompartments:
    """The autoloader's two compartments: ``Chamber`` on the chamber, ``Autoloader``
    on the sample loader."""

    COMPARTMENTS = {
        "Autoloader": FakeCompartment({"Dewar": 78.0, "CassetteGripper": 90.0}),
        "Chamber": FakeCompartment({"CompustageHolder": 85.0}),
    }

    def test_each_device_reads_its_own(self, microscope):
        chamber = _chamber(microscope, FakeState(), self.COMPARTMENTS)
        loader = _loader(microscope, self.COMPARTMENTS)
        assert chamber.compartment.get_value() == CryoCompartment(
            "Cryo", 0.72, {"CompustageHolder": 85.0}
        )
        assert loader.compartment.get_value() == CryoCompartment(
            "Cryo", 0.72, {"Dewar": 78.0, "CassetteGripper": 90.0}
        )
        assert not chamber.parameters["compartment"].metadata.settable
        assert not loader.parameters["compartment"].metadata.settable

    def test_reads_4_13_sensor_temperatures(self, microscope):
        """4.13 reads a sensor with ``get_sensor_temperature``; 4.14 renamed it."""
        compartment = FakeCompartment({"Dewar": 78.0}, renamed=False)
        loader = _loader(microscope, {"Autoloader": compartment})
        assert loader.compartment.get_value().temperatures == {"Dewar": 78.0}

    def test_a_sensor_that_does_not_answer_is_left_out(self, microscope):
        compartment = FakeCompartment({"Dewar": 78.0, "Docker": 0.0})
        loader = _loader(microscope, {"Autoloader": compartment})
        assert loader.compartment.get_value().temperatures == {"Dewar": 78.0}

    def test_an_autoscript_without_compartments_leaves_them_absent(self, microscope):
        loader = _loader(microscope, None)
        assert "compartment" not in loader.parameters
        assert "magazine" in loader.parameters
