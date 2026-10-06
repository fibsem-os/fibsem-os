"""Services have the device shape without being devices.

A service declares parameters, commands and roles as a device does, and they behave the
same: binding, change events, metadata, the microscope's resources and the role checks.
It is not a `Device`, and a device's own behaviour is unchanged by the split.
"""

import pytest

import fibsem.devices
from fibsem.devices.core import (
    Device,
    Parameter,
    ParameterMetadata,
    Resources,
    Role,
    RoleUnfilled,
    command,
)
from fibsem.services import Service


class Beam(Device):
    current = Parameter(float, unit="A")

    def __init__(self, name="ion", **kwargs):
        super().__init__(name, **kwargs)
        self._current = 1e-9

    def read_current(self):
        return self._current

    def write_current(self, value):
        self._current = value


class Milling(Service):
    beam = Role(Beam)
    state = Parameter(str)

    def __init__(self, **kwargs):
        super().__init__("milling", **kwargs)
        self._state = "idle"

    def read_state(self):
        return self._state

    def metadata_state(self):
        return ParameterMetadata(choices=["idle", "running"])

    @command
    def start(self) -> None:
        """Start milling."""
        self._state = "running"
        self.state.get_value()


def test_a_service_is_not_a_device():
    assert not issubclass(Service, Device)
    assert not hasattr(fibsem.devices, "Service")


def test_a_service_binds_parameters_commands_and_roles_as_a_device_does():
    beam = Beam().connect()
    milling = Milling().fill_roles(beam=beam).connect()
    assert sorted(milling.parameters) == ["state"]
    assert milling.state.choices == ["idle", "running"]
    assert not milling.state.writable
    assert milling.commands["start"].available
    assert milling.beam is beam
    assert milling.describe()["state"]["type"] == "str"


def test_a_service_emits_changes():
    milling = Milling().fill_roles(beam=Beam().connect()).connect()
    milling.state.get_value()  # a first read sets the cache, with no event
    seen = []
    milling.changed.connect(lambda name, value: seen.append((name, value)))
    milling.start()
    assert seen == [("state", "running")]


def test_a_service_uses_its_beam_through_the_role():
    beam = Beam().connect()
    milling = Milling().fill_roles(beam=beam).connect()
    milling.beam.current.set_value(2e-9)
    assert beam.current.cached == 2e-9


def test_a_service_needs_its_required_roles():
    with pytest.raises(RoleUnfilled):
        Milling().connect()


def test_a_service_shares_its_parents_resources():
    class Microscope:
        resources = Resources()

    parent = Microscope()
    assert Milling(parent=parent).resources is parent.resources


def test_a_service_cant_change_a_parameters_type():
    with pytest.raises(TypeError):

        class Bad(Milling):
            state = Parameter(int)
