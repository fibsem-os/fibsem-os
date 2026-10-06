"""Roles: a place on a device that another device fills, typed by what it must be.

A builder fills a device's roles before connecting it; the device reaches each part
through the attribute, and an unfilled one is an error, at connect for a required role
and on reading for any.
"""

import pytest

from fibsem.devices.core import Device, Parameter, Role, RoleUnfilled
from fibsem.devices.drivers.fm import bind_fm_devices
from fibsem.devices.fm import FM, Camera, FilterSet, LightSource, Objective
from fibsem.fm.microscope import FluorescenceMicroscope


class Sensor(Device):
    reading = Parameter(float)

    def read_reading(self):
        return 1.0


class Lamp(Device):
    pass


class Rig(Device):
    sensor = Role(Sensor)
    lamp = Role(Lamp, required=False)


class BiggerRig(Rig):
    spare = Role(Sensor, required=False)


def test_a_filled_role_is_the_device_in_it():
    sensor = Sensor("sensor").connect()
    rig = Rig("rig").fill_roles(sensor=sensor).connect()
    assert rig.sensor is sensor
    assert rig.sensor.reading.get_value() == 1.0
    assert rig.roles == {"sensor": sensor}


def test_the_class_lists_its_roles_and_a_subclass_inherits_them():
    assert list(Rig.declared_roles()) == ["sensor", "lamp"]
    assert list(BiggerRig.declared_roles()) == ["sensor", "lamp", "spare"]
    assert Rig.sensor.interface is Sensor
    assert Rig.sensor.required and not Rig.lamp.required


def test_connect_refuses_a_required_role_left_unfilled():
    with pytest.raises(RoleUnfilled, match="sensor"):
        Rig("rig").connect()


def test_an_optional_role_may_stay_empty_but_reading_it_raises():
    rig = Rig("rig").fill_roles(sensor=Sensor("sensor")).connect()
    assert "lamp" not in rig.roles
    with pytest.raises(RoleUnfilled, match="lamp"):
        rig.lamp
    # A missing part reads as unsupported, the way a missing parameter does.
    assert not hasattr(rig, "lamp")


def test_a_role_takes_only_its_interface():
    with pytest.raises(TypeError, match="takes a Sensor"):
        Rig("rig").fill_roles(sensor=Lamp("lamp"))


def test_a_role_the_class_does_not_declare_is_an_error():
    with pytest.raises(TypeError, match="no 'heater' role"):
        Rig("rig").fill_roles(heater=Lamp("heater"))


def test_the_fm_group_has_its_four_parts_as_roles():
    assert {name: role.interface for name, role in FM.declared_roles().items()} == {
        "camera": Camera,
        "light_source": LightSource,
        "filter_set": FilterSet,
        "objective": Objective,
    }
    assert all(role.required for role in FM.declared_roles().values())

    devices = bind_fm_devices(FluorescenceMicroscope())
    group = devices["fm"]
    for name in FM.declared_roles():
        assert getattr(group, name) is devices[name]
