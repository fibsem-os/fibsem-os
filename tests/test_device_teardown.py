"""Disconnecting lets go of every device, not only the vendor's client.

`FibsemMicroscope.disconnect` disconnects each device in reverse build order, then
runs what the builders registered with `BuildContext.on_disconnect` (a connection the
entries of one driver share), then the driver's own `_disconnect`. One that fails is
logged, and the rest still run.
"""

import logging
from types import SimpleNamespace

import pytest

from fibsem import utils
from fibsem.devices import Device
from fibsem.devices.entries import (
    DeviceBuildError,
    ResolvedEntry,
    build_device_entries,
)
from fibsem.drivers import registry
from fibsem.structures import DeviceEntry


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(setup_logging=False, manufacturer="Demo")
    return microscope


def _record_disconnects(microscope, calls, monkeypatch):
    for name, device in microscope.devices.items():
        monkeypatch.setattr(device, "disconnect", lambda name=name: calls.append(name))
    monkeypatch.setattr(microscope, "_disconnect", lambda: calls.append("driver"))


def test_a_device_has_nothing_to_close_by_default():
    assert Device(name="plain").disconnect() is None


def test_devices_go_in_reverse_build_order_then_cleanups_then_the_driver(
    microscope, monkeypatch
):
    calls = []
    _record_disconnects(microscope, calls, monkeypatch)
    microscope._disconnect_cleanups.extend(
        [lambda: calls.append("first"), lambda: calls.append("second")]
    )

    microscope.disconnect()

    devices = list(microscope.devices)
    assert calls == [*reversed(devices), "second", "first", "driver"]
    # a cleanup runs once: a second disconnect doesn't close it again
    calls.clear()
    microscope.disconnect()
    assert calls == [*reversed(devices), "driver"]


def test_one_that_fails_is_logged_and_the_rest_still_run(
    microscope, monkeypatch, caplog
):
    calls = []
    _record_disconnects(microscope, calls, monkeypatch)

    def stuck():
        raise RuntimeError("controller not answering")

    monkeypatch.setattr(microscope.devices["stage"], "disconnect", stuck)
    microscope._disconnect_cleanups.append(stuck)
    logging.getLogger().addHandler(caplog.handler)

    microscope.disconnect()

    assert "stage" not in calls
    assert set(calls) == set(microscope.devices) - {"stage"} | {"driver"}
    assert calls[-1] == "driver"
    assert "Device 'stage' did not disconnect cleanly" in caplog.text
    assert "did not close cleanly: controller not answering" in caplog.text


# -- what the builders register ------------------------------------------------


class _Builder:
    def __init__(self, build):
        self.build = build

    def load(self):
        return self.build


@pytest.fixture
def builders(monkeypatch):
    table = {}

    def device_builder(driver, type_):
        build = table.get(type_)
        return None if build is None else _Builder(build)

    monkeypatch.setattr(registry, "device_builder", device_builder)
    return table


def _item(name, type_, required=None):
    return ResolvedEntry(DeviceEntry(name=name, type=type_, required=required), "Test")


def test_a_builders_cleanups_are_kept_on_the_microscope(builders):
    closed = []

    def build(entry, context):
        connection = context.shared.setdefault("Test", object())
        if len(context.cleanups) == 0:
            context.on_disconnect(lambda: closed.append(connection))
        return Device(name=entry.name)

    builders.update(stage=build, chamber=build)
    scope = SimpleNamespace(_disconnect_cleanups=[])

    build_device_entries([_item("stage", "stage"), _item("chamber", "chamber")], scope)

    assert len(scope._disconnect_cleanups) == 1  # one shared connection, one close
    scope._disconnect_cleanups[0]()
    assert len(closed) == 1


def test_cleanups_are_kept_when_a_required_device_fails_the_connect(builders):
    def opens(entry, context):
        context.on_disconnect(lambda: None)
        return Device(name=entry.name)

    def fails(entry, context):
        raise RuntimeError("no controller")

    builders.update(stage=opens, chamber=fails)
    scope = SimpleNamespace(_disconnect_cleanups=[])

    with pytest.raises(DeviceBuildError):
        build_device_entries(
            [_item("stage", "stage"), _item("chamber", "chamber", required=True)],
            scope,
        )
    assert len(scope._disconnect_cleanups) == 1
