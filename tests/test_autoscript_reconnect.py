"""reconnect connects again on the port the first connect used, not AutoScript's
default: a configuration's ``info.port`` survives a reconnect."""

import types

from fibsem.drivers.autoscript.microscope import ThermoMicroscope


def _microscope(monkeypatch, port):
    microscope = object.__new__(ThermoMicroscope)
    microscope.connection = object()
    microscope.system = types.SimpleNamespace(
        info=types.SimpleNamespace(ip_address="10.0.0.1")
    )
    microscope._port = port
    calls = []
    monkeypatch.setattr(microscope, "disconnect", lambda: None)
    monkeypatch.setattr(
        microscope,
        "connect_to_microscope",
        lambda ip_address, **kwargs: calls.append((ip_address, kwargs)),
    )
    return microscope, calls


def test_reconnect_uses_the_port_it_connected_on(monkeypatch):
    microscope, calls = _microscope(monkeypatch, 7521)
    microscope.reconnect()
    assert calls == [("10.0.0.1", {"port": 7521})]


def test_reconnect_before_any_connect_uses_the_default(monkeypatch):
    microscope, calls = _microscope(monkeypatch, None)
    microscope.reconnect()
    assert calls == [("10.0.0.1", {})]
