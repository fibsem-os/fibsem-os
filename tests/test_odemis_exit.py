"""An odemis client process exits: each proxy's Pyro4 future daemon is shut down at
exit, before interpreter shutdown can leave it waiting on stopped threads.

No odemis installation: ``Pyro4.core`` is a stand-in module with a ``Proxy`` class.
Seen on the odemis METEOR simulator: a script that ran one focus move then hung at
exit every time; with the hook it exits.
"""

import atexit
import sys
import types

import pytest

from fibsem.drivers import odemis as odemis_driver


class _Daemon:
    def __init__(self, fails=False):
        self.shutdowns = 0
        self.fails = fails

    def shutdown(self):
        self.shutdowns += 1
        if self.fails:
            raise RuntimeError("socket already closed")


@pytest.fixture
def pyro(monkeypatch):
    module = types.ModuleType("Pyro4.core")

    class Proxy:
        def __init__(self, daemon=None):
            self._pyroFutureDaemon = daemon

        def __getattr__(self, name):  # a remote method, as on a real proxy
            raise AssertionError(f"remote call {name}")

    module.Proxy = Proxy
    monkeypatch.setitem(sys.modules, "Pyro4.core", module)
    return module


def test_each_proxys_future_daemon_is_shut_down_once(pyro):
    daemon = _Daemon()
    moved, idle = pyro.Proxy(daemon), pyro.Proxy(None)

    odemis_driver.shut_down_pyro_future_daemons()
    odemis_driver.shut_down_pyro_future_daemons()  # and again, as __del__ would

    assert daemon.shutdowns == 1
    assert moved._pyroFutureDaemon is None
    assert idle._pyroFutureDaemon is None


def test_a_daemon_that_fails_to_shut_down_does_not_stop_the_others(pyro):
    failing, working = _Daemon(fails=True), _Daemon()
    proxies = [pyro.Proxy(failing), pyro.Proxy(working)]

    odemis_driver.shut_down_pyro_future_daemons()

    assert (failing.shutdowns, working.shutdowns) == (1, 1)
    assert all(p._pyroFutureDaemon is None for p in proxies)


def test_without_odemis_it_does_nothing(monkeypatch):
    monkeypatch.delitem(sys.modules, "Pyro4.core", raising=False)
    odemis_driver.shut_down_pyro_future_daemons()


def test_adding_the_odemis_path_registers_the_hook_once(monkeypatch, tmp_path):
    registered = []
    monkeypatch.setattr(atexit, "register", registered.append)
    monkeypatch.setattr(odemis_driver, "_exit_hook_registered", False)

    odemis_driver.add_odemis_path(str(tmp_path / "no-odemis.conf"))
    odemis_driver.add_odemis_path(str(tmp_path / "no-odemis.conf"))

    assert registered == [odemis_driver.shut_down_pyro_future_daemons]
