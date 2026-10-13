"""The Odemis driver: a ThermoFisher microscope and a METEOR FM driven through Odemis.

The package holds everything for this driver: ``devices.py`` (its devices and their
builders), ``services.py`` (its services, such as milling) and ``microscope.py`` (its
``FibsemMicroscope``). This module holds only its ``DRIVER`` record, the registry's entry
for it (``fibsem.drivers.registry``), and ``add_odemis_path``, which puts an
installed odemis on the path before any module here imports it. It imports nothing
from the package, so listing
or looking up the drivers stays cheap and never needs the vendor SDK.
"""

import atexit
import gc
import logging
import sys

from fibsem.drivers.registry import DeviceBuilder, DriverEntry

# No port: Odemis reaches the instrument through its own back end.
DRIVER = DriverEntry(
    manufacturer="Odemis",
    microscope_class="fibsem.drivers.odemis.microscope:OdemisThermoMicroscope",
    devices={
        device_type: DeviceBuilder(
            f"fibsem.drivers.odemis.devices:build_odemis_{device_type}"
        )
        for device_type in ("beam", "stage", "chamber", "fm")
    },
)

_exit_hook_registered = False


def add_odemis_path(config_path: str = "/etc/odemis.conf"):
    """Add the odemis path to the python path.

    Safe to call on machines without an odemis installation (no config file,
    or a config without DEVPATH): it simply leaves sys.path unchanged so the
    subsequent `import odemis` fails with a regular ImportError.

    Every module calls this before importing odemis, so it also arranges for odemis's
    proxies to be shut down at exit (`shut_down_pyro_future_daemons`).
    """
    global _exit_hook_registered
    if not _exit_hook_registered:
        atexit.register(shut_down_pyro_future_daemons)
        _exit_hook_registered = True

    def parse_config(path) -> dict:
        """Parse the odemis config file and return a dict with the config values"""

        with open(path) as f:
            config = f.read()

        config = config.split("\n")
        config = [line.split("=") for line in config]
        config = {
            line[0]: line[1].replace('"', "") for line in config if len(line) == 2
        }
        return config

    try:
        config = parse_config(config_path)
    except Exception as e:
        logging.debug(f"Odemis config not available at {config_path}: {e}")
        return

    # Each module that imports odemis calls this, so a path is added once.
    devpath = config.get("DEVPATH")
    paths = [f"{devpath}/odemis/src"] if devpath else []  # dev version
    paths.append("/usr/lib/python3/dist-packages")  # release version + pyro4
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)


def shut_down_pyro_future_daemons() -> None:
    """Shut down the Pyro4 daemon of every odemis proxy that has one, at exit.

    A proxy that ran an asynchronous call (a move, a filter change) keeps a Pyro4
    daemon to receive the result. Pyro4 shuts it down in the proxy's ``__del__``,
    which at interpreter shutdown can run after Python has stopped the daemon's
    worker threads; it then waits for them forever, and the process never exits.
    ``atexit`` runs while they still run.
    """
    pyro = sys.modules.get("Pyro4.core")
    if pyro is None:
        return  # odemis was never imported
    for obj in gc.get_objects():
        # by type, so no object's own __class__ or __getattr__ runs
        if not issubclass(type(obj), pyro.Proxy):
            continue
        daemon = obj.__dict__.get("_pyroFutureDaemon")
        if not daemon:
            continue
        try:
            daemon.shutdown()
        except Exception as e:
            logging.debug(f"Odemis proxy's future daemon did not shut down: {e}")
        obj._pyroFutureDaemon = None
