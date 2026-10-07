"""The Odemis driver: a ThermoFisher microscope and a METEOR FM driven through Odemis.

The package holds everything for this driver: ``devices.py`` (its devices and their
builders), ``services.py`` (its services, such as milling) and ``microscope.py`` (its
``FibsemMicroscope``). This module holds only its ``DRIVER`` record, the registry's entry
for it (``fibsem.drivers.registry``), and ``add_odemis_path``, which puts an
installed odemis on the path before any module here imports it. It imports nothing
from the package, so listing
or looking up the drivers stays cheap and never needs the vendor SDK.
"""

import logging
import sys

from fibsem import manufacturers
from fibsem.drivers.registry import DeviceBuilder, DriverEntry

# No port: Odemis reaches the instrument through its own back end.
DRIVER = DriverEntry(
    manufacturer=manufacturers.ODEMIS,
    microscope_class="fibsem.drivers.odemis.microscope:OdemisThermoMicroscope",
    devices={
        device_type: DeviceBuilder(
            f"fibsem.drivers.odemis.devices:build_odemis_{device_type}"
        )
        for device_type in ("beam", "stage", "chamber")
    },
)


def add_odemis_path(config_path: str = "/etc/odemis.conf"):
    """Add the odemis path to the python path.

    Safe to call on machines without an odemis installation (no config file,
    or a config without DEVPATH): it simply leaves sys.path unchanged so the
    subsequent `import odemis` fails with a regular ImportError.
    """

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

    devpath = config.get("DEVPATH")
    if devpath:
        sys.path.append(f"{devpath}/odemis/src")  # dev version
    sys.path.append("/usr/lib/python3/dist-packages")  # release version + pyro4
