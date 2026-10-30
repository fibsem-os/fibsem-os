"""The JEOL driver, through JEOL's external control clients (the CryoLameller).

The package holds everything for this driver: ``connection.py`` (JEOL's two clients,
and the only place that calls them), ``devices.py`` (its beams and stage, and their
builders), ``services.py`` (milling and spot burn), ``settings.py`` (its own milling
settings), ``setup.py`` (what the guided setup offers for it) and ``microscope.py``
(its ``FibsemMicroscope``). This module holds only its ``DRIVER`` record, the
registry's entry for it (``fibsem.drivers.registry``), and imports nothing from the
package, so listing or looking up the drivers stays cheap and never needs JEOL's
library.

``port`` is the FIB client's; the SEM client's port and the folder of JEOL's client
library are the connection's (``fibsem.drivers.jeol.connection``).
"""

from fibsem.drivers.registry import DeviceBuilder, DriverEntry

DRIVER = DriverEntry(
    manufacturer="JEOL",
    microscope_class="fibsem.drivers.jeol.microscope:JeolMicroscope",
    port=22218,
    setup="fibsem.drivers.jeol.setup:SETUP",
    milling_settings="fibsem.drivers.jeol.settings:JeolMillingSettings",
    devices={
        device_type: DeviceBuilder(
            f"fibsem.drivers.jeol.devices:build_jeol_{device_type}"
        )
        for device_type in ("beam", "stage")
    },
)
