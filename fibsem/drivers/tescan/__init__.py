"""The Tescan driver, through the Tescan Automation SDK.

The package holds everything for this driver: ``devices.py`` (its devices and their
builders), ``services.py`` (its services, such as milling) and ``microscope.py`` (its
``FibsemMicroscope``). This module holds only its ``DRIVER`` record, the registry's entry
for it (``fibsem.drivers.registry``), and imports nothing from the package, so listing
or looking up the drivers stays cheap and never needs the vendor SDK.
"""

from fibsem import manufacturers
from fibsem.drivers.registry import DeviceBuilder, DriverEntry

DRIVER = DriverEntry(
    manufacturer=manufacturers.TESCAN,
    microscope_class="fibsem.drivers.tescan.microscope:TescanMicroscope",
    config={"port": 8300, "ion-column-tilt": 55, "electron-column-tilt": 0},
    devices={
        device_type: DeviceBuilder(
            f"fibsem.drivers.tescan.devices:build_tescan_{device_type}"
        )
        for device_type in ("beam", "stage")
    },
)
