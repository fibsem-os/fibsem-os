"""The Demo driver: a simulated microscope, the reference implementation of a driver.

The package holds everything for this driver: ``devices.py`` (its devices and their
builders), ``services.py`` (its services, such as milling), ``setup.py`` (what the guided
setup offers for it) and ``microscope.py`` (its
``FibsemMicroscope``). This module holds only its ``DRIVER`` record, the registry's entry
for it (``fibsem.drivers.registry``), and imports nothing from the package, so listing
or looking up the drivers stays cheap and never needs the vendor SDK.
"""

from fibsem.drivers.registry import DeviceBuilder, DriverEntry

DRIVER = DriverEntry(
    manufacturer="Demo",
    microscope_class="fibsem.drivers.demo.microscope:DemoMicroscope",
    config={"port": 7520, "ion-column-tilt": 52, "electron-column-tilt": 0},
    setup="fibsem.drivers.demo.setup:SETUP",
    devices={
        **{
            device_type: DeviceBuilder(
                f"fibsem.drivers.demo.devices:build_demo_{device_type}"
            )
            for device_type in (
                "beam",
                "stage",
                "chamber",
                "chamber_camera",
                "manipulator",
                "sample_loader",
                "gis",
                "sputter_coater",
                "fm",
            )
        },
        # An external scan generator a beam's ``scanner`` role can be bound to.
        "scan_generator": DeviceBuilder(
            "fibsem.drivers.demo.devices:build_demo_scan_generator"
        ),
    },
)
