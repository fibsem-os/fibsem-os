"""The AutoScript driver: ThermoFisher microscopes (Hydra, Helios, Arctis and others).

The package holds everything for this driver: ``devices.py`` (its devices and their
builders), ``services.py`` (its services, such as milling), ``setup.py`` (what the guided
setup offers for it) and ``microscope.py`` (its
``FibsemMicroscope``). This module holds only its ``DRIVER`` record, the registry's entry
for it (``fibsem.drivers.registry``), and imports nothing from the package, so listing
or looking up the drivers stays cheap and never needs the vendor SDK.
"""

from fibsem.drivers.registry import DeviceBuilder, DriverEntry

DRIVER = DriverEntry(
    manufacturer="ThermoFisher",
    aliases=("thermo", "thermo fisher", "thermo fisher scientific"),
    microscope_class="fibsem.drivers.autoscript.microscope:ThermoMicroscope",
    port=7520,
    setup="fibsem.drivers.autoscript.setup:SETUP",
    devices={
        device_type: DeviceBuilder(
            f"fibsem.drivers.autoscript.devices:build_autoscript_{device_type}"
        )
        for device_type in (
            "beam",
            "stage",
            "chamber",
            "chamber_camera",
            "manipulator",
            "sample_loader",
            "fm",
            "gis",
            "sputter_coater",
        )
    },
)
