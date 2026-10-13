"""The JEOL driver, through JEOL's external control clients (the CryoLameller).

The package holds everything for this driver: ``connection.py`` (JEOL's two clients,
and the only place that calls them), ``devices.py`` (its beams and stage, and their
builders), ``services.py`` (milling and spot burn) and ``microscope.py`` (its
``FibsemMicroscope``). This module holds only its ``DRIVER`` record, the registry's
entry for it (``fibsem.drivers.registry``), and imports nothing from the package, so
listing or looking up the drivers stays cheap and never needs JEOL's library.

``config``: ``port`` is the FIB client's and ``sem_port`` the SEM client's;
``api_path`` is the folder of JEOL's client library, None to take it from
``FIBSEM_JEOL_API_PATH``.
"""

from fibsem.drivers.registry import DeviceBuilder, DriverEntry

DRIVER = DriverEntry(
    manufacturer="JEOL",
    microscope_class="fibsem.drivers.jeol.microscope:JeolMicroscope",
    config={
        "port": 22218,
        "sem_port": 22300,
        "api_path": None,
        "ion-column-tilt": 53,
        "electron-column-tilt": 0,
    },
    devices={
        device_type: DeviceBuilder(
            f"fibsem.drivers.jeol.devices:build_jeol_{device_type}"
        )
        for device_type in ("beam", "stage")
    },
)
