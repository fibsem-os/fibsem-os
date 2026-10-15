"""A simulated external scan generator, packaged as a fibsemOS device plugin.

It is the shape a real one would take: a vendor SDK (here ``sdk.py``, a stand-in for
the unit's own library), the device class over it (``devices.py``), and this module,
which holds only the driver's record so listing the drivers imports nothing else.

Install it next to fibsem and name it on a ``hardware.devices`` entry::

    hardware:
      devices:
        - name: fake_scanner
          type: scan_generator
          driver: FakeScan
          address: 127.0.0.1
        - name: electron
          roles: {scanner: fake_scanner}
"""

from fibsem.drivers.registry import DeviceBuilder, DriverEntry

MANUFACTURER = "FakeScan"


def driver() -> DriverEntry:
    """The ``fibsem.drivers`` entry point: this driver's record."""
    return DriverEntry(
        manufacturer=MANUFACTURER,
        aliases=("fake scan", "fake-scan"),
        devices={
            "scan_generator": DeviceBuilder(
                "fibsem_fake_scanner.devices:build_fake_scanner"
            ),
        },
    )
