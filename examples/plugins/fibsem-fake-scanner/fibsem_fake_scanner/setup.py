"""What the guided setup offers for this driver, and the keys its entry takes.

Imported when the wizard lists the drivers, and when an entry naming this driver is
built (its keys are checked against :data:`SCANNER` first), so it imports nothing
but the setup records.
"""

from fibsem.drivers.setup import SetupDevice, SetupKey

SCANNER = SetupDevice(
    type="scan_generator",
    label="FakeScan scan generator",
    summary="A simulated external scan unit, which scans for the electron beam.",
    name="fake_scanner",
    keys=(
        SetupKey("address", "127.0.0.1", "Address", "Where the unit's controller is."),
        SetupKey("pattern", "gradient", "Pattern", "What the simulated unit images."),
        SetupKey(
            "simulate_delay",
            False,
            "Simulate scan time",
            "Wait as long as a real frame would take.",
        ),
    ),
    role=("electron", "scanner"),
)
