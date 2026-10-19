"""What the guided setup offers for the fixture's device-only driver, and the keys its
entries take."""

from fibsem.drivers.setup import SetupDevice, SetupKey

SCANNER = SetupDevice(
    type="scan_generator",
    label="Fixture scan generator",
    summary="A scan generator from a plugin.",
    name="fixture_scanner",
    keys=(SetupKey("address", "127.0.0.1", "Address"),),
    role=("electron", "scanner"),
)
