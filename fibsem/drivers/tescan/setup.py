"""What the guided setup offers for this driver: Tescan instruments."""

from fibsem.drivers.setup import DriverSetup, SetupModel

SETUP = DriverSetup(
    label="Tescan",
    summary="Driven by presets rather than beam currents.",
    control_software="Essence",
    api_module="tescanautomation",
    api_label="TESCAN Automation",
    models=(
        SetupModel(
            key="tescan",
            label="Tescan instrument",
            summary="Presets rather than beam currents",
            filename="tescan-configuration.yaml",
            knows_stage=False,
        ),
    ),
)
