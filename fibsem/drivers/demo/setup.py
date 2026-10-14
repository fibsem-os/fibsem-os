"""What the guided setup offers for this driver: the simulator, standing in for an
instrument."""

from fibsem.drivers.setup import DriverSetup, SetupModel

SETUP = DriverSetup(
    label="Simulator",
    summary="No instrument. Everything runs against the built-in simulator.",
    simulated=True,
    models=(
        # The generic simulator ships a 35 degree pre-tilt, so it stands in for an
        # ordinary shuttle system and is still asked about the stage. A run set up
        # against the simulator should behave like the instrument it stands in for.
        SetupModel(
            key="sim-demo",
            label="Demo microscope",
            summary="The default simulated instrument, on a 35° pre-tilted shuttle.",
            filename="microscope-configuration.yaml",
            knows_stage=False,
            config_overrides={"sim": {"is_compustage": False}},
        ),
        # The simulator reads `sim.is_compustage`, not the model name (it reports
        # "DemoMicroscope" on connect), so a simulated compustage says so.
        SetupModel(
            key="sim-arctis",
            label="Simulated Arctis",
            summary="A simulated compustage, for trying the Arctis workflow.",
            filename="sim-arctis-configuration.yaml",
            config_overrides={"sim": {"is_compustage": True}},
        ),
    ),
)
