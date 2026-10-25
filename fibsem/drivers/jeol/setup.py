"""What the guided setup offers for this driver: JEOL instruments."""

from fibsem.drivers.setup import DriverSetup, SetupModel

SETUP = DriverSetup(
    label="JEOL",
    summary="Through JEOL's external control clients.",
    control_software="the CryoLameller software",
    # one of the client library's top-level modules: found only once its folder is
    # importable
    api_module="command",
    api_label="JEOL external control client",
    devices=(
        {"name": "ion", "column_tilt": 53},
        {"name": "electron", "column_tilt": 0},
    ),
    models=(
        SetupModel(
            key="jeol",
            label="JEOL CryoLameller",
            summary="Dual view, SEM and FIB side by side",
            filename="jeol-configuration.yaml",
            knows_stage=False,
        ),
    ),
)
