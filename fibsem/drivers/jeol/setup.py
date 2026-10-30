"""What the guided setup offers for this driver: JEOL instruments."""

from fibsem.drivers.setup import DriverSetup, SetupLibrary, SetupModel

# JEOL's client library is a folder of top-level modules, not an installed package.
LIBRARY = SetupLibrary(
    label="JEOL client library",
    packages=(
        "common",
        "logger",
        "packet",
        "command",
        "ecommon",
        "ecommand",
        "models",
        "sem_live",
    ),
    env="FIBSEM_JEOL_API_PATH",
)

SETUP = DriverSetup(
    label="JEOL",
    summary="Through JEOL's external control clients.",
    control_software="the CryoLameller software",
    library=LIBRARY,
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
