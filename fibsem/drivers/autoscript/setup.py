"""What the guided setup offers for this driver: ThermoFisher and its instruments."""

from fibsem.drivers.setup import DriverSetup, SetupModel

# The frame this driver reads and drives an ordinary stage in (``microscope.py`` sets
# ``CoordinateSystem.RAW`` for it), while xT normally shows the linked frame. A
# compustage is driven in SPECIMEN, so it has no note.
STAGE_NOTE = (
    "Stage values here are Raw coordinates, which can differ from the ones xT "
    "displays. A mismatch between the two is expected rather than a sign something "
    "is wrong."
)

SETUP = DriverSetup(
    label="ThermoFisher",
    summary="Arctis, Hydra and Aquilos.",
    control_software="xT",
    api_module="autoscript_sdb_microscope_client",
    api_label="AutoScript",
    devices=(
        {"name": "ion", "column_tilt": 52},
        {"name": "electron", "column_tilt": 0},
    ),
    models=(
        # A compustage reaches the other side of the sample by tilting, with no
        # shuttle pre-tilt, so its configuration's stage values hold for every one.
        SetupModel(
            key="tfs-arctis",
            label="Arctis",
            summary="Compustage, no shuttle pre-tilt",
            filename="tfs-arctis-configuration.yaml",
        ),
        SetupModel(
            key="tfs-hydra",
            label="Hydra",
            summary="Plasma FIB, 35° pre-tilted shuttle",
            filename="tfs-hydra-configuration.yaml",
            knows_stage=False,
            stage_note=STAGE_NOTE,
        ),
        SetupModel(
            key="tfs-aquilos2",
            label="Aquilos 2",
            summary="Cryo-FIB, 35° pre-tilted shuttle",
            filename="tfs-aquilos2-configuration.yaml",
            knows_stage=False,
            stage_note=STAGE_NOTE,
        ),
        SetupModel(
            key="tfs-other",
            label="Another ThermoFisher instrument",
            summary="Start from the default configuration and enter the stage settings.",
            filename="microscope-configuration.yaml",
            knows_stage=False,
            stage_note=STAGE_NOTE,
        ),
    ),
)
