"""Which grid is in which slot of a fixed holder is session state.

It used to be `sample-holder-occupancy.yaml`, one file for the PC. It is now
`holder_occupancy` in the session state of the configuration the microscope was
connected with, imported from the old file the first time it is absent.
"""

import os
from pathlib import Path

import yaml

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscopes._stage import HOLDER_OCCUPANCY
from fibsem.session_state import SessionState, session_state_for
from fibsem.structures import SampleGrid

SHIPPED = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


def _connect(config_path=SHIPPED):
    microscope, _ = utils.setup_session(
        config_path=str(config_path), manufacturer="Demo", setup_logging=False
    )
    return microscope


def _slots(microscope) -> dict:
    return {
        name: slot.loaded_grid.name if slot.loaded_grid is not None else None
        for name, slot in microscope._stage.holder.slots.items()
    }


def test_the_first_connect_restores_the_grids():
    """The stage is built during the connect, before the microscope knows its
    configuration -- so the restore has to happen once it does, or every start
    forgets what is in the shuttle."""
    SessionState(SHIPPED, writable=True).save_section(
        HOLDER_OCCUPANCY, {"Slot-02": {"name": "grid-birch"}}
    )

    microscope = _connect()

    assert _slots(microscope) == {"Slot-01": None, "Slot-02": "grid-birch"}


def test_the_old_file_is_imported_and_left_alone():
    path = Path(cfg.SAMPLE_HOLDER_OCCUPANCY_PATH)
    path.write_text(yaml.safe_dump({"Slot-01": {"name": "grid-ash"}}))
    before = path.read_text()

    microscope = _connect()

    assert _slots(microscope)["Slot-01"] == "grid-ash"
    assert path.read_text() == before


def test_a_connect_only_reads():
    """Importing the old file does not write the session state; the application's
    first grid assignment does."""
    Path(cfg.SAMPLE_HOLDER_OCCUPANCY_PATH).write_text(
        yaml.safe_dump({"Slot-01": {"name": "grid-ash"}})
    )

    microscope = _connect()

    assert not session_state_for(microscope).path.exists()


def test_two_configurations_do_not_share_their_grids(tmp_path):
    other = tmp_path / "bay-2.yaml"
    other.write_text(Path(SHIPPED).read_text())
    first = _connect()
    first._stage.assign_grid("Slot-01", SampleGrid(name="grid-ash"), persist=True)

    second = _connect(other)

    assert _slots(second)["Slot-01"] is None
    assert _slots(_connect())["Slot-01"] == "grid-ash"


def test_a_rebuilt_stage_keeps_its_grids():
    """Apply rebuilds the stage when the configuration names a holder; the grids
    are restored from the session state then too."""
    from fibsem.microscopes._stage import _create_sample_stage

    microscope = _connect()
    session_state_for(microscope, writable=True).save_section(
        HOLDER_OCCUPANCY, {"Slot-02": {"name": "grid-birch"}}
    )
    for slot in microscope._stage.holder.slots.values():
        slot.loaded_grid = None  # only the session state knows

    microscope._stage = _create_sample_stage(microscope)

    assert _slots(microscope)["Slot-02"] == "grid-birch"
