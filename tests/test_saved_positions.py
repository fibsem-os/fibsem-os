"""Saved positions are session state: one list, per instrument configuration.

They were two files for the PC. The Saved Positions panel wrote
`saved-positions.yaml` and the cryo-deposition widget listed it, while the deposition
looked names up in the older `positions.yaml` -- so a position picked from the list
was "not found". Both are imported into `saved_positions`, and every reader uses it.
"""

import os
from pathlib import Path

import pytest
import yaml

import fibsem.config as cfg
from fibsem import gis, utils
from fibsem.saved_positions import (
    get_saved_position,
    load_saved_positions,
    save_saved_positions,
)
from fibsem.session_state import SessionState, session_state_for
from fibsem.structures import FibsemStagePosition

SHIPPED = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


def _position(name: str, x: float = 0.0) -> dict:
    return FibsemStagePosition(name=name, x=x, y=0.0, z=0.0, r=0.0, t=0.0).to_dict()


def _write(path: str, positions: list) -> str:
    Path(path).write_text(yaml.safe_dump(positions))
    return Path(path).read_text()


@pytest.fixture
def store(tmp_path) -> SessionState:
    return SessionState("bench.yaml", writable=True, directory=str(tmp_path))


# ---------------------------------------------------------------------------
# Importing the two old files
# ---------------------------------------------------------------------------


def test_both_old_files_are_imported_and_the_newer_wins_a_name(store):
    _write(cfg.POSITION_PATH, [_position("cryo", x=1e-3), _position("home")])
    _write(cfg.LEGACY_POSITIONS_PATH, [_position("cryo", x=9e-3), _position("old")])

    positions = load_saved_positions(store)

    assert [p.name for p in positions] == ["cryo", "home", "old"]
    assert positions[0].x == 1e-3  # saved-positions.yaml, not positions.yaml


def test_the_old_files_are_left_alone(store):
    saved = _write(cfg.POSITION_PATH, [_position("cryo")])
    legacy = _write(cfg.LEGACY_POSITIONS_PATH, [_position("old")])

    load_saved_positions(store)
    save_saved_positions([], store)

    assert Path(cfg.POSITION_PATH).read_text() == saved
    assert Path(cfg.LEGACY_POSITIONS_PATH).read_text() == legacy


def test_a_read_only_store_imports_without_writing(tmp_path):
    _write(cfg.POSITION_PATH, [_position("cryo")])
    reader = SessionState("bench.yaml", directory=str(tmp_path))

    assert [p.name for p in load_saved_positions(reader)] == ["cryo"]
    assert not reader.path.exists()


def test_nothing_saved_anywhere_is_an_empty_list(store):
    assert load_saved_positions(store) == []


def test_an_emptied_list_stays_empty(store):
    """Deleting every position must not bring the old files' ones back."""
    _write(cfg.POSITION_PATH, [_position("cryo")])
    load_saved_positions(store)

    save_saved_positions([], store)

    assert load_saved_positions(store) == []


def test_a_position_that_cannot_be_read_is_skipped(store):
    store.save_section("saved_positions", [_position("good"), {"x": "not a number"}])

    assert [p.name for p in load_saved_positions(store)] == ["good"]


# ---------------------------------------------------------------------------
# The deposition finds what the widgets list
# ---------------------------------------------------------------------------


@pytest.fixture
def microscope():
    microscope, _ = utils.setup_session(
        config_path=SHIPPED, manufacturer="Demo", setup_logging=False
    )
    yield microscope
    microscope.disconnect()


def test_the_deposition_finds_a_position_the_panel_saved(microscope):
    """The bug: listed from `saved-positions.yaml`, looked up in `positions.yaml`."""
    _write(cfg.POSITION_PATH, [_position("cryo", x=2e-3)])

    found = gis._saved_position(microscope, "cryo")

    assert found is not None and found.x == 2e-3


def test_the_deposition_still_finds_a_position_only_the_oldest_file_had(microscope):
    _write(cfg.LEGACY_POSITIONS_PATH, [_position("deposition")])

    assert gis._saved_position(microscope, "deposition") is not None


def test_an_unknown_name_is_none(microscope):
    assert gis._saved_position(microscope, "nowhere") is None


def test_two_configurations_do_not_share_positions(tmp_path):
    first = SessionState("bay-1.yaml", writable=True, directory=str(tmp_path))
    second = SessionState("bay-2.yaml", writable=True, directory=str(tmp_path))
    save_saved_positions([FibsemStagePosition(name="cryo")], first)

    assert get_saved_position("cryo", first) is not None
    assert get_saved_position("cryo", second) is None


def test_the_deposition_uses_the_microscope_s_configuration(microscope):
    save_saved_positions(
        [FibsemStagePosition(name="cryo", x=3e-3, y=0, z=0, r=0, t=0)],
        session_state_for(microscope, writable=True),
    )

    assert gis._saved_position(microscope, "cryo").x == 3e-3
