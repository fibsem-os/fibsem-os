"""Named stage positions the operator saved: session state.

They are stage coordinates, so they belong to one instrument configuration and live in
its session state (`fibsem.session_state`) under ``saved_positions``, a list of
positions in the order they were saved.

They used to be two files for the whole PC. ``saved-positions.yaml`` is what the Saved
Positions panel wrote and the cryo-deposition widget listed; ``positions.yaml`` is an
older file that the deposition itself looked names up in -- so a position picked from
the list could not be found when the deposition ran. Both are imported the first time
the section is absent, ``saved-positions.yaml`` winning a name both contain, and
neither is deleted. Every reader and the one writer now use the same list.
"""

import logging
import os
from typing import List, Optional

import yaml

from fibsem import config as cfg
from fibsem.session_state import SessionState
from fibsem.structures import FibsemStagePosition

SAVED_POSITIONS = "saved_positions"


def _read_list(path: str) -> list:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def _import_saved_positions() -> Optional[list]:
    """The positions from the files they lived in before, merged by name."""
    saved = _read_list(cfg.POSITION_PATH)
    legacy = _read_list(cfg.LEGACY_POSITIONS_PATH)
    names = {entry.get("name") for entry in saved if isinstance(entry, dict)}
    merged = saved + [
        entry
        for entry in legacy
        if isinstance(entry, dict) and entry.get("name") not in names
    ]
    return merged or None


def load_saved_positions(state: SessionState) -> List[FibsemStagePosition]:
    """The saved positions, in the order they were saved."""
    data = state.load_section(SAVED_POSITIONS, migrate=_import_saved_positions)
    positions = []
    for entry in data if isinstance(data, list) else []:
        try:
            positions.append(FibsemStagePosition.from_dict(entry))
        except Exception as e:
            logging.warning(f"Skipping a saved position that could not be read: {e}")
    return positions


def save_saved_positions(
    positions: List[FibsemStagePosition], state: SessionState
) -> bool:
    """Replace the saved positions. Returns whether anything was written."""
    return state.save_section(SAVED_POSITIONS, [p.to_dict() for p in positions])


def get_saved_position(name: str, state: SessionState) -> Optional[FibsemStagePosition]:
    """The saved position called *name*, or None."""
    for position in load_saved_positions(state):
        if position.name == name:
            return position
    return None
