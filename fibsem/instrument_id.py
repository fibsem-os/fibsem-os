"""This instrument's id: made once on its PC, kept there, and never changed.

A session, an index and a hub name an instrument by it. The serial number
can't do that: a simulator and some instruments report none, and the hardware
that carries it can be swapped while the instrument stays the same. The name
people call it by is separate and can change.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Optional

from fibsem import paths

__all__ = ["instrument_id", "instrument_id_file"]

log = logging.getLogger(__name__)


def instrument_id_file() -> Path:
    return paths.state_root() / "instrument.json"


def _read(path: Path) -> Optional[str]:
    try:
        with open(path) as f:
            found = json.load(f).get("id")
    except FileNotFoundError:
        return None
    except (OSError, ValueError, AttributeError):
        log.warning("Instrument id file %s unreadable", path)
        return None
    return found if isinstance(found, str) and found else None


def instrument_id(path: Optional[Path] = None) -> Optional[str]:
    """The id in *path* (this install's file by default), made now when there
    is none. None only when the file can be neither read nor written: an id
    that isn't kept would be a different one each time."""
    path = Path(path) if path is not None else instrument_id_file()
    found = _read(path)
    if found is not None:
        return found
    made = str(uuid.uuid4())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive, so two processes starting together keep one id.
        with open(path, "x") as f:
            json.dump({"id": made}, f)
    except FileExistsError:
        return _read(path)
    except OSError as error:
        log.warning("Instrument id not kept in %s: %s", path, error)
        return None
    return made
