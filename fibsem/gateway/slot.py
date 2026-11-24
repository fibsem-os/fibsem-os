"""The instrument's one agent slot, written down: who holds it, how they came
in, since when, at what level, in what state, and why it last changed.

One file the gateway keeps (``~/.fibsem/agent/slot.json``), so a restart, the
page and the Overview all read the same answer instead of working it out from
the paired list, the manager's memory and the App Server's. It records; it
never decides who may connect (pairing and the Allow card do that).

States: ``running`` (the gateway's agent runs, or an allowed one is
connected), ``waiting`` (the gateway's agent waits for its app), ``ended``
(the holder ended or was replaced; the slot is free).
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fibsem.util import timestamps

logger = logging.getLogger(__name__)

RUNNING, WAITING, ENDED = "running", "waiting", "ended"
# How a holder came in.
START, RESUME, ALLOW = "start", "resume", "allow"
FIELDS = ("holder", "session_id", "came_in", "since", "level", "state", "why")


class SlotRecord:
    """The slot's file: read whole, written whole (by rename)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def read(self) -> Dict[str, Any]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def take(
        self,
        holder: str,
        came_in: str,
        why: str,
        session_id: Optional[str] = None,
        level: Optional[str] = None,
        state: str = RUNNING,
    ) -> None:
        """A new holder: everything about the last one goes."""
        now = timestamps.now().isoformat()
        self._write(
            {
                "holder": holder,
                "session_id": session_id,
                "came_in": came_in,
                "since": now,
                "level": level,
                "state": state,
                "why": why,
                "changed": now,
            }
        )

    def change(self, session_id: Optional[str] = None, **changes: Any) -> bool:
        """Change the holder's *changes* (state, level, why); with
        *session_id*, only while that session holds it. Whether it changed."""
        with self._lock:
            slot = self.read()
            if not slot or (
                session_id is not None and slot.get("session_id") != session_id
            ):
                return False
            new = {k: v for k, v in changes.items() if k in FIELDS}
            if all(slot.get(k) == v for k, v in new.items()):
                return False
            slot.update(new, changed=timestamps.now().isoformat())
            self._write_unlocked(slot)
            return True

    def _write(self, slot: Dict[str, Any]) -> None:
        with self._lock:
            self._write_unlocked(slot)

    def _write_unlocked(self, slot: Dict[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            partial = self.path.with_name(self.path.name + ".partial")
            partial.write_text(json.dumps(slot, indent=2), encoding="utf-8")
            os.replace(partial, self.path)
        except OSError:
            logger.warning("The agent slot wasn't written down.", exc_info=True)
