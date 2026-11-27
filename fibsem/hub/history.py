"""The Hub's history: what was asked of it and what its agents did.

One line each time an agent is started, resumed or ends on this hub PC, an
instrument's app or agent is started or stopped through the Hub, an
instrument is added, paired or removed, the simulator is started or stopped,
and what the agents here run with is changed (a playbook, a connector, a key:
its name, never its value). Kept in ``~/.fibsem/hub/history.jsonl``, newest
read first. What happens at a microscope PC itself is in that gateway's own
history.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fibsem.gateway.gateway import _tail
from fibsem.hub.hub import TOKENS_DIR

__all__ = ["AGENT", "APP", "HISTORY_FILE", "INSTRUMENT", "SETUP", "Event", "HubHistory"]

log = logging.getLogger(__name__)

HISTORY_FILE = TOKENS_DIR / "history.jsonl"
HISTORY_SHOWN = 100
HISTORY_MAX = 1000
# What a line is about.
AGENT = "agent"
APP = "app"
INSTRUMENT = "instrument"
SETUP = "setup"


@dataclass
class Event:
    """One line of the Hub's history."""

    at: float
    kind: str
    # What happened, in words: "Started an agent (Watch)".
    what: str
    who: str
    instrument: Optional[str] = None
    # False for something asked and refused.
    ok: bool = True


class HubHistory:
    def __init__(
        self, file: Optional[Path] = None, clock: Callable[[], float] = time.time
    ) -> None:
        self._file = Path(file or HISTORY_FILE)
        self._clock = clock
        self._lock = threading.Lock()

    def write(
        self,
        kind: str,
        what: str,
        who: str,
        instrument: Optional[str] = None,
        ok: bool = True,
    ) -> None:
        """Add a line. Never raises: the history never stops what it records."""
        line = Event(self._clock(), kind, what, who or "someone", instrument, ok)
        try:
            with self._lock:
                self._file.parent.mkdir(parents=True, exist_ok=True)
                with open(self._file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(line)) + "\n")
        except OSError:
            log.exception("The Hub's history was not written")

    def read(self, limit: int = HISTORY_SHOWN) -> Dict[str, Any]:
        """``{events, more}``: the latest *limit* lines, newest first, and
        whether the file holds older ones."""
        limit = max(1, min(int(limit), HISTORY_MAX))
        try:
            lines = _tail(self._file, limit + 1)
        except OSError:
            return {"events": [], "more": False}
        events: List[Dict[str, Any]] = []
        for line in reversed(lines[-limit:]):
            try:
                events.append(asdict(Event(**json.loads(line))))
            except (ValueError, TypeError):
                continue
        return {"events": events, "more": len(lines) > limit}
