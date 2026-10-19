"""The readings milling monitors report, kept for reading on demand (FIB-1481).

The milling service keeps one `MillingReadings`: every reading of every step,
numbered across steps, up to ``keep`` of them. ``snapshot`` is the ``readings``
parameter: per monitor of the latest step, its latest reading and the last
minute of milled time, one reading a second. ``follow`` answers the readings
after a number, waiting for one when there are none yet.
"""

from __future__ import annotations

import itertools
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional

# The longest a follow waits for a reading, in seconds: under the App Server's
# timeouts, so it answers before the call gives up.
MAX_FOLLOW_TIMEOUT = 30.0


class MillingReadings:
    def __init__(self, keep: int = 5000, recent_seconds: float = 60.0):
        self._entries: Deque[Dict[str, Any]] = deque(maxlen=keep)
        self._recent_seconds = recent_seconds
        self._count = itertools.count(1)
        self._seq = 0
        self._step: Dict[str, Any] = {}
        self._changed = threading.Condition()

    def next_seq(self) -> int:
        """The next reading's number; numbers run on across steps."""
        with self._changed:
            self._seq = next(self._count)
            return self._seq

    def start_step(self, stage: Optional[str], index: int) -> None:
        """A step begins: the snapshot shows its readings from now."""
        with self._changed:
            self._step = {"stage": stage, "step": index, "from_seq": self._seq + 1}

    def add(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Keep *entry*, a step's reading, tagged with its stage and step."""
        entry = {
            "stage": self._step.get("stage"),
            "step": self._step.get("step"),
            **entry,
        }
        with self._changed:
            self._entries.append(entry)
            self._changed.notify_all()
        return entry

    def snapshot(self) -> Dict[str, Any]:
        with self._changed:
            step = dict(self._step)
            first = step.pop("from_seq", 0)
            entries = [e for e in self._entries if e["seq"] >= first]
            seq = self._entries[-1]["seq"] if self._entries else 0
        monitors: Dict[str, Dict[str, Any]] = {}
        for name in dict.fromkeys(e["monitor"] for e in entries):
            mine = [e for e in entries if e["monitor"] == name]
            latest = mine[-1]
            since = latest["milled_time"] - self._recent_seconds
            per_second: Dict[int, Dict[str, Any]] = {}
            for e in mine:
                if e["milled_time"] >= since:
                    per_second[int(e["milled_time"])] = e
            monitors[name] = {"latest": latest, "recent": list(per_second.values())}
        return {"seq": seq, **step, "monitors": monitors}

    def follow(self, since: int = 0, timeout: float = 0.0) -> Dict[str, Any]:
        """The readings numbered after *since*, waiting up to *timeout* seconds
        for one. ``missed`` says older ones after *since* are no longer kept."""
        timeout = max(0.0, min(float(timeout), MAX_FOLLOW_TIMEOUT))
        with self._changed:
            self._changed.wait_for(
                lambda: bool(self._entries) and self._entries[-1]["seq"] > since,
                timeout=timeout,
            )
            readings: List[Dict[str, Any]] = [
                e for e in self._entries if e["seq"] > since
            ]
            seq = self._entries[-1]["seq"] if self._entries else since
            missed = (
                bool(readings)
                and readings[0]["seq"] > since + 1
                and (len(self._entries) == self._entries.maxlen)
            )
        return {"seq": seq, "readings": readings, "missed": missed}
