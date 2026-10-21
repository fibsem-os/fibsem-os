"""Control of the microscope held across requests: a lease.

A remote app's run takes control once and holds it for its whole length, as a
local run does (`fibsem.control`). Its calls carry the lease (``LEASE_HEADER``)
and run inside it without taking control again. While its run waits on a
question, it lends the lease's control, as a local run lends its own.

A lease that nobody renews for ``idle_s`` is given up, so a client that went
away never holds the microscope.
"""

from __future__ import annotations

import secrets
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from fibsem.acting import ActorKind
from fibsem.control import Control, InControl

__all__ = ["LEASE_HEADER", "IDLE_S", "Lease", "Leases"]

LEASE_HEADER = "X-Fibsem-Control"
IDLE_S = 30.0


@dataclass
class Lease:
    id: str
    entry: InControl
    release: Callable[[], None]
    last: float = field(default_factory=time.monotonic)
    lending: Optional[AbstractContextManager] = None


class Leases:
    """The leases on one microscope's control."""

    def __init__(self, control: Control, idle_s: float = IDLE_S):
        self.control = control
        self.idle_s = idle_s
        self._lock = threading.Lock()
        self._leases: Dict[str, Lease] = {}
        self._reaper: Optional[threading.Thread] = None

    def take(self, by: "ActorKind | str", action: str) -> Lease:
        """Take control and hold it until released, or raise ``MicroscopeBusy``."""
        taking = self.control.take(by, action)
        entry = taking.__enter__()
        lease = Lease(
            id=secrets.token_urlsafe(16),
            entry=entry,
            release=lambda: taking.__exit__(None, None, None),
        )
        with self._lock:
            self._leases[lease.id] = lease
            if self._reaper is None:
                self._reaper = threading.Thread(target=self._reap, daemon=True)
                self._reaper.start()
        return lease

    def get(self, lease_id: Optional[str]) -> Optional[Lease]:
        """The live lease *lease_id*, renewed; None when there is none."""
        with self._lock:
            lease = self._leases.get(lease_id or "")
            if lease is not None:
                lease.last = time.monotonic()
            return lease

    def release(self, lease_id: str) -> bool:
        """Give the lease up: its lending first, which waits for the borrower."""
        with self._lock:
            lease = self._leases.pop(lease_id, None)
        if lease is None:
            return False
        self._end_lending(lease)
        lease.release()
        return True

    def lend(self, lease_id: str) -> bool:
        lease = self.get(lease_id)
        if lease is None:
            return False
        if lease.lending is None:
            lease.lending = self.control.lend()
            lease.lending.__enter__()
        return True

    def end_lend(self, lease_id: str) -> bool:
        lease = self.get(lease_id)
        if lease is None:
            return False
        self._end_lending(lease)
        return True

    def _end_lending(self, lease: Lease) -> None:
        lending, lease.lending = lease.lending, None
        if lending is not None:
            lending.__exit__(None, None, None)

    def _reap(self) -> None:
        while True:
            time.sleep(min(self.idle_s / 3, 5.0))
            now = time.monotonic()
            with self._lock:
                idle = [
                    lease.id
                    for lease in self._leases.values()
                    if now - lease.last > self.idle_s
                ]
            for lease_id in idle:
                self.release(lease_id)
            with self._lock:
                if not self._leases:
                    self._reaper = None
                    return
