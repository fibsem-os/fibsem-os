"""Control of a remote microscope: the App Server's, held by a lease.

A take here takes control on the server too (``POST /microscope/control``) and
holds it as a lease until the block ends, so a remote app's run holds the
microscope for its whole length, as a local run does, and its calls run inside
it. A lend lends the lease, so whoever answers the run's question may act.
Locally it is a `Control` as well: this app's own clicks and run follow the
same one-at-a-time rule before asking the server.

The server says who acts, by the token: the operator's app takes control as the
operator, the agent as the agent, whatever *by* says here.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Dict, Iterator, Optional

from fibsem.acting import ActorKind
from fibsem.client.transport import Transport, in_lease
from fibsem.control import Control, InControl

__all__ = ["RemoteControl"]

PATH = "/microscope/control"


class RemoteControl(Control):
    """Who is in control of the App Server's microscope, and this app's lease on it."""

    def __init__(self, transport: Transport) -> None:
        super().__init__()
        self._transport = transport
        # The lease each take holds, by its entry, for lend to find.
        self._leases: Dict[int, str] = {}

    @property
    def held(self) -> Optional[InControl]:
        """Who is in control on the server, whoever took it there."""
        held = self._transport.fetch("GET", PATH)["in_control"]
        if held is None:
            return None
        return InControl(
            by=ActorKind(held["by"]), action=held["action"], since=held["since"]
        )

    @contextmanager
    def take(
        self, by: "ActorKind | str", action: str, join: bool = False
    ) -> Iterator[InControl]:
        """Take control here and on the server, or raise ``MicroscopeBusy``
        naming who is in control there. A take that joins a click already
        holding control here runs inside that click's lease."""
        with super().take(by, action, join=join) as entry:
            held = self._leases.get(id(entry))
            if held is not None:
                with in_lease(held):
                    yield entry
                return
            taken = self._transport.fetch("POST", PATH, body={"action": action})
            lease = taken["lease"]
            stop = threading.Event()
            threading.Thread(
                target=self._keep, args=(lease, taken["idle_s"] / 3, stop), daemon=True
            ).start()
            self._leases[id(entry)] = lease
            try:
                with in_lease(lease):
                    yield entry
            finally:
                self._leases.pop(id(entry), None)
                stop.set()
                self._transport.request("DELETE", f"{PATH}/{lease}")

    @contextmanager
    def lend(self) -> Iterator[None]:
        owner = self._owner
        lease = self._leases.get(id(owner)) if owner is not None else None
        with super().lend():
            if lease is None:
                yield
                return
            self._transport.fetch("POST", f"{PATH}/{lease}/lend")
            try:
                yield
            finally:
                self._transport.request("DELETE", f"{PATH}/{lease}/lend")

    def _keep(self, lease: str, every: float, stop: threading.Event) -> None:
        while not stop.wait(every):
            try:
                self._transport.request("PUT", f"{PATH}/{lease}")
            except Exception:  # noqa: BLE001 - the next call says it, if gone
                pass
