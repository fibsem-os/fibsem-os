"""A ``uvicorn.Server`` that stops when asked, whoever connected as it did.

uvicorn's shutdown closes its listener, tells each open connection to close,
then waits for them all to be gone. A connection accepted just before the
listener closed registers a moment later, misses the telling, and keeps being
served for as long as its client keeps it busy: a long-polling agent keeps a
stopped server serving it, and the shutdown waiting, forever.
"""

import asyncio
import weakref

import uvicorn


class Server(uvicorn.Server):
    """``uvicorn.Server``, whose shutdown also tells the connections that
    register after it began."""

    async def shutdown(self, sockets=None) -> None:
        told = weakref.WeakSet(self.server_state.connections)
        late = asyncio.ensure_future(self._tell_late_connections(told))
        try:
            await super().shutdown(sockets)
        finally:
            late.cancel()

    async def _tell_late_connections(self, told) -> None:
        while True:
            await asyncio.sleep(0.05)
            for connection in list(self.server_state.connections):
                if connection not in told:
                    told.add(connection)
                    connection.shutdown()
