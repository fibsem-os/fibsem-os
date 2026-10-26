"""A stopped server stops, even for a connection made as it stopped."""

import asyncio
import socket
import threading
import time

import httpx
import pytest
import uvicorn

from fibsem.server.uvicorn_server import Server


async def _app(scope, receive, send):
    if scope["type"] != "http":
        return
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _late_connection_at_stop(server_class):
    """Whether the server ends when one keep-alive connection, busy all along,
    registers just after its shutdown began: the connection accepted as the
    listener closed."""
    port = _free_port()
    server = server_class(uvicorn.Config(_app, port=port, log_level="warning"))
    server.install_signal_handlers = lambda: None
    loop_ready = threading.Event()
    loop = None

    def run():
        nonlocal loop

        async def serve():
            nonlocal loop
            loop = asyncio.get_running_loop()
            loop_ready.set()
            await server.serve()

        asyncio.run(serve())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    loop_ready.wait(5)
    deadline = time.monotonic() + 5
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.01)

    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=2)
    assert client.get("/").status_code == 200
    connections = server.server_state.connections
    (connection,) = connections

    listener = server.servers[0]
    close = listener.close

    def close_then_register():
        # It registers once the listener has closed, as an accepted
        # connection's protocol does a moment after its accept.
        close()
        loop.call_soon(connections.add, connection)

    listener.close = close_then_register
    hidden = threading.Event()

    def hide():
        connections.discard(connection)
        hidden.set()

    loop.call_soon_threadsafe(hide)
    hidden.wait(5)
    server.should_exit = True

    # The client keeps it busy, as a long-poll does.
    stop_at = time.monotonic() + 3
    while thread.is_alive() and time.monotonic() < stop_at:
        try:
            client.get("/")
        except httpx.TransportError:
            pass
        time.sleep(0.05)
    ended = not thread.is_alive()
    client.close()
    server.force_exit = True
    thread.join(5)
    return ended


def test_a_connection_made_as_the_server_stops_does_not_keep_it_serving():
    assert _late_connection_at_stop(Server)


@pytest.mark.xfail(strict=True, reason="uvicorn's own shutdown misses it")
def test_plain_uvicorn_keeps_serving_it():
    assert _late_connection_at_stop(uvicorn.Server)
