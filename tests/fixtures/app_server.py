"""An App Server in this process, on a free port, for a test to reach over HTTP."""

import threading
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def serving(app) -> Iterator[str]:
    """Serve *app* (`build_server`'s) until the block ends; its url."""
    import uvicorn

    from fibsem.server.devices import WS_PROTOCOL

    server = uvicorn.Server(
        uvicorn.Config(
            app, host="127.0.0.1", port=0, log_level="warning", ws=WS_PROTOCOL
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        assert thread.is_alive(), "App Server failed to start"
        threading.Event().wait(0.01)
    try:
        yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
