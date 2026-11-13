"""Shared test helpers: a free local port, and waiting for a condition.

The default wait is long on purpose. A wait that passes returns as soon as its
condition holds, so only a failing test pays for it, and a short one is the
usual cause of a test that fails only on a slow runner.
"""

import socket
import time
from typing import Callable

WAIT_S = 10.0


def free_port() -> int:
    """A local TCP port nothing listens on, at the time of asking."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(
    condition: Callable[[], object], timeout: float = WAIT_S, interval: float = 0.02
) -> bool:
    """Poll `condition` until it is truthy or `timeout` seconds pass.

    Returns whether it held, so a test writes `assert wait_for(...), "why"`.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return bool(condition())


def qt_wait_for(condition: Callable[[], object], timeout: float = WAIT_S) -> bool:
    """`wait_for` that runs the Qt event loop while it waits, for conditions
    met by queued signals, timers or work handed back to the GUI thread."""
    from qtpy.QtCore import QEventLoop, QTimer

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()
    return bool(condition())
