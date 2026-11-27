"""A simulated instrument on the hub PC, to try the Hub with no microscope.

It is an ordinary ``fibsem-gateway`` on this PC, started with the Demo
microscope and no window, and listed as the instrument "Simulator" at its
address here. The Hub pairs with it and asks it as it does any instrument's
gateway. A PC runs one gateway, so one that already runs here is never
taken for the simulator: it is added by its address like any other.
Stopping the Hub stops the simulator it started.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fibsem import unreleased
from fibsem.gateway.gateway import GatewayRefused
from fibsem.hub.hub import Hub
from fibsem.server.discovery import gateway_port, read_gateway_file
from fibsem.util.process import end

__all__ = ["CONNECTION", "NAME", "Simulator"]

log = logging.getLogger(__name__)

NAME = "Simulator"
# What its gateway connects each app it starts to: the Demo microscope.
CONNECTION = ("--manufacturer", "Demo")
# How long its gateway is given to listen, and to stop.
START_WAIT_S = 30.0
STOP_WAIT_S = 30.0


def _spawn(command: List[str]) -> subprocess.Popen:
    return subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(Path.home()),
    )


class Simulator:
    """The simulator's gateway, while this Hub runs it."""

    def __init__(
        self,
        hub: Hub,
        spawn: Callable[[List[str]], Any] = _spawn,
        running: Callable[[], Optional[dict]] = read_gateway_file,
        port: Callable[[], int] = gateway_port,
        python: str = sys.executable,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._hub = hub
        self._spawn = spawn
        self._running = running
        self._port = port
        self._python = python
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._process: Optional[Any] = None

    def _ours(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def state(self) -> Dict[str, Any]:
        """``{name, running, listed}``: whether this Hub runs it now, and
        whether it is in the list."""
        return {
            "name": NAME,
            "running": self._ours(),
            "listed": any(i.name == NAME for i in self._hub.instruments()),
        }

    def start(self) -> None:
        """Start the simulator's gateway and list it. Refused while another
        gateway runs on this PC."""
        with self._lock:
            if not self._ours():
                self._start()
        url = f"http://127.0.0.1:{self._port()}"
        listed = next((i for i in self._hub.instruments() if i.name == NAME), None)
        if listed is None or listed.url != url:
            self._hub.add(NAME, url)
        else:
            self._hub.check()

    def _start(self) -> None:
        if not unreleased.enabled():
            raise GatewayRefused(
                409,
                "not_released",
                unreleased.NOT_RELEASED.format(what="The gateway"),
            )
        found = self._running()
        if found is not None:
            raise GatewayRefused(
                409,
                "gateway_here",
                f"A gateway already runs on this PC ({found['url']}), and a PC "
                "runs one. Add that one as an instrument by its address, or "
                "stop it first.",
            )
        try:
            process = self._spawn(
                [
                    self._python,
                    "-m",
                    "fibsem.gateway",
                    "run",
                    "--no-window",
                    *CONNECTION,
                ]
            )
        except OSError as error:
            raise GatewayRefused(
                500, "simulator_failed", f"The simulator didn't start: {error}"
            ) from None
        deadline = self._clock() + START_WAIT_S
        while self._running() is None:
            if process.poll() is not None:
                raise GatewayRefused(
                    500,
                    "simulator_failed",
                    "The simulator's gateway ended as it started. Run "
                    "`fibsem-gateway run --no-window --manufacturer Demo` in a "
                    "terminal on this PC to see why.",
                )
            if self._clock() >= deadline:
                end(process, 3.0)
                raise GatewayRefused(
                    500,
                    "simulator_failed",
                    "The simulator's gateway didn't start within "
                    f"{START_WAIT_S:.0f} s.",
                )
            self._sleep(0.2)
        self._process = process
        log.info("The Hub started the simulator's gateway (pid %s).", process.pid)

    def stop(self) -> None:
        """Stop the simulator's gateway, as ``fibsem-gateway stop`` does: an
        app it runs without a window stops with it. It stays in the list."""
        with self._lock:
            process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        try:
            asked = self._spawn([self._python, "-m", "fibsem.gateway", "stop"])
            asked.wait(STOP_WAIT_S)
        except (OSError, subprocess.TimeoutExpired):
            log.debug("Asking the simulator's gateway to stop failed.", exc_info=True)
        end(process, 3.0)
        log.info("The Hub stopped the simulator's gateway.")
