"""The agent host the gateway runs (`fibsem.server.agent.host`), so the agent
works with no app: it talks, plans and keeps its review list while the app is
off, and reaches the microscope once the app answers.

The gateway starts ``fibsem-agent-host`` beside itself, with no window
(``python -m fibsem.gateway.child``, stopped by its stop file), and starts it
again if it ends. It passes calls to it while no App Server answers
(`fibsem.gateway.passthrough`). A host already running on this PC, started by
hand, is used as it is and never stopped here. The gateway stops its own when
it closes.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from fibsem.gateway import child_secrets
from fibsem.server.discovery import (
    DISCOVERY_DIR,
    HOST_DISCOVERY_FILE,
    read_discovery_file,
)

logger = logging.getLogger(__name__)

__all__ = ["AgentHostProcess", "TARGET"]

TARGET = "fibsem.server.agent.host:serve_for_gateway"
# The least wait before starting a host again after one ended or never came
# up (its port taken, say), so a host that can't start isn't started every
# second.
RETRY_S = 30.0
STOP_WAIT_S = 5.0


def _host_file() -> Optional[dict]:
    return read_discovery_file(HOST_DISCOVERY_FILE)


class AgentHostProcess:
    """The gateway's agent host: started by `ensure` (a gateway watcher) while
    none runs on this PC, stopped by `close`."""

    def __init__(
        self,
        python: str,
        home: Optional[Path] = None,
        found: Callable[[], Optional[dict]] = _host_file,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
        args: tuple = (),
    ) -> None:
        self._python = python
        self._home = Path(home) if home is not None else DISCOVERY_DIR
        self._found = found
        self._popen = popen
        self._clock = clock
        self._args = list(args)
        self._proc: Optional[subprocess.Popen] = None
        self._stop_file: Optional[Path] = None
        self._log: Optional[Path] = None
        self._started = float("-inf")
        self._said_ended = False

    def found(self) -> Optional[dict]:
        """The running host's discovery file (``url``, ``pid``), None while
        none runs."""
        return self._found()

    def url(self) -> Optional[str]:
        """The running host's address (the gateway's own or one started by
        hand), None while none runs."""
        found = self._found()
        return found.get("url") if found else None

    def ensure(self) -> None:
        """Start a host when none runs on this PC and the last start was
        `RETRY_S` ago."""
        if self._found() is not None:
            return
        proc = self._proc
        if proc is not None and proc.poll() is None:
            return  # starting: its discovery file comes once it listens
        if proc is not None and not self._said_ended:
            self._said_ended = True
            logger.warning(
                "The gateway's agent host ended (%s); its log is %s",
                proc.returncode,
                self._log,
            )
        if self._clock() - self._started < RETRY_S:
            return
        self._start()

    def _start(self) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        logs = self._home / "logs" / "gateway"
        logs.mkdir(parents=True, exist_ok=True)
        self._log = logs / f"agent-host-{stamp}.log"
        self._stop_file = self._home / f"agent-host-stop-{stamp}"
        self._stop_file.unlink(missing_ok=True)
        command = [
            self._python,
            "-m",
            "fibsem.gateway.child",
            str(self._stop_file),
            TARGET,
            *self._args,
        ]
        self._started = self._clock()
        self._said_ended = False
        # A new secret for this start, through its environment alone: the
        # host refuses every call without it while the gateway runs.
        secret = child_secrets.make()
        try:
            with open(self._log, "w") as out:
                self._proc = self._popen(
                    command,
                    env=child_secrets.environment(secret),
                    stdout=out,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
        except OSError as error:
            self._proc = None
            logger.warning("The gateway couldn't start the agent host: %s", error)
            return
        child_secrets.keep(self._proc.pid, secret)
        logger.info("Gateway started the agent host (pid %d)", self._proc.pid)

    def close(self, wait_s: float = STOP_WAIT_S) -> None:
        """Stop the host the gateway started, by its stop file, else killed
        after *wait_s*. A host started by hand is left running."""
        proc, stop_file = self._proc, self._stop_file
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None and stop_file is not None:
            stop_file.touch()
            try:
                proc.wait(wait_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(2.0)
                except subprocess.TimeoutExpired:
                    pass
        if stop_file is not None:
            stop_file.unlink(missing_ok=True)
