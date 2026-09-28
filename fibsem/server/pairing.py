"""The device server's token file, and pairing a coordinator with a short code.

A device server has one bearer token (``fibsem.server.auth``). For a remote device it
must survive restarts, since the coordinator reconnects to it, so it lives in a file:
``load_or_create_token`` makes one the first time and reuses it after.

Pairing hands that token to a coordinator without anyone copying it. The person at
the device's computer opens a pairing window, which shows a 6-digit code; the person
at the microscope enters the code, and the coordinator trades it for the token
(``POST /pair``). A code works once, for two minutes, and five wrong guesses close
the window, so being on the network is not enough to pair: someone has to read the
code off the device computer's screen.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Optional, Union

PAIRING_SECONDS = 120.0
PAIRING_ATTEMPTS = 5


def load_or_create_token(path: Union[str, Path]) -> str:
    """The token in ``path``, or a new one written there (readable by its owner only)."""
    path = Path(path).expanduser()
    if path.exists():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    write_token(path, token)
    logging.info(f"Created a new device server token in {path}")
    return token


def write_token(path: Union[str, Path], token: str) -> Path:
    """Write a token file that only its owner can read, where the OS allows it."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8") as f:
        f.write(token + "\n")
    return path


class Pairing:
    """At most one open pairing window: one code, one use, a deadline and a few tries."""

    def __init__(
        self, seconds: float = PAIRING_SECONDS, attempts: int = PAIRING_ATTEMPTS
    ):
        self.seconds = seconds
        self.attempts = attempts
        self._lock = threading.Lock()
        self._code: Optional[str] = None
        self._deadline = 0.0
        self._left = 0

    def open(self) -> str:
        """Open a window and return its code, replacing any code already open."""
        with self._lock:
            self._code = f"{secrets.randbelow(10**6):06d}"
            self._deadline = time.monotonic() + self.seconds
            self._left = self.attempts
            return self._code

    def close(self) -> None:
        with self._lock:
            self._code = None

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._code is not None and time.monotonic() < self._deadline

    def redeem(self, code: str) -> bool:
        """True once, for the right code while the window is open. Closes the window
        on success, and after the last wrong try."""
        with self._lock:
            if self._code is None or time.monotonic() >= self._deadline:
                self._code = None
                return False
            if hmac.compare_digest(str(code), self._code):
                self._code = None
                return True
            self._left -= 1
            if self._left <= 0:
                self._code = None
            return False
