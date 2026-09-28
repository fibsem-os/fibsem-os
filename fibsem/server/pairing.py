"""The device server's token file, pairing a coordinator with a short code, and
checking a connection.

A device server has one bearer token (``fibsem.server.auth``). For a remote device it
must survive restarts, since the coordinator reconnects to it, so it lives in a file:
``load_or_create_token`` makes one the first time and reuses it after.

Pairing hands that token to a coordinator without anyone copying it. The person at
the device's computer opens a pairing window, which shows a 6-digit code; the person
at the microscope enters the code, and the coordinator trades it for the token
(``POST /pair``). A code works once, for two minutes, and five wrong guesses close
the window, so being on the network is not enough to pair: someone has to read the
code off the device computer's screen.

Both computers keep the token in a file, by default ``~/.fibsem/device-server-token``.
Pairing writes the coordinator's; copying the device computer's file across works too.

From the coordinator (the microscope PC), with only the standard library and requests:

    python -m fibsem.server.pairing pair 192.168.0.20 8765 123456
    python -m fibsem.server.pairing check 192.168.0.20 8765

``check`` walks the connection in order and stops at the first failure, saying which:
the server answers, it takes the token, what it allows, and which devices it serves.
"""

from __future__ import annotations

import argparse
import hmac
import logging
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

PAIRING_SECONDS = 120.0
PAIRING_ATTEMPTS = 5
DEFAULT_TOKEN_FILE = Path("~/.fibsem/device-server-token")
CHECK_TIMEOUT = 5.0


def read_token(path: Union[str, Path, None] = None) -> Optional[str]:
    """The token in ``path`` (the default file when None), or None if there is none."""
    path = Path(path or DEFAULT_TOKEN_FILE).expanduser()
    try:
        token = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    return token or None


def load_or_create_token(path: Union[str, Path]) -> str:
    """The token in ``path``, or a new one written there (readable by its owner only)."""
    path = Path(path).expanduser()
    token = read_token(path)
    if token is not None:
        return token
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


# -- the coordinator's side ---------------------------------------------------------


def pair(
    host: str, port: int, code: str, token_file: Union[str, Path, None] = None
) -> Path:
    """Trade a pairing code for the server's token, and write it to ``token_file``.

    Raises ``PermissionError`` when the server refuses the code (wrong, expired, used,
    or no pairing window open), and ``ConnectionError`` when it can't be reached.
    """
    import requests

    url = f"http://{host}:{port}/pair"
    try:
        response = requests.post(url, json={"code": code}, timeout=CHECK_TIMEOUT)
    except requests.RequestException as error:
        raise ConnectionError(f"{url}: {type(error).__name__}") from None
    if response.status_code == 403:
        raise PermissionError(
            f"{host}:{port} refused the code: it is wrong, has expired or was already "
            "used. Open a new pairing window on the device computer and try again."
        )
    response.raise_for_status()
    return write_token(token_file or DEFAULT_TOKEN_FILE, response.json()["token"])


@dataclass
class Check:
    """One step of ``check_connection``: its name, whether it passed, and what it found."""

    name: str
    ok: bool
    detail: str


def check_connection(
    host: str, port: int, token: Optional[str], timeout: float = CHECK_TIMEOUT
) -> List[Check]:
    """Walk the connection to a device server, stopping at the first step that fails.

    1. reachable: something answers HTTP at ``host:port``;
    2. token: it accepts ``token``;
    3. access: the scopes it has armed (only ``read`` means a read-only mirror);
    4. devices: what it serves, whether each reaches its hardware, and the round trip.
    """
    import requests

    base = f"http://{host}:{port}"
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    checks: List[Check] = []

    def get(path: str, **kwargs: Any) -> Any:
        return requests.get(f"{base}/{path}", timeout=timeout, **kwargs)

    try:
        get("access")  # any answer will do, a refusal included
    except requests.RequestException as error:
        reason = (
            f"no answer within {timeout} s"
            if isinstance(error, requests.Timeout)
            else "connection refused"
        )
        return [
            Check(
                "reachable",
                False,
                f"{base}: {reason}. Is the device server running, and is the port "
                "open in the firewall?",
            )
        ]
    checks.append(Check("reachable", True, base))

    if token is None:
        checks.append(
            Check("token", False, "No token file here: pair, or copy the token file.")
        )
        return checks
    response = get("access", headers=headers)
    if response.status_code == 401:
        checks.append(
            Check(
                "token",
                False,
                "The server refused the token: pair again, or copy its token file.",
            )
        )
        return checks
    response.raise_for_status()
    checks.append(Check("token", True, "accepted"))

    scopes = response.json()["scopes"]
    detail = ", ".join(scopes)
    if "hardware" not in scopes:
        detail += " (read only: start the server with --arm-hardware to allow control)"
    checks.append(Check("access", True, detail))

    start = time.monotonic()
    devices = get("devices", headers=headers).json()
    milliseconds = (time.monotonic() - start) * 1000
    health = get("health", headers=headers).json()["devices"]
    down = [f"{name} ({h['detail']})" for name, h in health.items() if not h["ok"]]
    served = ", ".join(sorted(devices)) or "none"
    detail = f"{served}; round trip {milliseconds:.0f} ms"
    if down:
        detail += f"; hardware not answering: {', '.join(down)}"
    checks.append(Check("devices", bool(devices) and not down, detail))
    return checks


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pair with a device server, or check the connection to one."
    )
    parser.add_argument("action", choices=("pair", "check"))
    parser.add_argument("host")
    parser.add_argument("port", type=int)
    parser.add_argument("code", nargs="?", help="the pairing code (pair only)")
    parser.add_argument("--token-file", default=str(DEFAULT_TOKEN_FILE))
    args = parser.parse_args(argv)
    if args.action == "pair":
        if args.code is None:
            parser.error("pair needs the code shown on the device computer")
        try:
            path = pair(args.host, args.port, args.code, args.token_file)
        except (PermissionError, ConnectionError) as error:
            print(error, file=sys.stderr)
            return 1
        print(f"Paired with {args.host}:{args.port}. The token is in {path}.")
        return 0
    checks = check_connection(args.host, args.port, read_token(args.token_file))
    for check in checks:
        print(f"{'ok  ' if check.ok else 'FAIL'} {check.name}: {check.detail}")
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
