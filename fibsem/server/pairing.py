"""Pairing a computer with a short code shown at the serving PC.

The operator opens pairing (the App Server dialog, the chat page, or the
headless server's terminal), and the other computer sends the code with a name
(``POST /pair``). It gets a token of its own, with the role pairing was opened
for: the operator's (the operator's app on another computer, or a browser), or
an agent's, which the agent level gates as the App Server's own agent token.
**Unpair** revokes one computer without touching the others.

A code is 4 digits, lives 5 minutes, and dies after 5 wrong tries, so one
opening is 1 in 2,000 to guess. Tokens are kept hashed on the serving PC
(``~/.fibsem/paired.json``), so the file alone pairs nothing.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fibsem.server.discovery import DISCOVERY_DIR

__all__ = [
    "AGENT",
    "CODE_TTL_S",
    "MAX_TRIES",
    "OPERATOR",
    "PairedComputer",
    "PairedTokens",
    "Pairing",
]

PAIRED_FILE = DISCOVERY_DIR / "paired.json"
CODE_DIGITS = 4
CODE_TTL_S = 300.0
MAX_TRIES = 5
# A code opened from an unpaired browser (headless) that died of wrong tries
# keeps the next one shut this long: guessing stays slow.
REQUEST_LOCKOUT_S = 3600.0
# What a paired token is: the operator's, or an agent's.
OPERATOR = "operator"
AGENT = "agent"
ROLES = (OPERATOR, AGENT)


class PairingRefused(Exception):
    """A pairing that failed: closed, expired, wrong or too many tries."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PairedComputer:
    """One paired computer: its id, the name it gave, when it paired, and
    whether it is the operator or an agent."""

    id: str
    name: str
    paired_at: float
    token_hash: str
    role: str = OPERATOR

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "paired_at": self.paired_at,
            "role": self.role,
        }


class PairedTokens:
    """The paired computers' tokens, hashed, kept in a file (``0600``)."""

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path is not None else PAIRED_FILE
        self._lock = threading.Lock()
        self._computers: Dict[str, PairedComputer] = self._load()

    def _load(self) -> Dict[str, PairedComputer]:
        try:
            with open(self._path) as f:
                rows = json.load(f).get("computers", [])
            return {row["id"]: PairedComputer(**row) for row in rows}
        except FileNotFoundError:
            return {}
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            logging.warning(
                "Paired computers file %s unreadable; none paired", self._path
            )
            return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(
                {"computers": [asdict(c) for c in self._computers.values()]},
                f,
                indent=2,
            )
        os.chmod(self._path, 0o600)

    def add(self, name: str, role: str = OPERATOR) -> "tuple[PairedComputer, str]":
        """Pair a computer named *name* as *role*: its record and its token,
        which is shown once and kept only hashed here."""
        token = secrets.token_urlsafe(32)
        computer = PairedComputer(
            id=secrets.token_hex(4),
            name=name,
            paired_at=time.time(),
            token_hash=_hash(token),
            role=role,
        )
        with self._lock:
            self._computers[computer.id] = computer
            self._save()
        return computer, token

    def remove(self, computer_id: str) -> bool:
        """Unpair one computer: its token stops working at once."""
        with self._lock:
            if self._computers.pop(computer_id, None) is None:
                return False
            self._save()
        return True

    def match(self, token: Optional[str]) -> Optional[PairedComputer]:
        """The paired computer whose token this is, or None."""
        if not token:
            return None
        hashed = _hash(token)
        for computer in list(self._computers.values()):
            if hmac.compare_digest(hashed, computer.token_hash):
                return computer
        return None

    def computers(self) -> List[PairedComputer]:
        return sorted(self._computers.values(), key=lambda c: c.paired_at)


class Pairing:
    """Whether pairing is open, and its code, tries and expiry.

    ``on_change`` listeners are called with the state after each change (open,
    close, a pairing, a code that died), from whichever thread made it."""

    def __init__(
        self,
        tokens: PairedTokens,
        ttl_s: float = CODE_TTL_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.tokens = tokens
        self._ttl_s = ttl_s
        self._clock = clock
        self._lock = threading.Lock()
        self._code: Optional[str] = None
        self._role = OPERATOR
        self._expires = 0.0
        self._tries = 0
        self._requested = False
        self._locked_until = 0.0
        self._listeners: List[Callable[[dict], None]] = []
        # Where the server shows a code an unpaired browser asked for (the
        # headless server's terminal); None means browsers can't ask.
        self.announce: Optional[Callable[[str], None]] = None

    def on_change(self, callback: Callable[[dict], None]) -> None:
        self._listeners.append(callback)

    def _changed(self) -> None:
        state = self.state()
        for callback in list(self._listeners):
            try:
                callback(state)
            except Exception:  # noqa: BLE001 - a listener never blocks pairing
                logging.exception("Pairing listener failed")

    def _live(self) -> bool:
        if self._code is not None and self._clock() >= self._expires:
            self._code = None
        return self._code is not None

    def open(self, role: str = OPERATOR) -> str:
        """Open pairing with a new code for *role*; one already open is replaced."""
        if role not in ROLES:
            raise ValueError(f"No pairing role {role!r}")
        return self._open(requested=False, role=role)

    def _open(self, requested: bool, role: str = OPERATOR) -> str:
        with self._lock:
            self._code = f"{secrets.randbelow(10**CODE_DIGITS):0{CODE_DIGITS}d}"
            self._role = role
            self._expires = self._clock() + self._ttl_s
            self._tries = 0
            self._requested = requested
            code = self._code
        self._changed()
        return code

    def request(self) -> bool:
        """Open pairing for an unpaired browser that asks (headless): its code
        is shown only by ``announce``, where the server runs. Never replaces an
        open code, and stays shut for a while after a requested code died of
        wrong tries. Whether a new code was opened."""
        announce = self.announce
        if announce is None:
            return False
        with self._lock:
            if self._live() or self._clock() < self._locked_until:
                return False
        announce(self._open(requested=True))
        return True

    def close(self) -> None:
        with self._lock:
            was_open = self._code is not None
            self._code = None
        if was_open:
            self._changed()

    def state(self) -> dict:
        """``{"open", "role", "code", "expires_in_s"}``; the code only while open."""
        with self._lock:
            live = self._live()
            return {
                "open": live,
                "role": self._role if live else None,
                "code": self._code if live else None,
                "expires_in_s": max(0.0, self._expires - self._clock())
                if live
                else 0.0,
            }

    def pair(self, code: str, name: str) -> "tuple[PairedComputer, str]":
        """Pair a computer with *code*, or raise `PairingRefused`. A right code
        closes pairing: one code pairs one computer."""
        name = " ".join(str(name or "").split())[:80] or "computer"
        with self._lock:
            if not self._live():
                raise PairingRefused(
                    "pairing_closed",
                    "Pairing isn't open: open it at the microscope PC and try "
                    "the new code.",
                )
            if not hmac.compare_digest(
                str(code).strip().encode("utf-8"), self._code.encode("utf-8")
            ):
                self._tries += 1
                if self._tries < MAX_TRIES:
                    raise PairingRefused("wrong_code", "That code is wrong.")
                self._code = None
                if self._requested:
                    self._locked_until = self._clock() + REQUEST_LOCKOUT_S
                died = True
            else:
                self._code = None
                role = self._role
                died = False
        if died:
            logging.warning("Pairing closed after %d wrong codes", MAX_TRIES)
            self._changed()
            raise PairingRefused(
                "pairing_closed",
                "Too many wrong codes: pairing closed. Open it again at the "
                "microscope PC.",
            )
        paired = self.tokens.add(name, role)
        logging.info("Paired %s as %s", paired[0].name, role)
        self._changed()
        return paired
