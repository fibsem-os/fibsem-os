"""API keys and tokens: from the environment, else this computer's own store.

The store is Windows Credential Manager or the macOS Keychain, through
``keyring``, under the service ``fibsem`` and the environment variable's name.
Elsewhere there is no store and only the environment counts: keyring's other
backends include plain-text files. A key is never logged or printed.

Besides `KEYS`, a key may be stored under the token name of a lab's connector
(``token:`` in ``mcp.yaml``, `fibsem.ai.agent.extra_mcp`). Such a token is
stored with where it may go (`pinned`): the connectors' addresses, or their
commands, when it was stored. The harness sends it nowhere else, so editing a
connector's address can't send a stored token to a new one. The pairing tokens
the programs make for themselves (`RESERVED`) are never stored by hand.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Iterable, List, Mapping, Optional

__all__ = [
    "KEYS",
    "RESERVED",
    "KeyNameRefused",
    "check_name",
    "pin",
    "pinned",
    "SERVICE",
    "SecretStoreError",
    "delete_secret",
    "get_secret",
    "set_secret",
    "store_name",
    "stored_secret",
]

SERVICE = "fibsem"
# The keys fibsemOS reads, by environment variable name.
KEYS = {
    "ANTHROPIC_API_KEY": "the agent's Claude API key",
    "FIBSEM_SLACK_TOKEN": "the Slack bot token",
}
# Tokens the programs make and pass for themselves: never typed in.
RESERVED = (
    "FIBSEM_OPERATOR_TOKEN",
    "FIBSEM_SERVER_TOKEN",
    "FIBSEM_HOST_TOKEN",
    "FIBSEM_HOST_OPERATOR_TOKEN",
)
# Where a connector's token may go is kept beside it, under this suffix.
PIN = "#for"
log = logging.getLogger(__name__)


class KeyNameRefused(ValueError):
    """A name no key may be stored under, with the reason in words."""


def check_name(name: str, connector_tokens: Iterable[str] = ()) -> None:
    """Refuse *name* unless it is one of `KEYS` or a connector's token
    (*connector_tokens*, the names ``mcp.yaml`` uses)."""
    if name in RESERVED:
        raise KeyNameRefused(
            f"{name} is made by fibsemOS itself when it pairs: it is never set by hand."
        )
    if name in KEYS or name in set(connector_tokens):
        return
    known = ", ".join(KEYS)
    raise KeyNameRefused(
        f"{name} is not a key fibsemOS reads. The keys are {known}, and the token "
        "name of each connector in mcp.yaml: add the connector first."
    )


class SecretStoreError(RuntimeError):
    """This computer has no store to keep a key in, or it refused."""


def _store():
    """The operating system's keyring backend, or None."""
    try:
        if sys.platform == "win32":
            from keyring.backends.Windows import WinVaultKeyring as Store
        elif sys.platform == "darwin":
            from keyring.backends.macOS import Keyring as Store
        else:
            return None
    except ImportError:
        return None
    return Store()


def store_name() -> Optional[str]:
    """Where `set_secret` keeps a key here, or None when it can't."""
    return (
        {"win32": "Windows Credential Manager", "darwin": "the Keychain"}.get(
            sys.platform
        )
        if _store() is not None
        else None
    )


def stored_secret(name: str) -> Optional[str]:
    """The key stored under *name*, or None."""
    store = _store()
    if store is None:
        return None
    try:
        return store.get_password(SERVICE, name) or None
    except Exception as e:  # a locked or missing store reads as no key
        log.debug("Couldn't read %s from the store: %s", name, type(e).__name__)
        return None


def get_secret(name: str, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The environment variable *name* if set, else the stored key."""
    environ = os.environ if environ is None else environ
    return environ.get(name) or stored_secret(name)


def set_secret(name: str, value: str) -> None:
    store = _store()
    if store is None:
        raise SecretStoreError(
            f"No key store on this computer: set the {name} environment variable."
        )
    try:
        store.set_password(SERVICE, name, value)
    except Exception as e:
        raise SecretStoreError(
            f"The store refused the key ({type(e).__name__})."
        ) from None


def delete_secret(name: str) -> bool:
    """Remove the stored key, and where it may go; False when none was stored."""
    if stored_secret(name) is None:
        return False
    try:
        _store().delete_password(SERVICE, name)
    except Exception as e:
        raise SecretStoreError(f"The store refused ({type(e).__name__}).") from None
    _unpin(name)
    return True


def pinned(name: str) -> Optional[List[str]]:
    """Where the stored token *name* may go (addresses or commands, as
    `fibsem.ai.agent.extra_mcp.where` writes them), None when nothing was kept
    (a token stored before this was kept, or none stored)."""
    store = _store()
    if store is None:
        return None
    try:
        kept = store.get_password(SERVICE, name + PIN)
    except Exception:  # noqa: BLE001 - a locked store reads as nothing kept
        return None
    if not kept:
        return None
    try:
        places = json.loads(kept)
    except ValueError:
        return None
    return [str(p) for p in places] if isinstance(places, list) else None


def pin(name: str, places: Iterable[str]) -> None:
    """Keep where the stored token *name* may go."""
    store = _store()
    if store is None:
        return  # no store: only the environment's token, which isn't pinned
    try:
        store.set_password(SERVICE, name + PIN, json.dumps(sorted(set(places))))
    except Exception as e:
        raise SecretStoreError(
            f"The store refused where {name} may go ({type(e).__name__})."
        ) from None


def _unpin(name: str) -> None:
    store = _store()
    try:
        if store is not None and store.get_password(SERVICE, name + PIN):
            store.delete_password(SERVICE, name + PIN)
    except Exception:  # noqa: BLE001 - a pin left over pins nothing stored
        log.debug("Couldn't remove where %s may go", name)
