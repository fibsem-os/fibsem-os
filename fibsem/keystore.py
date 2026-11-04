"""API keys and tokens: from the environment, else this computer's own store.

The store is Windows Credential Manager or the macOS Keychain, through
``keyring``, under the service ``fibsem`` and the environment variable's name.
Elsewhere there is no store and only the environment counts: keyring's other
backends include plain-text files. A key is never logged or printed.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Mapping, Optional

__all__ = [
    "KEYS",
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
log = logging.getLogger(__name__)


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
    """Remove the stored key; False when none was stored."""
    if stored_secret(name) is None:
        return False
    try:
        _store().delete_password(SERVICE, name)
    except Exception as e:
        raise SecretStoreError(f"The store refused ({type(e).__name__}).") from None
    return True
