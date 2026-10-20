"""The operator token, kept on disk: one per install, the same at every launch.

The web page is bookmarked with it, so it outlives the App Server, unlike the
agent's token. ``~/.fibsem/operator-token.json``, written ``0600`` (on Windows
the profile folder's ACLs are the real protection, as for the discovery file).
"New operator token" in Tools → App Server replaces it, which revokes the old.

``person`` is kept for per-person logins later; nothing reads it yet.
"""

import json
import logging
import os
import secrets
from pathlib import Path
from typing import Optional

from fibsem.server.discovery import DISCOVERY_DIR

OPERATOR_TOKEN_FILE = DISCOVERY_DIR / "operator-token.json"


def load_operator_token(path: Optional[Path] = None) -> str:
    """The install's operator token, made and saved the first time."""
    path = Path(path) if path is not None else OPERATOR_TOKEN_FILE
    try:
        with open(path) as f:
            token = json.load(f).get("token")
        if isinstance(token, str) and token:
            return token
    except FileNotFoundError:
        pass
    except (OSError, ValueError, AttributeError):
        logging.warning("Operator token file %s unreadable; making a new one", path)
    return new_operator_token(path)


def new_operator_token(path: Optional[Path] = None) -> str:
    """Make, save and return a new operator token; the old one stops working
    once the App Server takes this one."""
    path = Path(path) if path is not None else OPERATOR_TOKEN_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    with open(path, "w") as f:
        json.dump({"token": token, "person": None}, f, indent=2)
    os.chmod(path, 0o600)
    return token
