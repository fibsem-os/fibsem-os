"""Which of two things this PC's gateway is, kept in
``~/.fibsem/gateway-role.json``.

A *microscope PC*'s gateway (the default) runs the app for the instrument
beside it. A *hub*'s gateway is on another PC: it holds a list of instruments,
one or many, reaches each over a tunnel and runs agents for them. The role is
chosen by a person, at install or in the gateway's Settings: a gateway never
becomes a hub because of what it was sent.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from fibsem.gateway.gateway import GatewayRefused
from fibsem.server.discovery import DISCOVERY_DIR

__all__ = ["HUB", "MICROSCOPE", "ROLE_FILE", "is_hub", "read_role", "save_role"]

log = logging.getLogger(__name__)

ROLE_FILE = DISCOVERY_DIR / "gateway-role.json"
MICROSCOPE = "microscope"
HUB = "hub"
ROLES = (MICROSCOPE, HUB)


def read_role(path: Optional[Path] = None) -> str:
    """This gateway's role; a microscope PC's when none is saved or the file
    doesn't read."""
    path = path or ROLE_FILE
    try:
        with open(path) as f:
            role = json.load(f).get("role")
    except FileNotFoundError:
        return MICROSCOPE
    except (OSError, ValueError, AttributeError) as error:
        log.warning("The gateway's role in %s not read: %s", path, error)
        return MICROSCOPE
    return role if role in ROLES else MICROSCOPE


def save_role(role: str, path: Optional[Path] = None) -> str:
    """Keep *role*; raises `GatewayRefused` for one that isn't a role."""
    if role not in ROLES:
        raise GatewayRefused(422, "bad_role", f"The role is one of {', '.join(ROLES)}.")
    path = path or ROLE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w") as f:
        json.dump({"role": role}, f, indent=2)
    os.replace(partial, path)
    return role


def is_hub() -> bool:
    return read_role() == HUB
