"""What the gateway's page fills in for the next start, kept in
``~/.fibsem/gateway-defaults.json``: the microscope's configuration, the
app and its window, and the agent's playbook, level and budget.

A field left unset falls back to what the page did before there were
defaults: the default configuration, the first app, the playbook's own
level and budget. Nothing here starts anything."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional

from fibsem.ai.agent.levels import AgentLevel
from fibsem.gateway.gateway import GatewayRefused
from fibsem.server.discovery import DISCOVERY_DIR

__all__ = ["DEFAULTS_FILE", "GatewayDefaults", "read_defaults", "save_defaults"]

DEFAULTS_FILE = DISCOVERY_DIR / "gateway-defaults.json"
LEVELS = tuple(level.value for level in AgentLevel if level is not AgentLevel.OFF)
log = logging.getLogger(__name__)


@dataclass
class GatewayDefaults:
    """What the page fills in for the next start: the last app saved (*app*)
    and its choices here, and each app's own choices under *apps*, by app key:
    an app for another instrument keeps its playbook and settings apart from
    AutoLamella's."""

    configuration: Optional[str] = None
    connect: bool = True
    app: Optional[str] = None
    window: bool = True
    playbook: Optional[str] = None
    level: Optional[str] = None
    budget_usd: Optional[float] = None
    settings: Dict[str, str] = field(default_factory=dict)
    apps: Dict[str, "GatewayDefaults"] = field(default_factory=dict)

    def for_app(self, key: str) -> "GatewayDefaults":
        """*key*'s own choices: its saved ones, else these when they are its."""
        if key in self.apps:
            return self.apps[key]
        return replace(self, apps={}) if self.app == key else GatewayDefaults(app=key)

    def merged(self, saved: "GatewayDefaults") -> "GatewayDefaults":
        """These, saved over *saved*: its other apps' choices kept, and these
        kept as their app's own."""
        apps = {**saved.apps, **self.apps}
        if self.app:
            apps[self.app] = replace(self, apps={})
        return replace(self, apps=apps)

    @classmethod
    def checked(cls, body: Dict[str, Any]) -> "GatewayDefaults":
        """The defaults *body* names, refusing a value the page could not use."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(body) - known)
        if unknown:
            raise GatewayRefused(422, "bad_defaults", f"Unknown: {', '.join(unknown)}.")
        values = {
            name: (str(body[name]).strip() or None) if body.get(name) else None
            for name in ("configuration", "app", "playbook", "level")
        }
        if values["level"] is not None and values["level"] not in LEVELS:
            raise GatewayRefused(
                422, "bad_level", f"The level is one of {', '.join(LEVELS)}."
            )
        budget = body.get("budget_usd")
        try:
            budget = float(budget) if budget not in (None, "") else None
        except (TypeError, ValueError):
            raise GatewayRefused(
                422, "bad_budget", "The budget is a number of dollars."
            ) from None
        if budget is not None and budget <= 0:
            raise GatewayRefused(422, "bad_budget", "The budget is above $0.")
        settings = body.get("settings") or {}
        if not isinstance(settings, dict):
            raise GatewayRefused(422, "bad_defaults", "settings is {name: value}.")
        apps = body.get("apps") or {}
        if not isinstance(apps, dict) or not all(
            isinstance(v, dict) and not v.get("apps") for v in apps.values()
        ):
            raise GatewayRefused(
                422, "bad_defaults", "apps is {app: its defaults}, one level deep."
            )
        return cls(
            **values,
            connect=bool(body.get("connect", True)),
            window=bool(body.get("window", True)),
            budget_usd=budget,
            settings={str(k): str(v) for k, v in settings.items()},
            apps={
                str(key): cls.checked({**entry, "app": key})
                for key, entry in apps.items()
            },
        )


def read_defaults(path: Optional[Path] = None) -> GatewayDefaults:
    """The saved defaults; none saved, or a file that doesn't read, is all unset."""
    path = path or DEFAULTS_FILE
    try:
        with open(path) as f:
            return GatewayDefaults.checked(json.load(f))
    except FileNotFoundError:
        return GatewayDefaults()
    except (OSError, ValueError, TypeError, AttributeError, GatewayRefused) as e:
        log.warning("Gateway defaults in %s not read: %s", path, e)
        return GatewayDefaults()


def save_defaults(defaults: GatewayDefaults, path: Optional[Path] = None) -> None:
    path = path or DEFAULTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w") as f:
        json.dump(asdict(defaults), f, indent=2)
    os.replace(partial, path)
