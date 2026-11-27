"""What an agent started on this hub PC may be given beyond its own folder.

Three things, picked in the start form for an agent with an instrument or
without one:

- **Folder**: one data folder the agent may read, from the folders listed in
  the Hub's settings. It never writes there: what it makes stays in its own
  work folder, in its session.
- **Connectors**: which of this hub PC's connectors it may use. Their tokens
  stay in the hub PC's key store and are never shown.
- **Code mode**: the agent writes and runs its own code, always in the
  sandbox (`fibsem.ai.agent.code_mode`); there is no unsandboxed choice. Its
  network is off unless **Network** is ticked, which gives the sandbox the
  allowed sites and nothing else. The allowed sites are one list in the Hub's
  settings, never a local address.

The two lists, and whether any of this is offered at all
(``install.hub.offer_agent_settings``, off until the sandbox has been tried
on this PC), are changed only at the hub PC. Every start is checked against
them here, whatever page asked: a folder outside the list, a site not on it
and a local address are each refused by name.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fibsem import config
from fibsem.ai.agent import code_mode, extra_mcp
from fibsem.gateway.gateway import GatewayRefused

__all__ = ["launch", "read", "save"]

LIST_MAX = 50


def _hub():
    return config.load_user_preferences().install.hub


def _same(a: str, b: str) -> bool:
    return code_mode.workdir(a) == code_mode.workdir(b)


def read() -> Dict[str, Any]:
    """``{offered, data_folders, allowed_sites, connectors, code_mode}``:
    the settings, the usable connectors by name, and whether this PC has a
    sandbox for code mode (macOS and Linux)."""
    hub = _hub()
    return {
        "offered": bool(hub.offer_agent_settings),
        "data_folders": [str(folder) for folder in hub.data_folders or []],
        "allowed_sites": [str(site) for site in hub.allowed_sites or []],
        "connectors": sorted(extra_mcp.usable()[0]),
        "code_mode": sys.platform != "win32",
    }


def _folder(given: Any) -> str:
    """A data folder for the list, as it is kept: its whole path. Raises
    `GatewayRefused`."""
    if not isinstance(given, str) or not given.strip():
        raise GatewayRefused(422, "bad_folder", "A data folder is given by its path.")
    folder = code_mode.workdir(given.strip())
    if not folder.is_dir():
        raise GatewayRefused(
            422, "bad_folder", f"{given} is not a folder on this hub PC."
        )
    home = Path.home().resolve()
    if folder in (home, Path(folder.anchor)) or home.is_relative_to(folder):
        raise GatewayRefused(
            422,
            "bad_folder",
            f"{folder} holds everything of this account: list one data folder.",
        )
    return str(folder)


def _site(given: Any) -> str:
    """A site for the list, in lower case. Raises `GatewayRefused`: a local
    address by that name."""
    if not isinstance(given, str):
        raise GatewayRefused(422, "bad_site", "A site is a host name.")
    refused = code_mode.site_refusal(given)
    if refused:
        local = "local" in refused or "IP address" in refused or "inside" in refused
        raise GatewayRefused(422, "local_address" if local else "bad_site", refused)
    return given.strip().lower()


def _listed(given: Any, check) -> List[str]:
    if not isinstance(given, list) or len(given) > LIST_MAX:
        raise GatewayRefused(
            422, "bad_value", f"A list of at most {LIST_MAX} is expected."
        )
    found: List[str] = []
    for one in given:
        checked = check(one)
        if checked not in found:
            found.append(checked)
    return found


def save(body: Any) -> Dict[str, Any]:
    """Change the fields *body* names: ``data_folders``, ``allowed_sites``,
    ``offered``. Each list is checked whole. Raises `GatewayRefused`."""
    if not isinstance(body, dict) or not body:
        raise GatewayRefused(422, "no_change", "Nothing to change was given.")
    unknown = sorted(set(body) - {"data_folders", "allowed_sites", "offered"})
    if unknown:
        raise GatewayRefused(422, "unknown_field", f"No such setting: {unknown[0]}.")
    folders = _listed(body["data_folders"], _folder) if "data_folders" in body else None
    sites = _listed(body["allowed_sites"], _site) if "allowed_sites" in body else None
    if "offered" in body and not isinstance(body["offered"], bool):
        raise GatewayRefused(422, "bad_value", "offered is true or false.")

    def change(preferences) -> None:
        hub = preferences.install.hub
        if folders is not None:
            hub.data_folders = folders
        if sites is not None:
            hub.allowed_sites = sites
        if "offered" in body:
            hub.offer_agent_settings = body["offered"]

    config.update_user_preferences(change)
    return read()


def launch(asked: Optional[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    """What one start asked for (``{folder, connectors, code, network,
    sites}``), checked against this hub PC's settings as they are now:
    ``fibsem-agent``'s arguments for it, and the same in a few words for the
    session's record (which a resume is checked from again). Raises
    `GatewayRefused`."""
    asked = asked if isinstance(asked, dict) else {}
    folder = str(asked.get("folder") or "").strip()
    connectors = [str(name) for name in asked.get("connectors") or []]
    code = bool(asked.get("code"))
    network = bool(asked.get("network"))
    if not (folder or connectors or code or network):
        return [], {}
    now = read()
    if not now["offered"]:
        raise GatewayRefused(
            403,
            "not_offered",
            "A folder, connectors and code mode are not switched on for agents "
            "on this hub PC. They are switched on in the Hub's Settings, at "
            "the hub PC, once the sandbox has been tried there.",
        )
    args: List[str] = []
    said: Dict[str, Any] = {}
    if code:
        if not now["code_mode"]:
            raise GatewayRefused(
                409,
                "no_sandbox",
                "Code mode needs the sandbox, which Windows doesn't have: the "
                "hub PC must run macOS or Linux.",
            )
        args.append("--code")
        said["code"] = True
    if network:
        if not code:
            raise GatewayRefused(
                422,
                "network_needs_code",
                "Network is for the code sandbox: tick Code mode too.",
            )
        allowed = now["allowed_sites"]
        sites = [str(site).strip().lower() for site in asked.get("sites") or allowed]
        for site in sites:
            if site not in allowed:
                raise GatewayRefused(
                    403,
                    "site_not_allowed",
                    f"{site} is not on this Hub's list of allowed sites.",
                )
            refused = code_mode.site_refusal(site)
            if refused:
                raise GatewayRefused(403, "local_address", refused)
        if not sites:
            raise GatewayRefused(
                409,
                "no_sites",
                "No site is allowed yet: list the allowed sites in the Hub's "
                "Settings, at the hub PC.",
            )
        for site in sites:
            args += ["--allow-site", site]
        said["network"], said["sites"] = True, sites
    if folder:
        listed = next((f for f in now["data_folders"] if _same(f, folder)), None)
        if listed is None:
            raise GatewayRefused(
                403,
                "folder_not_listed",
                f"{folder} is not one of this Hub's data folders: an agent "
                "reads only a folder listed in the Hub's Settings.",
            )
        if not Path(listed).is_dir():
            raise GatewayRefused(
                409, "no_folder", f"{listed} is not on this hub PC any more."
            )
        if not code:
            # Its own work folder, which the read folder goes beside.
            args.append("--files")
        args += ["--read", listed]
        said["folder"] = listed
    if connectors:
        for name in connectors:
            if name not in now["connectors"]:
                raise GatewayRefused(
                    409,
                    "no_connector",
                    f"{name} is not a usable connector on this hub PC.",
                )
            args += ["--connector", name]
        said["connectors"] = connectors
    return args, said
