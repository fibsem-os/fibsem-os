"""The setup an agent runs to connect to this App Server: the ``fibsem-mcp``
sidecar beside this Python, and the text that registers and checks it. The
Connect an agent dialog and the operator's ``GET /app/agent/connect`` give the
same text.

The absolute path matters: a GUI app such as Claude Desktop does not inherit
the conda environment's PATH, so a bare ``fibsem-mcp`` is not found.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

from fibsem.server.discovery import dev_checkout

__all__ = ["agent_setup_text", "setup_text", "sidecar_path"]


def sidecar_path(executable: Optional[str] = None) -> Optional[Path]:
    """The ``fibsem-mcp`` script beside this Python, or None if it isn't installed.

    Found from the interpreter rather than PATH, which a GUI app may not have:
    ``<env>/bin`` on Linux and macOS, ``<env>/Scripts`` on Windows.
    """
    here = Path(executable or sys.executable).parent
    names = ("fibsem-mcp.exe",) if os.name == "nt" else ("fibsem-mcp",)
    for folder in (here, here / "Scripts", here / "bin"):
        for name in names:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


DEFAULT_URL = "http://127.0.0.1:8001"


def desktop_config_path(platform: Optional[str] = None) -> str:
    """Where Claude Desktop keeps its config on this platform."""
    platform = platform or sys.platform
    if platform.startswith("win"):
        return r"%APPDATA%\Claude\claude_desktop_config.json"
    if platform == "darwin":
        return "~/Library/Application Support/Claude/claude_desktop_config.json"
    return "~/.config/Claude/claude_desktop_config.json"


def agent_setup_text(
    path: Path,
    url: Optional[str] = None,
    platform: Optional[str] = None,
    pythonpath: Optional[str] = None,
) -> str:
    """Setup the agent runs itself: register the sidecar, check it, report the level.

    ``-s user`` registers it for every project; without it ``claude mcp add``
    covers only the current directory's. Claude Desktop can't edit its own
    config, so its line is for the person, and says to merge, not replace.
    ``pythonpath`` is the checkout the app runs from when it came from
    ``PYTHONPATH`` (:func:`~fibsem.server.discovery.dev_checkout`): the sidecar
    gets it too, or it imports the environment's own ``fibsem`` (FIB-1429).
    """
    platform = platform or sys.platform
    url = url or DEFAULT_URL
    config: dict = {"command": str(path)}
    env = ""
    check = f'"{path}" --check'
    if pythonpath:
        config["env"] = {"PYTHONPATH": pythonpath}
        env = f'-e PYTHONPATH="{pythonpath}" '
        check = (
            f'{check} with PYTHONPATH set to "{pythonpath}"'
            if platform.startswith("win")
            else f'PYTHONPATH="{pythonpath}" {check}'
        )
    entry = json.dumps(config)
    return (
        "Connect to my AutoLamella App Server (same machine).\n"
        "1. Register the fibsem MCP server for all my projects:\n"
        f'   claude mcp add -s user {env}fibsem -- "{path}"\n'
        "   If fibsem is already registered, remove it first: "
        "claude mcp remove -s user fibsem\n"
        "   (If you're Claude Desktop instead: add "
        f'"fibsem": {entry} under "mcpServers" in '
        f"{desktop_config_path(platform)}. Merge it into the servers already "
        "there; don't replace the file.)\n"
        f'2. Check the connection: {check} should print "connected to {url}".\n'
        "3. Tell me to start a new session, or reconnect fibsem in /mcp, so the "
        "tools load.\n"
        "4. Then call server_get_capabilities and tell me your agent level. I set it "
        "from the Agent chip in AutoLamella.\n"
        "5. If you ask me before each tool, tell me which fibsem tools I can allow "
        "for good: the ones that only read (marked read-only), and "
        "experiment_add_note, which only adds a note.\n"
    )


def setup_text(url: Optional[str] = None) -> Optional[str]:
    """The text for this install's sidecar, or None if it isn't installed."""
    path = sidecar_path()
    if path is None:
        return None
    return agent_setup_text(path, url, pythonpath=dev_checkout())
