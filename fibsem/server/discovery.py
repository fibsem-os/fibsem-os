"""Same-machine discovery of a running fibsem server.

The server writes ``~/.fibsem/app-server.json`` while it runs so local
clients (the ``fibsem-mcp`` sidecar in particular) can connect without the
user copying a token. The file is written ``0600``; on Windows the mode is
advisory and the real protection is the user-profile directory's ACLs — do
not claim more than that.

One file means one server per machine, which doubles as the startup guard
for the single-commander rule: refuse to start a second server while a live
one owns the file.
"""

import json
import os
from pathlib import Path
from typing import Optional

from fibsem.server.auth import AuthConfig

DISCOVERY_DIR = Path.home() / ".fibsem"
DISCOVERY_FILE = DISCOVERY_DIR / "app-server.json"
# The file's name before the App Server rename (FIB-1035). Read, never written,
# for one release, so a sidecar finds an App Server that predates the rename;
# delete after v0.6.x.
LEGACY_DISCOVERY_FILE = DISCOVERY_DIR / "agent-server.json"


def _client_host(bind_host: str) -> str:
    # A 0.0.0.0 bind is not a connectable address; local clients use loopback.
    return "127.0.0.1" if bind_host in ("0.0.0.0", "::", "") else bind_host


def write_discovery_file(
    host: str, port: int, auth: AuthConfig, path: Optional[Path] = None
) -> Path:
    path = Path(path) if path is not None else DISCOVERY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": f"http://{_client_host(host)}:{port}",
        "token": auth.token,
        "pid": os.getpid(),
        # No scopes or level: they change while the server runs and a file
        # would go stale. /capabilities is the live answer (FIB-1035).
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    os.chmod(path, 0o600)
    return path


def read_discovery_file(path: Optional[Path] = None) -> Optional[dict]:
    """The running server's connection info, or None if absent/stale/unreadable.

    With no ``path``, the current name is tried first, then the legacy one, so
    a live server under either name is found -- by clients, and by the
    one-server-per-machine guard.
    """
    if path is not None:
        return _read(Path(path))
    for candidate in (DISCOVERY_FILE, LEGACY_DISCOVERY_FILE):
        found = _read(candidate)
        if found is not None:
            return found
    return None


def _read(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "url" not in data or "token" not in data:
        return None
    if not _pid_alive(data.get("pid")):
        return None
    return data


def remove_discovery_file(path: Optional[Path] = None) -> None:
    path = Path(path) if path is not None else DISCOVERY_FILE
    try:
        path.unlink()
    except OSError:
        pass


def _pid_alive(pid) -> bool:
    if not isinstance(pid, int):
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
