"""Same-machine discovery of a running fibsem server.

The server writes ``~/.fibsem/app-server.json`` while it runs so local
clients (the ``fibsem-mcp`` sidecar in particular) find it without the
user copying anything. Its token only asks to connect: the operator allows
the agent (`fibsem.server.connections`). The file is written ``0600``; on Windows the mode is
advisory and the real protection is the user-profile directory's ACLs — do
not claim more than that.

One file means one server per machine, which doubles as the startup guard
for the single-commander rule: refuse to start a second server while a live
one owns the file.
"""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import fibsem
from fibsem import paths
from fibsem.util.process import pid_alive

if TYPE_CHECKING:
    # Typing only: auth needs FastAPI, and the sidecar and ``fibsem-mcp watch``
    # read this file without the [server] extra.
    from fibsem.server.auth import AuthConfig

DISCOVERY_DIR = paths.user_dir()
# Another App Server's file, for a client started beside a second host on
# this PC (an eval run starts one per run).
DISCOVERY_FILE_ENV = "FIBSEM_DISCOVERY_FILE"
_OVERRIDE = os.environ.get(DISCOVERY_FILE_ENV)
DISCOVERY_FILE = Path(_OVERRIDE) if _OVERRIDE else DISCOVERY_DIR / "app-server.json"
# The file's name before the App Server rename. Read, never written,
# for one release, so a sidecar finds an App Server that predates the rename;
# delete after v0.6.x.
LEGACY_DISCOVERY_FILE = (
    DISCOVERY_FILE if _OVERRIDE else DISCOVERY_DIR / "agent-server.json"
)

# The version of the App Server's routes. A change that an older client can't
# work with (a route or a field removed or renamed, an answer that changes
# meaning) raises the minor number, and clients built for another minor refuse
# to connect (`api_mismatch`). A change clients can ignore raises the patch.
API_VERSION = "0.3.0"


class ApiMismatch(ConnectionError):
    """The server's routes are another version's than this client's."""


def fibsem_folder() -> str:
    """The folder the running ``fibsem`` package was imported from: the App
    Server reports it, and the sidecar compares its own."""
    import fibsem

    return str(Path(fibsem.__file__).resolve().parent)


def code_mismatch(capabilities: dict, who: str) -> Optional[str]:
    """A warning when the App Server runs other ``fibsem`` code than *who*
    imports, from its /capabilities; None when they match. On the same
    machine the folders are compared; from another machine, whose
    folders never match, the versions."""
    from fibsem import __version__

    served = capabilities.get("fibsem") or {}
    path, version = served.get("path"), served.get("version")
    ours = fibsem_folder()
    if path and os.path.isdir(path):
        if os.path.normcase(path) == os.path.normcase(ours):
            return None
        return (
            f"{who}: warning: the app runs fibsem from {path}, but {who} "
            f"imports it from {ours}. Run it with "
            f"PYTHONPATH={os.path.dirname(path)} so its tools match the app."
        )
    if version and version != __version__:
        return (
            f"{who}: warning: the app runs fibsem {version}, but {who} runs "
            f"{__version__}. Tools may not match the app's routes: install "
            "the same version on both machines."
        )
    return None


def _minor(version: object) -> Optional[tuple]:
    """*version*'s (major, minor), or None when it isn't a version."""
    parts = str(version).split(".") if version else []
    if len(parts) < 2 or not all(part.isdigit() for part in parts[:2]):
        return None
    return int(parts[0]), int(parts[1])


def api_mismatch(capabilities: dict, who: str) -> Optional[str]:
    """Why *who* can't work with the server these /capabilities are from: its
    routes are another major or minor version's, or it names no version. None
    when they match."""
    served = capabilities.get("api_version")
    if _minor(served) == _minor(API_VERSION):
        return None
    theirs = (capabilities.get("fibsem") or {}).get("version")
    return (
        f"{who}: the App Server's routes are version {served or 'unknown'}"
        + (f" (fibsem {theirs})" if theirs else "")
        + f", and {who} is built for {API_VERSION} (fibsem {fibsem.__version__}). "
        "Install the same fibsem version on both computers."
    )


def check_api(capabilities: dict, who: str) -> None:
    """Raise `ApiMismatch` when *who* can't work with the server these
    /capabilities are from."""
    why = api_mismatch(capabilities, who)
    if why:
        raise ApiMismatch(why)


def dev_checkout(environ: Optional[dict] = None) -> Optional[str]:
    """The checkout holding the running ``fibsem`` when it came from
    ``PYTHONPATH`` (a development worktree), else None: a sidecar started
    without it imports the environment's installed ``fibsem`` instead."""
    environ = os.environ if environ is None else environ
    checkout = Path(fibsem_folder()).parent
    for entry in environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and Path(entry).resolve() == checkout:
            return str(checkout)
    return None


def _client_host(bind_host: str) -> str:
    # A 0.0.0.0 bind is not a connectable address; local clients use loopback.
    return "127.0.0.1" if bind_host in ("0.0.0.0", "::", "") else bind_host


def write_discovery_file(
    host: str, port: int, auth: "AuthConfig", path: Optional[Path] = None
) -> Path:
    path = Path(path) if path is not None else DISCOVERY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "url": f"http://{_client_host(host)}:{port}",
        # Only asks to connect: the operator allows an agent on this PC
        # (fibsem.server.connections). Never the agent's own token.
        "connect_token": auth.connect_token,
        "pid": os.getpid(),
        # No level: it changes while the server runs and a file
        # would go stale. /capabilities is the live answer.
    }
    # Written whole, then moved into place: a reader never sees it half written.
    partial = path.with_name(path.name + ".partial")
    with open(partial, "w") as f:
        json.dump(payload, f, indent=2)
    os.chmod(partial, 0o600)
    os.replace(partial, path)
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
    if not isinstance(data, dict) or "url" not in data:
        return None
    if "token" not in data and "connect_token" not in data:
        return None
    if not pid_alive(data.get("pid")):
        return None
    return data


def remove_discovery_file(path: Optional[Path] = None) -> None:
    path = Path(path) if path is not None else DISCOVERY_FILE
    try:
        path.unlink()
    except OSError:
        pass
