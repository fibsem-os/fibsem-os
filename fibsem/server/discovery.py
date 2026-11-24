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
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import fibsem
from fibsem import paths
from fibsem.util.process import process_started, same_process

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

# The agent host's (`fibsem.server.agent.host`), run on its own: the same
# shape, so the same functions write and read it. A host and an App Server
# may run on one PC.
HOST_DISCOVERY_FILE = DISCOVERY_DIR / "agent-host.json"
# The gateway's (`fibsem.gateway`): this PC's one way in. Its url, pid and the
# connect token an agent here asks to connect with.
GATEWAY_FILE = DISCOVERY_DIR / "gateway.json"
GATEWAY_PORT = 8010
# The agent token an App Server running alone takes, and its clients send (tests,
# live evals, scripts): with no gateway in front, nothing pairs or asks there.
SERVER_TOKEN_ENV = "FIBSEM_SERVER_TOKEN"


def this_process() -> dict:
    """This process, as a file it writes names it: its pid, when it started
    (so a reader never takes another process with the same pid for it,
    `same_process`) and its program."""
    return {
        "pid": os.getpid(),
        "started": process_started(os.getpid()),
        "program": Path(sys.argv[0]).name if sys.argv and sys.argv[0] else None,
    }


def gateway_port() -> int:
    """The port this PC's gateway listens on: the preferences'
    (``install.gateway.port``), else `GATEWAY_PORT`."""
    try:
        from fibsem.config import load_user_preferences

        port = int(load_user_preferences().install.gateway.port)
    except Exception:  # noqa: BLE001 - unreadable preferences are the default
        return GATEWAY_PORT
    return port if 0 < port < 65536 else GATEWAY_PORT


def gateway_url() -> str:
    """This PC's gateway: the running one's address, else its usual port."""
    found = read_gateway_file()
    return found["url"] if found is not None else f"http://127.0.0.1:{gateway_port()}"


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
        **this_process(),
        # No level: it changes while the server runs and a file
        # would go stale. /capabilities is the live answer.
    }
    if auth.connect_token is not None:
        # Only asks to connect: the operator allows an agent on this PC
        # (fibsem.server.connections). Never the agent's own token. The App
        # Server has none: an agent asks at the gateway.
        payload["connect_token"] = auth.connect_token
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
    if not same_process(data.get("pid"), data.get("started")):
        return None
    return data


def read_gateway_file(path: Optional[Path] = None) -> Optional[dict]:
    """The running gateway's ``{"url", "pid", "connect_token"}``, or None when
    none runs."""
    try:
        with open(path or GATEWAY_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "url" not in data:
        return None
    return data if same_process(data.get("pid"), data.get("started")) else None


def remove_discovery_file(path: Optional[Path] = None) -> None:
    path = Path(path) if path is not None else DISCOVERY_FILE
    try:
        path.unlink()
    except OSError:
        pass
