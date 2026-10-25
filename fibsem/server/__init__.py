"""The fibsem server package.

Exports resolve lazily (PEP 562): the factory and auth pull in FastAPI, but
sibling modules like ``fibsem.server.images`` (PIL-only) are deliberately
importable in environments without the [server] extra — the app's prompt/image serializers
run in the GUI process whether or not a server can be built there. An eager
import here would make ``from fibsem.server.images import ...`` require
fastapi transitively, which is exactly what bit the ui-tests CI leg (no
fastapi installed, preview silently degraded to None).
"""

from importlib.util import find_spec
from typing import TYPE_CHECKING, Optional, Sequence

__all__ = ["AuthConfig", "build_server", "missing_server_extra"]

# What serving needs beyond the base install: the [server] extra.
SERVER_EXTRA = ("fastapi", "uvicorn")

_EXPORTS = {
    "AuthConfig": "fibsem.server.auth",
    "build_server": "fibsem.server.server",
}

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fibsem.server.auth import AuthConfig  # noqa: F401
    from fibsem.server.server import build_server  # noqa: F401


def missing_server_extra(packages: Sequence[str] = SERVER_EXTRA) -> Optional[str]:
    """Why this install can't use *packages* (serve, by default), or None when
    the [server] extra is in."""
    missing = [name for name in packages if find_spec(name) is None]
    if not missing:
        return None
    return (
        f"the [server] extra ({', '.join(missing)} not installed): "
        "pip install fibsem[server]"
    )


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
