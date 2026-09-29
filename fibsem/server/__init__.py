"""The fibsem server package.

Exports resolve lazily (PEP 562): the factory and auth pull in FastAPI, but
sibling modules like ``fibsem.server.images`` (PIL-only) and
``fibsem.server.catalog`` (stdlib-only) are deliberately importable in
environments without the [server] extra — the app's prompt/image serializers
run in the GUI process whether or not a server can be built there. An eager
import here would make ``from fibsem.server.images import ...`` require
fastapi transitively, which is exactly what bit the ui-tests CI leg (no
fastapi installed, preview silently degraded to None).
"""

from typing import TYPE_CHECKING

__all__ = ["AuthConfig", "FibsemClient", "FibsemServer", "Scope", "build_server"]

_EXPORTS = {
    "AuthConfig": "fibsem.server.auth",
    "Scope": "fibsem.server.auth",
    "FibsemClient": "fibsem.server.client",
    "FibsemServer": "fibsem.server.server",
    "build_server": "fibsem.server.server",
}

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fibsem.server.auth import AuthConfig, Scope  # noqa: F401
    from fibsem.server.client import FibsemClient  # noqa: F401
    from fibsem.server.server import FibsemServer, build_server  # noqa: F401


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
