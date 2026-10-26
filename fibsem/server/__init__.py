"""The fibsem server package.

Exports resolve lazily (PEP 562), so importing a light sibling such as
``fibsem.server.parts`` does not build the whole server.
"""

from typing import TYPE_CHECKING

__all__ = ["AuthConfig", "build_server"]

_EXPORTS = {
    "AuthConfig": "fibsem.server.auth",
    "build_server": "fibsem.server.server",
}

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fibsem.server.auth import AuthConfig  # noqa: F401
    from fibsem.server.server import build_server  # noqa: F401


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
