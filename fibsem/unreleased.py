"""Whether this install runs the parts of fibsemOS that are not released yet: the
agent and the gateway.

On when running from a git checkout of fibsem-os, off in an install from a
release. ``FIBSEM_UNRELEASED=1`` or ``0`` overrides either way.
"""

import os

from fibsem.versioning import _source_checkout_root

ENV = "FIBSEM_UNRELEASED"

NOT_RELEASED = "{what} is not part of this release of fibsemOS."


def enabled() -> bool:
    """Whether the agent and the gateway run here."""
    value = os.environ.get(ENV, "").strip()
    if value in ("0", "1"):
        return value == "1"
    return _source_checkout_root() is not None


def refuse(what: str) -> None:
    """Exit with a message when *what* is not part of this release."""
    if not enabled():
        raise SystemExit(NOT_RELEASED.format(what=what))
