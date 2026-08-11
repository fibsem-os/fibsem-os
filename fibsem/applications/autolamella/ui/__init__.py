"""AutoLamella's UI package.

`AutoLamellaUI` is exported lazily (PEP 562) rather than imported here.

Importing it eagerly meant that importing *any* leaf widget in this package pulled
the whole application in first, because the package `__init__` runs before the
submodule. Once widgets under `fibsem/ui/` began importing widgets from here, that
closed a loop: `fibsem.ui.<widget>` -> this package -> `AutoLamellaUI` -> back into
`fibsem.ui`. The `except ImportError` below then swallowed it and bound
`AutoLamellaUI` to None, so the failure showed up as a dead application rather than
a traceback, and only for some import orders.

Deferring the import to first attribute access keeps `from ... .ui import
AutoLamellaUI` working exactly as before while leaving leaf-widget imports cheap and
loop-free.
"""

import logging
from typing import TYPE_CHECKING

__all__ = ["AutoLamellaUI"]

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI


def __getattr__(name: str):
    if name != "AutoLamellaUI":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
    except ImportError as e:
        logging.info(f"Error importing autolamella.ui: {e}, using dummy instead.")
        AutoLamellaUI = None

    # Cache on the module so later accesses skip this hook entirely, matching the
    # old behaviour where the name was bound once at import time.
    globals()["AutoLamellaUI"] = AutoLamellaUI
    return AutoLamellaUI
