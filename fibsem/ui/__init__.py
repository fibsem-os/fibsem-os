"""Lazy re-exports for the fibsem UI widgets.

Importing these eagerly meant that reaching *any* module under ``fibsem.ui`` — including
leaves with no display code, such as ``fibsem.ui.qt.threading`` — executed this file and so
imported every widget, one of which imports napari. Resolving the names on first access
(PEP 562) lets Qt-free and napari-free modules be imported on their own.

**Prefer importing from the concrete module** (``from fibsem.ui.FibsemMovementWidget import
FibsemMovementWidget``), which is what code in this repo now does.

The re-exports are kept, and keep returning the *class*. That needs one piece of machinery:
each widget class shares its name with the module defining it, and Python binds an imported
submodule as an attribute of its package — so once ``fibsem.ui.FibsemMovementWidget`` (the
module) has been imported by anything, that name would resolve to the module, and PEP 562's
``__getattr__`` would never run because the attribute is no longer missing. The eager version
papered over this by overwriting the attribute with the class at import time. ``_LazyExports``
below restores that guarantee by unwrapping the shadowing module on access.
"""
import importlib
import sys
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # keep static analysis and IDEs resolving the names
    from fibsem.ui.FibsemCryoDepositionWidget import FibsemCryoDepositionWidget
    from fibsem.ui.FibsemEmbeddedDetectionWidget import FibsemEmbeddedDetectionUI
    from fibsem.ui.FibsemImageSettingsWidget import FibsemImageSettingsWidget
    from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget
    from fibsem.ui.FibsemMinimapWidget import FibsemMinimapWidget
    from fibsem.ui.FibsemMovementWidget import FibsemMovementWidget
    from fibsem.ui.FibsemSpotBurnWidget import FibsemSpotBurnWidget
    from fibsem.ui.FibsemSystemSetupWidget import FibsemSystemSetupWidget
    from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget

# exported name -> module that defines it
_WIDGETS = {
    "FibsemImageSettingsWidget": "fibsem.ui.FibsemImageSettingsWidget",
    "FibsemMovementWidget": "fibsem.ui.FibsemMovementWidget",
    "FibsemSystemSetupWidget": "fibsem.ui.FibsemSystemSetupWidget",
    "FibsemCryoDepositionWidget": "fibsem.ui.FibsemCryoDepositionWidget",
    "FibsemMinimapWidget": "fibsem.ui.FibsemMinimapWidget",
    "FibsemManipulatorWidget": "fibsem.ui.FibsemManipulatorWidget",
    "FibsemSpotBurnWidget": "fibsem.ui.FibsemSpotBurnWidget",
    "MillingTaskViewerWidget": "fibsem.ui.widgets.milling_task_viewer_widget",
    # Optional — needs the segmentation stack; see _OPTIONAL and DETECTION_AVAILABLE.
    "FibsemEmbeddedDetectionUI": "fibsem.ui.FibsemEmbeddedDetectionWidget",
}

# Requires optional dependencies (torch et al). The eager __init__ imported it under
# try/except ImportError, so the name simply did not exist when the stack was missing.
# It is therefore kept out of __all__: advertising it unconditionally would make
# ``from fibsem.ui import *`` and ``inspect.getmembers`` force the import and raise.
_OPTIONAL = "FibsemEmbeddedDetectionUI"

__all__ = [n for n in _WIDGETS if n != _OPTIONAL] + ["DETECTION_AVAILABLE"]

_detection: "bool | None" = None  # memoised; the probe import is not cheap


def _resolve(name: str):
    """Import the module that defines *name* and return the attribute itself."""
    return getattr(importlib.import_module(_WIDGETS[name]), name)


def _detection_available() -> bool:
    global _detection
    if _detection is None:
        try:
            _resolve(_OPTIONAL)
        except ImportError:
            import logging

            logging.debug("Could not import FibsemEmbeddedDetectionWidget")
            _detection = False
        else:
            _detection = True
    return _detection


def __getattr__(name: str):
    """Resolve an export on first access, then cache it in the module namespace."""
    if name == "DETECTION_AVAILABLE":
        value = _detection_available()
    elif name in _WIDGETS:
        try:
            value = _resolve(name)
        except ImportError:
            if name != _OPTIONAL:
                raise
            # Mirror the eager behaviour: the name is absent, not broken. AttributeError
            # (not ImportError) keeps hasattr/getmembers/dir degrading gracefully.
            raise AttributeError(
                f"{name!r} is unavailable: the optional segmentation dependencies are "
                f"not installed (see fibsem.ui.DETECTION_AVAILABLE)"
            ) from None
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value  # cache — __getattr__ runs once per name
    return value


def __dir__():
    names = {*globals(), *__all__}
    if _detection_available():  # memoised, so dir() stays cheap after the first call
        names.add(_OPTIONAL)
    return sorted(names)


class _LazyExports(ModuleType):
    """Module type that unwraps a submodule shadowing an exported class name.

    ``ModuleType.__getattribute__`` already falls back to the module-level ``__getattr__``
    above when a name is missing, so PEP 562 still does the lazy loading. This only fixes
    the case where the name is *present* but bound to the submodule.
    """

    def __getattribute__(self, name):
        value = ModuleType.__getattribute__(self, name)
        if name in _WIDGETS and isinstance(value, ModuleType):
            value = _resolve(name)  # the class, not the module that defines it
            ModuleType.__setattr__(self, name, value)
        return value


# _WIDGETS / _resolve are read from this module's globals inside __getattribute__ (not via
# self), so there is no recursion.
sys.modules[__name__].__class__ = _LazyExports
