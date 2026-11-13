import os
import sys
import warnings


def _default_qt_binding() -> None:
    """Pick PySide6 unless QT_API names a binding (FIB-1565). An environment upgraded
    from 0.6 can still hold PyQt5, which qtpy would otherwise take first. Must run
    before qtpy is imported, hence here."""
    if sys.platform == "win32" and sys.getwindowsversion().build < 17763:
        warnings.warn(
            "The fibsemOS GUI needs Qt 6, which needs Windows 10 build 1809 or later. "
            "On this PC use fibsemOS 0.6.",
            stacklevel=2,
        )
    os.environ.setdefault("QT_API", "pyside6")


_default_qt_binding()

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("fibsem")
except ModuleNotFoundError:
    __version__ = "unknown"
