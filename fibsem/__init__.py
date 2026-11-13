import os
import sys


def _default_qt_binding() -> None:
    """Pick PySide6 unless QT_API names a binding (FIB-1565). qtpy would otherwise
    take PyQt5 wherever both are installed. Windows 10 before build 1809 cannot run
    Qt 6, so PyQt5 there; it needs the [pyqt5] extra. Must run before qtpy is
    imported, hence here."""
    if os.environ.get("QT_API"):
        return
    old_windows = sys.platform == "win32" and sys.getwindowsversion().build < 17763
    os.environ["QT_API"] = "pyqt5" if old_windows else "pyside6"


_default_qt_binding()

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("fibsem")
except ModuleNotFoundError:
    __version__ = "unknown"
