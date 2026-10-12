"""Moved to ``fibsem.drivers.demo.microscope``; this name is kept for scripts that still import it.

Importing it warns, and gives the moved module itself, so its names, and patches made
through this name, are the same objects.
"""

import importlib
import sys
import warnings

warnings.warn(
    "fibsem.microscopes.device_demo has moved to fibsem.drivers.demo.microscope",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = importlib.import_module("fibsem.drivers.demo.microscope")
