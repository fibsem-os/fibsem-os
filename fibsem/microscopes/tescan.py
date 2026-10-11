"""Moved to ``fibsem.drivers.tescan.microscope``; this name is kept for scripts that still import it.

Importing it warns, and gives the moved module itself, so its names, and patches made
through this name, are the same objects.
"""

import importlib
import sys
import warnings

warnings.warn(
    "fibsem.microscopes.tescan has moved to fibsem.drivers.tescan.microscope",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = importlib.import_module("fibsem.drivers.tescan.microscope")
