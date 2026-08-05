"""Simulated-hardware timing.

The Demo microscope (``fibsem.microscopes.simulator``) and the simulated
fluorescence-microscope base classes (``fibsem.fm.microscope``) sleep to emulate
real acquisition, stage, objective and milling timing. Those sleeps dominate the
test suite's wall-clock — most of it is spent asleep rather than computing — so
the tests disable them via the ``FIBSEM_SIM_NO_DELAY`` environment variable (set
in ``tests/conftest.py``).

Real drivers (e.g. the odemis fluorescence microscope) override these methods and
never call :func:`sim_sleep`, so the flag only affects simulated timing.
"""

import os
import time


def sim_sleep(seconds: float) -> None:
    """Sleep to emulate simulated-hardware timing.

    A no-op when ``FIBSEM_SIM_NO_DELAY=1`` (or when ``seconds`` is non-positive),
    which is how the test suite skips the simulator's artificial delays. Reads the
    environment on every call so it can be toggled per run without re-importing.
    """
    if seconds > 0 and os.environ.get("FIBSEM_SIM_NO_DELAY") != "1":
        time.sleep(seconds)
