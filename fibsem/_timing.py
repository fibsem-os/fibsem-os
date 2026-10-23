"""Simulated-hardware timing.

The Demo microscope and the simulated fluorescence microscope
(``fibsem.drivers.demo.simulator``, and the Demo devices) sleep to emulate
real acquisition, stage, objective and milling timing. Those sleeps dominate the
test suite's wall-clock — most of it is spent asleep rather than computing — so
the tests disable them via the ``FIBSEM_SIM_NO_DELAY`` environment variable (set
in ``tests/conftest.py``).

Real drivers never call :func:`sim_sleep`, so the flag only affects simulated
timing.
"""

import os
import time


def sim_delays_off() -> bool:
    """Whether ``FIBSEM_SIM_NO_DELAY=1`` turns the simulated delays off. Read on
    every call so it can be toggled per run without re-importing."""
    return os.environ.get("FIBSEM_SIM_NO_DELAY") == "1"


def sim_sleep(seconds: float) -> None:
    """Sleep to emulate simulated-hardware timing.

    A no-op when ``FIBSEM_SIM_NO_DELAY=1`` (or when ``seconds`` is non-positive),
    which is how the test suite skips the simulator's artificial delays.
    """
    if seconds > 0 and not sim_delays_off():
        time.sleep(seconds)
