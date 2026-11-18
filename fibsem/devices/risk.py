"""What a device parameter's write or a command risks: one label each, declared
next to its code (``DeviceParameter(..., risk=...)``, ``@command(risk=...)``).

A label says what the call does to the instrument or the sample, not who may
make it. A parameter's read and a query are always ``read_only``.
"""

from enum import Enum
from typing import Optional


class Risk(str, Enum):
    """What a write or a command risks, from least to most."""

    READ_ONLY = "read_only"  # reads; moves and changes nothing
    LOCAL = "local"  # the app's own records or a computation; no hardware
    STOPS = "stops"  # stops, pauses or resumes what is running
    OPTICS = "optics"  # beam, lens, detector and scan settings
    ACQUIRES = "acquires"  # takes an image, exposing the sample to a beam or light
    MOVES_HARDWARE = "moves_hardware"  # moves the stage, the objective, a grid
    ALTERS_SAMPLE = "alters_sample"  # mills, burns, deposits or sputters
    HIGH_IMPACT = "high_impact"  # instrument-level and hard to undo
    NEVER = "never"  # the agent may not, at any level


RISK_WORDS = {
    Risk.READ_ONLY: "reads only",
    Risk.LOCAL: "changes the app's own records, no hardware",
    Risk.STOPS: "stops or pauses what is running",
    Risk.OPTICS: "changes a beam, lens, detector or scan setting",
    Risk.ACQUIRES: "takes an image",
    Risk.MOVES_HARDWARE: "moves hardware",
    Risk.ALTERS_SAMPLE: "changes the sample",
    Risk.HIGH_IMPACT: "acts on the whole instrument and is hard to undo",
    Risk.NEVER: "the agent may never do this",
}


def checked(risk: Optional[str], what: str) -> Optional[Risk]:
    """*risk* as a `Risk`, or None when it is None; a label that isn't one is
    an error where it is declared."""
    if risk is None:
        return None
    try:
        return Risk(risk)
    except ValueError:
        labels = ", ".join(label.value for label in Risk)
        raise ValueError(f"{what}: {risk!r} is not a risk label ({labels})") from None
