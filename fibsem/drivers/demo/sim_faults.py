"""Things that go wrong on the simulator, when the sample settings say so.

``faults`` in the sample settings is a list, one entry per fault::

    faults:
      - {kind: beam_off, beam: ion, at: 3}        # the third FIB image
      - {kind: mill_drift, at: [2, 5], amount: 4e-7}   # a seeded pass from 2 to 5
      - {kind: stage_fails, at: 10}                # the tenth stage move

A fault fires once, on the `at`-th event of the kind it goes with, counted from
the start of the session (for a fault with a `beam`, that beam's events). `at`
as a pair is drawn between the two from the scene's seed. What fired is kept in
``log``, and goes out with the scene's record.

=============  =====  ============================================  =========
kind           event  what happens                                  amount
=============  =====  ============================================  =========
beam_off       image  the beam turns off: black frames until it is  -
               mill   turned on again; a pass mills nothing
current_drop   image  the beam current falls to `amount` of itself  0.3
               mill
charging       image  this frame charges `amount` more              3.0
blank_image    image  this frame is black                           -
noisy_image    image  this frame has `amount` times the noise       6.0
stage_fails    move   the move raises, the stage where it was       -
stage_off      move   the stage lands `amount` m off in x, and      5e-6
                      reads as if it had not
gis_closed     gis    the valve does not open: nothing is laid      -
                      down
mill_drift     mill   this pass drifts `amount` m more, up the      3e-7
                      image
=============  =====  ============================================  =========

beam_off and current_drop go with an image unless `on: mill` is given.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# kind: (the event it goes with by default, the others it may, its amount)
KINDS = {
    "beam_off": ("image", ("mill",), 0.0),
    "current_drop": ("image", ("mill",), 0.3),
    "charging": ("image", (), 3.0),
    "blank_image": ("image", (), 0.0),
    "noisy_image": ("image", (), 6.0),
    "stage_fails": ("move", (), 0.0),
    "stage_off": ("move", (), 5e-6),
    "gis_closed": ("gis", (), 0.0),
    "mill_drift": ("mill", (), 3e-7),
}
BEAMS = ("electron", "ion")
KEYS = {"kind", "at", "on", "beam", "amount"}


class SimulatedFault(RuntimeError):
    """A command the simulator was set to fail."""


@dataclass
class Fault:
    kind: str
    on: str
    at: int
    beam: Optional[str]
    amount: float
    fired: bool = False


class Faults:
    """A sample's faults: asked at each event which are due."""

    def __init__(self, entries=None, seed: int = 0) -> None:
        self.faults: List[Fault] = []
        self.counts: Dict[tuple, int] = {}
        self.log: List[dict] = []
        for n, entry in enumerate(entries or []):
            unknown = set(entry) - KEYS
            kind = entry.get("kind")
            if unknown or kind not in KINDS:
                raise ValueError(f"Not a simulator fault: {dict(entry)}")
            event, others, amount = KINDS[kind]
            on = entry.get("on", event)
            beam = entry.get("beam")
            if on not in (event,) + others or beam not in (None,) + BEAMS:
                raise ValueError(f"Not a simulator fault: {dict(entry)}")
            at = entry.get("at", 1)
            if isinstance(at, (list, tuple)):
                rng = np.random.default_rng([int(seed) & 0x7FFFFFFF, 59, n])
                at = int(rng.integers(int(at[0]), int(at[1]) + 1))
            amount = float(entry.get("amount", amount))
            self.faults.append(Fault(kind, on, int(at), beam, amount))

    def due(self, event: str, beam: Optional[str] = None) -> List[Fault]:
        """Count one `event` (an image or a pass of `beam`, a stage move, a
        valve opening) and give the faults that fire on it."""
        counts = []
        for key in {(event, None), (event, beam)}:
            self.counts[key] = self.counts.get(key, 0) + 1
            counts.append((key[1], self.counts[key]))
        out = []
        for fault in self.faults:
            if fault.fired or fault.on != event:
                continue
            if (fault.beam, fault.at) not in counts:
                continue
            fault.fired = True
            entry = {
                "kind": fault.kind,
                "on": event,
                "at": fault.at,
                "beam": beam,
                "amount": fault.amount,
                "time": time.time(),
            }
            self.log.append(entry)
            logging.info({"msg": "sim_fault", **entry})
            out.append(fault)
        return out
