"""What a tiled acquisition says about itself while it runs.

`FibsemMicroscope.tiled_acquisition_signal` carries a bare dict and has no declared
contract, so a consumer works out what it received from which keys are present. That was
survivable while `imaging.tiled` was its only emitter: three payload shapes, all from one
class, all carrying the same keys.

It stops being survivable with a second producer. The fluorescence tileset runner reports
the same thing -- tile *n* of *N*, and a mosaic so far -- and reports it somewhere else
entirely, on a signal hanging off the detector that also carries z-stacks, channel
acquisitions and autofocus sweeps (FIB-725). Bringing it here needs consumers to be able
to tell whose run they are looking at, because most of them want exactly one:

* `FibsemMinimapWidget` and `FibsemOverviewWidget` each drive a *beam* overview, and hand
  the payload's mosaic straight to their own canvas. Handed a fluorescence one they would
  draw it into the beam mosaic -- and the fluorescence preview is keyed `image` already,
  deliberately, to match this signal.
* `AutoLamellaSingleWindowUI` shows whichever is running in the status bar, and needs to
  say *which*, or a glance tells you a run is going and not what it is.

Hence `modality`: the one thing that differs, in the word this codebase already uses for
it. Everything on this signal is a tiled run by definition -- that is what the signal is
named for -- so the kind of *work* needs no discriminator; what varies is what is imaging.

# Reading a payload

Use :func:`is_modality`, not `payload["modality"]`. The key is new, and a consumer that
demands it would ignore every payload emitted by an older producer -- including anything
outside this repository subscribing to a public signal. Absent means beam, which is what
it was before this existed.
"""

from __future__ import annotations


# The beam tiler: SEM or FIB, since no consumer distinguishes them -- both draw into the
# same overview.
MODALITY_BEAM = "beam"
# The fluorescence tileset runner.
MODALITY_FLUORESCENCE = "fluorescence"

__all__ = ["MODALITY_BEAM", "MODALITY_FLUORESCENCE", "modality_of", "is_modality"]


def modality_of(payload: dict) -> str:
    """Which imaging modality produced *payload*.

    Defaults to the beam rather than raising or returning None: this signal carried
    nothing else for its whole life, so an unlabelled payload is a beam run from a
    producer that predates the key.
    """
    return payload.get("modality") or MODALITY_BEAM


def is_modality(payload: dict, modality: str) -> bool:
    """Whether *payload* came from *modality*, treating unlabelled as beam.

    The form consumers should use. `payload.get("modality") == MODALITY_BEAM` looks
    equivalent and is not: it drops every payload from a producer that has not been
    taught the key, which on a public signal is not only the ones in this repository.
    """
    return modality_of(payload) == modality
