"""Record the AutoScript calls of Thermo's spot burn through the spot burn service.

Run as a script, in its own interpreter, for the same reason as
``autoscript_beam_parity.py``, whose fake SDK, microscope and recorder it reuses. It
writes JSON to the path it is given: ``cases``, each holding, for one burn on a
microscope with the service built as connect builds it, what it returned, the SDK
calls and writes it made, the messages it logged and the progress it reported; and
``facts``, what the service is.
"""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import autoscript_beam_parity as B  # noqa: E402  (installs the fake SDK)

from fibsem.imaging.spot import SpotBurnSettings  # noqa: E402
from fibsem.structures import Point  # noqa: E402

# The exposure countdown: nothing here needs to wait.
time.sleep = lambda seconds: None


def make():
    microscope = B.make(plasma=False)
    microscope._build_beams()
    microscope._build_spot_burn()
    return microscope


def _settings(*points, exposure_time=2.0):
    return SpotBurnSettings(
        coordinates=list(points), exposure_time=exposure_time, milling_current=1e-9
    )


def _stopped():
    event = threading.Event()
    event.set()
    return event


CASES = (
    (
        "two points",
        lambda m: m.run_spot_burn(_settings(Point(0.2, 0.3), Point(0.7, 0.4))),
    ),
    (
        "a point outside the image",
        lambda m: m.run_spot_burn(_settings(Point(1.5, 0.5), Point(0.5, 0.5))),
    ),
    (
        "one point, one second",
        lambda m: m.run_spot_burn(_settings(Point(0.5, 0.5), exposure_time=1.0)),
    ),
    (
        "stopped before the first point",
        lambda m: m.run_spot_burn(_settings(Point(0.5, 0.5)), stop_event=_stopped()),
    ),
)


def _burn(call):
    microscope = make()
    reports = []
    microscope.spot_burn_progress_signal.connect(
        lambda p: reports.append({**vars(p), "status": p.status.value})
    )
    result, calls, messages = B.run(lambda: call(microscope))
    return [result, calls, messages, reports]


def cases():
    return [{"key": key, "burn": _burn(call)} for key, call in CASES]


def facts():
    from fibsem.services.spot_burn import SpotBurn

    spot_burn = make().spot_burn
    no_ion = B.make(plasma=False, ion=False)
    no_ion._build_beams()
    no_ion._build_spot_burn()
    return {
        "is_spot_burn": isinstance(spot_burn, SpotBurn),
        "roles": sorted(spot_burn.declared_roles()),
        "ion": spot_burn.ion.name,
        "supported": sorted(spot_burn.supported_settings()),
        "no_ion": no_ion.spot_burn is None,
    }


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump({"cases": cases(), "facts": facts()}, f, indent=1, default=repr)
