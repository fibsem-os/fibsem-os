"""Record ThermoMicroscope's get_available_values over the fake AutoScript SDK.

Run as a script, in its own interpreter, for the reason ``autoscript_beam_parity.py``
gives: the fake SDK must be installed before ``fibsem.microscopes.autoscript`` is
imported. It writes JSON to the path it is given: for a microscope with and without a
plasma column, connected as the app connects it (beams built), the answer to every
key in ``KEYS`` for no beam type and for each beam (or what it raised).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import autoscript_beam_parity as P  # noqa: E402  (installs the fake SDK)

from fibsem.structures import BeamType  # noqa: E402

KEYS = json.loads(sys.argv[2])


def record():
    out = {}
    for plasma in (False, True):
        microscope = P.routed(plasma)
        P._preset(
            microscope.connection,
            "patterning.list_all_application_files",
            lambda: ["Si", "Si-multipass", "C"],
        )
        for beam_type in (None, BeamType.ELECTRON, BeamType.ION):
            name = "None" if beam_type is None else beam_type.name
            for key in KEYS:
                try:
                    answer = P.S._plain(microscope.get_available_values(key, beam_type))
                except Exception as e:  # a raise is an answer too
                    answer = f"EXC {type(e).__name__}: {e}"
                out[f"thermo plasma={plasma} {name} {key}"] = answer
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump(record(), f, default=str)
