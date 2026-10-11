"""The driver messages that moved from INFO to DEBUG after the parity recordings.

Each is logged once per parameter written, beam blanked or frame acquired, and
together they buried the terminal: a two-lamella run of the default protocol on
Demo printed 539 lines, 254 of them "acquiring new ELECTRON image.". They are
still in the logfile, which records DEBUG. The recordings kept INFO and above, so
a parity test drops these from the old side before it compares.
"""

import re
from typing import List

_NOW_DEBUG = re.compile(
    r"^(?:ELECTRON|ION) (?:.+ set to |beam (?:un)?blanked\.$)"
    r"|^Electron beam (?:working distance|current|voltage) set to "
    r"|^Detector (?:type|mode|brightness|contrast) set to "
    r"|^Angular correction angle set to "
    r"|^acquiring new (?:ELECTRON|ION) image\.$"
    r"|^Acquired Image: "
    r"|^Running autocontrast on "
)


def as_logged_now(messages: List[List[str]]) -> List[List[str]]:
    """The recorded ``[level, message]`` pairs without the ones now at DEBUG."""
    return [m for m in messages if not (m[0] == "INFO" and _NOW_DEBUG.search(m[1]))]
