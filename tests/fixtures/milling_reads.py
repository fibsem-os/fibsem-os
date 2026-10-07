"""Which recipe fields a milling service's ``setup`` reads.

A driver's ``setting_names`` (with ``COMMON_SETTINGS``) says which
`FibsemMillingSettings` fields it mills with, and the milling form shows only those.
`fields_setup_reads` runs ``setup`` on a recipe that records each field it is asked
for, so a test can hold the list to the code. Whole-recipe reads for a log message
(``to_dict``) don't count.
"""

from dataclasses import fields
from typing import Set

from fibsem.structures import FibsemMillingSettings

FIELDS = frozenset(f.name for f in fields(FibsemMillingSettings))


def fields_setup_reads(milling, settings: FibsemMillingSettings) -> Set[str]:
    read: Set[str] = set()
    quiet = []

    class Recording(FibsemMillingSettings):
        def __getattribute__(self, name):
            if name in FIELDS and not quiet:
                read.add(name)
            return object.__getattribute__(self, name)

        def to_dict(self):
            quiet.append(True)
            try:
                return super().to_dict()
            finally:
                quiet.pop()

    quiet.append(True)
    recording = Recording(**{name: getattr(settings, name) for name in FIELDS})
    quiet.pop()
    milling.setup(recording)
    return read
