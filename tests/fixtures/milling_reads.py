"""Which recipe fields a milling service's ``setup`` reads.

A driver's ``setting_names`` (with ``COMMON_SETTINGS``) says which
`FibsemMillingSettings` fields it mills with, and the milling form shows only those.
`fields_setup_reads` runs ``setup`` on a recipe that records each field it is asked
for, so a test can hold the list to the code; a driver's own field counts by its path
(``driver.Tescan.preset``). Whole-recipe reads for a log message (``to_dict``) and the
ignored-fields warning don't count.
"""

from dataclasses import fields, is_dataclass
from typing import Set

from fibsem.structures import FibsemMillingSettings

FIELDS = frozenset(f.name for f in fields(FibsemMillingSettings))


def fields_setup_reads(milling, settings: FibsemMillingSettings) -> Set[str]:
    read: Set[str] = set()
    quiet = []

    def recording_block(prefix, block):
        if not is_dataclass(block):
            return block
        names = {f.name for f in fields(block)}

        class RecordingBlock(type(block)):
            def __getattribute__(self, name):
                if name in names and not quiet:
                    read.add(f"driver.{prefix}.{name}")
                return object.__getattribute__(self, name)

        return RecordingBlock(**{name: getattr(block, name) for name in names})

    class Recording(FibsemMillingSettings):
        def __getattribute__(self, name):
            value = object.__getattribute__(self, name)
            if name in FIELDS and not quiet:
                if name != "driver":
                    read.add(name)
            return value

        def to_dict(self):
            quiet.append(True)
            try:
                return super().to_dict()
            finally:
                quiet.pop()

    def ignored_settings(recipe):
        quiet.append(True)
        try:
            return type(milling).ignored_settings(milling, recipe)
        finally:
            quiet.pop()

    quiet.append(True)
    recording = Recording(**{name: getattr(settings, name) for name in FIELDS})
    recording.driver = {
        prefix: recording_block(prefix, block)
        for prefix, block in settings.driver.items()
    }
    quiet.pop()
    milling.ignored_settings = ignored_settings
    try:
        milling.setup(recording)
    finally:
        del milling.ignored_settings
    return read
