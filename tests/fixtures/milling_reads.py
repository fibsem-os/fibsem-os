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


def own_milling_code(microscope):
    """*microscope*, milling with its backend's own code (``ThermoMilling``,
    ``TescanDrawBeam``, ``OdemisPatterning``): the code its milling service drives,
    which it milled with before the service. The parity tests hold the service to it.

    ``ServiceMilling`` raises without a service now, rather than falling through to
    that code, so the microscope's class becomes a subclass whose milling methods
    are the ones after ``ServiceMilling`` in its order.
    """
    from fibsem.services.milling import ServiceMilling

    cls = type(microscope)
    after = cls.__mro__[cls.__mro__.index(ServiceMilling) + 1 :]
    own = {}
    for name, value in vars(ServiceMilling).items():
        if not callable(value) or name.startswith("__") or name == "_milling_service":
            continue
        own[name] = next(vars(base)[name] for base in after if name in vars(base))
    microscope.__class__ = type(f"Own{cls.__name__}", (cls,), own)
    microscope.milling = None
    return microscope
