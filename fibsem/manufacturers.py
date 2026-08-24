"""Canonical manufacturer names and the one alias map (FIB-300).

The same vendor arrives spelled differently depending on the source: live TESCAN
hardware reports "TESCAN" in image headers while the driver property and configs say
"Tescan"; ThermoFisher appears as "Thermo" (configs), "ThermoFisher" (driver
property) and "Thermo Fisher Scientific" (session setup). Comparisons scattered
across the codebase each coped locally -- defensive ``.upper()``, hand-rolled alias
lists -- or didn't cope and silently took the wrong branch (a live inverted-UI bug
on PR #115). One canonical form, normalised at the boundaries (``SystemInfo.from_dict``
and the driver assignments), retires all of that.

Canonical *strings* rather than an Enum, on purpose: CI runs Python 3.8 (no
``StrEnum``), the ``(str, Enum)`` mixin diverges between ``str()`` and ``format()``
on 3.8-3.10 exactly where YAML/dict serialisation would hit it, and strings keep
every existing ``to_dict``/``from_dict``/config round-trip shape-identical.
"""

from typing import Optional

THERMOFISHER = "ThermoFisher"
TESCAN = "Tescan"
DEMO = "Demo"
ODEMIS = "Odemis"
ZEISS = "Zeiss"

# every known spelling, lowercased -> canonical
_ALIASES = {
    "thermo": THERMOFISHER,
    "thermofisher": THERMOFISHER,
    "thermo fisher": THERMOFISHER,
    "thermo fisher scientific": THERMOFISHER,
    "tescan": TESCAN,
    "demo": DEMO,
    "odemis": ODEMIS,
    "zeiss": ZEISS,
}


def normalize_manufacturer(raw: Optional[str]) -> Optional[str]:
    """Return the canonical spelling for any known manufacturer alias.

    Unknown, empty and non-string values pass through unchanged -- normalisation is
    lenient so reading old data can never fail. Validation sites keep their own
    membership checks against the canonical names.
    """
    if not isinstance(raw, str):
        return raw
    return _ALIASES.get(raw.strip().lower(), raw)


def is_tescan(value: Optional[str]) -> bool:
    """Whether *value* names TESCAN, in any known spelling."""
    return normalize_manufacturer(value) == TESCAN


def is_thermo(value: Optional[str]) -> bool:
    """Whether *value* names ThermoFisher, in any known spelling."""
    return normalize_manufacturer(value) == THERMOFISHER
