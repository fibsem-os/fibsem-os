"""How a value is shown: label, scale, unit, step and decimals.

Declared once per device ``Parameter`` (``Parameter(display=Display(...))``) and per
dataclass field (``field_meta(display=Display(...))``), so a form renders device
parameters and recipe fields alike. What a value may be (its unit, limits and
choices) is not here: the parameter or field declares that, in SI, and an instrument
reports its own.

It imports nothing from fibsem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Union


@dataclass(frozen=True)
class Display:
    """How to show a parameter. Every field is optional; a form falls back to its own
    default for the ones left out.

    ``scale`` turns the SI value into the shown one (1e6: metres to µm). ``unit`` is the
    shown unit; left out, it is the parameter's unit with the scale's SI prefix (1e6 and
    "m" give "µm"). Give it when the scale isn't an SI prefix: 180/π for degrees, 100
    for percent. ``step`` is in shown units.
    """

    label: Optional[str] = None
    scale: Optional[float] = None
    unit: Optional[str] = None
    step: Optional[float] = None
    decimals: Optional[int] = None
    advanced: bool = False
    # never shown in a form: set by code, not by hand (a pattern's drawn shapes)
    hidden: bool = False


# A composite value (a Point, a stage position) takes one hint per field.
DisplayHint = Union[Display, Mapping[str, Display]]


def display_to_dict(hint: Optional[DisplayHint]) -> Optional[Dict[str, Any]]:
    """A hint as JSON-ready data, with only the fields it sets: one dict, or one per
    field for a composite value. ``None`` for no hint. What a device's description
    carries, so a client with no copy of the class (an agent) shows it the same way."""
    if hint is None:
        return None
    if isinstance(hint, Mapping):
        return {name: display_to_dict(field) for name, field in hint.items()}
    fields = {
        "label": hint.label,
        "scale": hint.scale,
        "unit": hint.unit,
        "step": hint.step,
        "decimals": hint.decimals,
    }
    shown = {key: value for key, value in fields.items() if value is not None}
    if hint.advanced:
        shown["advanced"] = True
    if hint.hidden:
        shown["hidden"] = True
    return shown
