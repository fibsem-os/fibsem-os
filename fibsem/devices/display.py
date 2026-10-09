"""How a value is shown: label, scale, unit, step and decimals.

Declared once per device ``Parameter`` (``Parameter(display=Display(...))``), in the
same vocabulary as a dataclass field's ``field_meta``, so a form renders device
parameters and recipe fields alike. What a value may be (limits, choices, settable) is
not here: the instrument reports that, in ``ParameterMetadata``.

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

    def as_field_metadata(self, unit: Optional[str] = None) -> Dict[str, Any]:
        """The hint as ``field_meta`` keys, for a parameter in ``unit`` (its SI unit).

        Only the keys this hint sets, so they can be laid over a form's defaults.
        """
        keys = {
            "label": self.label,
            "scale": self.scale,
            "unit": unit,
            "display_unit": self.unit,
            "step": self.step,
            "decimals": self.decimals,
        }
        metadata = {key: value for key, value in keys.items() if value is not None}
        if self.advanced:
            metadata["advanced"] = True
        return metadata


# A composite value (a Point, a stage position) takes one hint per field.
DisplayHint = Union[Display, Mapping[str, Display]]
