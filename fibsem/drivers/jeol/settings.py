"""JEOL's own milling settings: the ones only its process queue mills with.

A recipe keeps them in its ``driver`` block under ``JEOL``
(``fibsem.structures.settings.FibsemMillingSettings``). Imports only
``fibsem.structures``, so any PC reads the block, JEOL or not.
"""

from __future__ import annotations

from dataclasses import dataclass

from fibsem.structures.params import Display, RangeLimit, param
from fibsem.structures.settings import DriverMillingSettings


@dataclass
class JeolMillingSettings(DriverMillingSettings):
    # 1 nC/µm³, the CryoLameller driver's default; not yet measured on a JEOL.
    milling_rate: float = param(
        1e9,
        unit="C/m³",
        limits=RangeLimit(1e6, 1e11),
        doc="The charge that mills away one unit of volume. JEOL takes a dose per "
        "area, so each pattern's dose is its depth times this.",
        display=Display("Milling Rate", scale=1e-9, unit="nC/µm³", decimals=4),
    )
