"""Tescan's own milling settings: the ones only DrawBeam mills with.

A recipe keeps them in its ``driver`` block under ``Tescan``
(``fibsem.structures.FibsemMillingSettings``). Imports only ``fibsem.structures``, so
any PC reads the block, Tescan or not.
"""

from __future__ import annotations

from dataclasses import dataclass

from fibsem.structures import Display, DriverMillingSettings, RangeLimit, param


@dataclass
class TescanMillingSettings(DriverMillingSettings):
    preset: str = param(
        "30 keV; 2nA",
        device_parameter="beam.preset",
        doc="The preset used for milling. Presets define the beam settings for different milling conditions.",
        display=Display("Preset"),
    )
    # 1 µm is the value the cryo lamella milling was validated at on hardware
    # (2026-07-22).
    spot_size: float = param(
        1.0e-6,
        unit="m",
        limits=RangeLimit(1e-9, 1e-4),
        doc="The spot size for the ion beam during milling.",
        display=Display("Spot Size", scale=1e6, step=0.01, decimals=3),
    )
    rate: float = param(
        1.3e-8,
        unit="m³/A/s",
        doc="Ion etching rate — how much material one amp removes per "
        "second. Equivalently µm³/nA/s, which is how TESCAN quotes "
        "it. Default is the cryo lamella value; silicon is 0.3.",
        display=Display("Rate", scale=1e9, unit="mm³/A/s"),
    )
    dwell_time: float = param(
        1.0e-6,
        unit="s",
        doc="The dwell time for the ion beam during milling (µs).",
        display=Display("Dwell Time", scale=1e6),
    )
    spacing: float = param(
        0.005,
        limits=RangeLimit(0.0, 100.0),
        doc="Exposition mesh spacing — how finely the pattern is filled "
        "with exposure points. Dimensionless; the TESCAN default is "
        "1.0 and smaller values mill more finely and take longer.",
        display=Display("Spacing", step=0.001, decimals=4),
    )
