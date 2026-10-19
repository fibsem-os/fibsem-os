"""Tescan's own milling settings: what a recipe keeps in its ``tescan`` block.

Alone in this module, which imports only the structures, so the driver registry can
name the class (``DriverEntry.milling_settings``) and a recipe's Tescan block is read
and checked on any PC without importing the driver.
"""

from dataclasses import dataclass

from fibsem.structures import (
    Display,
    DriverMillingSettings,
    RangeLimit,
    param,
    shared_milling_setting,
)


@dataclass
class TescanMillingSettings(DriverMillingSettings):
    """What Tescan mills with: the preset sets the current and voltage, and the rest
    go into the DrawBeam layer. The recipe saves its own in its ``tescan`` block."""

    preset: str = param(
        "30 keV; 2nA",
        device_parameter="beam.preset",
        doc="The preset used for milling. Presets define the beam settings for different milling conditions.",
        display=Display("Preset"),
    )
    # 1 µm is the value the cryo lamella milling was validated at on hardware
    # (2026-07-22). Protocols do not set spot_size, so this default is what every
    # milling stage actually uses -- the `milling:` block in the system config is
    # not consulted for it (only milling_current is read from there).
    spot_size: float = param(
        1.0e-6,
        unit="m",
        # Real spot sizes are tens of nm -- the TESCAN default is 50 nm -- so a
        # 1 µm minimum would clamp every real value up to 1 µm.
        limits=RangeLimit(1e-9, 1e-4),
        doc="The spot size for the ion beam during milling.",
        display=Display("Spot Size", scale=1e6, step=0.01, decimals=3),
    )
    rate: float = param(
        1.3e-8,
        # mm³/A/s and TESCAN's own µm³/nA/s are numerically identical.
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
    patterning_mode: str = shared_milling_setting("patterning_mode")
