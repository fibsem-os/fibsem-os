"""Which instrument a Demo stands in for: its profile.

Every Demo runs the same simulation (the images, the sample scene, milling, timing).
A profile holds what differs from one vendor's instrument to another's: the
manufacturer the microscope reports, and the values its beams offer.

A configuration picks one by its manufacturer and ``sim: {enabled: true}``
(``fibsem.drivers.registry.connect_microscope``): ``info.manufacturer: ThermoFisher``
with the sim enabled is the Demo with the ThermoFisher profile. ``info.manufacturer:
Demo`` is the same Demo, so the simulator configurations written before profiles keep
working.

ThermoFisher is the only profile so far, and it is what the Demo always simulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

from fibsem import manufacturers
from fibsem.structures import BeamType


@dataclass(frozen=True)
class DemoProfile:
    """What a Demo shows of the instrument it stands in for."""

    # What ``microscope.manufacturer`` reports.
    manufacturer: str
    # The beams' choices.
    voltages: Mapping[BeamType, Tuple[float, ...]]
    detector_types: Tuple[str, ...]
    detector_modes: Tuple[str, ...]


THERMOFISHER_PROFILE = DemoProfile(
    manufacturer=manufacturers.THERMOFISHER,
    voltages={
        BeamType.ELECTRON: (2000, 5000, 10000, 20000, 30000),
        BeamType.ION: (500, 1000, 2000, 8000, 16000, 30000),
    },
    detector_types=("ETD", "TLD", "EDS"),
    detector_modes=("SecondaryElectrons", "BackscatteredElectrons", "EDS"),
)

# Each profile, by the manufacturer a configuration names. Demo is ThermoFisher's,
# as it always was.
_PROFILES: Mapping[str, DemoProfile] = {
    manufacturers.THERMOFISHER: THERMOFISHER_PROFILE,
    manufacturers.DEMO: THERMOFISHER_PROFILE,
}


def simulated_profile(manufacturer: Optional[str]) -> DemoProfile:
    """The profile that simulates a configuration's ``info.manufacturer``, in any
    spelling, for a configuration with the sim enabled.

    Raises ``NotImplementedError`` for an instrument the Demo cannot simulate yet.
    """
    profile = _PROFILES.get(manufacturers.normalize_manufacturer(manufacturer))
    if profile is None:
        simulated = sorted(n for n in _PROFILES if n != manufacturers.DEMO)
        raise NotImplementedError(
            f"The Demo cannot simulate a {manufacturer} instrument yet. "
            f"Simulated instruments: {', '.join(simulated)}."
        )
    return profile


def demo_profile(manufacturer: Optional[str]) -> DemoProfile:
    """The profile a Demo shows for a configuration's ``info.manufacturer``.

    A manufacturer with no profile gets ThermoFisher's, the Demo as it always was: a
    plugin driver may build the Demo under its own name, and the sim refuses the
    instruments it cannot simulate before it builds one (``simulated_profile``).
    """
    return _PROFILES.get(
        manufacturers.normalize_manufacturer(manufacturer), THERMOFISHER_PROFILE
    )


def demo_profile_of(microscope: object) -> DemoProfile:
    """The profile a Demo device shows: its Demo microscope's. A Demo device on
    another backend (an entry naming ``driver: Demo``) shows ThermoFisher's, as Demo
    devices always did."""
    return getattr(microscope, "profile", None) or THERMOFISHER_PROFILE
