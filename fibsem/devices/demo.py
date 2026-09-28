"""The Demo backend's beams as devices, bound beside the untouched Demo microscope.

Each binding reads and writes what the matching branch of ``DemoMicroscope._get`` and
``_set`` reads and writes, so the old call and the new parameter touch the same state.
This is step 1 of moving a key ("declare and bind"); the Demo chain itself is unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from fibsem.devices.beam import Beam
from fibsem.devices.core import ParamMeta, Resources
from fibsem.structures import BeamType

if TYPE_CHECKING:
    from fibsem.microscopes.simulator import DemoMicroscope


def _attribute(owner: Any, name: str, cast: Any = None) -> Dict[str, Any]:
    """read/write for a plain attribute, as the Demo branches do it."""

    def read() -> Any:
        return getattr(owner, name)

    def write(value: Any) -> None:
        setattr(owner, name, cast(value) if cast is not None else value)

    return {"read": read, "write": write}


def _choices(microscope: DemoMicroscope, key: str, beam_type: BeamType) -> Any:
    """Metadata from the Demo's get_available_values, re-read when a dependency changes."""
    return lambda: ParamMeta(choices=microscope.get_available_values(key, beam_type))


def _blank_writer(microscope: DemoMicroscope, system: Any, beam_type: BeamType) -> Any:
    """The Demo "blanked" branch as it stands, spot burn included."""

    def write(value: bool) -> None:
        system.blanked = value
        if not value and system.scanning_mode == "spot":
            microscope._burn_into_sample_scene(beam_type)

    return write


def _plasma_gas_writer(microscope: DemoMicroscope) -> Any:
    """The Demo "plasma_gas" branch as it stands: an unavailable gas logs and is ignored."""

    def write(value: str) -> None:
        if not microscope.check_available_values("plasma_gas", value, BeamType.ION):
            logging.warning(
                f"Plasma gas {value} not available. Available values: "
                f"{microscope.get_available_values('plasma_gas', BeamType.ION)}"
            )
            return
        logging.info(f"Setting plasma gas to {value}... this may take some time...")
        microscope.system.ion.plasma_gas = value
        logging.info(f"Plasma gas set to {value}.")

    return write


def bind_demo_beams(
    microscope: DemoMicroscope, resources: Optional[Resources] = None
) -> Dict[BeamType, Beam]:
    """Build ``beams[BeamType]`` for a connected Demo microscope."""
    resources = resources if resources is not None else Resources()
    beams = {}
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        system = (
            microscope.electron_system
            if beam_type is BeamType.ELECTRON
            else microscope.ion_system
        )
        beam = Beam(beam_type, parent=microscope, resources=resources)

        beam.bind(
            "voltage",
            **_attribute(system.beam, "voltage"),
            meta=_choices(microscope, "voltage", beam_type),
        )
        beam.bind(
            "current",
            **_attribute(system.beam, "beam_current"),
            meta=_choices(microscope, "current", beam_type),
        )
        beam.bind("working_distance", **_attribute(system.beam, "working_distance"))
        beam.bind("hfw", **_attribute(system.beam, "hfw"))
        # Demo's get returns float(scan_rotation) and its set stores float(value).
        scan_rotation = _attribute(system.beam, "scan_rotation", cast=float)
        beam.bind(
            "scan_rotation",
            read=lambda beam=system.beam: float(beam.scan_rotation),
            write=scan_rotation["write"],
        )
        beam.bind(
            "blanked",
            read=lambda system=system: system.blanked,
            write=_blank_writer(microscope, system, beam_type),
        )
        beam.bind(
            "detector_type",
            **_attribute(system.detector, "type"),
            meta=_choices(microscope, "detector_type", beam_type),
        )
        beam.bind(
            "detector_mode",
            **_attribute(system.detector, "mode"),
            meta=_choices(microscope, "detector_mode", beam_type),
        )
        # Demo has no presets: "preset" stays unbound, so it is absent on the new API
        # while the old set("preset", ...) keeps its no-op through the Demo chain.
        if beam_type is BeamType.ION and microscope.system.ion.plasma:
            beam.bind(
                "plasma_gas",
                read=lambda: microscope.system.ion.plasma_gas,
                write=_plasma_gas_writer(microscope),
                meta=_choices(microscope, "plasma_gas", beam_type),
            )
        beams[beam_type] = beam
    return beams
