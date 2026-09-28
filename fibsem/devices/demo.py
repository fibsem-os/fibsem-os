"""The Demo backend's beams as devices, beside the untouched Demo microscope.

``DemoBeam`` implements each parameter with what the matching branch of
``DemoMicroscope._get`` and ``_set`` reads and writes, so the old call and the new
parameter touch the same state. This is step 1 of moving a key ("declare and
implement"); the Demo chain itself is unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional

from fibsem.devices.beam import Beam
from fibsem.devices.core import ParamMeta, Resources
from fibsem.structures import BeamType

if TYPE_CHECKING:
    from fibsem.microscopes.simulator import DemoMicroscope


class DemoBeam(Beam):
    def __init__(
        self,
        beam_type: BeamType,
        parent: DemoMicroscope,
        resources: Optional[Resources] = None,
    ):
        super().__init__(beam_type, parent=parent, resources=resources)
        self._system = (
            parent.electron_system
            if beam_type is BeamType.ELECTRON
            else parent.ion_system
        )

    def _choices(self, key: str) -> ParamMeta:
        return ParamMeta(choices=self.parent.get_available_values(key, self.beam_type))

    # Plain attributes: the Demo branches read and assign them directly.
    working_distance = Beam.working_distance.attribute("_system.beam.working_distance")
    hfw = Beam.hfw.attribute("_system.beam.hfw")
    # Demo's get returns float(scan_rotation) and its set stores float(value).
    scan_rotation = Beam.scan_rotation.attribute(
        "_system.beam.scan_rotation", cast=float
    )

    voltage = Beam.voltage.attribute("_system.beam.voltage")

    @voltage.meta
    def voltage(self) -> ParamMeta:
        return self._choices("voltage")

    current = Beam.current.attribute("_system.beam.beam_current")

    @current.meta
    def current(self) -> ParamMeta:
        return self._choices("current")

    detector_type = Beam.detector_type.attribute("_system.detector.type")

    @detector_type.meta
    def detector_type(self) -> ParamMeta:
        return self._choices("detector_type")

    detector_mode = Beam.detector_mode.attribute("_system.detector.mode")

    @detector_mode.meta
    def detector_mode(self) -> ParamMeta:
        return self._choices("detector_mode")

    # The Demo "blanked" branch as it stands, spot burn included.
    @Beam.blanked.reader
    def blanked(self) -> bool:
        return self._system.blanked

    @blanked.writer
    def blanked(self, value: bool) -> None:
        self._system.blanked = value
        if not value and self._system.scanning_mode == "spot":
            self.parent._burn_into_sample_scene(self.beam_type)

    # Only a plasma ion column has a gas. An unavailable gas logs and is ignored,
    # as the Demo branch does.
    @Beam.plasma_gas.reader
    def plasma_gas(self) -> str:
        return self.parent.system.ion.plasma_gas

    @plasma_gas.writer
    def plasma_gas(self, value: str) -> None:
        microscope = self.parent
        if not microscope.check_available_values("plasma_gas", value, BeamType.ION):
            logging.warning(
                f"Plasma gas {value} not available. Available values: "
                f"{microscope.get_available_values('plasma_gas', BeamType.ION)}"
            )
            return
        logging.info(f"Setting plasma gas to {value}... this may take some time...")
        microscope.system.ion.plasma_gas = value
        logging.info(f"Plasma gas set to {value}.")

    @plasma_gas.meta
    def plasma_gas(self) -> ParamMeta:
        return self._choices("plasma_gas")

    @plasma_gas.available
    def plasma_gas(self) -> bool:
        return self.beam_type is BeamType.ION and self.parent.system.ion.plasma

    # "preset" is not implemented: Demo has no presets, so it is absent on the new
    # API while the old set("preset", ...) keeps its no-op through the Demo chain.


def bind_demo_beams(
    microscope: DemoMicroscope, resources: Optional[Resources] = None
) -> Dict[BeamType, Beam]:
    """Build ``beams[BeamType]`` for a connected Demo microscope."""
    resources = resources if resources is not None else Resources()
    return {
        beam_type: DemoBeam(beam_type, microscope, resources).connect()
        for beam_type in (BeamType.ELECTRON, BeamType.ION)
    }
