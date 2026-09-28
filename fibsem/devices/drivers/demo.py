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
from fibsem.devices.core import ParameterMetadata, Resources
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

    def _choices(self, key: str) -> ParameterMetadata:
        return ParameterMetadata(
            choices=self.parent.get_available_values(key, self.beam_type)
        )

    # Each parameter is the matching Demo branch as it stands.

    def read_voltage(self) -> float:
        return self._system.beam.voltage

    def write_voltage(self, value: float) -> None:
        self._system.beam.voltage = value

    def metadata_voltage(self) -> ParameterMetadata:
        return self._choices("voltage")

    def read_current(self) -> float:
        return self._system.beam.beam_current

    def write_current(self, value: float) -> None:
        self._system.beam.beam_current = value

    def metadata_current(self) -> ParameterMetadata:
        return self._choices("current")

    def read_working_distance(self) -> float:
        return self._system.beam.working_distance

    def write_working_distance(self, value: float) -> None:
        self._system.beam.working_distance = value

    def read_hfw(self) -> float:
        return self._system.beam.hfw

    def write_hfw(self, value: float) -> None:
        self._system.beam.hfw = value

    def read_scan_rotation(self) -> float:
        return float(self._system.beam.scan_rotation)

    def write_scan_rotation(self, value: float) -> None:
        self._system.beam.scan_rotation = float(value)

    def read_blanked(self) -> bool:
        return self._system.blanked

    def write_blanked(self, value: bool) -> None:
        self._system.blanked = value
        if not value and self._system.scanning_mode == "spot":
            self.parent._burn_into_sample_scene(self.beam_type)  # the spot burn

    def read_detector_type(self) -> str:
        return self._system.detector.type

    def write_detector_type(self, value: str) -> None:
        self._system.detector.type = value

    def metadata_detector_type(self) -> ParameterMetadata:
        return self._choices("detector_type")

    def read_detector_mode(self) -> str:
        return self._system.detector.mode

    def write_detector_mode(self, value: str) -> None:
        self._system.detector.mode = value

    def metadata_detector_mode(self) -> ParameterMetadata:
        return self._choices("detector_mode")

    # Only a plasma ion column has a gas.
    def available_plasma_gas(self) -> bool:
        return self.beam_type is BeamType.ION and self.parent.system.ion.plasma

    def read_plasma_gas(self) -> str:
        return self.parent.system.ion.plasma_gas

    def write_plasma_gas(self, value: str) -> None:
        # An unavailable gas logs and is ignored, as the Demo branch does.
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

    def metadata_plasma_gas(self) -> ParameterMetadata:
        return self._choices("plasma_gas")

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
