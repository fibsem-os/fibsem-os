"""The Demo backend, rebuilt from devices.

``DeviceDemoMicroscope`` is the demo microscope the device migration grows, one
device at a time, beside the untouched ``DemoMicroscope``. Each device it builds
takes over its keys through ``FibsemMicroscope.get``/``set``; every key that has
not moved yet still goes to the Demo chain, which this class inherits for now.
When every key and method has moved, the inheritance goes, and this class takes
the ``DemoMicroscope`` name.

``tests/test_microscope_contract.py`` runs the same contract against both, and
compares them call by call, so the two can't drift while both exist.

Select it with ``sim: {devices: true}`` in a Demo configuration.

Devices so far: the beams (``DemoBeam``) and the stage (``DemoStage``), from
``fibsem.devices.drivers.demo``. The stage is ``stage_device``, a temporary name
until the stage redesign settles it; its keys are not routed, the stage methods
use it directly.
"""

from __future__ import annotations

from types import MappingProxyType

from fibsem.devices.beam import BEAM_ROUTES
from fibsem.devices.drivers.demo import bind_demo_beams, bind_demo_stage
from fibsem.microscope import _records_stage_move
from fibsem.microscopes.simulator import DemoMicroscope
from fibsem.structures import FibsemStagePosition


class DeviceDemoMicroscope(DemoMicroscope):
    def connect_to_microscope(
        self, ip_address: str, port: int = 8080, reset_beam_shift: bool = True
    ) -> None:
        super().connect_to_microscope(ip_address, port, reset_beam_shift)
        self._connect_devices()

    def _connect_devices(self) -> None:
        """Build the devices and route their keys to them."""
        self.beams = MappingProxyType(bind_demo_beams(self))
        self._beam_routes = MappingProxyType(dict(BEAM_ROUTES))
        self.stage_device = bind_demo_stage(self)

    # The old moves go through the device without its limit check, as today.

    @_records_stage_move
    def move_stage_absolute(self, position: FibsemStagePosition) -> FibsemStagePosition:
        self.stage_device.move_through(position)
        return self.get_stage_position()

    def home(self) -> None:
        # Demo's own override returns None, and the old API keeps that.
        self.stage_device.home()

    @_records_stage_move
    def move_stage_relative(self, position: FibsemStagePosition) -> FibsemStagePosition:
        self.stage_device.move_through(position, relative=True)
        return self.get_stage_position()
