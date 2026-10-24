"""The AutoScript connection the Thermo devices and services share.

``ThermoMicroscope`` makes one ``SdbMicroscopeClient`` and wraps it in an
`AutoScriptConnection`, which it hands to every device and service it builds. They make
their SDK calls through it rather than through the microscope.

The beams, the FM and the chamber camera share one active view and one active device,
so whoever sets it last owns it. ``lock`` is the microscope's
``imaging_channel`` lock (its ``_threading_lock``), the same one the devices claim the
channel with; the connection does not make one of its own.

This module imports nothing from the SDK; the Thermo backend's guarded import is the
only one.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

from fibsem.structures.settings import BeamType


class AutoScriptConnection:
    """One AutoScript client: the SDK's ``SdbMicroscopeClient`` and the channel lock.

    A plain driver object, not a device: it has no parameters. ``sdk`` is the client.
    ``vendor_stage`` is the stage connect found (``specimen.stage`` or
    ``specimen.compustage``), with ``compustage_installed`` and the coordinate system
    its positions are read in.
    """

    def __init__(self, sdk: Any, lock: Any):
        self.sdk = sdk
        self.lock = lock
        self.vendor_stage: Any = None
        self.compustage_installed: bool = False
        self.default_stage_coordinate_system: Any = None

    def connect(self, host: str, port: int) -> None:
        self.sdk.connect(host=host, port=port)

    def disconnect(self) -> None:
        self.sdk.disconnect()

    def beam(self, beam_type: BeamType) -> Any:
        """The vendor beam for ``beam_type``: ``beams.electron_beam`` or ``.ion_beam``."""
        if beam_type is BeamType.ELECTRON:
            return self.sdk.beams.electron_beam
        elif beam_type is BeamType.ION:
            return self.sdk.beams.ion_beam
        else:
            raise ValueError(f"Unknown beam type: {beam_type}")

    def set_channel(self, channel: BeamType) -> None:
        """Point the active view and device at ``channel``'s beam."""
        self.sdk.imaging.set_active_view(channel.value)
        self.sdk.imaging.set_active_device(channel.value)
        logging.debug(f"Set active channel to {channel.name}")

    def find_stage(self) -> None:
        """Find the stage, the compustage first, and read it in its default
        coordinate system from now on: specimen on a compustage, raw otherwise."""
        from fibsem.drivers.autoscript.microscope import CoordinateSystem

        specimen = self.sdk.specimen
        if specimen.compustage.is_installed:
            self.vendor_stage = specimen.compustage
            self.compustage_installed = True
            self.default_stage_coordinate_system = CoordinateSystem.SPECIMEN
        elif specimen.stage.is_installed:
            self.vendor_stage = specimen.stage
            self.compustage_installed = False
            self.default_stage_coordinate_system = CoordinateSystem.RAW
        else:
            raise Exception(
                "No stage installed. Please check the microscope configuration."
            )

        # set default coordinate system
        self.vendor_stage.set_default_coordinate_system(
            self.default_stage_coordinate_system
        )

    def read_compucentric_offset(self) -> Optional[Tuple[float, float]]:
        """Specimen minus raw stage coordinates, (x, y) metres, as xT reports them:
        where xT centres its compucentric rotation. None on a compustage, which does
        not rotate."""
        from fibsem.drivers.autoscript.microscope import (
            CoordinateSystem,
            stage_position_from_autoscript,
        )

        if self.compustage_installed:
            return None
        stage = self.vendor_stage
        try:
            stage.set_default_coordinate_system(CoordinateSystem.SPECIMEN)
            specimen = stage_position_from_autoscript(stage.current_position)
            stage.set_default_coordinate_system(CoordinateSystem.RAW)
            raw = stage_position_from_autoscript(stage.current_position)
        finally:
            stage.set_default_coordinate_system(self.default_stage_coordinate_system)
        offset = (specimen.x - raw.x, specimen.y - raw.y)
        logging.info(
            f"Compucentric rotation offset (specimen - raw): "
            f"x={offset[0] * 1e6:.2f} um, y={offset[1] * 1e6:.2f} um."
        )
        return offset
